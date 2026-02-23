def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.nn import functional as F
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet, PolicyHead
    from torch_compile import maybe_compile
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']
    online_policy_cfg = config.get('online_policy', {})
    online_policy_enabled = online and bool(online_policy_cfg.get('enabled', False))
    if online_policy_enabled and online_policy_cfg.get('method', 'awac') != 'awac':
        raise ValueError(f"unsupported online_policy.method: {online_policy_cfg.get('method')}")
    freeze_mortal = bool(online_policy_cfg.get('freeze_mortal', True))
    freeze_dqn = bool(online_policy_cfg.get('freeze_dqn', True))
    train_aux = bool(online_policy_cfg.get('train_aux', False))
    actor_lr = float(online_policy_cfg.get('actor_lr', config['optim']['scheduler']['peak']))
    actor_weight = float(online_policy_cfg.get('actor_weight', 1.0))
    awac_lambda = float(online_policy_cfg.get('awac_lambda', 1.0))
    awac_max_weight = float(online_policy_cfg.get('awac_max_weight', 20.0))
    entropy_coef = float(online_policy_cfg.get('entropy_coef', 0.002))
    bc_coef = float(online_policy_cfg.get('bc_coef', 0.05))
    bc_warmup_steps = int(online_policy_cfg.get('bc_warmup_steps', 5000))
    if online_policy_enabled:
        if awac_lambda <= 0:
            raise ValueError('online_policy.awac_lambda must be > 0')
        if awac_max_weight <= 0:
            raise ValueError('online_policy.awac_max_weight must be > 0')
        if actor_lr <= 0:
            raise ValueError('online_policy.actor_lr must be > 0')

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    policy = PolicyHead(version=version).to(device) if online_policy_enabled else None
    all_models = [mortal, dqn, aux_net]
    if policy is not None:
        all_models.append(policy)
    if enable_compile:
        for m in all_models:
            maybe_compile(m, enable=True, device=device, label=f"train:{m.__class__.__name__}")

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')
    if policy is not None:
        logging.info(f'policy params: {parameter_count(policy):,}')
        logging.info(f'online policy mode: awac, actor_lr={actor_lr:g}, freeze_mortal={freeze_mortal}, freeze_dqn={freeze_dqn}, train_aux={train_aux}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])
    if online_policy_enabled:
        if freeze_mortal:
            mortal.requires_grad_(False)
            mortal.eval()
        if freeze_dqn:
            dqn.requires_grad_(False)
            dqn.eval()
        if not train_aux:
            aux_net.requires_grad_(False)
            aux_net.eval()

    optim_models = [mortal, dqn, aux_net]
    if online_policy_enabled:
        optim_models = [policy]
        if not freeze_mortal:
            optim_models.append(mortal)
        if train_aux:
            optim_models.append(aux_net)
    decay_params = []
    no_decay_params = []
    for model in optim_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler_cfg = dict(config['optim']['scheduler'])
    if online_policy_enabled:
        scheduler_cfg['peak'] = actor_lr
        scheduler_cfg['final'] = min(float(scheduler_cfg['final']), actor_lr)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **scheduler_cfg)
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    policy_meta = {'method': 'awac', 'initialized_from': 'random'} if policy is not None else None
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if policy is not None:
            policy_state = state.get('policy')
            if policy_state is None:
                logging.warning('online_policy is enabled but checkpoint has no `policy`; using random initialization')
            else:
                policy.load_state_dict(policy_state)
                policy_meta = state.get('policy_meta', {'method': 'awac', 'initialized_from': 'checkpoint'})
        if not online or state['config']['control']['online']:
            try:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
            except Exception as ex:
                logging.warning(f'skipped optimizer/scheduler restore due to mismatch: {ex}')
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, policy=policy, is_idle=True)
        logging.info('param has been submitted')

    # Avoid mixed/retrograde curves when resuming from checkpoints in the same
    # tensorboard directory.
    writer = SummaryWriter(config['control']['tensorboard_dir'], purge_step=steps)
    if online_policy_enabled:
        stats = {
            'actor_loss': 0,
            'policy_entropy': 0,
            'awac_weight_mean': 0,
            'awac_weight_max': 0,
            'policy_kl_to_behavior': 0,
        }
        if train_aux:
            stats['next_rank_loss'] = 0
        all_q = None
        all_q_target = None
    else:
        stats = {
            'dqn_loss': 0,
            'cql_loss': 0,
            'next_rank_loss': 0,
        }
        all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
        all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                if online_policy_enabled:
                    policy_logits = policy(phi).masked_fill(~masks, -torch.inf)
                    log_probs = F.log_softmax(policy_logits, dim=-1)
                    probs = log_probs.exp()
                    safe_log_probs = torch.where(
                        masks,
                        log_probs,
                        torch.zeros_like(log_probs),
                    )
                    chosen_logp = log_probs[range(batch_size), actions]
                    policy_kl_to_behavior = -chosen_logp.mean()
                    bc_loss = (
                        policy_kl_to_behavior
                        if steps < bc_warmup_steps
                        else chosen_logp.new_zeros(())
                    )
                    # Use no_grad (not inference_mode) here: awac_weight is
                    # consumed by actor_loss backward as a constant tensor.
                    with torch.no_grad():
                        safe_q = q_out.masked_fill(~masks, 0.0)
                        v_est = (probs.detach() * safe_q.detach()).sum(-1)
                        adv = q.detach() - v_est
                        awac_weight = torch.exp(adv / awac_lambda).clamp(max=awac_max_weight)
                    actor_loss = -(awac_weight * chosen_logp).mean()
                    entropy = -(probs * safe_log_probs).sum(-1).mean()
                    loss = actor_weight * actor_loss - entropy_coef * entropy
                    next_rank_loss = chosen_logp.new_zeros(())
                    if steps < bc_warmup_steps and bc_coef > 0:
                        loss = loss + bc_coef * bc_loss
                    if train_aux:
                        next_rank_logits, = aux_net(phi)
                        next_rank_loss = ce(next_rank_logits, player_ranks)
                        loss = loss + next_rank_loss * next_rank_weight
                else:
                    q_target_mc = gamma ** steps_to_done * kyoku_rewards
                    q_target_mc = q_target_mc.to(torch.float32)

                    dqn_loss = 0.5 * mse(q, q_target_mc)
                    cql_loss = 0
                    if not online:
                        cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                    next_rank_logits, = aux_net(phi)
                    next_rank_loss = ce(next_rank_logits, player_ranks)

                    loss = sum((
                        dqn_loss,
                        cql_loss * min_q_weight,
                        next_rank_loss * next_rank_weight,
                    ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                if online_policy_enabled:
                    stats['actor_loss'] += actor_loss
                    stats['policy_entropy'] += entropy
                    stats['awac_weight_mean'] += awac_weight.mean()
                    stats['awac_weight_max'] += awac_weight.max()
                    stats['policy_kl_to_behavior'] += policy_kl_to_behavior
                    if train_aux:
                        stats['next_rank_loss'] += next_rank_loss
                else:
                    stats['dqn_loss'] += dqn_loss
                    stats['cql_loss'] += cql_loss
                    stats['next_rank_loss'] += next_rank_loss
                    all_q[idx] = q
                    all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, policy=policy, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                if online_policy_enabled:
                    writer.add_scalar('loss/actor_loss', stats['actor_loss'] / save_every, steps)
                    writer.add_scalar('loss/policy_loss', stats['actor_loss'] / save_every, steps)
                    writer.add_scalar('policy/entropy', stats['policy_entropy'] / save_every, steps)
                    writer.add_scalar('policy/awac_weight_mean', stats['awac_weight_mean'] / save_every, steps)
                    writer.add_scalar('policy/awac_weight_max', stats['awac_weight_max'] / save_every, steps)
                    writer.add_scalar('policy/kl_to_behavior', stats['policy_kl_to_behavior'] / save_every, steps)
                    if train_aux:
                        writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                    writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                else:
                    # downsample to reduce tensorboard event size
                    all_q_1d = all_q.cpu().numpy().flatten()[::128]
                    all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]
                    writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                    writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                    writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                    writer.add_histogram('q_predicted', all_q_1d, steps)
                    writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                if policy is not None:
                    state['policy'] = policy.state_dict()
                    state['policy_meta'] = policy_meta
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, policy=policy, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, policy, device)
                    mortal.train()
                    dqn.train()
                    if policy is not None:
                        policy.train()
                    if online_policy_enabled:
                        if freeze_mortal:
                            mortal.eval()
                        if freeze_dqn:
                            dqn.eval()
                        if not train_aux:
                            aux_net.eval()

                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    # Always keep a test checkpoint with explicit step suffix.
                    state['best_perf'] = best_perf
                    state['timestamp'] = datetime.now().timestamp()
                    torch.save(state, state_file)
                    snapshot_file = f'{path.splitext(state_file)[0]}_step{steps}.pth'
                    torch.save(state, snapshot_file)
                    logging.info(f'saved test snapshot: {snapshot_file}')

                    if better:
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(mortal, dqn, policy=policy, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
