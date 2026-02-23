import prelude


def _load_manifest(manifest_file):
    import json
    from os import path

    out = {}
    if not manifest_file or not path.isfile(manifest_file):
        return out
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if 'file' in row:
                out[row['file']] = row
    return out


def train():
    import gc
    import logging
    import os
    from os import path

    import torch
    from torch import nn, optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from ach_rvr_utils import (
        checkpoint_snapshot_name,
        load_legacy_states,
        maybe_load_state,
        save_ach_rvr_state,
    )
    from common import drain, submit_param
    from config import config
    from dataloader_ach_rvr import AchRvrFileDataset, collate_ach_rvr, worker_init_fn
    from dataloader_ach_rvr_strict import (
        StrictAchRvrFileDataset,
        collate_ach_rvr_strict,
    )
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import DQN, PolicyHead
    from model_ach_rvr import (
        AchRvrModel,
        centered_clipped_y,
        gather_log_prob,
        hedge_policy_from_y,
        masked_entropy,
        zero_sum_utility,
    )
    from periodic_eval import run_periodic_eval

    device = torch.device(config['control']['device'])
    enable_amp = bool(config['control']['enable_amp'])
    torch.backends.cudnn.benchmark = bool(config['control']['enable_cudnn_benchmark'])

    version = int(config['control']['version'])
    batch_size = int(config['control']['batch_size'])
    opt_step_every = int(config['control']['opt_step_every'])
    save_every = int(config['control']['save_every'])
    submit_every = int(config['control'].get('submit_every', save_every))
    round_steps_limit = int(config['control'].get('online_round_steps', save_every))
    max_steps = int(config['optim']['scheduler']['max_steps'])
    max_grad_norm = float(config['optim'].get('max_grad_norm', 0.0))

    ach_cfg = config.get('ach', {})
    eta = float(ach_cfg.get('eta', 1.0))
    imp_clip = float(ach_cfg.get('imp_clip', 0.2))
    logit_clip = float(ach_cfg.get('logit_clip', 8.0))
    entropy_coef = float(ach_cfg.get('entropy_coef', 0.003))
    value_coef = float(ach_cfg.get('value_coef', 0.5))
    gamma = float(ach_cfg.get('gamma', 1.0))
    gae_lambda = float(ach_cfg.get('gae_lambda', 0.95))
    adv_norm = bool(ach_cfg.get('adv_norm', True))
    freeze_private_brain = bool(ach_cfg.get('freeze_private_brain', False))
    freeze_value_head = bool(ach_cfg.get('freeze_value_head', False))

    strict_cfg = config.get('strict', {})
    strict_enabled = bool(strict_cfg.get('enabled', False))
    strict_require_trace = bool(strict_cfg.get('require_trace', True))
    strict_require_old_logp = bool(strict_cfg.get('require_old_logp', True))
    strict_fail_fast = bool(strict_cfg.get('fail_fast', True))
    strict_allow_synth_trace = bool(strict_cfg.get('allow_synth_trace', False))
    strict_allow_missing_old_logp = bool(strict_cfg.get('allow_missing_old_logp', False))
    strict_synth_mismatch_policy = str(strict_cfg.get('synth_mismatch_policy', 'fail'))
    strict_mismatch_report_file = str(strict_cfg.get('mismatch_report_file', ''))
    if strict_enabled:
        logging.info(
            "strict mode enabled: Algorithm-2 loss + trace-required + old_logp-required"
        )
        if not strict_require_trace:
            raise RuntimeError('strict mode requires require_trace=true')
        if not strict_require_old_logp:
            raise RuntimeError('strict mode requires require_old_logp=true')
        if strict_allow_synth_trace:
            raise RuntimeError('strict mode requires allow_synth_trace=false')
        if strict_allow_missing_old_logp:
            raise RuntimeError('strict mode requires allow_missing_old_logp=false')
        if strict_synth_mismatch_policy != 'fail':
            raise RuntimeError("strict mode requires synth_mismatch_policy='fail'")

    rvr_cfg = config.get('rvr', {})
    rvr_enabled = bool(rvr_cfg.get('enabled', True))
    use_oracle_value = bool(rvr_cfg.get('use_oracle_value', False))
    use_expected_reward = bool(rvr_cfg.get('use_expected_reward', False))
    ern_preterminal_only = bool(rvr_cfg.get('ern_preterminal_only', True))
    reward_blend_alpha = float(rvr_cfg.get('reward_blend_alpha', 0.0))
    zero_sum_coef = float(rvr_cfg.get('zero_sum_coef', 0.05))
    rv_coef = float(rvr_cfg.get('rv_coef', 0.5))
    ern_coef = float(rvr_cfg.get('ern_coef', 0.5))
    include_oracle_obs = bool(config.get('oracle_dataset', {}).get('include_invisible_obs', True))
    if rvr_enabled and (use_oracle_value or use_expected_reward) and not include_oracle_obs:
        raise ValueError('rvr oracle heads are enabled but oracle_dataset.include_invisible_obs=false')

    model = AchRvrModel(version=version, **config['resnet']).to(device)

    model.private_brain.requires_grad_(not freeze_private_brain)
    model.policy_y.requires_grad_(True)
    model.value_head.requires_grad_(not freeze_value_head)
    model.oracle_brain.requires_grad_(rvr_enabled and (use_oracle_value or use_expected_reward))
    model.rv_head.requires_grad_(rvr_enabled and use_oracle_value)
    model.ern.requires_grad_(rvr_enabled and use_expected_reward)
    logging.info(
        "online joint update: private_brain=%s value_head=%s policy_y=True",
        str(not freeze_private_brain),
        str(not freeze_value_head),
    )

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        train_params,
        lr=1.0,
        betas=tuple(config['optim']['betas']),
        eps=float(config['optim']['eps']),
        weight_decay=float(config['optim']['weight_decay']),
    )
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    state_file = config['control']['state_file']
    best_state_file = config['control'].get('best_state_file', state_file)
    bootstrap_file = config['control'].get('bootstrap_state_file', '')

    steps = 0
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}
    policy_meta = {
        'method': 'ach+rvr',
        'initialized_from': 'random',
        'eta': eta,
        'imp_clip': imp_clip,
        'logit_clip': logit_clip,
    }
    league_meta = {}

    proxy_dqn = DQN(version=version).to(device).eval()
    actor_policy = PolicyHead(version=version).to(device).eval()

    if path.exists(state_file):
        steps, best_perf, policy_meta, league_meta = maybe_load_state(
            state_file,
            map_location=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        loaded = torch.load(state_file, weights_only=False, map_location='cpu')
        if 'current_dqn' in loaded:
            proxy_dqn.load_state_dict(loaded['current_dqn'])
    elif bootstrap_file and path.exists(bootstrap_file):
        loaded = torch.load(bootstrap_file, weights_only=False, map_location='cpu')
        load_legacy_states(loaded, model, strict=False)
        if 'current_dqn' in loaded:
            proxy_dqn.load_state_dict(loaded['current_dqn'])
        policy_meta = {
            'method': 'ach+rvr',
            'initialized_from': bootstrap_file,
            'eta': eta,
            'imp_clip': imp_clip,
            'logit_clip': logit_clip,
        }
        logging.info(f'bootstrapped from: {bootstrap_file}')
    else:
        logging.warning('No existing or bootstrap checkpoint. DQN proxy is random; use policy decision_head for clients.')

    # Keep actor policy head synchronized from policy_y for client-side rollout.
    actor_policy.load_state_dict(model.policy_y.state_dict(), strict=False)

    def submit_actor(is_idle):
        actor_policy.load_state_dict(model.policy_y.state_dict(), strict=False)
        submit_param(model.private_brain, proxy_dqn, policy=actor_policy, is_idle=is_idle)

    submit_actor(is_idle=True)
    logging.info('initial params submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'], purge_step=steps)
    mse = nn.MSELoss()

    stats = {
        'actor_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'ratio_mean': 0.0,
    }
    if strict_enabled:
        stats['strict_ratio_mean'] = 0.0
        stats['strict_c_gate_rate_pos'] = 0.0
        stats['strict_c_gate_rate_neg'] = 0.0
        stats['strict_old_logp_coverage'] = 0.0
        stats['strict_four_seat_coverage'] = 0.0
    if rvr_enabled and use_oracle_value:
        stats['rv_loss'] = 0.0
        stats['rv_zero_sum'] = 0.0
    if rvr_enabled and use_expected_reward:
        stats['ern_loss'] = 0.0
        stats['ern_zero_sum'] = 0.0
        stats['ern_preterminal_coverage'] = 0.0

    optimizer.zero_grad(set_to_none=True)

    while steps < max_steps:
        drain_dir, manifest_file = drain(with_manifest=True)
        file_list = [
            path.join(drain_dir, name)
            for name in os.listdir(drain_dir)
            if name.endswith('.json.gz')
        ]
        if not file_list:
            logging.info('drain returned no log files, waiting next round')
            continue
        manifest_map = _load_manifest(manifest_file)
        if manifest_map:
            opponent_ids = {}
            for row in manifest_map.values():
                oid = row.get('opponent_id', '')
                opponent_ids[oid] = opponent_ids.get(oid, 0) + 1
            league_meta['last_round'] = {
                'files': len(file_list),
                'manifest_rows': len(manifest_map),
                'opponents': opponent_ids,
            }
        logging.info(f'online round files: {len(file_list):,}, manifest rows: {len(manifest_map):,}')
        writer.add_scalar('online/round_files', len(file_list), steps)
        writer.add_scalar('online/manifest_rows', len(manifest_map), steps)

        if strict_enabled:
            ds = StrictAchRvrFileDataset(
                version=version,
                file_list=file_list,
                pts=config['env']['pts'],
                gamma=gamma,
                gae_lambda=gae_lambda,
                reward_source=rvr_cfg.get('reward_source', 'grp_plus_ern'),
                oracle=include_oracle_obs,
                file_batch_size=int(config['dataset']['file_batch_size']),
                reserve_ratio=float(config['dataset'].get('reserve_ratio', 0.0)),
                player_names=[],
                excludes=None,
                num_epochs=int(config['dataset']['num_epochs']),
                enable_augmentation=bool(config['dataset'].get('enable_augmentation', False)),
                augmented_first=bool(config['dataset'].get('augmented_first', False)),
                include_terminal_context=bool(config.get('oracle_dataset', {}).get('include_terminal_context', True)),
                manifest_map=manifest_map,
                strict_require_trace=strict_require_trace,
                strict_require_old_logp=strict_require_old_logp,
                allow_synth_trace=strict_allow_synth_trace,
                allow_missing_old_logp=strict_allow_missing_old_logp,
                synth_mismatch_policy=strict_synth_mismatch_policy,
                mismatch_report_file=strict_mismatch_report_file,
            )
            collate_fn = collate_ach_rvr_strict
        else:
            ds = AchRvrFileDataset(
                version=version,
                file_list=file_list,
                pts=config['env']['pts'],
                gamma=gamma,
                reward_source=rvr_cfg.get('reward_source', 'grp_plus_ern'),
                oracle=include_oracle_obs,
                file_batch_size=int(config['dataset']['file_batch_size']),
                reserve_ratio=float(config['dataset'].get('reserve_ratio', 0.0)),
                player_names=['trainee'],
                excludes=None,
                num_epochs=int(config['dataset']['num_epochs']),
                enable_augmentation=bool(config['dataset'].get('enable_augmentation', False)),
                augmented_first=bool(config['dataset'].get('augmented_first', False)),
                include_terminal_context=bool(config.get('oracle_dataset', {}).get('include_terminal_context', True)),
                manifest_map=manifest_map,
            )
            collate_fn = collate_ach_rvr
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=int(config['dataset']['num_workers']),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

        round_steps = 0
        for batch in dl:
            obs = batch['obs'].to(device)
            actions = batch['actions'].to(device)
            masks = batch['masks'].to(device)
            returns = batch['returns'].to(device)
            advantages = batch['advantages'].to(device)
            can_act = batch['can_act'].to(device) if 'can_act' in batch else (actions >= 0)
            valid_actions = can_act & (actions >= 0)
            if strict_enabled and 'decision_head' in batch:
                policy_rows = torch.as_tensor(
                    [h == 'policy' for h in batch['decision_head']],
                    dtype=torch.bool,
                    device=device,
                )
                valid_actions = valid_actions & policy_rows
            if strict_enabled and not bool(valid_actions.any()):
                continue
            fallback_actions = masks.to(torch.int64).argmax(-1)
            safe_actions = torch.where(valid_actions, actions, fallback_actions)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = model.forward_private(obs)
                y = model.policy_y(phi)
                y_center = centered_clipped_y(y, masks, logit_clip)
                probs, log_probs = hedge_policy_from_y(y, masks, eta=eta, logit_clip=logit_clip)
                chosen_logp = gather_log_prob(log_probs, safe_actions)

                value_pred = model.value_head(phi).squeeze(-1)
                target_return = returns
                if (
                    rvr_enabled
                    and use_expected_reward
                    and reward_blend_alpha > 0
                    and 'invisible_obs' in batch
                ):
                    keep = batch['is_preterminal'].to(device)
                    if bool(keep.any()):
                        phi_oracle_for_blend = model.forward_oracle(obs[keep], batch['invisible_obs'].to(device)[keep])
                        ern_pred = model.ern(phi_oracle_for_blend)
                        seat = batch['seat'].to(device)[keep]
                        ern_self = ern_pred.gather(-1, seat.unsqueeze(-1)).squeeze(-1)
                        target_return = target_return.clone()
                        target_return[keep] = (
                            (1.0 - reward_blend_alpha) * target_return[keep]
                            + reward_blend_alpha * ern_self.detach()
                        )

                adv = advantages.clone()
                if adv_norm and bool(valid_actions.any()):
                    adv_valid = adv[valid_actions]
                    adv_normed = (adv_valid - adv_valid.mean()) / adv_valid.std(unbiased=False).clamp_min(1e-6)
                    adv = torch.zeros_like(adv)
                    adv[valid_actions] = adv_normed

                if strict_enabled:
                    if 'old_logp' not in batch:
                        if strict_fail_fast:
                            raise RuntimeError('strict mode requires old_logp in every batch')
                        old_logp = chosen_logp.detach()
                    else:
                        old_logp = batch['old_logp'].to(device)

                    finite_old = torch.isfinite(old_logp)
                    if (
                        strict_require_old_logp
                        and strict_fail_fast
                        and (not strict_allow_missing_old_logp)
                        and bool(valid_actions.any())
                    ):
                        if not bool(finite_old[valid_actions].all()):
                            raise RuntimeError('strict mode detected missing/invalid old_logp on can_act rows')

                    safe_old_logp = torch.where(valid_actions & finite_old, old_logp, chosen_logp.detach())
                    ratio_raw = torch.exp(chosen_logp - safe_old_logp)
                    ratio = ratio_raw.clamp(min=1.0 - imp_clip, max=1.0 + imp_clip)

                    batch_idx = torch.arange(actions.shape[0], device=device)
                    y_selected = y_center[batch_idx, safe_actions]
                    old_prob = safe_old_logp.exp().clamp_min(1e-8)

                    pos_adv = adv >= 0
                    c_gate = torch.where(
                        pos_adv,
                        (ratio_raw < (1.0 + imp_clip)) & (y_selected < logit_clip),
                        (ratio_raw > (1.0 - imp_clip)) & (y_selected > -logit_clip),
                    ) & valid_actions

                    actor_term = torch.zeros_like(adv)
                    gated = c_gate
                    actor_term[gated] = (y_selected[gated] / old_prob[gated]) * adv[gated].detach()
                    denom = valid_actions.float().sum().clamp_min(1.0)
                    actor_loss = -(actor_term.sum() / denom)

                    if bool(valid_actions.any()):
                        value_loss = 0.5 * (value_pred[valid_actions] - target_return[valid_actions]).pow(2).mean()
                    else:
                        value_loss = target_return.new_zeros(())

                    safe_log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
                    entropy_vec = (probs * safe_log_probs).sum(-1)
                    if bool(valid_actions.any()):
                        entropy = entropy_vec[valid_actions].mean()
                    else:
                        entropy = target_return.new_zeros(())
                    total_loss = actor_loss + value_coef * value_loss + entropy_coef * entropy

                    strict_ratio_mean = (
                        ratio[valid_actions].mean() if bool(valid_actions.any()) else target_return.new_ones(())
                    )
                    pos_mask = valid_actions & (adv >= 0)
                    neg_mask = valid_actions & (adv < 0)
                    strict_c_gate_rate_pos = (
                        c_gate[pos_mask].to(torch.float32).mean()
                        if bool(pos_mask.any())
                        else target_return.new_ones(())
                    )
                    strict_c_gate_rate_neg = (
                        c_gate[neg_mask].to(torch.float32).mean()
                        if bool(neg_mask.any())
                        else target_return.new_ones(())
                    )
                    strict_old_cov = (
                        finite_old[valid_actions].to(torch.float32).mean()
                        if bool(valid_actions.any())
                        else target_return.new_ones(())
                    )
                    strict_four_cov = target_return.new_tensor(
                        1.0 if bool(batch.get('aligned_four_seat', False)) else 0.0
                    )
                else:
                    if batch.get('has_old_logp', False) and 'old_logp' in batch:
                        old_logp = batch['old_logp'].to(device)
                        ratio = torch.exp(chosen_logp - old_logp)
                        ratio = ratio.clamp(min=1.0 - imp_clip, max=1.0 + imp_clip)
                    else:
                        ratio = torch.ones_like(chosen_logp)

                    if bool(valid_actions.any()):
                        actor_loss = -(ratio[valid_actions] * adv[valid_actions].detach() * chosen_logp[valid_actions]).mean()
                        value_loss = 0.5 * mse(value_pred[valid_actions], target_return[valid_actions])
                        entropy = masked_entropy(probs, log_probs, masks)[valid_actions].mean()
                    else:
                        actor_loss = target_return.new_zeros(())
                        value_loss = target_return.new_zeros(())
                        entropy = target_return.new_zeros(())
                    total_loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

                rv_loss = value_loss.new_zeros(())
                rv_zs = value_loss.new_zeros(())
                ern_loss = value_loss.new_zeros(())
                ern_zs = value_loss.new_zeros(())
                ern_preterminal_cov = value_loss.new_zeros(())
                if rvr_enabled and 'invisible_obs' in batch:
                    invisible = batch['invisible_obs'].to(device)
                    phi_oracle = model.forward_oracle(obs, invisible)

                    if use_oracle_value:
                        target_vec = zero_sum_utility(batch['final_reward_vec'].to(device))
                        rv_pred = model.rv_head(phi_oracle)
                        rv_loss = mse(rv_pred, target_vec)
                        rv_zs = rv_pred.sum(-1).pow(2).mean()
                        total_loss = total_loss + rv_coef * rv_loss + zero_sum_coef * rv_zs

                    if use_expected_reward:
                        target_vec = zero_sum_utility(batch['final_reward_vec'].to(device))
                        if ern_preterminal_only:
                            keep = batch['is_preterminal'].to(device)
                            ern_preterminal_cov = keep.to(torch.float32).mean()
                            if bool(keep.any()):
                                ern_pred = model.ern(phi_oracle[keep])
                                ern_target = target_vec[keep]
                                ern_loss = mse(ern_pred, ern_target)
                                ern_zs = ern_pred.sum(-1).pow(2).mean()
                                total_loss = total_loss + ern_coef * ern_loss + zero_sum_coef * ern_zs
                            else:
                                ern_loss = value_loss.new_zeros(())
                                ern_zs = value_loss.new_zeros(())
                        else:
                            ern_preterminal_cov = value_loss.new_ones(())
                            ern_pred = model.ern(phi_oracle)
                            ern_loss = mse(ern_pred, target_vec)
                            ern_zs = ern_pred.sum(-1).pow(2).mean()
                            total_loss = total_loss + ern_coef * ern_loss + zero_sum_coef * ern_zs

            if not torch.isfinite(total_loss).item():
                raise RuntimeError(
                    f'non-finite total_loss at step={steps + 1}: '
                    f'actor={float(actor_loss.detach())}, '
                    f'value={float(value_loss.detach())}, '
                    f'entropy={float(entropy.detach())}, '
                    f'ratio={float(ratio.mean().detach())}, '
                    f'rv={float(rv_loss.detach())}, '
                    f'ern={float(ern_loss.detach())}'
                )

            scaler.scale(total_loss / opt_step_every).backward()

            steps += 1
            round_steps += 1
            if steps % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(train_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            stats['actor_loss'] += float(actor_loss.detach())
            stats['value_loss'] += float(value_loss.detach())
            stats['entropy'] += float(entropy.detach())
            stats['ratio_mean'] += float(ratio.mean().detach())
            if strict_enabled:
                stats['strict_ratio_mean'] += float(strict_ratio_mean.detach())
                stats['strict_c_gate_rate_pos'] += float(strict_c_gate_rate_pos.detach())
                stats['strict_c_gate_rate_neg'] += float(strict_c_gate_rate_neg.detach())
                stats['strict_old_logp_coverage'] += float(strict_old_cov.detach())
                stats['strict_four_seat_coverage'] += float(strict_four_cov.detach())
            if rvr_enabled and use_oracle_value:
                stats['rv_loss'] += float(rv_loss.detach())
                stats['rv_zero_sum'] += float(rv_zs.detach())
            if rvr_enabled and use_expected_reward:
                stats['ern_loss'] += float(ern_loss.detach())
                stats['ern_zero_sum'] += float(ern_zs.detach())
                stats['ern_preterminal_coverage'] += float(ern_preterminal_cov.detach())

            if steps % submit_every == 0:
                submit_actor(is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                writer.add_scalar('loss/actor_loss', stats['actor_loss'] / save_every, steps)
                writer.add_scalar('loss/value_loss', stats['value_loss'] / save_every, steps)
                writer.add_scalar('policy/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('policy/ratio_mean', stats['ratio_mean'] / save_every, steps)
                if strict_enabled:
                    writer.add_scalar('strict_ach/ratio_mean', stats['strict_ratio_mean'] / save_every, steps)
                    writer.add_scalar('strict_ach/c_gate_rate_pos', stats['strict_c_gate_rate_pos'] / save_every, steps)
                    writer.add_scalar('strict_ach/c_gate_rate_neg', stats['strict_c_gate_rate_neg'] / save_every, steps)
                    writer.add_scalar('strict_ach/old_logp_coverage', stats['strict_old_logp_coverage'] / save_every, steps)
                    writer.add_scalar('aligned/four_seat_coverage', stats['strict_four_seat_coverage'] / save_every, steps)
                if rvr_enabled and use_oracle_value:
                    writer.add_scalar('rv/loss', stats['rv_loss'] / save_every, steps)
                    writer.add_scalar('rv/zero_sum_penalty', stats['rv_zero_sum'] / save_every, steps)
                if rvr_enabled and use_expected_reward:
                    writer.add_scalar('ern/loss', stats['ern_loss'] / save_every, steps)
                    writer.add_scalar('ern/zero_sum_penalty', stats['ern_zero_sum'] / save_every, steps)
                    writer.add_scalar('ern/preterminal_coverage', stats['ern_preterminal_coverage'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.flush()

                save_ach_rvr_state(
                    filename=state_file,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    steps=steps,
                    best_perf=best_perf,
                    policy_meta=policy_meta,
                    league_meta=league_meta,
                    extra={'current_dqn': proxy_dqn.state_dict()},
                )
                snapshot = checkpoint_snapshot_name(state_file, steps)
                save_ach_rvr_state(
                    filename=snapshot,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    steps=steps,
                    best_perf=best_perf,
                    policy_meta=policy_meta,
                    league_meta=league_meta,
                    extra={'current_dqn': proxy_dqn.state_dict()},
                )
                try:
                    run_periodic_eval(
                        config=config,
                        stage_name='online_strict',
                        steps=steps,
                        current_model_path=snapshot,
                        writer=writer,
                    )
                except Exception:
                    logging.exception(f'periodic eval failed at step={steps}')

                if best_state_file and steps > 0:
                    # Keep latest as best placeholder until external evaluator updates best_perf.
                    save_ach_rvr_state(
                        filename=best_state_file,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_perf=best_perf,
                        policy_meta=policy_meta,
                        league_meta=league_meta,
                        extra={'current_dqn': proxy_dqn.state_dict()},
                    )

                stats = {k: 0.0 for k in stats}
                logging.info(f'total steps: {steps:,}')

            if steps >= max_steps or round_steps >= round_steps_limit:
                break

        submit_actor(is_idle=True)
        logging.info('idle params submitted')

        gc.collect()

    save_ach_rvr_state(
        filename=state_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        steps=steps,
        best_perf=best_perf,
        policy_meta=policy_meta,
        league_meta=league_meta,
        extra={'current_dqn': proxy_dqn.state_dict()},
    )


def main():
    train()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
