import prelude


def _blend_alpha(rvr_cfg: dict, step: int) -> float:
    end = float(rvr_cfg.get('reward_blend_alpha', 0.0))
    start = float(rvr_cfg.get('reward_blend_alpha_start', 0.0))
    warmup = int(rvr_cfg.get('reward_blend_alpha_warmup_steps', 0))
    if warmup <= 0:
        return end
    t = min(1.0, max(0.0, step / float(warmup)))
    return start + (end - start) * t


def train():
    import gc
    import logging
    from os import path

    import torch
    from torch import nn, optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from ach_rvr_utils import (
        build_file_list,
        checkpoint_snapshot_name,
        load_legacy_states,
        maybe_load_state,
        save_ach_rvr_state,
    )
    from config import config
    from dataloader_ach_rvr import AchRvrFileDataset, collate_ach_rvr, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model_ach_rvr import (
        AchRvrModel,
        gather_log_prob,
        hedge_policy_from_y,
        zero_sum_utility,
    )
    from periodic_eval import run_periodic_eval

    device = torch.device(config['control']['device'])
    enable_amp = bool(config['control']['enable_amp'])
    torch.backends.cudnn.benchmark = bool(config['control']['enable_cudnn_benchmark'])

    if bool(config.get('strict', {}).get('enabled', False)):
        raise RuntimeError('offline_teacher requires strict.enabled=false')

    version = int(config['control']['version'])
    batch_size = int(config['control']['batch_size'])
    opt_step_every = int(config['control']['opt_step_every'])
    save_every = int(config['control']['save_every'])
    max_steps = int(config['optim']['scheduler']['max_steps'])
    max_grad_norm = float(config['optim'].get('max_grad_norm', 0.0))

    ach_cfg = config.get('ach', {})
    eta = float(ach_cfg.get('eta', 1.0))
    logit_clip = float(ach_cfg.get('logit_clip', 6.0))
    value_coef = float(ach_cfg.get('value_coef', 0.5))
    gamma = float(ach_cfg.get('gamma', 1.0))
    gae_lambda = float(ach_cfg.get('gae_lambda', 0.95))
    adv_norm = bool(ach_cfg.get('adv_norm', True))
    entropy_coef = float(ach_cfg.get('entropy_coef', 1e-2))

    teacher_cfg = config.get('ach_teacher', {})
    awr_beta = float(teacher_cfg.get('awr_beta', 1.0))
    awr_clip = float(teacher_cfg.get('awr_clip', 20.0))
    actor_weight = float(teacher_cfg.get('actor_weight', 1.0))
    bc_coef = float(teacher_cfg.get('bc_coef', 0.05))

    rvr_cfg = config.get('rvr', {})
    rvr_enabled = bool(rvr_cfg.get('enabled', True))
    use_oracle_value = bool(rvr_cfg.get('use_oracle_value', True))
    use_expected_reward = bool(rvr_cfg.get('use_expected_reward', True))
    ern_preterminal_only = bool(rvr_cfg.get('ern_preterminal_only', True))
    zero_sum_coef = float(rvr_cfg.get('zero_sum_coef', 0.05))
    rv_coef = float(rvr_cfg.get('rv_coef', 0.5))
    ern_coef = float(rvr_cfg.get('ern_coef', 0.5))
    include_oracle_obs = bool(config.get('oracle_dataset', {}).get('include_invisible_obs', True))
    if rvr_enabled and (use_oracle_value or use_expected_reward) and not include_oracle_obs:
        raise ValueError('rvr oracle heads are enabled but oracle_dataset.include_invisible_obs=false')

    model = AchRvrModel(version=version, **config['resnet']).to(device)
    model.private_brain.requires_grad_(True)
    model.policy_y.requires_grad_(True)
    model.value_head.requires_grad_(True)
    model.oracle_brain.requires_grad_(rvr_enabled and (use_oracle_value or use_expected_reward))
    model.rv_head.requires_grad_(rvr_enabled and use_oracle_value)
    model.ern.requires_grad_(rvr_enabled and use_expected_reward)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=1.0,
        betas=tuple(config['optim']['betas']),
        eps=float(config['optim']['eps']),
        weight_decay=float(config['optim']['weight_decay']),
    )
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    state_file = config['control']['state_file']
    bootstrap_file = config['control'].get('bootstrap_state_file', '')

    steps = 0
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}
    policy_meta = {
        'method': 'ach+rvr_teacher',
        'initialized_from': 'random',
        'eta': eta,
        'logit_clip': logit_clip,
        'awr_beta': awr_beta,
    }
    league_meta = {}
    legacy_current_dqn = None
    legacy_aux_net = None

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
        legacy_current_dqn = loaded.get('current_dqn')
        legacy_aux_net = loaded.get('aux_net')
    elif bootstrap_file and path.exists(bootstrap_file):
        boot = torch.load(bootstrap_file, weights_only=False, map_location='cpu')
        load_legacy_states(boot, model, strict=False)
        policy_meta = {
            'method': 'ach+rvr_teacher',
            'initialized_from': bootstrap_file,
            'eta': eta,
            'logit_clip': logit_clip,
            'awr_beta': awr_beta,
        }
        legacy_current_dqn = boot.get('current_dqn')
        legacy_aux_net = boot.get('aux_net')
        logging.info(f'bootstrapped from: {bootstrap_file}')

    file_list = build_file_list(config['dataset'], shuffle=True)
    logging.info('offline teacher mode: privileged learning with raw_score + RVN + ERN')
    logging.info(f'device: {device}')
    logging.info(f'file list size: {len(file_list):,}')

    ds = AchRvrFileDataset(
        version=version,
        file_list=file_list,
        pts=config['env']['pts'],
        gamma=gamma,
        reward_source=rvr_cfg.get('reward_source', 'raw_score'),
        oracle=include_oracle_obs,
        file_batch_size=int(config['dataset']['file_batch_size']),
        reserve_ratio=float(config['dataset']['reserve_ratio']),
        player_names=[],
        excludes=None,
        num_epochs=int(config['dataset']['num_epochs']),
        enable_augmentation=bool(config['dataset']['enable_augmentation']),
        augmented_first=bool(config['dataset']['augmented_first']),
        include_terminal_context=bool(config.get('oracle_dataset', {}).get('include_terminal_context', True)),
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=int(config['dataset']['num_workers']),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_ach_rvr,
    )

    writer = SummaryWriter(config['control']['tensorboard_dir'], purge_step=steps)
    mse = nn.MSELoss()
    stats = {
        'actor_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'bc_loss': 0.0,
        'aw_weight_mean': 0.0,
        'rv_loss': 0.0,
        'rv_zero_sum': 0.0,
        'ern_loss': 0.0,
        'ern_zero_sum': 0.0,
        'ern_preterminal_coverage': 0.0,
    }

    optimizer.zero_grad(set_to_none=True)

    for batch in dl:
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)
        masks = batch['masks'].to(device)
        returns = batch['returns'].to(device)
        valid_actions = actions >= 0
        if not bool(valid_actions.any()):
            continue
        fallback_actions = masks.to(torch.int64).argmax(-1)
        safe_actions = torch.where(valid_actions, actions, fallback_actions)

        with torch.autocast(device.type, enabled=enable_amp):
            phi = model.forward_private(obs)
            y = model.policy_y(phi)
            probs, log_probs = hedge_policy_from_y(y, masks, eta=eta, logit_clip=logit_clip)
            chosen_logp = gather_log_prob(log_probs, safe_actions)

            value_pred = model.value_head(phi).squeeze(-1)
            target_return = returns
            if (
                rvr_enabled
                and use_expected_reward
                and 'invisible_obs' in batch
            ):
                blend_alpha = _blend_alpha(rvr_cfg, steps)
                if blend_alpha > 0:
                    keep = batch['is_preterminal'].to(device)
                    if bool(keep.any()):
                        phi_oracle_for_blend = model.forward_oracle(obs[keep], batch['invisible_obs'].to(device)[keep])
                        ern_pred = model.ern(phi_oracle_for_blend)
                        seat = batch['seat'].to(device)[keep]
                        ern_self = ern_pred.gather(-1, seat.unsqueeze(-1)).squeeze(-1)
                        target_return = target_return.clone()
                        target_return[keep] = (
                            (1.0 - blend_alpha) * target_return[keep]
                            + blend_alpha * ern_self.detach()
                        )

            adv = target_return - value_pred.detach()
            if adv_norm:
                adv_valid = adv[valid_actions]
                adv_normed = (adv_valid - adv_valid.mean()) / adv_valid.std(unbiased=False).clamp_min(1e-6)
                adv = torch.zeros_like(adv)
                adv[valid_actions] = adv_normed

            aw_weight = torch.exp(adv / max(awr_beta, 1e-6)).clamp_max(awr_clip)
            actor_loss = -(aw_weight[valid_actions] * chosen_logp[valid_actions]).mean()
            value_loss = 0.5 * mse(value_pred[valid_actions], target_return[valid_actions])
            entropy = (probs * torch.where(masks, log_probs, torch.zeros_like(log_probs))).sum(-1)[valid_actions].mean()
            bc_loss = -chosen_logp[valid_actions].mean()

            total_loss = (
                actor_weight * actor_loss
                + value_coef * value_loss
                + entropy_coef * entropy
                + bc_coef * bc_loss
            )

            rv_loss = value_loss.new_zeros(())
            rv_zs = value_loss.new_zeros(())
            ern_loss = value_loss.new_zeros(())
            ern_zs = value_loss.new_zeros(())
            ern_preterminal_cov = value_loss.new_zeros(())
            if rvr_enabled and 'invisible_obs' in batch:
                invisible = batch['invisible_obs'].to(device)
                phi_oracle = model.forward_oracle(obs, invisible)
                target_vec = zero_sum_utility(batch['final_reward_vec'].to(device))

                if use_oracle_value:
                    rv_pred = model.rv_head(phi_oracle)
                    rv_loss = mse(rv_pred, target_vec)
                    rv_zs = rv_pred.sum(-1).pow(2).mean()
                    total_loss = total_loss + rv_coef * rv_loss + zero_sum_coef * rv_zs

                if use_expected_reward:
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
                f'bc={float(bc_loss.detach())}, '
                f'rv={float(rv_loss.detach())}, '
                f'ern={float(ern_loss.detach())}'
            )

        scaler.scale(total_loss / opt_step_every).backward()

        steps += 1
        if steps % opt_step_every == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        stats['actor_loss'] += float(actor_loss.detach())
        stats['value_loss'] += float(value_loss.detach())
        stats['entropy'] += float(entropy.detach())
        stats['bc_loss'] += float(bc_loss.detach())
        stats['aw_weight_mean'] += float(aw_weight[valid_actions].mean().detach())
        stats['rv_loss'] += float(rv_loss.detach())
        stats['rv_zero_sum'] += float(rv_zs.detach())
        stats['ern_loss'] += float(ern_loss.detach())
        stats['ern_zero_sum'] += float(ern_zs.detach())
        stats['ern_preterminal_coverage'] += float(ern_preterminal_cov.detach())

        if steps % save_every == 0:
            writer.add_scalar('loss/actor_loss', stats['actor_loss'] / save_every, steps)
            writer.add_scalar('loss/value_loss', stats['value_loss'] / save_every, steps)
            writer.add_scalar('loss/bc_anchor', stats['bc_loss'] / save_every, steps)
            writer.add_scalar('policy/aw_weight_mean', stats['aw_weight_mean'] / save_every, steps)
            writer.add_scalar('policy/entropy', stats['entropy'] / save_every, steps)
            writer.add_scalar('rv/loss', stats['rv_loss'] / save_every, steps)
            writer.add_scalar('rv/zero_sum_penalty', stats['rv_zero_sum'] / save_every, steps)
            writer.add_scalar('ern/loss', stats['ern_loss'] / save_every, steps)
            writer.add_scalar('ern/zero_sum_penalty', stats['ern_zero_sum'] / save_every, steps)
            writer.add_scalar('ern/preterminal_coverage', stats['ern_preterminal_coverage'] / save_every, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
            writer.add_scalar('hparam/reward_blend_alpha', _blend_alpha(rvr_cfg, steps), steps)
            writer.flush()

            extra = {}
            if legacy_current_dqn is not None:
                extra['current_dqn'] = legacy_current_dqn
            if legacy_aux_net is not None:
                extra['aux_net'] = legacy_aux_net

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
                extra=extra,
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
                extra=extra,
            )
            try:
                run_periodic_eval(
                    config=config,
                    stage_name='offline_teacher',
                    steps=steps,
                    current_model_path=snapshot,
                    writer=writer,
                )
            except Exception:
                logging.exception(f'periodic eval failed at step={steps}')
            stats = {k: 0.0 for k in stats}
            logging.info(f'total steps: {steps:,}')

        if steps >= max_steps:
            break

    extra = {}
    if legacy_current_dqn is not None:
        extra['current_dqn'] = legacy_current_dqn
    if legacy_aux_net is not None:
        extra['aux_net'] = legacy_aux_net
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
        extra=extra,
    )
    gc.collect()


def main():
    train()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
