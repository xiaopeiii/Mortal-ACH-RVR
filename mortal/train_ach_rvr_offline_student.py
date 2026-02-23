import prelude


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
        load_legacy_states,
        maybe_load_state,
        checkpoint_snapshot_name,
        save_ach_rvr_state,
    )
    from config import config
    from dataloader_ach_rvr import AchRvrFileDataset, collate_ach_rvr, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model_ach_rvr import AchRvrModel, gather_log_prob, hedge_policy_from_y
    from periodic_eval import run_periodic_eval

    device = torch.device(config['control']['device'])
    enable_amp = bool(config['control']['enable_amp'])
    torch.backends.cudnn.benchmark = bool(config['control']['enable_cudnn_benchmark'])

    if bool(config.get('strict', {}).get('enabled', False)):
        raise RuntimeError('offline_student requires strict.enabled=false')

    version = int(config['control']['version'])
    batch_size = int(config['control']['batch_size'])
    opt_step_every = int(config['control']['opt_step_every'])
    save_every = int(config['control']['save_every'])
    max_steps = int(config['optim']['scheduler']['max_steps'])
    max_grad_norm = float(config['optim'].get('max_grad_norm', 0.0))

    ach_cfg = config.get('ach', {})
    eta = float(ach_cfg.get('eta', 1.0))
    logit_clip = float(ach_cfg.get('logit_clip', 6.0))
    gamma = float(ach_cfg.get('gamma', 1.0))

    distill_cfg = config.get('distill', {})
    teacher_state_file = str(
        distill_cfg.get(
            'teacher_state_file',
            config['control'].get('bootstrap_state_file', ''),
        )
    )
    if not teacher_state_file or not path.exists(teacher_state_file):
        raise FileNotFoundError(f'distill.teacher_state_file not found: {teacher_state_file}')

    kl_coef = float(distill_cfg.get('kl_coef', 1.0))
    value_distill_coef = float(distill_cfg.get('value_distill_coef', 1.0))
    bc_coef = float(distill_cfg.get('bc_coef', 0.05))
    entropy_coef = float(distill_cfg.get('entropy_coef', 1e-2))
    temperature = float(distill_cfg.get('temperature', 1.0))
    include_oracle_obs = bool(config.get('oracle_dataset', {}).get('include_invisible_obs', True))

    model = AchRvrModel(version=version, **config['resnet']).to(device)
    model.private_brain.requires_grad_(True)
    model.policy_y.requires_grad_(True)
    model.value_head.requires_grad_(True)
    model.oracle_brain.requires_grad_(False)
    model.rv_head.requires_grad_(False)
    model.ern.requires_grad_(False)

    teacher = AchRvrModel(version=version, **config['resnet']).to(device).eval()
    teacher_state = torch.load(teacher_state_file, weights_only=False, map_location='cpu')
    load_legacy_states(teacher_state, teacher, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)

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
        'method': 'ach+rvr_student',
        'initialized_from': teacher_state_file,
        'eta': eta,
        'logit_clip': logit_clip,
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
        legacy_current_dqn = boot.get('current_dqn')
        legacy_aux_net = boot.get('aux_net')
        logging.info(f'student bootstrapped from: {bootstrap_file}')

    file_list = build_file_list(config['dataset'], shuffle=True)
    logging.info(f'offline student distillation from teacher: {teacher_state_file}')
    logging.info(f'device: {device}')
    logging.info(f'file list size: {len(file_list):,}')

    ds = AchRvrFileDataset(
        version=version,
        file_list=file_list,
        pts=config['env']['pts'],
        gamma=gamma,
        reward_source=config.get('rvr', {}).get('reward_source', 'raw_score'),
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
        'distill_kl': 0.0,
        'value_distill': 0.0,
        'bc_loss': 0.0,
        'entropy': 0.0,
    }
    optimizer.zero_grad(set_to_none=True)

    for batch in dl:
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)
        masks = batch['masks'].to(device)
        valid_actions = actions >= 0
        if not bool(valid_actions.any()):
            continue
        fallback_actions = masks.to(torch.int64).argmax(-1)
        safe_actions = torch.where(valid_actions, actions, fallback_actions)

        with torch.autocast(device.type, enabled=enable_amp):
            phi_student = model.forward_private(obs)
            y_student = model.policy_y(phi_student) / max(temperature, 1e-6)
            probs_student, log_probs_student = hedge_policy_from_y(
                y_student,
                masks,
                eta=eta,
                logit_clip=logit_clip,
            )
            v_student = model.value_head(phi_student).squeeze(-1)
            chosen_logp = gather_log_prob(log_probs_student, safe_actions)

            with torch.no_grad():
                phi_teacher = teacher.forward_private(obs)
                y_teacher = teacher.policy_y(phi_teacher) / max(temperature, 1e-6)
                probs_teacher, log_probs_teacher = hedge_policy_from_y(
                    y_teacher,
                    masks,
                    eta=eta,
                    logit_clip=logit_clip,
                )
                v_teacher = teacher.value_head(phi_teacher).squeeze(-1)

            safe_logp_teacher = torch.where(masks, log_probs_teacher, torch.zeros_like(log_probs_teacher))
            safe_logp_student = torch.where(masks, log_probs_student, torch.zeros_like(log_probs_student))
            kl_each = (probs_teacher * (safe_logp_teacher - safe_logp_student)).sum(-1)
            distill_kl = kl_each[valid_actions].mean()
            value_distill = mse(v_student[valid_actions], v_teacher[valid_actions])
            bc_loss = -chosen_logp[valid_actions].mean()
            entropy = (probs_student * safe_logp_student).sum(-1)[valid_actions].mean()

            total_loss = (
                kl_coef * distill_kl
                + value_distill_coef * value_distill
                + bc_coef * bc_loss
                + entropy_coef * entropy
            )

        if not torch.isfinite(total_loss).item():
            raise RuntimeError(
                f'non-finite total_loss at step={steps + 1}: '
                f'kl={float(distill_kl.detach())}, '
                f'value_distill={float(value_distill.detach())}, '
                f'bc={float(bc_loss.detach())}, '
                f'entropy={float(entropy.detach())}'
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

        stats['distill_kl'] += float(distill_kl.detach())
        stats['value_distill'] += float(value_distill.detach())
        stats['bc_loss'] += float(bc_loss.detach())
        stats['entropy'] += float(entropy.detach())

        if steps % save_every == 0:
            writer.add_scalar('distill/kl', stats['distill_kl'] / save_every, steps)
            writer.add_scalar('distill/value_mse', stats['value_distill'] / save_every, steps)
            writer.add_scalar('distill/bc_anchor', stats['bc_loss'] / save_every, steps)
            writer.add_scalar('policy/entropy', stats['entropy'] / save_every, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
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
                    stage_name='offline_student',
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
