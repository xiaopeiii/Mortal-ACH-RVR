import prelude


def train():
    import gc
    import logging

    import torch
    from torch import nn, optim
    from torch.amp import GradScaler
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from ach_rvr_utils import (
        build_file_list,
        checkpoint_snapshot_name,
        maybe_load_state,
        save_ach_rvr_state,
    )
    from config import config
    from dataloader_ach_rvr import AchRvrFileDataset, collate_ach_rvr, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model_ach_rvr import AchRvrModel, zero_sum_utility

    device = torch.device(config['control']['device'])
    enable_amp = bool(config['control']['enable_amp'])
    torch.backends.cudnn.benchmark = bool(config['control']['enable_cudnn_benchmark'])

    version = int(config['control']['version'])
    batch_size = int(config['control']['batch_size'])
    opt_step_every = int(config['control']['opt_step_every'])
    save_every = int(config['control']['save_every'])
    max_steps = int(config['optim']['scheduler']['max_steps'])

    rvr_cfg = config.get('rvr', {})
    preterminal_only = bool(rvr_cfg.get('ern_preterminal_only', True))
    zero_sum_coef = float(rvr_cfg.get('zero_sum_coef', 0.1))

    model = AchRvrModel(version=version, **config['resnet']).to(device)
    model.private_brain.requires_grad_(False)
    model.policy_y.requires_grad_(False)
    model.value_head.requires_grad_(False)
    model.rv_head.requires_grad_(False)

    train_modules = [model.oracle_brain, model.ern]
    params = [p for m in train_modules for p in m.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=1.0,
        betas=tuple(config['optim']['betas']),
        eps=float(config['optim']['eps']),
        weight_decay=float(config['optim']['weight_decay']),
    )
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    steps, best_perf, policy_meta, league_meta = maybe_load_state(
        config['control']['state_file'],
        map_location=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    file_list = build_file_list(config['dataset'], shuffle=True)
    logging.info(f'device: {device}')
    logging.info(f'file list size: {len(file_list):,}')

    ds = AchRvrFileDataset(
        version=version,
        file_list=file_list,
        pts=config['env']['pts'],
        gamma=float(config['ach']['gamma']),
        reward_source=config.get('rvr', {}).get('reward_source', 'grp_plus_ern'),
        oracle=True,
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
    run = {
        'ern_loss': 0.0,
        'ern_zero_sum_penalty': 0.0,
    }

    optimizer.zero_grad(set_to_none=True)

    for batch in dl:
        if 'invisible_obs' not in batch:
            raise ValueError('ExpectedReward training requires oracle invisible_obs.')

        obs = batch['obs'].to(device)
        invisible = batch['invisible_obs'].to(device)
        target = zero_sum_utility(batch['final_reward_vec'].to(device))
        if preterminal_only:
            keep = batch['is_preterminal'].to(device)
            if not bool(keep.any()):
                continue
            obs = obs[keep]
            invisible = invisible[keep]
            target = target[keep]

        with torch.autocast(device.type, enabled=enable_amp):
            phi_oracle = model.forward_oracle(obs, invisible)
            pred = model.ern(phi_oracle)
            ern_loss = mse(pred, target)
            zs_loss = pred.sum(-1).pow(2).mean()
            loss = ern_loss + zero_sum_coef * zs_loss

        scaler.scale(loss / opt_step_every).backward()

        steps += 1
        if steps % opt_step_every == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        run['ern_loss'] += float(ern_loss.detach())
        run['ern_zero_sum_penalty'] += float(zs_loss.detach())

        if steps % save_every == 0:
            writer.add_scalar('ern/loss', run['ern_loss'] / save_every, steps)
            writer.add_scalar('ern/zero_sum_penalty', run['ern_zero_sum_penalty'] / save_every, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
            writer.flush()

            save_ach_rvr_state(
                filename=config['control']['state_file'],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                steps=steps,
                best_perf=best_perf,
                policy_meta=policy_meta,
                league_meta=league_meta,
            )
            snapshot = checkpoint_snapshot_name(config['control']['state_file'], steps)
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
            )

            run = {k: 0.0 for k in run}
            logging.info(f'total steps: {steps:,}')

        if steps >= max_steps:
            break

    save_ach_rvr_state(
        filename=config['control']['state_file'],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        steps=steps,
        best_perf=best_perf,
        policy_meta=policy_meta,
        league_meta=league_meta,
    )

    gc.collect()


def main():
    train()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
