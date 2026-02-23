import prelude

def train():
    import logging
    import gc
    import random
    import torch

    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.amp import GradScaler

    from config import config
    from common import tqdm, parameter_count
    from dataloader_ach_rvr import AchRvrFileDataset, collate_ach_rvr, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model_ach_rvr import AchRvrModel, zero_sum_utility
    from ach_rvr_utils import build_file_list, maybe_load_state, save_ach_rvr_state

    device = torch.device(config['control']['device'])
    enable_amp = bool(config['control']['enable_amp'])
    torch.backends.cudnn.benchmark = bool(config['control']['enable_cudnn_benchmark'])

    version = config['control']['version']
    resnet_cfg = config['resnet']

    batch_size = int(config['control']['batch_size'])
    save_every = int(config['control']['save_every'])
    max_steps = int(config['optim']['scheduler']['max_steps'])
    opt_step_every = int(config['control']['opt_step_every'])

    rvr_cfg = config.get('rvr', {})
    zero_sum_coef = float(rvr_cfg.get('zero_sum_coef', 0.1))

    model = AchRvrModel(version=version, **resnet_cfg).to(device)
    model.private_brain.requires_grad_(False)
    model.policy_y.requires_grad_(False)
    model.value_head.requires_grad_(False)
    model.ern.requires_grad_(False)

    params = list(model.oracle_brain.parameters()) + list(model.rv_head.parameters())
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

    logging.info(f"device: {device}")
    logging.info(f"oracle_brain params: {parameter_count(model.oracle_brain):,}")
    logging.info(f"rv_head params: {parameter_count(model.rv_head):,}")

    file_list = build_file_list(config['dataset'], shuffle=True)
    logging.info(f"file list size: {len(file_list):,}")

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
    run = {'rv_loss': 0.0, 'zs_loss': 0.0}

    optimizer.zero_grad(set_to_none=True)

    for batch in dl:
        obs = batch['obs'].to(device)
        invisible = batch['invisible_obs'].to(device)
        target = zero_sum_utility(batch['final_reward_vec'].to(device))

        with torch.autocast(device.type, enabled=enable_amp):
            phi_oracle = model.forward_oracle(obs, invisible)
            pred = model.rv_head(phi_oracle)
            rv_loss = mse(pred, target)
            zs_loss = (pred.sum(-1).pow(2)).mean()
            loss = rv_loss + zero_sum_coef * zs_loss

        scaler.scale(loss / opt_step_every).backward()

        steps += 1
        if steps % opt_step_every == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        run['rv_loss'] += float(rv_loss.detach())
        run['zs_loss'] += float(zs_loss.detach())

        if steps % save_every == 0:
            writer.add_scalar('rv/loss', run['rv_loss'] / save_every, steps)
            writer.add_scalar('rv/zero_sum_penalty', run['zs_loss'] / save_every, steps)
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

