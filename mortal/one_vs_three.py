import prelude

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN, PolicyHead
from engine import MortalEngine
from torch_compile import maybe_compile
from libriichi.arena import OneVsThree
from config import config

def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cham_cfg = state['config']
        version = cham_cfg['control'].get('version', 1)
        conv_channels = cham_cfg['resnet']['conv_channels']
        num_blocks = cham_cfg['resnet']['num_blocks']
        decision_head = cfg['champion'].get('decision_head', 'value')
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        dqn = None
        policy = None
        mortal.load_state_dict(state['mortal'])
        if 'current_dqn' in state:
            dqn = DQN(version=version).eval()
            dqn.load_state_dict(state['current_dqn'])
        if decision_head == 'policy':
            policy_state = state.get('policy')
            if policy_state is None:
                raise KeyError('1v3.champion.decision_head=policy but checkpoint has no `policy` state')
            policy = PolicyHead(version=version).eval()
            policy.load_state_dict(policy_state)
        elif dqn is None:
            raise KeyError('1v3.champion.decision_head=value but checkpoint has no `current_dqn` state')
        if cfg['champion']['enable_compile']:
            mortal = maybe_compile(
                mortal,
                enable=True,
                device=cfg['champion']['device'],
                label=f"{cfg['champion']['name']}:brain",
            )
            if dqn is not None:
                dqn = maybe_compile(
                    dqn,
                    enable=True,
                    device=cfg['champion']['device'],
                    label=f"{cfg['champion']['name']}:dqn",
                )
            if policy is not None:
                policy = maybe_compile(
                    policy,
                    enable=True,
                    device=cfg['champion']['device'],
                    label=f"{cfg['champion']['name']}:policy",
                )
        engine_cham = MortalEngine(
            mortal,
            dqn,
            policy = policy,
            is_oracle = False,
            version = version,
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            name = cfg['champion']['name'],
            decision_head = decision_head,
            policy_temp = cfg['champion'].get('policy_temp', 1.0),
            policy_top_p = cfg['champion'].get('policy_top_p', 1.0),
            policy_epsilon = cfg['champion'].get('policy_epsilon', 0.0),
        )

    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    chal_cfg = state['config']
    version = chal_cfg['control'].get('version', 1)
    conv_channels = chal_cfg['resnet']['conv_channels']
    num_blocks = chal_cfg['resnet']['num_blocks']
    decision_head = cfg['challenger'].get('decision_head', 'value')
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = None
    policy = None
    mortal.load_state_dict(state['mortal'])
    if 'current_dqn' in state:
        dqn = DQN(version=version).eval()
        dqn.load_state_dict(state['current_dqn'])
    if decision_head == 'policy':
        policy_state = state.get('policy')
        if policy_state is None:
            raise KeyError('1v3.challenger.decision_head=policy but checkpoint has no `policy` state')
        policy = PolicyHead(version=version).eval()
        policy.load_state_dict(policy_state)
    elif dqn is None:
        raise KeyError('1v3.challenger.decision_head=value but checkpoint has no `current_dqn` state')
    if cfg['challenger']['enable_compile']:
        mortal = maybe_compile(
            mortal,
            enable=True,
            device=cfg['challenger']['device'],
            label=f"{cfg['challenger']['name']}:brain",
        )
        if dqn is not None:
            dqn = maybe_compile(
                dqn,
                enable=True,
                device=cfg['challenger']['device'],
                label=f"{cfg['challenger']['name']}:dqn",
            )
        if policy is not None:
            policy = maybe_compile(
                policy,
                enable=True,
                device=cfg['challenger']['device'],
                label=f"{cfg['challenger']['name']}:policy",
            )
    engine_chal = MortalEngine(
        mortal,
        dqn,
        policy = policy,
        is_oracle = False,
        version = version,
        device = torch.device(cfg['challenger']['device']),
        enable_amp = cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
        name = cfg['challenger']['name'],
        decision_head = decision_head,
        policy_temp = cfg['challenger'].get('policy_temp', 1.0),
        policy_top_p = cfg['challenger'].get('policy_top_p', 1.0),
        policy_epsilon = cfg['challenger'].get('policy_epsilon', 0.0),
    )

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine = engine_chal,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
