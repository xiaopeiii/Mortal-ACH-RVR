import prelude

import argparse
import os
import random
import shutil

import torch
from libriichi.arena import OneVsThree, TwoVsTwo

from eval_match import load_engine, summarize


def run():
    parser = argparse.ArgumentParser(description='Evaluate ACH+RVR checkpoints (policy head by default).')
    parser.add_argument('--mode', choices=['1v3', '2v2'], default='1v3')
    parser.add_argument('--model-a', required=True)
    parser.add_argument('--model-b', required=True)
    parser.add_argument('--name-a', default='model_a')
    parser.add_argument('--name-b', default='model_b')
    parser.add_argument('--challenger', choices=['a', 'b'], default='a')
    parser.add_argument('--games', type=int, default=20000)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--seed-key', type=int, default=-1)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--enable-compile', action='store_true')
    parser.add_argument('--disable-rule-based-agari-guard', action='store_true')
    parser.add_argument('--decision-head-a', choices=['value', 'policy'], default='policy')
    parser.add_argument('--decision-head-b', choices=['value', 'policy'], default='policy')
    parser.add_argument('--policy-temp-a', type=float, default=1.0)
    parser.add_argument('--policy-temp-b', type=float, default=1.0)
    parser.add_argument('--policy-top-p-a', type=float, default=1.0)
    parser.add_argument('--policy-top-p-b', type=float, default=1.0)
    parser.add_argument('--policy-epsilon-a', type=float, default=0.0)
    parser.add_argument('--policy-epsilon-b', type=float, default=0.0)
    parser.add_argument('--policy-stochastic-a', action='store_true')
    parser.add_argument('--policy-stochastic-b', action='store_true')
    parser.add_argument('--pts', type=int, nargs=4, default=[90, 45, 0, -135])
    args = parser.parse_args()

    device = torch.device(args.device)
    enable_rule_based_agari_guard = not args.disable_rule_based_agari_guard

    if os.path.isdir(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    engine_a = load_engine(
        state_file=args.model_a,
        name=args.name_a,
        device=device,
        enable_amp=args.enable_amp,
        enable_compile=args.enable_compile,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        decision_head=args.decision_head_a,
        policy_temp=args.policy_temp_a,
        policy_top_p=args.policy_top_p_a,
        policy_epsilon=args.policy_epsilon_a,
        policy_stochastic=args.policy_stochastic_a,
    )
    engine_b = load_engine(
        state_file=args.model_b,
        name=args.name_b,
        device=device,
        enable_amp=args.enable_amp,
        enable_compile=args.enable_compile,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        decision_head=args.decision_head_b,
        policy_temp=args.policy_temp_b,
        policy_top_p=args.policy_top_p_b,
        policy_epsilon=args.policy_epsilon_b,
        policy_stochastic=args.policy_stochastic_b,
    )

    if args.challenger == 'a':
        challenger, champion = engine_a, engine_b
    else:
        challenger, champion = engine_b, engine_a

    unit = 4 if args.mode == '1v3' else 2
    seed_count = args.games // unit
    if seed_count <= 0:
        raise ValueError(f'--games must be >= {unit} for {args.mode}')
    actual_games = seed_count * unit
    seed_key = args.seed_key if args.seed_key != -1 else random.getrandbits(64)

    print(f'mode={args.mode} requested_games={args.games} actual_games={actual_games}')
    if args.mode == '1v3':
        env = OneVsThree(disable_progress_bar=False, log_dir=args.log_dir)
        env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
        )
    else:
        env = TwoVsTwo(disable_progress_bar=False, log_dir=args.log_dir)
        env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
        )

    print('\n=== Summary ===')
    summarize(args.log_dir, args.name_a, args.pts, False)
    summarize(args.log_dir, args.name_b, args.pts, False)


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass
