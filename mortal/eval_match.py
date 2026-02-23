import prelude

import argparse
import os
import random
import shutil
import torch
from model import Brain, DQN, PolicyHead
from engine import MortalEngine
from torch_compile import maybe_compile
from libriichi.arena import OneVsThree, TwoVsTwo
from libriichi.stat import Stat


def parse_pts(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--pts must have 4 integers, got {len(parts)}: {text}"
        )
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --pts: {text}") from exc


def ensure_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")


def load_engine(
    state_file: str,
    name: str,
    device: torch.device,
    enable_amp: bool,
    enable_compile: bool,
    enable_rule_based_agari_guard: bool,
    decision_head: str,
    policy_temp: float,
    policy_top_p: float,
    policy_epsilon: float,
    policy_stochastic: bool,
) -> MortalEngine:
    ensure_file(state_file)
    state = torch.load(state_file, weights_only=False, map_location=torch.device("cpu"))

    cfg = state.get("config")
    if cfg is None:
        raise KeyError(f"`config` not found in checkpoint: {state_file}")
    if "mortal" not in state:
        raise KeyError(f"checkpoint missing mortal: {state_file}")

    version = cfg["control"].get("version", 1)
    conv_channels = cfg["resnet"]["conv_channels"]
    num_blocks = cfg["resnet"]["num_blocks"]

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = None
    policy = None
    mortal.load_state_dict(state["mortal"])
    if "current_dqn" in state:
        dqn = DQN(version=version).eval()
        dqn.load_state_dict(state["current_dqn"])
    if decision_head == "policy":
        policy_state = state.get("policy")
        if policy_state is None:
            raise KeyError(
                f"decision_head=policy but checkpoint has no `policy`: {state_file}"
            )
        policy = PolicyHead(version=version).eval()
        policy.load_state_dict(policy_state)
    elif dqn is None:
        raise KeyError(f"decision_head=value but checkpoint has no `current_dqn`: {state_file}")

    if enable_compile:
        mortal = maybe_compile(mortal, enable=True, device=device, label=f"{name}:brain")
        if dqn is not None:
            dqn = maybe_compile(dqn, enable=True, device=device, label=f"{name}:dqn")
        if policy is not None:
            policy = maybe_compile(policy, enable=True, device=device, label=f"{name}:policy")

    return MortalEngine(
        mortal,
        dqn,
        policy=policy,
        is_oracle=False,
        version=version,
        device=device,
        stochastic_latent=False,
        enable_amp=enable_amp,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name,
        decision_head=decision_head,
        policy_temp=policy_temp,
        policy_top_p=policy_top_p,
        policy_epsilon=policy_epsilon,
        policy_stochastic=policy_stochastic,
    )


def summarize(log_dir: str, player_name: str, pts: list[int], disable_progress_bar: bool) -> None:
    stat = Stat.from_dir(log_dir, player_name, disable_progress_bar)
    if stat.game == 0:
        print(f"[{player_name}] no games found in logs: {log_dir}")
        return

    print(f"[{player_name}]")
    print(f"games={stat.game}")
    print(
        "rank_counts="
        f"[{stat.rank_1}, {stat.rank_2}, {stat.rank_3}, {stat.rank_4}]"
    )
    print(
        "rank_rates="
        f"[{stat.rank_1_rate:.6f}, {stat.rank_2_rate:.6f}, "
        f"{stat.rank_3_rate:.6f}, {stat.rank_4_rate:.6f}]"
    )
    print(f"avg_rank={stat.avg_rank:.6f}")
    print(f"avg_pt={stat.avg_pt(pts):.6f} (pts={pts})")
    print(f"avg_score={stat.avg_point_per_game:.6f}")


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mortal checkpoints with existing arena code "
            "in 1v3 or 2v2 mode."
        )
    )
    parser.add_argument("--mode", choices=["1v3", "2v2"], required=True)
    parser.add_argument("--model-a", required=True, help="checkpoint path for model A")
    parser.add_argument("--model-b", required=True, help="checkpoint path for model B")
    parser.add_argument(
        "--challenger",
        choices=["a", "b"],
        default="a",
        help=(
            "which model is challenger side. "
            "For 1v3: challenger=single side. For 2v2: challenger=2-player side."
        ),
    )
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--games", type=int, default=4000, help="target total hanchans")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-key", type=int, default=-1, help="-1 means random 64-bit key")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pts", type=parse_pts, default=[90, 45, 0, -135])
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--decision-head-a", choices=["value", "policy"], default="value")
    parser.add_argument("--decision-head-b", choices=["value", "policy"], default="value")
    parser.add_argument("--policy-temp-a", type=float, default=1.0)
    parser.add_argument("--policy-temp-b", type=float, default=1.0)
    parser.add_argument("--policy-top-p-a", type=float, default=1.0)
    parser.add_argument("--policy-top-p-b", type=float, default=1.0)
    parser.add_argument("--policy-epsilon-a", type=float, default=0.0)
    parser.add_argument("--policy-epsilon-b", type=float, default=0.0)
    parser.add_argument("--policy-stochastic-a", action="store_true")
    parser.add_argument("--policy-stochastic-b", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--keep-log-dir", action="store_true")
    parser.add_argument(
        "--disable-rule-based-agari-guard",
        action="store_true",
        help="disable rule-based agari guard for both sides",
    )
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.name_a == args.name_b:
        raise ValueError("--name-a and --name-b must be different for stat separation")
    if args.seed_key < -1:
        raise ValueError("--seed-key must be -1 or >= 0")

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"device: {device}")

    if os.path.isdir(args.log_dir) and not args.keep_log_dir:
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    enable_rule_based_agari_guard = not args.disable_rule_based_agari_guard
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

    if args.challenger == "a":
        challenger, champion = engine_a, engine_b
        challenger_name, champion_name = args.name_a, args.name_b
    else:
        challenger, champion = engine_b, engine_a
        challenger_name, champion_name = args.name_b, args.name_a

    seed_key = args.seed_key if args.seed_key != -1 else random.getrandbits(64)
    unit = 4 if args.mode == "1v3" else 2
    seed_count = args.games // unit
    if seed_count <= 0:
        raise ValueError(f"--games is too small for {args.mode}; need at least {unit}")
    actual_games = seed_count * unit
    if actual_games != args.games:
        print(
            f"warning: --games={args.games} is not divisible by {unit}; "
            f"actual_games={actual_games}"
        )

    print(
        f"mode={args.mode} challenger={challenger_name} champion={champion_name} "
        f"seed_start={args.seed_start} seed_count={seed_count} actual_games={actual_games}"
    )
    print(f"log_dir={os.path.abspath(args.log_dir)}")

    if args.mode == "1v3":
        env = OneVsThree(
            disable_progress_bar=args.disable_progress_bar,
            log_dir=args.log_dir,
        )
        rankings = env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
        )
        print(f"challenger_rank_counts(raw_1st_to_4th)={list(rankings)}")
    else:
        env = TwoVsTwo(
            disable_progress_bar=args.disable_progress_bar,
            log_dir=args.log_dir,
        )
        env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
        )

    print("\n=== Summary ===")
    summarize(args.log_dir, args.name_a, args.pts, args.disable_progress_bar)
    summarize(args.log_dir, args.name_b, args.pts, args.disable_progress_bar)

    if args.mode == "1v3":
        print(
            "\nNote: in 1v3, challenger model appears once per hanchan, "
            "champion model appears three times per hanchan."
        )
    else:
        print(
            "\nNote: in 2v2, each model appears twice per hanchan."
        )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
