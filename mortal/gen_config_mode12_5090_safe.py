#!/usr/bin/env python3
import argparse
import copy
import os
from pathlib import Path

import toml


def detect_root(cli_root: str | None) -> Path:
    candidates = [
        cli_root,
        os.environ.get("PRJ"),
        "/root/autodl-tmp/Mahjong",
        "/root/Mahjong",
        "/workspace/Mahjong",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).resolve()
        if p.exists():
            return p
    raise FileNotFoundError("Cannot detect Mahjong root. Pass --root explicitly.")


def apply_common_off(cfg: dict, root: str, tag: str) -> None:
    cfg["control"]["online"] = False
    cfg["control"]["device"] = "cuda:0"
    cfg["control"]["enable_amp"] = True
    cfg["control"]["enable_compile"] = False
    cfg["control"]["state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}.pth"
    cfg["control"]["best_state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}_best.pth"
    cfg["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/mortal_mode12_5090_{tag}"

    cfg["dataset"]["globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["dataset"]["file_index"] = f"{root}/output/mode12/file_index_mode12_5090_{tag}.pth"
    cfg["dataset"]["enable_augmentation"] = True
    cfg["dataset"]["augmented_first"] = True

    cfg["env"]["pts"] = [100.0, 50.0, -5.0, -200.0]
    cfg["env"]["reward_scale"] = 0.01
    cfg["env"]["q_target_clip"] = 4.0

    cfg["cql"]["min_q_weight"] = 3.0
    cfg["cql"]["online_q_weight"] = 0.2
    cfg["aux"]["next_rank_weight"] = 0.2
    cfg["freeze_bn"]["mortal"] = False

    cfg["baseline"]["train"]["state_file"] = f"{root}/output/checkpoints/baseline_mode1.pth"
    cfg["baseline"]["test"]["state_file"] = f"{root}/output/checkpoints/baseline_mode1.pth"

    cfg["grp"]["state_file"] = f"{root}/output/checkpoints/grp_mode12_5090_{tag}.pth"
    cfg["grp"]["control"]["device"] = "cuda:0"
    cfg["grp"]["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/grp_mode12_5090_{tag}"
    cfg["grp"]["dataset"]["train_globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["val_globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["file_index"] = f"{root}/output/mode12/grp_file_index_mode12_5090_{tag}.pth"


def apply_common_on(cfg: dict, root: str, tag: str) -> None:
    cfg["control"]["online"] = True
    cfg["control"]["device"] = "cuda:0"
    cfg["control"]["enable_amp"] = True
    cfg["control"]["enable_compile"] = False
    cfg["control"]["state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}_online_bootstrap.pth"
    cfg["control"]["best_state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}_online_best.pth"
    cfg["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/mortal_mode12_5090_{tag}_online"

    cfg["dataset"]["globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["dataset"]["file_index"] = f"{root}/output/mode12/file_index_mode12_5090_{tag}.pth"
    cfg["dataset"]["enable_augmentation"] = True
    cfg["dataset"]["augmented_first"] = True

    cfg["env"]["pts"] = [100.0, 50.0, -5.0, -200.0]
    cfg["env"]["reward_scale"] = 0.01
    cfg["env"]["q_target_clip"] = 4.0

    cfg["cql"]["min_q_weight"] = 3.0
    cfg["aux"]["next_rank_weight"] = 0.2
    cfg["freeze_bn"]["mortal"] = True

    cfg["baseline"]["train"]["state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}_best.pth"
    cfg["baseline"]["test"]["state_file"] = f"{root}/output/checkpoints/mortal_mode12_5090_{tag}_best.pth"

    cfg["online"]["remote"]["host"] = "127.0.0.1"
    cfg["online"]["remote"]["port"] = 5000
    cfg["online"]["server"]["buffer_dir"] = f"{root}/output/online_buffer_mode12_5090_{tag}"
    cfg["online"]["server"]["drain_dir"] = f"{root}/output/online_drain_mode12_5090_{tag}"

    cfg["grp"]["state_file"] = f"{root}/output/checkpoints/grp_mode12_5090_{tag}.pth"
    cfg["grp"]["control"]["device"] = "cuda:0"
    cfg["grp"]["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/grp_mode12_5090_{tag}"
    cfg["grp"]["dataset"]["train_globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["val_globs"] = [f"{root}/output/mode12/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["file_index"] = f"{root}/output/mode12/grp_file_index_mode12_5090_{tag}.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mode12 5090 SAFE offline/online configs.")
    parser.add_argument("--root", help="Project root path, e.g. /root/Mahjong")
    args = parser.parse_args()

    root = detect_root(args.root)
    cfg_dir = root / "Mortal" / "mortal"

    base_off = toml.load(cfg_dir / "config.mode1.toml")
    base_on = toml.load(cfg_dir / "config.mode1.online.toml")

    tag = "safe"
    off = copy.deepcopy(base_off)
    on = copy.deepcopy(base_on)
    apply_common_off(off, str(root), tag)
    apply_common_on(on, str(root), tag)

    off["control"]["batch_size"] = 384
    off["control"]["opt_step_every"] = 2
    off["dataset"]["num_workers"] = 8
    off["dataset"]["file_batch_size"] = 18
    off["optim"]["max_grad_norm"] = 0.8
    off["optim"]["weight_decay"] = 0.1
    off["optim"]["scheduler"]["peak"] = 1.2e-4
    off["optim"]["scheduler"]["final"] = 1.2e-5
    off["optim"]["scheduler"]["warm_up_steps"] = 4000
    off["optim"]["scheduler"]["max_steps"] = 300000
    off["grp"]["control"]["batch_size"] = 1024

    on["control"]["batch_size"] = 256
    on["control"]["opt_step_every"] = 2
    on["control"]["submit_every"] = 400
    on["dataset"]["num_workers"] = 8
    on["dataset"]["file_batch_size"] = 18
    on["train_play"]["default"]["games"] = 400
    on["train_play"]["default"]["boltzmann_epsilon"] = 0.0
    on["train_play"]["default"]["boltzmann_temp"] = 1.0
    on["train_play"]["default"]["top_p"] = 1.0
    on["cql"]["online_q_weight"] = 0.2
    on["optim"]["max_grad_norm"] = 0.3
    on["optim"]["scheduler"]["peak"] = 8e-6
    on["optim"]["scheduler"]["final"] = 2e-6
    on["optim"]["scheduler"]["warm_up_steps"] = 30000
    on["optim"]["scheduler"]["max_steps"] = 300000
    on["online"]["server"]["capacity"] = 1600
    on["online"]["server"]["force_sequential"] = True

    off_out = cfg_dir / "config.mode12.5090.safe.toml"
    on_out = cfg_dir / "config.mode12.5090.safe.online.toml"
    with open(off_out, "w", encoding="utf-8") as f:
        toml.dump(off, f)
    with open(on_out, "w", encoding="utf-8") as f:
        toml.dump(on, f)

    print("generated:")
    print(f" - {off_out}")
    print(f" - {on_out}")


if __name__ == "__main__":
    main()
