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
    cfg["control"]["state_file"] = f"{root}/output/checkpoints/mortal_mode16_konten_5090_{tag}.pth"
    cfg["control"]["best_state_file"] = f"{root}/output/checkpoints/mortal_mode16_konten_5090_{tag}_best.pth"
    cfg["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/mortal_mode16_konten_5090_{tag}"

    cfg["dataset"]["globs"] = [f"{root}/output/mode16_konten_4p/mjai_rust/*.json.gz"]
    cfg["dataset"]["file_index"] = f"{root}/output/mode16_konten_4p/file_index_mode16_konten_5090_{tag}.pth"
    cfg["dataset"]["player_names_files"] = []
    cfg["dataset"]["enable_augmentation"] = True
    cfg["dataset"]["augmented_first"] = True

    # Throne-point style target with scaling for stable Q regression.
    cfg["env"]["gamma"] = 1
    cfg["env"]["pts"] = [100.0, 50.0, -5.0, -200.0]
    cfg["env"]["reward_scale"] = 0.01
    cfg["env"]["q_target_clip"] = 4.0

    cfg["cql"]["min_q_weight"] = 3.5
    cfg["aux"]["next_rank_weight"] = 0.2
    cfg["freeze_bn"]["mortal"] = False

    # Keep baseline/GRP dependencies explicit.
    cfg["baseline"]["train"]["state_file"] = f"{root}/output/checkpoints/mortal_mode1_best.pth"
    cfg["baseline"]["test"]["state_file"] = f"{root}/output/checkpoints/mortal_mode1_best.pth"

    cfg["grp"]["state_file"] = f"{root}/output/checkpoints/grp_mode16_konten_5090_{tag}.pth"
    cfg["grp"]["control"]["device"] = "cuda:0"
    cfg["grp"]["control"]["tensorboard_dir"] = f"{root}/output/tensorboard/grp_mode16_konten_5090_{tag}"
    cfg["grp"]["dataset"]["train_globs"] = [f"{root}/output/mode16_konten_4p/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["val_globs"] = [f"{root}/output/mode16_konten_4p/mjai_rust/*.json.gz"]
    cfg["grp"]["dataset"]["file_index"] = f"{root}/output/mode16_konten_4p/grp_file_index_mode16_konten_5090_{tag}.pth"

    cfg["test_play"]["log_dir"] = f"{root}/output/test_play_mode16_konten_5090_{tag}"
    cfg["train_play"]["default"]["log_dir"] = f"{root}/output/train_play_mode16_konten_5090_{tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mode16 konten (4-soul-ten) 5090 SUPER offline config.")
    parser.add_argument("--root", help="Project root path, e.g. /root/Mahjong")
    args = parser.parse_args()

    root = detect_root(args.root)
    cfg_dir = root / "Mortal" / "mortal"

    base_off = toml.load(cfg_dir / "config.mode1.toml")

    tag = "super"
    off = copy.deepcopy(base_off)
    apply_common_off(off, str(root), tag)

    # 5090 super profile: big model + high throughput.
    off["control"]["batch_size"] = 768
    off["control"]["opt_step_every"] = 1
    off["control"]["save_every"] = 500
    off["control"]["test_every"] = 25000
    off["test_play"]["games"] = 400

    off["dataset"]["file_batch_size"] = 32
    off["dataset"]["reserve_ratio"] = 0.05
    off["dataset"]["num_workers"] = 16
    off["dataset"]["num_epochs"] = 8

    off["resnet"]["conv_channels"] = 256
    off["resnet"]["num_blocks"] = 40

    off["optim"]["eps"] = 1e-8
    off["optim"]["betas"] = [0.9, 0.999]
    off["optim"]["weight_decay"] = 0.12
    off["optim"]["max_grad_norm"] = 0.5
    off["optim"]["scheduler"]["peak"] = 1.5e-4
    off["optim"]["scheduler"]["final"] = 8e-6
    off["optim"]["scheduler"]["warm_up_steps"] = 8000
    off["optim"]["scheduler"]["max_steps"] = 200000

    off["grp"]["control"]["batch_size"] = 1536
    off["grp"]["control"]["save_every"] = 2000
    off["grp"]["control"]["val_steps"] = 400
    off["grp"]["optim"]["lr"] = 1e-5

    off_out = cfg_dir / "config.mode16.konten.5090.super.toml"
    with open(off_out, "w", encoding="utf-8") as f:
        toml.dump(off, f)

    print("generated:")
    print(f" - {off_out}")


if __name__ == "__main__":
    main()
