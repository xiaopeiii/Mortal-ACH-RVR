import prelude


def _collect_stat(log_dir: str, player_name: str, pts: list[int], disable_progress_bar: bool) -> dict:
    from libriichi.stat import Stat

    stat = Stat.from_dir(log_dir, player_name, disable_progress_bar)
    if stat.game <= 0:
        raise RuntimeError(f"no games found in eval logs for {player_name}: {log_dir}")

    rank_counts = [int(stat.rank_1), int(stat.rank_2), int(stat.rank_3), int(stat.rank_4)]
    rank_rates = [
        float(stat.rank_1_rate),
        float(stat.rank_2_rate),
        float(stat.rank_3_rate),
        float(stat.rank_4_rate),
    ]
    pt_distribution = {
        str(int(pts[0])): rank_counts[0],
        str(int(pts[1])): rank_counts[1],
        str(int(pts[2])): rank_counts[2],
        str(int(pts[3])): rank_counts[3],
    }
    return {
        "games": int(stat.game),
        "rank_counts": rank_counts,
        "rank_rates": rank_rates,
        "avg_rank": float(stat.avg_rank),
        "avg_pt": float(stat.avg_pt(pts)),
        "avg_score": float(stat.avg_point_per_game),
        "pt_distribution": pt_distribution,
    }


def run_periodic_eval(
    *,
    config: dict,
    stage_name: str,
    steps: int,
    current_model_path: str,
    writer=None,
) -> dict | None:
    import json
    import logging
    import os
    import random
    import shutil
    from os import path

    import torch

    from eval_match import load_engine
    from libriichi.arena import OneVsThree, TwoVsTwo

    eval_cfg = config.get("periodic_eval", {})
    if not bool(eval_cfg.get("enabled", False)):
        return None

    every_steps = int(eval_cfg.get("every_steps", 0))
    if every_steps <= 0 or steps % every_steps != 0:
        return None

    compare_model = str(eval_cfg.get("compare_model", "")).strip()
    if not compare_model:
        logging.warning("periodic_eval enabled but compare_model is empty, skip")
        return None
    if not path.isfile(compare_model):
        logging.warning(f"periodic_eval compare_model not found, skip: {compare_model}")
        return None
    if not path.isfile(current_model_path):
        logging.warning(f"periodic_eval current model snapshot not found, skip: {current_model_path}")
        return None

    mode = str(eval_cfg.get("mode", "2v2")).lower()
    if mode not in ("1v3", "2v2"):
        raise ValueError(f"periodic_eval.mode must be 1v3 or 2v2, got: {mode}")
    unit = 4 if mode == "1v3" else 2

    games = int(eval_cfg.get("games", 2000))
    if games <= 0:
        raise ValueError("periodic_eval.games must be > 0")
    seed_count = games // unit
    if seed_count <= 0:
        raise ValueError(f"periodic_eval.games too small for {mode}, need at least {unit}")
    actual_games = seed_count * unit
    if actual_games != games:
        logging.warning(
            f"periodic_eval games={games} not divisible by {unit}, actual_games={actual_games}"
        )

    seed_start = int(eval_cfg.get("seed_start", 10000))
    if bool(eval_cfg.get("seed_key_random", False)):
        seed_key = random.getrandbits(64)
    else:
        seed_key = int(eval_cfg.get("seed_key", 20260218))

    pts = eval_cfg.get("pts", [90, 45, 0, -135])
    if len(pts) != 4:
        raise ValueError(f"periodic_eval.pts must have 4 entries, got {len(pts)}")
    pts = [int(x) for x in pts]

    current_name = str(eval_cfg.get("name_current", "current_model"))
    compare_name = str(eval_cfg.get("name_compare", "compare_model"))
    if current_name == compare_name:
        compare_name = f"{compare_name}_ref"

    challenger = str(eval_cfg.get("challenger", "a")).lower()
    if challenger not in ("a", "b"):
        raise ValueError(f"periodic_eval.challenger must be a or b, got: {challenger}")

    disable_progress_bar = bool(eval_cfg.get("disable_progress_bar", True))
    keep_log_dir = bool(eval_cfg.get("keep_log_dir", False))

    log_dir_root = str(eval_cfg.get("log_dir_root", "")).strip()
    if not log_dir_root:
        ckpt_dir = path.dirname(str(config["control"]["state_file"]))
        log_dir_root = path.normpath(path.join(ckpt_dir, "..", "eval", f"periodic_{stage_name}"))
    log_dir = path.join(log_dir_root, f"step_{steps}")
    if path.isdir(log_dir) and not keep_log_dir:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    enable_amp = bool(eval_cfg.get("enable_amp", config["control"].get("enable_amp", False)))
    enable_compile = bool(eval_cfg.get("enable_compile", False))
    enable_rule_based_agari_guard = bool(eval_cfg.get("enable_rule_based_agari_guard", True))
    device = torch.device(str(eval_cfg.get("device", config["control"]["device"])))

    current_head = str(eval_cfg.get("decision_head_current", config["control"].get("decision_head", "policy")))
    compare_head = str(eval_cfg.get("decision_head_compare", "value"))

    current_policy_temp = float(eval_cfg.get("policy_temp_current", config["control"].get("policy_temp", 1.0)))
    current_policy_top_p = float(eval_cfg.get("policy_top_p_current", config["control"].get("policy_top_p", 1.0)))
    current_policy_epsilon = float(eval_cfg.get("policy_epsilon_current", config["control"].get("policy_epsilon", 0.0)))
    current_policy_stochastic = bool(eval_cfg.get("policy_stochastic_current", False))

    compare_policy_temp = float(eval_cfg.get("policy_temp_compare", 1.0))
    compare_policy_top_p = float(eval_cfg.get("policy_top_p_compare", 1.0))
    compare_policy_epsilon = float(eval_cfg.get("policy_epsilon_compare", 0.0))
    compare_policy_stochastic = bool(eval_cfg.get("policy_stochastic_compare", False))

    engine_current = load_engine(
        state_file=current_model_path,
        name=current_name,
        device=device,
        enable_amp=enable_amp,
        enable_compile=enable_compile,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        decision_head=current_head,
        policy_temp=current_policy_temp,
        policy_top_p=current_policy_top_p,
        policy_epsilon=current_policy_epsilon,
        policy_stochastic=current_policy_stochastic,
    )
    engine_compare = load_engine(
        state_file=compare_model,
        name=compare_name,
        device=device,
        enable_amp=enable_amp,
        enable_compile=enable_compile,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        decision_head=compare_head,
        policy_temp=compare_policy_temp,
        policy_top_p=compare_policy_top_p,
        policy_epsilon=compare_policy_epsilon,
        policy_stochastic=compare_policy_stochastic,
    )

    if challenger == "a":
        challenger_engine = engine_current
        champion_engine = engine_compare
    else:
        challenger_engine = engine_compare
        champion_engine = engine_current

    if mode == "1v3":
        env = OneVsThree(disable_progress_bar=disable_progress_bar, log_dir=log_dir)
        env.py_vs_py(
            challenger=challenger_engine,
            champion=champion_engine,
            seed_start=(seed_start, seed_key),
            seed_count=seed_count,
        )
    else:
        env = TwoVsTwo(disable_progress_bar=disable_progress_bar, log_dir=log_dir)
        env.py_vs_py(
            challenger=challenger_engine,
            champion=champion_engine,
            seed_start=(seed_start, seed_key),
            seed_count=seed_count,
        )

    current_stat = _collect_stat(log_dir, current_name, pts, disable_progress_bar)
    compare_stat = _collect_stat(log_dir, compare_name, pts, disable_progress_bar)

    summary = {
        "stage": stage_name,
        "steps": int(steps),
        "mode": mode,
        "games_requested": int(games),
        "games_actual": int(actual_games),
        "seed_start": int(seed_start),
        "seed_key": int(seed_key),
        "current_model": current_model_path,
        "compare_model": compare_model,
        "current": current_stat,
        "compare": compare_stat,
        "log_dir": log_dir,
    }

    os.makedirs(log_dir_root, exist_ok=True)
    summary_file = path.join(log_dir_root, f"summary_step_{steps}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if writer is not None:
        writer.add_scalar("periodic_eval/current_avg_pt", current_stat["avg_pt"], steps)
        writer.add_scalar("periodic_eval/current_avg_rank", current_stat["avg_rank"], steps)
        writer.add_scalar("periodic_eval/compare_avg_pt", compare_stat["avg_pt"], steps)
        writer.add_scalar("periodic_eval/compare_avg_rank", compare_stat["avg_rank"], steps)
        writer.add_scalar("periodic_eval/delta_avg_pt", current_stat["avg_pt"] - compare_stat["avg_pt"], steps)
        writer.add_scalar("periodic_eval/delta_avg_rank", compare_stat["avg_rank"] - current_stat["avg_rank"], steps)
        for pt_value, count in current_stat["pt_distribution"].items():
            writer.add_scalar(f"periodic_eval/current_pt_count/{pt_value}", count, steps)
        for pt_value, count in compare_stat["pt_distribution"].items():
            writer.add_scalar(f"periodic_eval/compare_pt_count/{pt_value}", count, steps)
        writer.flush()

    logging.info(
        "[periodic_eval] step=%s current(avg_pt=%.4f avg_rank=%.4f pt_dist=%s) "
        "vs compare(avg_pt=%.4f avg_rank=%.4f pt_dist=%s)",
        steps,
        current_stat["avg_pt"],
        current_stat["avg_rank"],
        current_stat["pt_distribution"],
        compare_stat["avg_pt"],
        compare_stat["avg_rank"],
        compare_stat["pt_distribution"],
    )
    return summary
