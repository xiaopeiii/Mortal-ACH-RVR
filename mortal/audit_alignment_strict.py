import argparse
import json
import logging
import os
from os import path

import prelude


def main():
    parser = argparse.ArgumentParser(description="Audit strict ACH+RVR alignment without training.")
    parser.add_argument("--config", type=str, default="", help="Optional config TOML path.")
    parser.add_argument("--limit", type=int, default=0, help="Only audit first N files.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Summary JSON path. Default: <dataset_dir>/strict_alignment_audit_summary.json",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Failure JSONL path. Default: <dataset_dir>/strict_alignment_audit_failures.jsonl",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["MORTAL_CFG"] = args.config

    from ach_rvr_utils import build_file_list
    from config import config
    from dataloader_ach_rvr_strict import (
        StrictAchRvrFileDataset,
        StrictAlignmentError,
    )
    from libriichi.dataset import GameplayLoader

    strict_cfg = config.get("strict", {})
    if not bool(strict_cfg.get("enabled", False)):
        raise RuntimeError("this auditor is for strict mode; set strict.enabled=true in config")
    if not bool(strict_cfg.get("require_trace", True)):
        raise RuntimeError("strict audit requires strict.require_trace=true")
    if not bool(strict_cfg.get("require_old_logp", True)):
        raise RuntimeError("strict audit requires strict.require_old_logp=true")

    file_list = build_file_list(config["dataset"], shuffle=False)
    if args.limit > 0:
        file_list = file_list[: args.limit]
    if not file_list:
        raise RuntimeError("no files found from dataset.globs")

    dataset_dir = path.dirname(file_list[0])
    output_json = args.output_json or path.join(dataset_dir, "strict_alignment_audit_summary.json")
    output_jsonl = args.output_jsonl or path.join(dataset_dir, "strict_alignment_audit_failures.jsonl")
    if path.isfile(output_jsonl):
        os.remove(output_jsonl)

    ds = StrictAchRvrFileDataset(
        version=int(config["control"]["version"]),
        file_list=[],
        pts=config["env"]["pts"],
        gamma=float(config.get("ach", {}).get("gamma", 1.0)),
        gae_lambda=float(config.get("ach", {}).get("gae_lambda", 0.95)),
        reward_source="raw_score",
        oracle=bool(config.get("oracle_dataset", {}).get("include_invisible_obs", True)),
        file_batch_size=1,
        reserve_ratio=0.0,
        player_names=[],
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        include_terminal_context=bool(config.get("oracle_dataset", {}).get("include_terminal_context", True)),
        manifest_map={},
        strict_require_trace=True,
        strict_require_old_logp=True,
        allow_synth_trace=False,
        allow_missing_old_logp=False,
        synth_mismatch_policy="fail",
        mismatch_report_file="",
    )

    loader = GameplayLoader(
        version=int(config["control"]["version"]),
        oracle=bool(config.get("oracle_dataset", {}).get("include_invisible_obs", True)),
        player_names=[],
        excludes=None,
        augmented=False,
    )

    counts = {
        "missing_old_logp": 0,
        "ambiguous_mapping": 0,
        "missing_trace": 0,
        "missing_kan_fields": 0,
        "action_mismatch": 0,
        "other": 0,
    }
    total_samples = 0
    mapped_samples = 0
    ok_files = 0
    failed_files = 0

    for i, log_file in enumerate(file_list, start=1):
        try:
            rep = ds.audit_file(loader, log_file, meta={})
            total_samples += int(rep["total_samples"])
            mapped_samples += int(rep["mapped_samples"])
            ok_files += 1
        except StrictAlignmentError as ex:
            failed_files += 1
            total_samples += int(ex.total_samples)
            mapped_samples += int(ex.mapped_samples)
            kind = ex.kind if ex.kind in counts else "other"
            counts[kind] += 1
            row = {
                "file": ex.filename,
                "seat": int(ex.seat),
                "kind": ex.kind,
                "detail": ex.detail,
                "total_samples": int(ex.total_samples),
                "mapped_samples": int(ex.mapped_samples),
            }
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if i % 200 == 0 or i == len(file_list):
            logging.info("audit progress: %s/%s", i, len(file_list))

    mapping_coverage = float(mapped_samples / total_samples) if total_samples > 0 else 1.0
    summary = {
        "files_total": len(file_list),
        "files_ok": ok_files,
        "files_failed": failed_files,
        "total_samples": int(total_samples),
        "mapped_samples": int(mapped_samples),
        "mapping_coverage": mapping_coverage,
        **counts,
        "failure_jsonl": output_jsonl,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {output_json}")
    print(f"failures: {output_jsonl}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
