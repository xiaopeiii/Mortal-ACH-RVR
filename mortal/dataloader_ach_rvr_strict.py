import gzip
import json
import logging
import math
import random
from os import path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from config import config
from grp_loader import load_grp_from_cfg
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader


def _zero_sum_utility(scores: np.ndarray) -> np.ndarray:
    return scores - scores.mean()


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(rewards.shape[0])
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    next_adv = 0.0
    next_value = 0.0
    for i in range(n - 1, -1, -1):
        mask = 0.0 if bool(dones[i]) else 1.0
        delta = float(rewards[i]) + gamma * mask * float(next_value) - float(values[i])
        next_adv = delta + gamma * lam * mask * next_adv
        adv[i] = next_adv
        ret[i] = next_adv + float(values[i])
        next_value = float(values[i])
    return adv, ret


def _trace_path_for_log(log_file: str) -> str:
    if not log_file.endswith(".json.gz"):
        raise ValueError(f"expected .json.gz log file, got: {log_file}")
    return f"{log_file[:-8]}.trace.jsonl.gz"


def _load_trace_rows(trace_file: str) -> List[dict]:
    rows = []
    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty trace file: {trace_file}")
    return rows


def _opt_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(fv):
        return float("nan")
    return fv


class StrictAlignmentError(ValueError):
    def __init__(
        self,
        *,
        kind: str,
        filename: str,
        seat: int = -1,
        detail: str = "",
        total_samples: int = 0,
        mapped_samples: int = 0,
    ):
        self.kind = kind
        self.filename = filename
        self.seat = int(seat)
        self.detail = detail
        self.total_samples = int(total_samples)
        self.mapped_samples = int(mapped_samples)
        super().__init__(f"{filename}: kind={kind} seat={seat} detail={detail}")


class StrictAchRvrFileDataset(IterableDataset):
    def __init__(
        self,
        *,
        version: int,
        file_list: List[str],
        pts: List[float],
        gamma: float,
        gae_lambda: float,
        reward_source: str = "grp_plus_ern",
        oracle: bool = True,
        file_batch_size: int = 20,
        reserve_ratio: float = 0.0,
        player_names: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        num_epochs: int = 1,
        enable_augmentation: bool = False,
        augmented_first: bool = False,
        include_terminal_context: bool = True,
        manifest_map: Optional[Dict[str, Dict]] = None,
        strict_require_trace: bool = True,
        strict_require_old_logp: bool = True,
        allow_synth_trace: bool = False,
        allow_missing_old_logp: bool = False,
        synth_mismatch_policy: str = "fail",
        mismatch_report_file: str = "",
    ):
        super().__init__()
        self.version = version
        self.file_list = list(file_list)
        self.pts = pts
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_source = reward_source
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.include_terminal_context = include_terminal_context
        self.manifest_map = manifest_map or {}
        self.strict_require_trace = strict_require_trace
        self.strict_require_old_logp = strict_require_old_logp
        self.allow_synth_trace = allow_synth_trace
        self.allow_missing_old_logp = allow_missing_old_logp
        self.synth_mismatch_policy = str(synth_mismatch_policy).strip().lower()
        self.mismatch_report_file = mismatch_report_file
        self.iterator = None

        if not self.strict_require_trace:
            raise ValueError("strict dataset requires strict_require_trace=true")
        if self.allow_synth_trace:
            raise ValueError("strict dataset requires allow_synth_trace=false")
        if self.synth_mismatch_policy != "fail":
            raise ValueError("strict dataset requires synth_mismatch_policy='fail'")
        if self.strict_require_old_logp and self.allow_missing_old_logp:
            raise ValueError(
                "strict dataset requires allow_missing_old_logp=false when strict_require_old_logp=true"
            )

    def _report_alignment_issue(self, *, filename: str, seat: int, kind: str, detail: str):
        msg = f"{filename}: alignment issue kind={kind} seat={seat} detail={detail}"
        logging.warning(msg)
        if self.mismatch_report_file:
            row = {
                "file": filename,
                "seat": int(seat),
                "kind": kind,
                "detail": detail,
                "policy": "fail",
            }
            with open(self.mismatch_report_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _raise_alignment(
        self,
        *,
        filename: str,
        seat: int,
        kind: str,
        detail: str,
        total_samples: int = 0,
        mapped_samples: int = 0,
    ):
        self._report_alignment_issue(filename=filename, seat=seat, kind=kind, detail=detail)
        raise StrictAlignmentError(
            kind=kind,
            filename=filename,
            seat=seat,
            detail=detail,
            total_samples=total_samples,
            mapped_samples=mapped_samples,
        )

    def build_iter(self):
        if self.reward_source in ("grp", "grp_plus_ern"):
            self.grp, _ = load_grp_from_cfg(config["grp"], map_location=torch.device("cpu"))
            self.reward_calc = RewardCalculator(self.grp, self.pts)
        elif self.reward_source == "raw_score":
            self.grp = None
            self.reward_calc = RewardCalculator(None, self.pts)
        else:
            raise ValueError(f"unknown reward_source: {self.reward_source}")

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented: bool):
        random.shuffle(self.file_list)
        loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )

        buffer: List[dict] = []
        for start in range(0, len(self.file_list), self.file_batch_size):
            old_size = len(buffer)
            chunk = self.file_list[start:start + self.file_batch_size]
            self.populate_buffer(loader, chunk, buffer)
            cur_size = len(buffer)

            reserved = int((cur_size - old_size) * self.reserve_ratio)
            if reserved > cur_size:
                continue
            random.shuffle(buffer)
            yield from buffer[reserved:]
            del buffer[reserved:]

        random.shuffle(buffer)
        yield from buffer

    def _extract_gameplay(self, game) -> dict:
        obs = game.take_obs()
        invisible_obs = game.take_invisible_obs() if self.oracle else None
        actions = game.take_actions()
        masks = game.take_masks()
        at_kyoku = game.take_at_kyoku()
        dones = game.take_dones()
        apply_gamma = game.take_apply_gamma()
        seat_decision_idx = game.take_seat_decision_idx()
        at_kan_select = game.take_at_kan_select()
        player_id = int(game.take_player_id())
        grp = game.take_grp()

        if not (
            len(obs)
            == len(actions)
            == len(masks)
            == len(at_kyoku)
            == len(dones)
            == len(apply_gamma)
            == len(seat_decision_idx)
            == len(at_kan_select)
        ):
            raise ValueError(f"gameplay field length mismatch for seat={player_id}")

        grp_feature = grp.take_feature()
        rank_by_player = grp.take_rank_by_player()
        final_scores = np.asarray(grp.take_final_scores(), dtype=np.float32)
        final_utility = _zero_sum_utility(final_scores)

        if self.reward_source == "raw_score":
            raw = self.reward_calc.calc_delta_points(player_id, grp_feature, final_scores)
            step_reward = np.asarray([raw[k] for k in at_kyoku], dtype=np.float32)
        elif self.reward_source in ("grp", "grp_plus_ern"):
            grp_reward_by_kyoku = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            step_reward = np.asarray([grp_reward_by_kyoku[k] for k in at_kyoku], dtype=np.float32)
        else:
            raise ValueError(f"unknown reward_source: {self.reward_source}")

        return {
            "player_id": player_id,
            "obs": obs,
            "invisible_obs": invisible_obs,
            "actions": actions,
            "masks": masks,
            "at_kyoku": at_kyoku,
            "dones": dones,
            "apply_gamma": apply_gamma,
            "seat_decision_idx": np.asarray(seat_decision_idx, dtype=np.int64),
            "at_kan_select": np.asarray(at_kan_select, dtype=np.bool_),
            "step_reward": step_reward,
            "final_reward_vec": final_utility.astype(np.float32),
        }

    def _group_trace_rows_strict(
        self,
        rows: List[dict],
        *,
        filename: str,
    ) -> Tuple[Dict[int, Dict[int, dict]], Dict[int, Dict[int, dict]]]:
        table_steps: Dict[int, Dict[int, dict]] = {}
        seat_decision_map: Dict[int, Dict[int, dict]] = {0: {}, 1: {}, 2: {}, 3: {}}
        for raw in rows:
            try:
                table_step_idx = int(raw["table_step_idx"])
                seat = int(raw["seat"])
            except Exception as ex:
                self._raise_alignment(
                    filename=filename,
                    seat=-1,
                    kind="invalid_trace_shape",
                    detail=f"bad table_step_idx/seat: {ex}",
                )
            if seat not in (0, 1, 2, 3):
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="invalid_trace_shape",
                    detail="seat must be 0..3",
                )

            row = {
                "table_step_idx": table_step_idx,
                "seat": seat,
                "can_act": bool(raw.get("can_act", False)),
                "seat_decision_idx": int(raw.get("seat_decision_idx", -1)),
                "action_idx": None if raw.get("action_idx", None) is None else int(raw.get("action_idx")),
                "selected_logp": _opt_float(raw.get("selected_logp", None)),
                "selected_value": _opt_float(raw.get("selected_value", None)),
                "kan_action_idx": None
                if raw.get("kan_action_idx", None) is None
                else int(raw.get("kan_action_idx")),
                "kan_selected_logp": _opt_float(raw.get("kan_selected_logp", None)),
                "kan_selected_value": _opt_float(raw.get("kan_selected_value", None)),
                "decision_head": str(raw.get("decision_head", "")),
            }

            table_steps.setdefault(table_step_idx, {})
            if seat in table_steps[table_step_idx]:
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="invalid_trace_shape",
                    detail=f"duplicate row in table_step_idx={table_step_idx}",
                )
            table_steps[table_step_idx][seat] = row

            if row["can_act"]:
                decision_idx = row["seat_decision_idx"]
                if decision_idx < 0:
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="invalid_trace_shape",
                        detail="can_act row missing seat_decision_idx",
                    )
                if decision_idx in seat_decision_map[seat]:
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="ambiguous_mapping",
                        detail=f"duplicate seat_decision_idx={decision_idx}",
                    )
                seat_decision_map[seat][decision_idx] = row

        for table_step_idx, row_map in table_steps.items():
            seats = sorted(row_map.keys())
            if seats != [0, 1, 2, 3]:
                self._raise_alignment(
                    filename=filename,
                    seat=-1,
                    kind="invalid_trace_shape",
                    detail=f"table_step_idx={table_step_idx} does not have 4 seats, got={seats}",
                )
        return table_steps, seat_decision_map

    def _build_seat_lookup_and_targets(
        self,
        *,
        g: dict,
        trace_by_decision: Dict[int, dict],
        filename: str,
        seat: int,
        total_samples: int,
        mapped_samples: int,
    ) -> Tuple[dict, int]:
        n = len(g["actions"])
        lookup: Dict[int, Dict[str, int]] = {}
        for i in range(n):
            decision_idx = int(g["seat_decision_idx"][i])
            at_kan = bool(g["at_kan_select"][i])
            key = "kan" if at_kan else "main"
            entry = lookup.setdefault(decision_idx, {})
            if key in entry:
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="ambiguous_mapping",
                    detail=f"duplicate {key} sample at seat_decision_idx={decision_idx}",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples,
                )
            entry[key] = i

        old_logp = np.full(n, np.nan, dtype=np.float32)
        old_value = np.zeros(n, dtype=np.float32)
        mapped_here = 0
        for decision_idx, entry in lookup.items():
            row = trace_by_decision.get(decision_idx)
            if row is None:
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="ambiguous_mapping",
                    detail=f"missing trace for seat_decision_idx={decision_idx}",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples + mapped_here,
                )
            if "main" not in entry:
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="ambiguous_mapping",
                    detail=f"missing main gameplay sample at seat_decision_idx={decision_idx}",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples + mapped_here,
                )
            main_idx = entry["main"]
            row_action = row.get("action_idx", None)
            gp_action = int(g["actions"][main_idx])
            if row_action is None or int(row_action) != gp_action:
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="action_mismatch",
                    detail=f"main seat_decision_idx={decision_idx}, trace={row_action}, gameplay={gp_action}",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples + mapped_here,
                )

            sl = float(row["selected_logp"])
            sv = float(row["selected_value"])
            if self.strict_require_old_logp and not math.isfinite(sl):
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="missing_old_logp",
                    detail=f"main seat_decision_idx={decision_idx} selected_logp is missing/NaN",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples + mapped_here,
                )
            if not math.isfinite(sv):
                self._raise_alignment(
                    filename=filename,
                    seat=seat,
                    kind="missing_old_logp",
                    detail=f"main seat_decision_idx={decision_idx} selected_value is missing/NaN",
                    total_samples=total_samples,
                    mapped_samples=mapped_samples + mapped_here,
                )
            old_logp[main_idx] = sl
            old_value[main_idx] = sv
            mapped_here += 1

            kan_idx = entry.get("kan", None)
            if kan_idx is not None:
                row_kan_action = row.get("kan_action_idx", None)
                gp_kan_action = int(g["actions"][kan_idx])
                if row_kan_action is None:
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="missing_kan_fields",
                        detail=f"kan seat_decision_idx={decision_idx} missing kan_action_idx",
                        total_samples=total_samples,
                        mapped_samples=mapped_samples + mapped_here,
                    )
                if int(row_kan_action) != gp_kan_action:
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="action_mismatch",
                        detail=f"kan seat_decision_idx={decision_idx}, trace={row_kan_action}, gameplay={gp_kan_action}",
                        total_samples=total_samples,
                        mapped_samples=mapped_samples + mapped_here,
                    )

                ksl = float(row["kan_selected_logp"])
                ksv = float(row["kan_selected_value"])
                if self.strict_require_old_logp and not math.isfinite(ksl):
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="missing_old_logp",
                        detail=f"kan seat_decision_idx={decision_idx} kan_selected_logp is missing/NaN",
                        total_samples=total_samples,
                        mapped_samples=mapped_samples + mapped_here,
                    )
                if not math.isfinite(ksv):
                    self._raise_alignment(
                        filename=filename,
                        seat=seat,
                        kind="missing_kan_fields",
                        detail=f"kan seat_decision_idx={decision_idx} kan_selected_value is missing/NaN",
                        total_samples=total_samples,
                        mapped_samples=mapped_samples + mapped_here,
                    )
                old_logp[kan_idx] = ksl
                old_value[kan_idx] = ksv
                mapped_here += 1

        rewards = g["step_reward"].astype(np.float32)
        dones = np.asarray(g["dones"], dtype=np.bool_)
        adv, ret = _compute_gae(
            rewards=rewards,
            dones=dones,
            values=old_value,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        return {
            **g,
            "lookup": lookup,
            "old_logp": old_logp,
            "old_value": old_value,
            "advantage": adv,
            "return": ret,
        }, mapped_here

    def _infer_dummy_tensors(self, seat_seq: Dict[int, dict]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        obs_shape = None
        mask_shape = None
        inv_shape = None
        for seq in seat_seq.values():
            if len(seq["obs"]) > 0:
                obs_shape = seq["obs"][0].shape
                mask_shape = seq["masks"][0].shape
                if self.oracle and seq["invisible_obs"]:
                    inv_shape = seq["invisible_obs"][0].shape
                break
        if obs_shape is None or mask_shape is None:
            raise ValueError("cannot infer dummy tensor shapes from empty gameplay")
        dummy_obs = np.zeros(obs_shape, dtype=np.float32)
        dummy_mask = np.zeros(mask_shape, dtype=np.bool_)
        fallback_idx = 45 if mask_shape[0] > 45 else 0
        dummy_mask[fallback_idx] = True
        dummy_inv = np.zeros(inv_shape, dtype=np.float32) if inv_shape is not None else None
        return dummy_obs, dummy_mask, dummy_inv

    def _process_loaded_file(
        self,
        log_file: str,
        file_games,
        *,
        meta: Dict[str, Any],
        buffer: Optional[List[dict]],
    ) -> dict:
        filename = path.basename(log_file)
        trace_file = _trace_path_for_log(log_file)
        if not path.isfile(trace_file):
            self._raise_alignment(
                filename=filename,
                seat=-1,
                kind="missing_trace",
                detail=f"trace file not found: {trace_file}",
            )
        trace_rows = _load_trace_rows(trace_file)
        table_steps, trace_decision_map = self._group_trace_rows_strict(trace_rows, filename=filename)

        seat_to_game: Dict[int, dict] = {}
        for game in file_games:
            g = self._extract_gameplay(game)
            seat_to_game[g["player_id"]] = g
        if len(seat_to_game) != 4:
            self._raise_alignment(
                filename=filename,
                seat=-1,
                kind="invalid_trace_shape",
                detail=f"strict aligned dataset expects 4 seats, got {sorted(seat_to_game.keys())}",
            )

        total_samples = sum(len(g["actions"]) for g in seat_to_game.values())
        mapped_samples = 0
        seat_seq: Dict[int, dict] = {}
        for seat in range(4):
            seq, mapped_here = self._build_seat_lookup_and_targets(
                g=seat_to_game[seat],
                trace_by_decision=trace_decision_map[seat],
                filename=filename,
                seat=seat,
                total_samples=total_samples,
                mapped_samples=mapped_samples,
            )
            mapped_samples += mapped_here
            seat_seq[seat] = seq

        report = {
            "file": filename,
            "total_samples": int(total_samples),
            "mapped_samples": int(mapped_samples),
            "mapping_coverage": float(mapped_samples / total_samples) if total_samples > 0 else 1.0,
            "missing_old_logp": 0,
            "ambiguous_mapping": 0,
            "missing_trace": 0,
            "missing_kan_fields": 0,
            "action_mismatch": 0,
            "ok": True,
        }
        if mapped_samples != total_samples:
            self._raise_alignment(
                filename=filename,
                seat=-1,
                kind="ambiguous_mapping",
                detail=f"mapped_samples={mapped_samples}, total_samples={total_samples}",
                total_samples=total_samples,
                mapped_samples=mapped_samples,
            )

        if buffer is None:
            return report

        dummy_obs, dummy_mask, dummy_inv = self._infer_dummy_tensors(seat_seq)
        max_table_step = max(table_steps.keys()) if table_steps else 0
        game_id = f"{filename}::0"

        for table_step_idx in sorted(table_steps.keys()):
            row_map = table_steps[table_step_idx]
            is_preterminal = table_step_idx == (max_table_step - 1)
            step_items = []
            step_done = False
            for seat in range(4):
                row = row_map[seat]
                can_act = bool(row["can_act"])
                decision_idx = int(row["seat_decision_idx"])
                seq = seat_seq[seat]
                lookup = seq["lookup"].get(decision_idx, {})
                main_idx = lookup.get("main", None)

                if can_act and main_idx is not None:
                    idx = int(main_idx)
                    obs = seq["obs"][idx]
                    invisible_obs = (
                        seq["invisible_obs"][idx]
                        if self.oracle and seq["invisible_obs"] is not None
                        else dummy_inv
                    )
                    action = int(seq["actions"][idx])
                    mask = seq["masks"][idx]
                    reward = float(seq["step_reward"][idx])
                    ret = float(seq["return"][idx])
                    adv = float(seq["advantage"][idx])
                    old_logp = float(seq["old_logp"][idx])
                    old_value = float(seq["old_value"][idx])
                    done = bool(seq["dones"][idx])
                    at_kyoku = int(seq["at_kyoku"][idx])
                else:
                    obs = dummy_obs
                    invisible_obs = dummy_inv
                    action = -1
                    mask = dummy_mask
                    reward = 0.0
                    ret = 0.0
                    adv = 0.0
                    old_logp = float(row["selected_logp"]) if can_act else float("nan")
                    old_value = float(row["selected_value"]) if can_act else float("nan")
                    if can_act and self.strict_require_old_logp and not math.isfinite(old_logp):
                        self._raise_alignment(
                            filename=filename,
                            seat=seat,
                            kind="missing_old_logp",
                            detail=f"table_step_idx={table_step_idx} can_act main row missing selected_logp",
                            total_samples=total_samples,
                            mapped_samples=mapped_samples,
                        )
                    done = False
                    at_kyoku = 0

                step_done = step_done or done
                step_items.append(
                    {
                        "obs": obs,
                        "invisible_obs": invisible_obs,
                        "action": action,
                        "mask": mask,
                        "reward": reward,
                        "return": ret,
                        "advantage": adv,
                        "done": done,
                        "is_preterminal": is_preterminal,
                        "at_kyoku": at_kyoku,
                        "final_reward_vec": seq["final_reward_vec"],
                        "game_id": game_id,
                        "step_idx": int(table_step_idx),
                        "table_step_idx": int(table_step_idx),
                        "seat": int(seat),
                        "can_act": can_act,
                        "at_kan_select": False,
                        "decision_head": str(row.get("decision_head", "")),
                        "old_logp": old_logp,
                        "old_value": old_value,
                        "param_version": int(meta.get("param_version", -1)),
                        "opponent_id": str(meta.get("opponent_id", "")),
                        "profile": str(meta.get("profile", "")),
                        "client_id": str(meta.get("client_id", "")),
                        "aligned_four_seat": True,
                    }
                )

                kan_idx = lookup.get("kan", None)
                if can_act and kan_idx is not None:
                    k = int(kan_idx)
                    step_done = step_done or bool(seq["dones"][k])
                    step_items.append(
                        {
                            "obs": seq["obs"][k],
                            "invisible_obs": (
                                seq["invisible_obs"][k]
                                if self.oracle and seq["invisible_obs"] is not None
                                else dummy_inv
                            ),
                            "action": int(seq["actions"][k]),
                            "mask": seq["masks"][k],
                            "reward": float(seq["step_reward"][k]),
                            "return": float(seq["return"][k]),
                            "advantage": float(seq["advantage"][k]),
                            "done": bool(seq["dones"][k]),
                            "is_preterminal": is_preterminal,
                            "at_kyoku": int(seq["at_kyoku"][k]),
                            "final_reward_vec": seq["final_reward_vec"],
                            "game_id": game_id,
                            "step_idx": int(table_step_idx),
                            "table_step_idx": int(table_step_idx),
                            "seat": int(seat),
                            "can_act": True,
                            "at_kan_select": True,
                            "decision_head": str(row.get("decision_head", "")),
                            "old_logp": float(seq["old_logp"][k]),
                            "old_value": float(seq["old_value"][k]),
                            "param_version": int(meta.get("param_version", -1)),
                            "opponent_id": str(meta.get("opponent_id", "")),
                            "profile": str(meta.get("profile", "")),
                            "client_id": str(meta.get("client_id", "")),
                            "aligned_four_seat": True,
                        }
                    )

            if (not self.include_terminal_context) and step_done:
                continue
            buffer.extend(step_items)
        return report

    def populate_buffer(self, loader: GameplayLoader, file_chunk: List[str], buffer: List[dict]):
        data = loader.load_gz_log_files(file_chunk)
        for chunk_idx, file_games in enumerate(data):
            log_file = file_chunk[chunk_idx]
            filename = path.basename(log_file)
            meta = self.manifest_map.get(filename, {})
            self._process_loaded_file(log_file, file_games, meta=meta, buffer=buffer)

    def audit_file(self, loader: GameplayLoader, log_file: str, *, meta: Optional[Dict[str, Any]] = None) -> dict:
        data = loader.load_gz_log_files([log_file])
        file_games = data[0]
        return self._process_loaded_file(log_file, file_games, meta=meta or {}, buffer=None)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def collate_ach_rvr_strict(batch: List[dict]) -> Dict[str, torch.Tensor]:
    obs = torch.as_tensor(np.stack([b["obs"] for b in batch], axis=0), dtype=torch.float32)

    has_oracle = batch[0]["invisible_obs"] is not None
    invisible_obs = None
    if has_oracle:
        invisible_obs = torch.as_tensor(
            np.stack([b["invisible_obs"] for b in batch], axis=0),
            dtype=torch.float32,
        )

    masks = torch.as_tensor(np.stack([b["mask"] for b in batch], axis=0), dtype=torch.bool)
    actions = torch.as_tensor([b["action"] for b in batch], dtype=torch.int64)
    rewards = torch.as_tensor([b["reward"] for b in batch], dtype=torch.float32)
    returns = torch.as_tensor([b["return"] for b in batch], dtype=torch.float32)
    adv = torch.as_tensor([b["advantage"] for b in batch], dtype=torch.float32)
    dones = torch.as_tensor([b["done"] for b in batch], dtype=torch.bool)
    is_preterminal = torch.as_tensor([b["is_preterminal"] for b in batch], dtype=torch.bool)
    at_kyoku = torch.as_tensor([b["at_kyoku"] for b in batch], dtype=torch.int64)
    final_reward_vec = torch.as_tensor(
        np.stack([b["final_reward_vec"] for b in batch], axis=0),
        dtype=torch.float32,
    )
    can_act = torch.as_tensor([b["can_act"] for b in batch], dtype=torch.bool)
    at_kan_select = torch.as_tensor([b.get("at_kan_select", False) for b in batch], dtype=torch.bool)

    old_logp = torch.as_tensor([b["old_logp"] for b in batch], dtype=torch.float32)
    old_value = torch.as_tensor([b["old_value"] for b in batch], dtype=torch.float32)
    valid_old = can_act & torch.isfinite(old_logp)
    can_act_count = int(can_act.sum().item())
    old_logp_coverage = (
        float(valid_old.sum().item()) / float(can_act_count) if can_act_count > 0 else 1.0
    )

    out = {
        "obs": obs,
        "actions": actions,
        "masks": masks,
        "rewards": rewards,
        "returns": returns,
        "advantages": adv,
        "dones": dones,
        "is_preterminal": is_preterminal,
        "at_kyoku": at_kyoku,
        "final_reward_vec": final_reward_vec,
        "game_id": [b["game_id"] for b in batch],
        "step_idx": torch.as_tensor([b["step_idx"] for b in batch], dtype=torch.int64),
        "table_step_idx": torch.as_tensor([b["table_step_idx"] for b in batch], dtype=torch.int64),
        "seat": torch.as_tensor([b["seat"] for b in batch], dtype=torch.int64),
        "can_act": can_act,
        "at_kan_select": at_kan_select,
        "old_logp": old_logp,
        "old_value": old_value,
        "old_logp_coverage": old_logp_coverage,
        "aligned_four_seat": True,
        "param_version": torch.as_tensor([b.get("param_version", -1) for b in batch], dtype=torch.int64),
        "opponent_id": [b.get("opponent_id", "") for b in batch],
        "profile": [b.get("profile", "") for b in batch],
        "client_id": [b.get("client_id", "") for b in batch],
        "decision_head": [b.get("decision_head", "") for b in batch],
    }
    if invisible_obs is not None:
        out["invisible_obs"] = invisible_obs
    return out


def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
