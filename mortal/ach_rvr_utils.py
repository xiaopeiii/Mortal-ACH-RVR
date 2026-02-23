import json
import logging
import os
import random
from datetime import datetime
from glob import glob
from os import path
from typing import Dict, Iterable, List, Optional

import torch

from config import config


def now_ts() -> float:
    return datetime.now().timestamp()


def ensure_parent(filename: str) -> None:
    os.makedirs(path.dirname(path.abspath(filename)), exist_ok=True)


def build_file_list(dataset_cfg: dict, *, shuffle: bool = False) -> List[str]:
    file_index = dataset_cfg['file_index']
    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        file_list = index['file_list']
    else:
        file_list = []
        for pat in dataset_cfg['globs']:
            file_list.extend(glob(pat, recursive=True))
        file_list.sort(reverse=True)
        ensure_parent(file_index)
        torch.save({'file_list': file_list}, file_index)
    if shuffle:
        random.shuffle(file_list)
    return file_list


def load_legacy_states(state: dict, model, *, strict: bool = False) -> None:
    if 'mortal' in state:
        model.private_brain.load_state_dict(state['mortal'], strict=strict)
    if 'policy_y' in state:
        model.policy_y.load_state_dict(state['policy_y'], strict=False)
    elif 'policy' in state:
        model.policy_y.load_state_dict(state['policy'], strict=False)
    if 'value_head' in state:
        model.value_head.load_state_dict(state['value_head'], strict=False)
    if 'rv_head' in state:
        model.rv_head.load_state_dict(state['rv_head'], strict=False)
    if 'ern' in state:
        model.ern.load_state_dict(state['ern'], strict=False)


class LeaguePool:
    def __init__(self, *, cfg: dict, state_file: str, best_state_file: str):
        self.enabled = bool(cfg.get('enabled', False))
        self.pool_size = int(cfg.get('pool_size', 8))
        self.p_best = float(cfg.get('p_best', 0.5))
        self.p_latest = float(cfg.get('p_latest', 0.3))
        self.p_history = float(cfg.get('p_history', 0.2))
        self.refresh_every = int(cfg.get('refresh_every', 2000))
        self.min_eval_games = int(cfg.get('min_eval_games', 2000))

        self.state_file = state_file
        self.best_state_file = best_state_file
        self.history = []

    def _refresh(self):
        prefix = path.splitext(self.state_file)[0] + '_step*.pth'
        cands = sorted(glob(prefix))
        if len(cands) > self.pool_size:
            cands = cands[-self.pool_size:]
        self.history = cands

    def sample(self) -> Optional[str]:
        if not self.enabled:
            return None
        self._refresh()

        choices = []
        probs = []

        if path.exists(self.best_state_file):
            choices.append(self.best_state_file)
            probs.append(self.p_best)
        if path.exists(self.state_file):
            choices.append(self.state_file)
            probs.append(self.p_latest)
        if self.history:
            choices.append(random.choice(self.history))
            probs.append(self.p_history)

        if not choices:
            return None

        s = sum(probs)
        probs = [p / s for p in probs]
        return random.choices(choices, weights=probs, k=1)[0]


def save_ach_rvr_state(
    *,
    filename: str,
    model,
    optimizer,
    scheduler,
    scaler,
    steps: int,
    best_perf: dict,
    policy_meta: dict,
    league_meta: dict,
    extra: Optional[dict] = None,
):
    ensure_parent(filename)
    state = {
        'mortal': model.private_brain.state_dict(),
        'policy_y': model.policy_y.state_dict(),
        'policy': model.policy_y.state_dict(),
        'value_head': model.value_head.state_dict(),
        'rv_head': model.rv_head.state_dict(),
        'ern': model.ern.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': steps,
        'timestamp': now_ts(),
        'best_perf': best_perf,
        'config': config,
        'policy_meta': policy_meta,
        'league_meta': league_meta,
    }
    if extra:
        state.update(extra)
    torch.save(state, filename)


def checkpoint_snapshot_name(state_file: str, steps: int) -> str:
    return f"{path.splitext(state_file)[0]}_step{steps}.pth"


def maybe_load_state(state_file: str, *, map_location, model, optimizer, scheduler, scaler):
    steps = 0
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}
    policy_meta = {'method': 'ach+rvr', 'initialized_from': 'random'}
    league_meta = {}

    if not path.exists(state_file):
        return steps, best_perf, policy_meta, league_meta

    state = torch.load(state_file, weights_only=False, map_location=map_location)
    load_legacy_states(state, model, strict=False)
    if 'optimizer' in state:
        try:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        except Exception as ex:
            logging.warning(f'skip optimizer/scheduler restore: {ex}')
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    steps = int(state.get('steps', 0))
    best_perf = state.get('best_perf', best_perf)
    policy_meta = state.get('policy_meta', policy_meta)
    league_meta = state.get('league_meta', league_meta)
    ts = state.get('timestamp')
    if ts is not None:
        logging.info(f"loaded: {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}")
    return steps, best_perf, policy_meta, league_meta
