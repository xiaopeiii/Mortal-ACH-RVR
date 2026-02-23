import random
from os import path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from config import config
from grp_loader import load_grp_from_cfg
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader


class AchRvrFileDataset(IterableDataset):
    """Step-level dataset for ACH + RVR.

    Each yielded item is a dict containing private/oracle obs, action, masks,
    reward/return/advantage and metadata for online traceability.
    """

    def __init__(
        self,
        *,
        version: int,
        file_list: List[str],
        pts: List[float],
        gamma: float,
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
    ):
        super().__init__()
        self.version = version
        self.file_list = list(file_list)
        self.pts = pts
        self.gamma = gamma
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
        self.iterator = None

    def build_iter(self):
        if self.reward_source in ('grp', 'grp_plus_ern'):
            self.grp, _ = load_grp_from_cfg(config['grp'], map_location=torch.device('cpu'))
            self.reward_calc = RewardCalculator(self.grp, self.pts)
        elif self.reward_source == 'raw_score':
            self.grp = None
            self.reward_calc = RewardCalculator(None, self.pts)
        else:
            raise ValueError(f'unknown reward_source: {self.reward_source}')

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

        buffer = []
        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_size = len(buffer)
            chunk = self.file_list[start_idx:start_idx + self.file_batch_size]
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

    def populate_buffer(self, loader: GameplayLoader, file_chunk: List[str], buffer: List[Dict]):
        data = loader.load_gz_log_files(file_chunk)
        for chunk_idx, file_games in enumerate(data):
            filename = path.basename(file_chunk[chunk_idx])
            meta = self.manifest_map.get(filename, {})
            for local_game_idx, game in enumerate(file_games):
                obs = game.take_obs()
                invisible_obs = game.take_invisible_obs() if self.oracle else None
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                grp = game.take_grp()
                player_id = int(game.take_player_id())

                game_size = len(obs)
                if game_size == 0:
                    continue

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                final_scores = np.asarray(grp.take_final_scores(), dtype=np.float32)
                final_utility = final_scores - final_scores.mean()

                # per-kyoku reward
                if self.reward_source == 'raw_score':
                    raw = self.reward_calc.calc_delta_points(player_id, grp_feature, final_scores)
                    step_reward = np.asarray([raw[k] for k in at_kyoku], dtype=np.float32)
                elif self.reward_source in ('grp', 'grp_plus_ern'):
                    grp_reward_by_kyoku = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                    step_reward = np.asarray([grp_reward_by_kyoku[k] for k in at_kyoku], dtype=np.float32)
                else:
                    raise ValueError(f'unknown reward_source: {self.reward_source}')

                returns = np.zeros(game_size, dtype=np.float32)
                run = 0.0
                for i in range(game_size - 1, -1, -1):
                    if dones[i]:
                        run = 0.0
                    discount = self.gamma if apply_gamma[i] else 1.0
                    run = float(step_reward[i]) + discount * run
                    returns[i] = run
                advantage = returns - returns.mean()

                preterminal_idx = max(0, game_size - 2)
                game_id = f"{filename}::{local_game_idx}"

                for step_idx in range(game_size):
                    if (not self.include_terminal_context) and dones[step_idx]:
                        continue

                    item = {
                        'obs': obs[step_idx],
                        'invisible_obs': invisible_obs[step_idx] if invisible_obs is not None else None,
                        'action': int(actions[step_idx]),
                        'mask': masks[step_idx],
                        'reward': float(step_reward[step_idx]),
                        'return': float(returns[step_idx]),
                        'advantage': float(advantage[step_idx]),
                        'done': bool(dones[step_idx]),
                        'is_preterminal': bool(step_idx == preterminal_idx),
                        'at_kyoku': int(at_kyoku[step_idx]),
                        'final_reward_vec': final_utility.astype(np.float32),
                        'game_id': game_id,
                        'step_idx': int(step_idx),
                        'seat': player_id,
                        'pi_old_logp': meta.get('pi_old_logp'),
                        'param_version': int(meta.get('param_version', -1)),
                        'opponent_id': str(meta.get('opponent_id', '')),
                        'profile': str(meta.get('profile', '')),
                        'client_id': str(meta.get('client_id', '')),
                    }
                    buffer.append(item)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def collate_ach_rvr(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    obs = torch.as_tensor(np.stack([b['obs'] for b in batch], axis=0), dtype=torch.float32)

    has_oracle = batch[0]['invisible_obs'] is not None
    invisible_obs = None
    if has_oracle:
        invisible_obs = torch.as_tensor(np.stack([b['invisible_obs'] for b in batch], axis=0), dtype=torch.float32)

    masks = torch.as_tensor(np.stack([b['mask'] for b in batch], axis=0), dtype=torch.bool)
    actions = torch.as_tensor([b['action'] for b in batch], dtype=torch.int64)
    rewards = torch.as_tensor([b['reward'] for b in batch], dtype=torch.float32)
    returns = torch.as_tensor([b['return'] for b in batch], dtype=torch.float32)
    adv = torch.as_tensor([b['advantage'] for b in batch], dtype=torch.float32)
    dones = torch.as_tensor([b['done'] for b in batch], dtype=torch.bool)
    is_preterminal = torch.as_tensor([b['is_preterminal'] for b in batch], dtype=torch.bool)
    at_kyoku = torch.as_tensor([b['at_kyoku'] for b in batch], dtype=torch.int64)
    final_reward_vec = torch.as_tensor(np.stack([b['final_reward_vec'] for b in batch], axis=0), dtype=torch.float32)

    pi_old = [b['pi_old_logp'] for b in batch]
    has_old = all(v is not None for v in pi_old)
    old_logp = torch.as_tensor(pi_old, dtype=torch.float32) if has_old else None

    out = {
        'obs': obs,
        'actions': actions,
        'masks': masks,
        'rewards': rewards,
        'returns': returns,
        'advantages': adv,
        'dones': dones,
        'is_preterminal': is_preterminal,
        'at_kyoku': at_kyoku,
        'final_reward_vec': final_reward_vec,
        'game_id': [b['game_id'] for b in batch],
        'step_idx': torch.as_tensor([b['step_idx'] for b in batch], dtype=torch.int64),
        'seat': torch.as_tensor([b['seat'] for b in batch], dtype=torch.int64),
        'param_version': torch.as_tensor([b.get('param_version', -1) for b in batch], dtype=torch.int64),
        'opponent_id': [b.get('opponent_id', '') for b in batch],
        'profile': [b.get('profile', '') for b in batch],
        'client_id': [b.get('client_id', '') for b in batch],
        'has_old_logp': has_old,
    }
    if invisible_obs is not None:
        out['invisible_obs'] = invisible_obs
    if old_logp is not None:
        out['old_logp'] = old_logp
    return out


def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
