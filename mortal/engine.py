import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        policy = None,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
        decision_head = 'value',
        policy_temp = 1,
        policy_top_p = 1,
        policy_epsilon = 0,
        policy_stochastic = False,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = None if dqn is None else dqn.to(self.device).eval()
        self.policy = None if policy is None else policy.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        self.decision_head = decision_head
        self.policy_temp = policy_temp
        self.policy_top_p = policy_top_p
        self.policy_epsilon = policy_epsilon
        self.policy_stochastic = policy_stochastic
        if self.decision_head not in ('value', 'policy'):
            raise ValueError(f'unknown decision_head: {self.decision_head}')
        if self.decision_head == 'value' and self.dqn is None:
            raise ValueError('decision_head=value requires a dqn model')
        if self.decision_head == 'policy' and self.policy is None:
            raise ValueError('decision_head=policy requires a policy model')

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        batch_size = obs.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                feat = latent
            case 2 | 3 | 4:
                feat = self.brain(obs)

        if self.decision_head == 'policy':
            policy_logits = self.policy(feat).masked_fill(~masks, -torch.inf)
            temp = max(float(self.policy_temp), 1e-6)
            sample_logits = policy_logits / temp
            sample_log_probs = sample_logits.log_softmax(-1)
            sample_probs = sample_log_probs.exp()

            if self.policy_stochastic:
                is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                actions = sample_top_p(sample_logits, self.policy_top_p)
            elif self.policy_epsilon > 0:
                is_greedy = torch.full(
                    (batch_size,),
                    1 - self.policy_epsilon,
                    device=self.device,
                ).bernoulli().to(torch.bool)
                sampled = sample_top_p(sample_logits, self.policy_top_p)
                greedy = policy_logits.argmax(-1)
                actions = torch.where(is_greedy, greedy, sampled)
            else:
                if self.policy_top_p < 1 or temp != 1:
                    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    actions = sample_top_p(sample_logits, self.policy_top_p)
                else:
                    is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    actions = policy_logits.argmax(-1)

            batch_index = torch.arange(batch_size, device=self.device)
            selected_logp = sample_log_probs[batch_index, actions]

            if self.dqn is not None:
                q_out = self.dqn(feat, masks)
                selected_value = (sample_probs * q_out.masked_fill(~masks, 0.0)).sum(-1)
            else:
                selected_value = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

            return (
                actions.tolist(),
                policy_logits.tolist(),
                masks.tolist(),
                is_greedy.tolist(),
                selected_logp.tolist(),
                selected_value.tolist(),
            )

        q_out = self.dqn(feat, masks)
        logits = (q_out / max(float(self.boltzmann_temp), 1e-6)).masked_fill(~masks, -torch.inf)
        log_probs = logits.log_softmax(-1)
        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        batch_index = torch.arange(batch_size, device=self.device)
        selected_logp = log_probs[batch_index, actions]
        selected_value = q_out.max(-1).values

        return (
            actions.tolist(),
            q_out.tolist(),
            masks.tolist(),
            is_greedy.tolist(),
            selected_logp.tolist(),
            selected_value.tolist(),
        )

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
