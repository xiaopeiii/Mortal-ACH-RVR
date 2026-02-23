import torch
import numpy as np
import os
import shutil
import secrets
import logging
from os import path
from model import Brain, DQN, PolicyHead
from engine import MortalEngine
from torch_compile import maybe_compile
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

def _load_policy_from_state(state, version, *, enable_compile, device, label):
    policy_state = state.get('policy')
    if policy_state is None:
        return None
    policy = PolicyHead(version=version).eval()
    policy.load_state_dict(policy_state)
    if enable_compile:
        policy = maybe_compile(policy, enable=True, device=device, label=f'{label}:policy')
    return policy

def _reset_log_dir(log_dir):
    if path.isdir(log_dir):
        # On Windows, files in log dir may disappear during traversal
        # (e.g. race with prior cleanup/IO flush). Treat as benign.
        shutil.rmtree(log_dir, ignore_errors=True)
    elif path.exists(log_dir):
        try:
            os.remove(log_dir)
        except FileNotFoundError:
            pass

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = None
        stable_mortal.load_state_dict(state['mortal'])
        if 'current_dqn' in state:
            stable_dqn = DQN(version=version).eval()
            stable_dqn.load_state_dict(state['current_dqn'])
        baseline_decision_head = baseline_cfg.get('decision_head', 'value')
        baseline_policy = None
        if baseline_decision_head == 'policy':
            baseline_policy = _load_policy_from_state(
                state,
                version,
                enable_compile=baseline_cfg['enable_compile'],
                device=device,
                label='baseline',
            )
            if baseline_policy is None:
                raise KeyError('baseline.test.decision_head=policy but checkpoint has no `policy` state')
        elif stable_dqn is None:
            raise KeyError('baseline.test.decision_head=value but checkpoint has no `current_dqn` state')
        if baseline_cfg['enable_compile']:
            stable_mortal = maybe_compile(
                stable_mortal,
                enable=True,
                device=device,
                label="baseline:brain",
            )
            if stable_dqn is not None:
                stable_dqn = maybe_compile(
                    stable_dqn,
                    enable=True,
                    device=device,
                    label="baseline:dqn",
                )

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            policy = baseline_policy,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_quick_eval = baseline_cfg.get('enable_quick_eval', True),
            enable_rule_based_agari_guard = True,
            name = 'baseline',
            decision_head = baseline_decision_head,
            policy_temp = baseline_cfg.get('policy_temp', 1.0),
            policy_top_p = baseline_cfg.get('policy_top_p', 1.0),
            policy_epsilon = baseline_cfg.get('policy_epsilon', 0.0),
            policy_stochastic = baseline_cfg.get('policy_stochastic', False),
        )
        self.chal_version = config['control']['version']
        test_cfg = config['test_play']
        self.log_dir = path.abspath(test_cfg['log_dir'])
        self.chal_decision_head = test_cfg.get('decision_head', config['control'].get('decision_head', 'value'))
        self.chal_policy_temp = test_cfg.get('policy_temp', 1.0)
        self.chal_policy_top_p = test_cfg.get('policy_top_p', 1.0)
        self.chal_policy_epsilon = test_cfg.get('policy_epsilon', 0.0)

    def test_play(self, seed_count, mortal, dqn, policy, device):
        torch.backends.cudnn.benchmark = False
        if self.chal_decision_head == 'policy' and policy is None:
            raise ValueError('test_play.decision_head=policy but no policy is provided')
        engine_chal = MortalEngine(
            mortal,
            dqn,
            policy = policy,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            enable_quick_eval = config['test_play'].get('enable_quick_eval', True),
            name = 'mortal',
            decision_head = self.chal_decision_head,
            policy_temp = self.chal_policy_temp,
            policy_top_p = self.chal_policy_top_p,
            policy_epsilon = self.chal_policy_epsilon,
            policy_stochastic = config['test_play'].get('policy_stochastic', False),
        )

        _reset_log_dir(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])
        self._baseline_cfg = baseline_cfg
        self._baseline_device = device

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = None
        stable_mortal.load_state_dict(state['mortal'])
        if 'current_dqn' in state:
            stable_dqn = DQN(version=version).eval()
            stable_dqn.load_state_dict(state['current_dqn'])
        baseline_decision_head = baseline_cfg.get('decision_head', 'value')
        baseline_policy = None
        if baseline_decision_head == 'policy':
            baseline_policy = _load_policy_from_state(
                state,
                version,
                enable_compile=baseline_cfg['enable_compile'],
                device=device,
                label='baseline',
            )
            if baseline_policy is None:
                raise KeyError('baseline.train.decision_head=policy but checkpoint has no `policy` state')
        elif stable_dqn is None:
            raise KeyError('baseline.train.decision_head=value but checkpoint has no `current_dqn` state')
        if baseline_cfg['enable_compile']:
            stable_mortal = maybe_compile(
                stable_mortal,
                enable=True,
                device=device,
                label="baseline:brain",
            )
            if stable_dqn is not None:
                stable_dqn = maybe_compile(
                    stable_dqn,
                    enable=True,
                    device=device,
                    label="baseline:dqn",
                )

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            policy = baseline_policy,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_quick_eval = baseline_cfg.get('enable_quick_eval', True),
            enable_rule_based_agari_guard = True,
            name = 'baseline',
            decision_head = baseline_decision_head,
            policy_temp = baseline_cfg.get('policy_temp', 1.0),
            policy_top_p = baseline_cfg.get('policy_top_p', 1.0),
            policy_epsilon = baseline_cfg.get('policy_epsilon', 0.0),
            policy_stochastic = baseline_cfg.get('policy_stochastic', False),
        )
        self._league_pool = None
        league_cfg = config.get('league', {})
        if league_cfg.get('enabled', False):
            from ach_rvr_utils import LeaguePool
            state_file = config.get('control', {}).get('state_file', baseline_cfg['state_file'])
            best_state_file = config.get('control', {}).get('best_state_file', baseline_cfg['state_file'])
            self._league_pool = LeaguePool(
                cfg=league_cfg,
                state_file=state_file,
                best_state_file=best_state_file,
            )
            logging.info(f'league pool enabled, source={state_file}')

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        self.profile = profile
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000
        self.opponent_id = baseline_cfg.get('state_file', 'baseline')

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']
        self.decision_head = cfg.get('decision_head', config['control'].get('decision_head', 'value'))
        self.policy_temp = cfg.get('policy_temp', 1.0)
        self.policy_top_p = cfg.get('policy_top_p', 1.0)
        self.policy_epsilon = cfg.get('policy_epsilon', 0.0)
        self.uses_policy = self.decision_head == 'policy'

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

    def _build_baseline_engine_from_file(self, state_file):
        baseline_cfg = self._baseline_cfg
        device = self._baseline_device

        state = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = None
        stable_mortal.load_state_dict(state['mortal'])
        if 'current_dqn' in state:
            stable_dqn = DQN(version=version).eval()
            stable_dqn.load_state_dict(state['current_dqn'])

        baseline_decision_head = baseline_cfg.get('decision_head', 'value')
        baseline_policy = None
        if baseline_decision_head == 'policy':
            baseline_policy = _load_policy_from_state(
                state,
                version,
                enable_compile=baseline_cfg['enable_compile'],
                device=device,
                label='baseline',
            )
            if baseline_policy is None:
                raise KeyError('baseline.train.decision_head=policy but checkpoint has no `policy` state')
        elif stable_dqn is None:
            raise KeyError('baseline.train.decision_head=value but checkpoint has no `current_dqn` state')

        if baseline_cfg['enable_compile']:
            stable_mortal = maybe_compile(
                stable_mortal,
                enable=True,
                device=device,
                label='baseline:brain',
            )
            if stable_dqn is not None:
                stable_dqn = maybe_compile(
                    stable_dqn,
                    enable=True,
                    device=device,
                    label='baseline:dqn',
                )
            if baseline_policy is not None:
                baseline_policy = maybe_compile(
                    baseline_policy,
                    enable=True,
                    device=device,
                    label='baseline:policy',
                )

        return MortalEngine(
            stable_mortal,
            stable_dqn,
            policy=baseline_policy,
            is_oracle=False,
            version=version,
            device=device,
            enable_amp=True,
            enable_quick_eval=baseline_cfg.get('enable_quick_eval', True),
            enable_rule_based_agari_guard=True,
            name='baseline',
            decision_head=baseline_decision_head,
            policy_temp=baseline_cfg.get('policy_temp', 1.0),
            policy_top_p=baseline_cfg.get('policy_top_p', 1.0),
            policy_epsilon=baseline_cfg.get('policy_epsilon', 0.0),
            policy_stochastic=baseline_cfg.get('policy_stochastic', False),
        )

    def _refresh_league_baseline(self):
        if self._league_pool is None:
            return
        sampled = self._league_pool.sample()
        if not sampled or sampled == self.opponent_id:
            return
        try:
            self.baseline_engine = self._build_baseline_engine_from_file(sampled)
            self.opponent_id = sampled
            logging.info(f'league opponent switched to: {sampled}')
        except Exception as ex:
            logging.warning(f'failed to load sampled league opponent {sampled}: {ex}')

    def train_play(self, mortal, dqn, policy, device):
        torch.backends.cudnn.benchmark = False
        self._refresh_league_baseline()
        if self.decision_head == 'policy' and policy is None:
            raise ValueError('train_play.decision_head=policy but no policy is provided')
        engine_chal = MortalEngine(
            mortal,
            dqn,
            policy = policy,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = True,
            enable_quick_eval = config['train_play'][self.profile].get('enable_quick_eval', True),
            name = 'trainee',
            decision_head = self.decision_head,
            policy_temp = self.policy_temp,
            policy_top_p = self.policy_top_p,
            policy_epsilon = self.policy_epsilon,
            policy_stochastic = config['train_play'][self.profile].get('policy_stochastic', False),
        )

        _reset_log_dir(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
