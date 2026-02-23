import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
import os
from os import path
from model import Brain, DQN, PolicyHead
from player import TrainPlayer
from common import send_msg, recv_msg, submit_replay
from config import config
from torch_compile import maybe_compile

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    policy = None
    if config['online']['enable_compile']:
        mortal = maybe_compile(mortal, enable=True, device=device, label="mortal:brain")
        dqn = maybe_compile(dqn, enable=True, device=device, label="mortal:dqn")

    train_player = TrainPlayer()
    if train_player.uses_policy:
        policy = PolicyHead(version=version).to(device).eval()
        if config['online']['enable_compile']:
            policy = maybe_compile(policy, enable=True, device=device, label="mortal:policy")
    strict_cfg = config.get('strict', {})
    strict_enabled = bool(strict_cfg.get('enabled', False))
    if strict_enabled:
        if not train_player.uses_policy:
            raise RuntimeError('strict mode requires train_play.decision_head=policy')
        if train_player.policy_epsilon > 0:
            raise RuntimeError('strict mode requires policy_epsilon=0 for behavior policy')
        if not config['train_play'][train_player.profile].get('policy_stochastic', False):
            raise RuntimeError('strict mode requires train_play.*.policy_stochastic=true')
        if config['train_play'][train_player.profile].get('enable_quick_eval', True):
            raise RuntimeError('strict mode requires train_play.*.enable_quick_eval=false')
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []
    client_id = os.environ.get('MORTAL_CLIENT_ID', f'{socket.gethostname()}:{os.getpid()}')

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        if 'dqn' in rsp:
            dqn.load_state_dict(rsp['dqn'])
        elif not train_player.uses_policy:
            raise RuntimeError('server/trainer did not provide dqn params, but train_play.decision_head=value')
        if policy is not None:
            if 'policy' not in rsp:
                raise RuntimeError('server/trainer did not provide policy params, but train_play.decision_head=policy')
            policy.load_state_dict(rsp['policy'])
        logging.info('param has been updated')

        rankings, file_list = train_player.train_play(mortal, dqn, policy, device)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        submit_replay(
            logs,
            param_version=param_version,
            opponent_id=train_player.opponent_id,
            profile=train_player.profile,
            client_id=client_id,
        )
        logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
