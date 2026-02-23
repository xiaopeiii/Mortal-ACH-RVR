import torch
import socket
import struct
import time
from typing import *
from io import BytesIO
from functools import partial
from config import config

try:
    from tqdm.auto import tqdm as orig_tqdm
except ModuleNotFoundError:
    class _DummyTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

        def set_description(self, desc=None):
            return None

    def orig_tqdm(iterable=None, *args, **kwargs):
        return _DummyTqdm(iterable, *args, **kwargs)

tqdm = partial(orig_tqdm, unit='batch', dynamic_ncols=True, ascii=True)

def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def iter_grads(parameters, take=False):
    for p in parameters:
        if p.grad is not None:
            if take:
                # Set to zero instead of None to preserve the layout and make it
                # easier to assign back later
                yield p.grad.clone()
                p.grad.zero_()
            else:
                yield p.grad

def drain(*, with_manifest=False):
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {'type': 'drain'})
            msg = recv_msg(conn)
        if msg['count'] == 0:
            time.sleep(5)
            continue
        if with_manifest:
            return msg['drain_dir'], msg.get('manifest')
        return msg['drain_dir']

def submit_param(mortal, dqn, policy=None, is_idle=False):
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    with socket.socket() as conn:
        conn.connect(remote)
        msg = {
            'type': 'submit_param',
            'mortal': mortal.state_dict(),
            'is_idle': is_idle,
        }
        if dqn is not None:
            msg['dqn'] = dqn.state_dict()
        if policy is not None:
            msg['policy'] = policy.state_dict()
        send_msg(conn, msg)


def submit_replay(
    logs,
    *,
    param_version=-1,
    opponent_id='',
    profile='default',
    client_id='',
):
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    with socket.socket() as conn:
        conn.connect(remote)
        send_msg(conn, {
            'type': 'submit_replay',
            'logs': logs,
            'param_version': int(param_version),
            'opponent_id': str(opponent_id),
            'profile': str(profile),
            'client_id': str(client_id),
        })

def send_msg(conn: socket.socket, msg, packed=False):
    if packed:
        tx = msg
    else:
        buf = BytesIO()
        torch.save(msg, buf)
        tx = buf.getbuffer()
    conn.sendall(struct.pack('<Q', len(tx)))
    conn.sendall(tx)

def recv_msg(conn: socket.socket, map_location='cpu'):
    rx = recv_binary(conn, 8)
    (size,) = struct.unpack('<Q', rx)
    rx = recv_binary(conn, size)
    return torch.load(BytesIO(rx), weights_only=True, map_location=map_location)

def recv_binary(conn: socket.socket, size):
    assert size > 0
    ret = bytearray(size)
    buf = memoryview(ret)

    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)

class UnexpectedEOF(Exception):
    def __init__(self):
        super().__init__('unexpected EOF')
