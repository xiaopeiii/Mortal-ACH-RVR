import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple

from libriichi.consts import ACTION_SPACE
from model import Brain


def _hidden_dim(version: int) -> int:
    return 512 if version == 1 else 1024


class PolicyYHead(nn.Module):
    """Policy logit head y(a|s;theta) used by ACH/Hedge."""

    def __init__(self, *, version: int = 4):
        super().__init__()
        self.version = version
        in_dim = _hidden_dim(version)
        if version in (2, 3):
            hidden = 512 if version == 2 else 256
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Mish(inplace=True),
                nn.Linear(hidden, ACTION_SPACE),
            )
        else:
            self.net = nn.Linear(in_dim, ACTION_SPACE)
        if isinstance(self.net, nn.Linear):
            nn.init.constant_(self.net.bias, 0)

    def forward(self, phi: Tensor) -> Tensor:
        return self.net(phi)


class ValueHead(nn.Module):
    def __init__(self, *, version: int = 4):
        super().__init__()
        in_dim = _hidden_dim(version)
        if version in (2, 3):
            hidden = 512 if version == 2 else 256
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Mish(inplace=True),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)
        if isinstance(self.net, nn.Linear):
            nn.init.constant_(self.net.bias, 0)

    def forward(self, phi: Tensor) -> Tensor:
        return self.net(phi)


class RelativeValueHead(nn.Module):
    """Predict utility vector for all 4 seats from oracle feature."""

    def __init__(self, *, version: int = 4):
        super().__init__()
        in_dim = _hidden_dim(version)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 4),
        )

    def forward(self, phi_oracle: Tensor) -> Tensor:
        return self.net(phi_oracle)


class ExpectedRewardNet(nn.Module):
    """Predict expected final utility vector from pre-terminal global state."""

    def __init__(self, *, version: int = 4):
        super().__init__()
        in_dim = _hidden_dim(version)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, phi_oracle: Tensor) -> Tensor:
        return self.net(phi_oracle)


class AchRvrModel(nn.Module):
    """Composite container for ACH + RVR experiments."""

    def __init__(self, *, version: int, conv_channels: int, num_blocks: int):
        super().__init__()
        self.version = version
        self.private_brain = Brain(
            version=version,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            is_oracle=False,
        )
        self.oracle_brain = Brain(
            version=version,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            is_oracle=True,
        )

        self.policy_y = PolicyYHead(version=version)
        self.value_head = ValueHead(version=version)
        self.rv_head = RelativeValueHead(version=version)
        self.ern = ExpectedRewardNet(version=version)

    def forward_private(self, obs: Tensor) -> Tensor:
        return self.private_brain(obs)

    def forward_oracle(self, obs: Tensor, invisible_obs: Tensor) -> Tensor:
        return self.oracle_brain(obs, invisible_obs)


def centered_clipped_y(y: Tensor, mask: Tensor, logit_clip: float) -> Tensor:
    safe = y.masked_fill(~mask, 0.0)
    denom = mask.sum(-1, keepdim=True).clamp_min(1)
    centered = y - safe.sum(-1, keepdim=True) / denom
    return centered.clamp(min=-logit_clip, max=logit_clip)


def hedge_policy_from_y(
    y: Tensor,
    mask: Tensor,
    *,
    eta: float,
    logit_clip: float,
) -> Tuple[Tensor, Tensor]:
    y_used = centered_clipped_y(y, mask, logit_clip)
    logits = (eta * y_used).masked_fill(~mask, -torch.inf)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return probs, log_probs


def gather_log_prob(log_probs: Tensor, actions: Tensor) -> Tensor:
    return log_probs[torch.arange(actions.shape[0], device=actions.device), actions]


def masked_entropy(probs: Tensor, log_probs: Tensor, mask: Tensor) -> Tensor:
    safe_log = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log).sum(-1)


def zero_sum_utility(raw: Tensor) -> Tensor:
    return raw - raw.mean(dim=-1, keepdim=True)
