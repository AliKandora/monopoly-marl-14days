"""Neural Network Architectures for Decentralized Actors and Centralized Critics."""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.utils.action_masking import MaskedCategorical


def layer_init(layer: nn.Linear, std: float = 1.41421356, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization with constant bias (PPO best practice)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MaskedActor(nn.Module):
    """Discrete Policy Network respecting boolean/binary action masks."""

    def __init__(self, obs_dim: int = 144, num_actions: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return action, log_prob, and entropy."""
        logits = self.net(obs)
        dist = MaskedCategorical(logits=logits, mask=mask)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate given action under policy, return log_prob and entropy."""
        logits = self.net(obs)
        dist = MaskedCategorical(logits=logits, mask=mask)
        return dist.log_prob(action), dist.entropy()


class DecentralizedCritic(nn.Module):
    """Value Network estimating V(o_i) from local agent observation (IPPO)."""

    def __init__(self, obs_dim: int = 144, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class CentralizedCritic(nn.Module):
    """Centralized Value Network estimating V(s_global, i) for MAPPO (CTDE).

    Inputs: global_state (146) + agent_id_one_hot (4) = 150 features.
    """

    def __init__(self, global_dim: int = 146, num_agents: int = 4, hidden_dim: int = 128):
        super().__init__()
        in_dim = global_dim + num_agents
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor, agent_one_hot: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([global_state, agent_one_hot], dim=-1)
        return self.net(critic_input).squeeze(-1)
