"""Opponent Pool and League Matchmaking for Self-Play Training."""

from __future__ import annotations
import copy
from typing import List, Optional
import numpy as np
import torch

from src.agents.networks import MaskedActor
from src.agents.heuristic_agent import HeuristicAgent


class OpponentPool:
    """Maintains a league reservoir of historical actor checkpoints to prevent policy cyclicity."""

    def __init__(self, max_size: int = 10, obs_dim: int = 144, num_actions: int = 7):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.checkpoints: List[dict] = []

    def add_checkpoint(self, actor: MaskedActor):
        """Add deepcopy of current actor state dict to pool."""
        state_dict = copy.deepcopy(actor.state_dict())
        self.checkpoints.append(state_dict)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)  # FIFO replacement

    def sample_opponent_model(self, device: str = "cpu") -> Optional[MaskedActor]:
        """Sample a frozen historical actor model from pool."""
        if not self.checkpoints:
            return None
        idx = np.random.randint(0, len(self.checkpoints))
        model = MaskedActor(self.obs_dim, self.num_actions).to(device)
        model.load_state_dict(self.checkpoints[idx])
        model.eval()
        return model

    def __len__(self) -> int:
        return len(self.checkpoints)
