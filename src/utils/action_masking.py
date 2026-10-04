"""Action Masking Utilities for Multi-Agent Reinforcement Learning."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical


class MaskedCategorical(Categorical):
    """Categorical distribution that respects boolean/binary action masks by setting

    unmasked logits to -inf (or very large negative numbers).
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor, neg_inf: float = -1e8):
        """Initializes MaskedCategorical.

        Args:
            logits: Unnormalized action logits from actor network [batch_size, num_actions].
            mask: Binary mask where 1 indicates legal action, 0 illegal [batch_size, num_actions].
            neg_inf: Large negative constant to zero out probabilities for illegal actions.
        """
        mask = mask.bool()
        masked_logits = torch.where(mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
        super().__init__(logits=masked_logits)


def sample_masked_action(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample action and return (action, log_prob) using MaskedCategorical."""
    dist = MaskedCategorical(logits=logits, mask=mask)
    action = dist.sample()
    return action, dist.log_prob(action)


def get_random_legal_action(action_mask: np.ndarray) -> int:
    """Select a random action strictly chosen from the legal action set."""
    legal_indices = np.where(action_mask == 1)[0]
    if len(legal_indices) == 0:
        return 0  # Fallback to default action (e.g. PASS / ROLL)
    return int(np.random.choice(legal_indices))
