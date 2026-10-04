"""Random Baseline Agent for Monopoly MARL Environment."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class RandomAgent:
    """Agent that samples actions uniformly from the set of valid/legal actions."""

    def __init__(self, agent_id: str, seed: int | None = None):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select a random action strictly among the valid actions defined by action_mask."""
        action_mask = observation.get("action_mask")
        if action_mask is not None:
            legal_actions = np.where(action_mask == 1)[0]
            if len(legal_actions) > 0:
                return int(self.rng.choice(legal_actions))
        # Fallback if no mask or no valid actions found
        return 0

    def reset(self):
        """Reset agent internal memory between episodes."""
        pass
