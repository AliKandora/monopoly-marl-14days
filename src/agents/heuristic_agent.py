"""Rule-Based Heuristic Baseline Agent for Monopoly MARL."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.envs.monopoly_env import ActionType


class HeuristicAgent:
    """Rule-based heuristic agent for Monopoly.

    Applies common human gameplay heuristics:
    1. Buy properties whenever available if cash buffer is safe (> $200).
    2. Build houses when monopoly is achieved and sufficient cash buffer exists.
    3. Roll dice whenever turn is active.
    4. Only mortgage properties when in immediate financial distress (cash < $50).
    5. Pass turn when no further constructive action is available.
    """

    def __init__(self, agent_id: str, cash_safety_margin: float = 200.0):
        self.agent_id = agent_id
        self.cash_safety_margin = cash_safety_margin

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select action following hierarchical rule priority."""
        action_mask = observation.get("action_mask")
        obs_vec = observation.get("observation")

        if action_mask is None:
            return int(ActionType.PASS_TURN)

        # Extract current cash estimate from observation if available
        # Player cash is stored at indices 120, 124, 128, 132 (normalized by 1000)
        try:
            player_idx = int(self.agent_id.split("_")[-1])
            cash = obs_vec[120 + player_idx * 4] * 1000.0 if obs_vec is not None else 1000.0
        except Exception:
            cash = 1000.0

        # Heuristic 1: If can buy property and have safe cash reserve, buy it!
        if action_mask[ActionType.BUY_PROPERTY] == 1 and cash >= self.cash_safety_margin:
            return int(ActionType.BUY_PROPERTY)

        # Heuristic 2: If can build house and cash > 2 * margin, build!
        if action_mask[ActionType.BUILD_HOUSE] == 1 and cash >= (self.cash_safety_margin * 2):
            return int(ActionType.BUILD_HOUSE)

        # Heuristic 3: Roll dice if it's roll phase
        if action_mask[ActionType.ROLL_DICE] == 1:
            return int(ActionType.ROLL_DICE)

        # Heuristic 4: Emergency mortgage if running out of cash
        if cash < 50.0 and action_mask[ActionType.MORTGAGE] == 1:
            return int(ActionType.MORTGAGE)

        # Heuristic 5: Pass turn / End phase
        if action_mask[ActionType.PASS_TURN] == 1:
            return int(ActionType.PASS_TURN)

        # Fallback: Pick any legal action
        legal_actions = np.where(action_mask == 1)[0]
        if len(legal_actions) > 0:
            return int(legal_actions[0])

        return int(ActionType.PASS_TURN)

    def reset(self):
        """Reset agent internal memory between episodes."""
        pass
