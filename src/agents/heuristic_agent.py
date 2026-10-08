"""Rule-Based Heuristic Baseline Agent for Monopoly MARL."""

from __future__ import annotations
from typing import Any, Dict
import numpy as np

from src.envs.monopoly_env import ActionType
from src.envs.board_constants import TILE_PRICES
from src.utils.feature_extraction import extract_player_cash


class HeuristicAgent:
    """Rule-based heuristic agent for Monopoly.

    Applies common human gameplay heuristics with phase awareness:
    1. ROLL phase: Roll dice (or pay bail).
    2. BUY_OR_PASS phase: Buy property if cash buffer >= $150.
    3. MANAGE_OR_END phase:
       a. Build houses on completed monopolies whenever cash >= $250.
       b. Unmortgage properties when cash reserves are healthy (>= $400).
       c. Emergency mortgage when cash drops below $50.
       d. Pass turn to proceed.
    """

    def __init__(self, agent_id: str, cash_safety_margin: float = 150.0):
        self.agent_id = agent_id
        self.cash_safety_margin = cash_safety_margin
        try:
            self.player_idx = int(agent_id.split("_")[-1])
        except Exception:
            self.player_idx = 0

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select action following hierarchical rule priority."""
        action_mask = observation.get("action_mask")
        obs_vec = observation.get("observation")

        if action_mask is None:
            return int(ActionType.PASS_TURN)

        # Extract current cash estimate from observation vector
        cash = 1000.0
        if obs_vec is not None and len(obs_vec) >= 136:
            try:
                cash = extract_player_cash(obs_vec, self.player_idx)
            except Exception:
                cash = 1000.0

        # Phase 1: Roll phase
        if action_mask[ActionType.ROLL_DICE] == 1:
            return int(ActionType.ROLL_DICE)

        # Phase 2: Buy decision
        if action_mask[ActionType.BUY_PROPERTY] == 1:
            position = int(round(float(obs_vec[121 + self.player_idx * 4]) * 40)) if obs_vec is not None else 0
            price = TILE_PRICES[position]
            if cash - price >= self.cash_safety_margin:
                return int(ActionType.BUY_PROPERTY)
            else:
                return int(ActionType.PASS_TURN)

        # Phase 3: Property management
        if action_mask[ActionType.BUILD_HOUSE] == 1 and cash >= (self.cash_safety_margin + 100.0):
            return int(ActionType.BUILD_HOUSE)

        if action_mask[ActionType.UNMORTGAGE] == 1 and cash >= 400.0:
            return int(ActionType.UNMORTGAGE)

        if cash < 50.0 and action_mask[ActionType.MORTGAGE] == 1:
            return int(ActionType.MORTGAGE)

        # End turn
        if action_mask[ActionType.PASS_TURN] == 1:
            return int(ActionType.PASS_TURN)

        # Fallback to any valid legal action
        legal = np.where(action_mask == 1)[0]
        if len(legal) > 0:
            return int(legal[0])

        return int(ActionType.PASS_TURN)

    def reset(self):
        """Reset agent internal memory between episodes."""
        pass
