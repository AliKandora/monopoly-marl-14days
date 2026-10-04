"""Reward Shaping Wrappers and Utility Functions for Monopoly MARL."""

from __future__ import annotations
from typing import Dict, List


class RewardConfig:
    """Configuration container for reward shaping coefficients."""

    def __init__(
        self,
        net_worth_scale: float = 0.001,
        step_penalty: float = 0.005,
        win_bonus: float = 100.0,
        loss_penalty: float = 50.0,
        sparse_mode: bool = False,
    ):
        self.net_worth_scale = net_worth_scale
        self.step_penalty = step_penalty
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.sparse_mode = sparse_mode


def compute_shaped_rewards(
    net_worth_before: List[float],
    net_worth_after: List[float],
    terminations: Dict[str, bool],
    possible_agents: List[str],
    active_agents: List[str],
    config: RewardConfig | None = None,
) -> Dict[str, float]:
    """Compute dense net-worth delta or sparse competitive rewards.

    Args:
        net_worth_before: List of 4 floats before env step.
        net_worth_after: List of 4 floats after env step.
        terminations: Dict mapping agent_id -> bool.
        possible_agents: List of all 4 agent names.
        active_agents: List of currently remaining active agents.
        config: Optional RewardConfig.

    Returns:
        Dict mapping active agent -> shaped reward.
    """
    if config is None:
        config = RewardConfig()

    rewards: Dict[str, float] = {}

    for i, agent in enumerate(possible_agents):
        if agent not in active_agents and not terminations.get(agent, False):
            continue

        if config.sparse_mode:
            # Sparse: Only terminal outcomes rewarded
            reward = 0.0
            if terminations.get(agent, False):
                reward -= config.loss_penalty
            rewards[agent] = reward
        else:
            # Dense: Delta Net-Worth - Step Cost
            delta_nw = net_worth_after[i] - net_worth_before[i]
            reward = (delta_nw * config.net_worth_scale) - config.step_penalty

            if terminations.get(agent, False):
                reward -= config.loss_penalty

            rewards[agent] = float(reward)

    # Award win bonus if single survivor remains
    active_survivors = [a for a in active_agents if not terminations.get(a, False)]
    if len(active_survivors) == 1 and len(active_agents) > 1:
        winner = active_survivors[0]
        rewards[winner] = rewards.get(winner, 0.0) + config.win_bonus

    return rewards
