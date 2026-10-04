"""CTDE Feature Extraction Pipelines for Local Observations and Global Critic States."""

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def extract_player_cash(obs_vector: np.ndarray, player_idx: int) -> float:
    """Extract and unnormalize player cash from 144-dim observation vector."""
    base = 120 + player_idx * 4
    return float(obs_vector[base] * 1000.0)


def extract_player_net_worth(obs_vector: np.ndarray, player_idx: int) -> float:
    """Extract and unnormalize player net worth from 144-dim observation vector."""
    base = 120 + player_idx * 4 + 3
    return float(obs_vector[base] * 1000.0)


def extract_global_state(env: Any) -> np.ndarray:
    """Extract comprehensive global state vector s_global for CTDE MAPPO Critic.

    Combines:
    - 40 tile owners (-1..3)
    - 40 tile house counts (0..5)
    - 40 tile mortgage status (0..1)
    - 4 player cash levels (normalized by 1000.0)
    - 4 player positions (normalized by 40.0)
    - 4 player jail turns
    - 4 player net worths (normalized by 1000.0)
    - Total money supply in circulation
    - Turn progress
    Total global features = 146
    """
    global_features = np.zeros(146, dtype=np.float32)

    # Tile states (0..119)
    global_features[0:40] = (env.property_owner + 1.0) / 4.0
    global_features[40:80] = env.property_houses / 5.0
    global_features[80:120] = env.property_mortgaged

    # Players (120..135)
    total_cash = 0.0
    for i in range(4):
        base = 120 + i * 4
        global_features[base] = env.player_cash[i] / 1000.0
        global_features[base + 1] = env.player_pos[i] / 40.0
        global_features[base + 2] = float(env.player_in_jail[i]) / 3.0
        global_features[base + 3] = env.calculate_net_worth(i) / 1000.0
        total_cash += max(0.0, float(env.player_cash[i]))

    # Global environment dynamics (136..145)
    global_features[136] = env.turn_count / float(env.max_turns)
    global_features[137 + env.current_agent_idx] = 1.0
    global_features[141 + int(env.current_phase)] = 1.0
    global_features[144] = total_cash / 6000.0  # Normalized total circulating cash
    global_features[145] = len(env.agents) / 4.0

    return global_features
