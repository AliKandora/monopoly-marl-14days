"""Monopoly Multi-Agent Reinforcement Learning Environment.

Complies with PettingZoo ParallelEnv API standard.
Designed for 4-player games with CTDE (Centralized Training, Decentralized Execution).
"""

from __future__ import annotations

import functools
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class ActionType(IntEnum):
    """Discrete action enumeration for the Monopoly Environment."""
    ROLL_DICE = 0
    BUY_PROPERTY = 1
    PASS_TURN = 2
    BUILD_HOUSE = 3
    MORTGAGE = 4
    UNMORTGAGE = 5
    PROPOSE_TRADE = 6


NUM_TILES = 40
STARTING_CASH = 1500
MAX_TURNS_PER_EPISODE = 200


class MonopolyEnv(ParallelEnv):
    """Monopoly PettingZoo Parallel Environment.

    Attributes:
        metadata: Gym metadata dictionary.
        possible_agents: List of static agent identifiers.
        agents: Active agents in current episode.
    """

    metadata = {
        "name": "monopoly_marl_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }

    def __init__(self, render_mode: Optional[str] = None, max_turns: int = MAX_TURNS_PER_EPISODE):
        super().__init__()
        self.render_mode = render_mode
        self.max_turns = max_turns

        self.possible_agents: List[str] = [f"player_{i}" for i in range(4)]
        self.agents: List[str] = self.possible_agents[:]

        self.num_actions = len(ActionType)

        # Observation dimension:
        # 40 tiles * (owner_id + houses + is_mortgaged) = 120
        # 4 players * (cash + position + in_jail + net_worth) = 16
        # Current turn + active player one-hot (4) = 5
        # Total base features = 141
        self.obs_dim = 141

        # Spaces per agent
        self.action_spaces: Dict[str, spaces.Discrete] = {
            agent: spaces.Discrete(self.num_actions) for agent in self.possible_agents
        }

        self.observation_spaces: Dict[str, spaces.Dict] = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.num_actions,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }

        # Game state tracking
        self.current_agent_idx = 0
        self.turn_count = 0
        self.player_cash = np.zeros(4, dtype=np.float32)
        self.player_pos = np.zeros(4, dtype=np.int32)
        self.player_in_jail = np.zeros(4, dtype=np.int32)
        self.property_owner = np.full(NUM_TILES, -1, dtype=np.int32)
        self.property_houses = np.zeros(NUM_TILES, dtype=np.int32)
        self.property_mortgaged = np.zeros(NUM_TILES, dtype=np.int32)

        # Basic property costs (simplification of standard board)
        self.property_prices = np.full(NUM_TILES, 150, dtype=np.float32)
        self.property_prices[0] = 0   # GO
        self.property_prices[10] = 0  # Jail
        self.property_prices[20] = 0  # Free Parking
        self.property_prices[30] = 0  # Go to Jail

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """Reset the environment to the initial game state."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.turn_count = 0
        self.current_agent_idx = 0

        self.player_cash = np.full(4, STARTING_CASH, dtype=np.float32)
        self.player_pos = np.zeros(4, dtype=np.int32)
        self.player_in_jail = np.zeros(4, dtype=np.int32)
        self.property_owner = np.full(NUM_TILES, -1, dtype=np.int32)
        self.property_houses = np.zeros(NUM_TILES, dtype=np.int32)
        self.property_mortgaged = np.zeros(NUM_TILES, dtype=np.int32)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"legal_actions": self._get_action_mask(agent)} for agent in self.agents}

        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment forward by executing actions of all active agents."""
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}

        current_agent = self.possible_agents[self.current_agent_idx]

        if current_agent in actions and current_agent in self.agents:
            action = actions[current_agent]
            mask = self._get_action_mask(current_agent)

            # Penalize invalid action execution
            if mask[action] == 0:
                rewards[current_agent] -= 10.0
                action = ActionType.PASS_TURN

            # Process action
            if action == ActionType.ROLL_DICE:
                roll = int(np.random.randint(1, 7) + np.random.randint(1, 7))
                prev_pos = self.player_pos[self.current_agent_idx]
                new_pos = (prev_pos + roll) % NUM_TILES
                self.player_pos[self.current_agent_idx] = new_pos

                # Pass GO bonus
                if new_pos < prev_pos:
                    self.player_cash[self.current_agent_idx] += 200.0
                    rewards[current_agent] += 1.0

                # Rent deduction if owned by opponent
                owner = self.property_owner[new_pos]
                if owner != -1 and owner != self.current_agent_idx:
                    rent = 25.0 * (1 + self.property_houses[new_pos])
                    self.player_cash[self.current_agent_idx] -= rent
                    self.player_cash[owner] += rent
                    rewards[current_agent] -= 0.5
                    rewards[self.possible_agents[owner]] += 0.5

            elif action == ActionType.BUY_PROPERTY:
                pos = self.player_pos[self.current_agent_idx]
                cost = self.property_prices[pos]
                if self.property_owner[pos] == -1 and self.player_cash[self.current_agent_idx] >= cost:
                    self.player_cash[self.current_agent_idx] -= cost
                    self.property_owner[pos] = self.current_agent_idx
                    rewards[current_agent] += 2.0

            elif action == ActionType.PASS_TURN:
                pass

        # Check bankruptcies
        for idx, agent in enumerate(self.possible_agents):
            if agent in self.agents and self.player_cash[idx] < 0:
                terminations[agent] = True
                rewards[agent] -= 50.0

        # Advance turn
        self.turn_count += 1
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.possible_agents)

        # Check truncation (max episode steps)
        if self.turn_count >= self.max_turns:
            for agent in self.agents:
                truncations[agent] = True

        # Check winner if only one agent remains
        active_remaining = [a for a in self.agents if not terminations[a]]
        if len(active_remaining) <= 1 and len(self.agents) > 1:
            for a in active_remaining:
                rewards[a] += 100.0
                terminations[a] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"cash": self.player_cash[i]} for i, agent in enumerate(self.possible_agents) if agent in self.agents}

        # Filter out terminated agents
        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]

        return observations, rewards, terminations, truncations, infos

    def _get_action_mask(self, agent: str) -> np.ndarray:
        """Returns 1 for legal actions and 0 for illegal actions."""
        mask = np.zeros(self.num_actions, dtype=np.int8)
        agent_idx = self.possible_agents.index(agent)

        if agent_idx != self.current_agent_idx:
            # Not active player's turn: pass is the only valid action
            mask[ActionType.PASS_TURN] = 1
            return mask

        # Active player turn
        mask[ActionType.ROLL_DICE] = 1
        mask[ActionType.PASS_TURN] = 1

        curr_pos = self.player_pos[agent_idx]
        if (
            self.property_owner[curr_pos] == -1
            and self.property_prices[curr_pos] > 0
            and self.player_cash[agent_idx] >= self.property_prices[curr_pos]
        ):
            mask[ActionType.BUY_PROPERTY] = 1

        return mask

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """Constructs the agent's observation vector and action mask."""
        agent_idx = self.possible_agents.index(agent)
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Board state (indices 0..119)
        obs[0:40] = self.property_owner
        obs[40:80] = self.property_houses
        obs[80:120] = self.property_mortgaged

        # Player stats (indices 120..135)
        for i in range(4):
            base = 120 + i * 4
            obs[base] = self.player_cash[i] / 1000.0
            obs[base + 1] = self.player_pos[i] / 40.0
            obs[base + 2] = float(self.player_in_jail[i])
            obs[base + 3] = (self.player_cash[i] + np.sum(self.property_prices[self.property_owner == i])) / 1000.0

        # Global turn state (indices 136..140)
        obs[136] = self.turn_count / float(self.max_turns)
        obs[137 + self.current_agent_idx] = 1.0

        return {
            "observation": obs,
            "action_mask": self._get_action_mask(agent),
        }

    def render(self) -> Optional[str]:
        """Renders text state representation of the board."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            output = [
                f"--- Turn {self.turn_count} (Active: {self.possible_agents[self.current_agent_idx]}) ---"
            ]
            for i, p in enumerate(self.possible_agents):
                output.append(f"  {p}: Cash=${self.player_cash[i]:.0f}, Pos={self.player_pos[i]}")
            rendered = "\n".join(output)
            if self.render_mode == "human":
                print(rendered)
            return rendered
        return None

    def close(self):
        """Cleanup environment resources."""
        pass
