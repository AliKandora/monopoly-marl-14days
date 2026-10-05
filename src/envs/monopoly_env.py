"""Monopoly Multi-Agent Reinforcement Learning Environment.

Complies with PettingZoo ParallelEnv API standard.
Features full 40-tile board topology, micro-phase state machine, complete action handling,
jail mechanics and strict Net-Worth reward calculation.
"""

from __future__ import annotations

import functools
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.envs.board_constants import (
    COLOR_GROUPS,
    GROUP_RAILROAD,
    GROUP_UTILITY,
    HOUSE_COSTS,
    NUM_TILES,
    RENT_TABLE,
    TILE_NAMES,
    TILE_PRICES,
    TILE_TO_GROUP,
)


class ActionType(IntEnum):
    """Discrete action enumeration for the Monopoly Environment."""
    ROLL_DICE = 0
    BUY_PROPERTY = 1
    PASS_TURN = 2
    BUILD_HOUSE = 3
    MORTGAGE = 4
    UNMORTGAGE = 5
    PROPOSE_TRADE = 6


class TurnPhase(IntEnum):
    """Micro-phases within a player's turn to prevent action deadlocks."""
    ROLL = 0            # Must roll dice or pay bail
    BUY_OR_PASS = 1     # Landed on unowned property: Buy or Pass
    MANAGE_OR_END = 2   # Post-movement: Build, Mortgage, Unmortgage, Trade, or End Turn


STARTING_CASH = 1500.0
MAX_TURNS_PER_EPISODE = 200
BAIL_COST = 50.0

# ANSI color codes for terminal rendering
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"


class MonopolyEnv(ParallelEnv):
    """PettingZoo Parallel Environment for Monopoly Multi-Agent RL.

    Attributes:
        metadata: Gym metadata dictionary.
        possible_agents: List of static agent identifiers.
        agents: Active agents in current episode.
    """

    metadata = {
        "name": "monopoly_marl_v1",
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
        # 40 tiles * (owner_id [norm] + houses + is_mortgaged) = 120
        # 4 players * (cash/1000 + pos/40 + in_jail + net_worth/1000) = 16
        # Global: turn_count/max_turns (1), active_player one-hot (4), current_phase one-hot (3) = 8
        # Total observation features = 144
        self.obs_dim = 144

        # Action and Observation spaces per agent
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

        # Board specifications
        self.tile_prices = np.array(TILE_PRICES, dtype=np.float32)
        self.house_costs = np.array(HOUSE_COSTS, dtype=np.float32)

        # Game state tracking
        self.current_agent_idx = 0
        self.current_phase = TurnPhase.ROLL
        self.turn_count = 0
        self.last_dice_roll = 0

        self.player_cash = np.zeros(4, dtype=np.float32)
        self.player_pos = np.zeros(4, dtype=np.int32)
        self.player_in_jail = np.zeros(4, dtype=np.int32)  # turns spent in jail (0 = free)
        self.property_owner = np.full(NUM_TILES, -1, dtype=np.int32)
        self.property_houses = np.zeros(NUM_TILES, dtype=np.int32)
        self.property_mortgaged = np.zeros(NUM_TILES, dtype=np.int32)

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
        self.current_phase = TurnPhase.ROLL
        self.last_dice_roll = 0

        self.player_cash = np.full(4, STARTING_CASH, dtype=np.float32)
        self.player_pos = np.zeros(4, dtype=np.int32)
        self.player_in_jail = np.zeros(4, dtype=np.int32)
        self.property_owner = np.full(NUM_TILES, -1, dtype=np.int32)
        self.property_houses = np.zeros(NUM_TILES, dtype=np.int32)
        self.property_mortgaged = np.zeros(NUM_TILES, dtype=np.int32)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"phase": int(self.current_phase)} for agent in self.agents}

        return observations, infos

    def calculate_net_worth(self, player_idx: int) -> float:
        """Calculate total net worth: Cash + Property value + House value - Mortgage debt."""
        if self.player_cash[player_idx] < 0:
            return 0.0
        nw = float(self.player_cash[player_idx])
        owned = np.where(self.property_owner == player_idx)[0]
        for t in owned:
            if self.property_mortgaged[t] == 1:
                nw += float(self.tile_prices[t]) * 0.5
            else:
                nw += float(self.tile_prices[t])
                nw += float(self.property_houses[t]) * float(self.house_costs[t])
        return nw

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment forward by executing actions."""
        net_worth_before = [self.calculate_net_worth(i) for i in range(4)]

        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}

        active_agent = self.possible_agents[self.current_agent_idx]

        if active_agent in self.agents and active_agent in actions:
            action = actions[active_agent]
            mask = self._get_action_mask(active_agent)

            # Strict Mask Enforcement
            if mask[action] == 0:
                rewards[active_agent] -= 2.0
                action = ActionType.PASS_TURN

            # Execute action depending on current turn phase
            self._execute_player_action(self.current_agent_idx, action)

        # Net-Worth Delta Rewards
        for i, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                nw_after = self.calculate_net_worth(i)
                nw_delta = (nw_after - net_worth_before[i]) / 1000.0
                # Small step penalty to encourage proactive gameplay
                rewards[agent] += nw_delta - 0.005

        # Check Bankruptcies
        for idx, agent in enumerate(self.possible_agents):
            if agent in self.agents and self.player_cash[idx] < 0:
                terminations[agent] = True
                rewards[agent] -= 50.0
                # Release properties back to bank (capture mask before mutating)
                owned_mask = self.property_owner == idx
                self.property_owner[owned_mask] = -1
                self.property_houses[owned_mask] = 0
                self.property_mortgaged[owned_mask] = 0

        # Check Truncation (Max turns reached)
        if self.turn_count >= self.max_turns:
            for agent in self.agents:
                truncations[agent] = True

        # Check for single survivor
        active_remaining = [a for a in self.agents if not terminations[a]]
        if len(active_remaining) <= 1 and len(self.agents) > 1:
            for a in active_remaining:
                rewards[a] += 100.0
                terminations[a] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {
            agent: {
                "cash": float(self.player_cash[self.possible_agents.index(agent)]),
                "net_worth": float(self.calculate_net_worth(self.possible_agents.index(agent))),
                "phase": int(self.current_phase),
            }
            for agent in self.agents
        }

        # Filter out terminated agents
        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]

        # Ensure current_agent_idx points to an active player if possible
        if self.agents and self.possible_agents[self.current_agent_idx] not in self.agents:
            self._advance_to_next_active_player()

        return observations, rewards, terminations, truncations, infos

    def _execute_player_action(self, p_idx: int, action: int):
        """Micro-phase state machine execution."""
        pos = self.player_pos[p_idx]

        if self.current_phase == TurnPhase.ROLL:
            if action == ActionType.ROLL_DICE or action == ActionType.PASS_TURN:
                # Handle Jail
                if self.player_in_jail[p_idx] > 0:
                    d1, d2 = int(np.random.randint(1, 7)), int(np.random.randint(1, 7))
                    self.last_dice_roll = d1 + d2
                    if d1 == d2:
                        # Freed by doubles!
                        self.player_in_jail[p_idx] = 0
                        self._move_player(p_idx, self.last_dice_roll)
                    else:
                        self.player_in_jail[p_idx] += 1
                        if self.player_in_jail[p_idx] > 3:
                            # Mandatory bail
                            self.player_cash[p_idx] -= BAIL_COST
                            self.player_in_jail[p_idx] = 0
                            self._move_player(p_idx, self.last_dice_roll)
                        else:
                            # Remain in jail, end turn
                            self._advance_to_next_active_player()
                            return
                else:
                    # Normal Roll
                    d1, d2 = int(np.random.randint(1, 7)), int(np.random.randint(1, 7))
                    self.last_dice_roll = d1 + d2
                    self._move_player(p_idx, self.last_dice_roll)

        elif self.current_phase == TurnPhase.BUY_OR_PASS:
            if action == ActionType.BUY_PROPERTY:
                cost = self.tile_prices[pos]
                if self.property_owner[pos] == -1 and cost > 0 and self.player_cash[p_idx] >= cost:
                    self.player_cash[p_idx] -= cost
                    self.property_owner[pos] = p_idx
            self.current_phase = TurnPhase.MANAGE_OR_END

        elif self.current_phase == TurnPhase.MANAGE_OR_END:
            if action == ActionType.BUILD_HOUSE:
                self._execute_build_house(p_idx)
            elif action == ActionType.MORTGAGE:
                self._execute_mortgage(p_idx)
            elif action == ActionType.UNMORTGAGE:
                self._execute_unmortgage(p_idx)
            elif action == ActionType.PROPOSE_TRADE:
                self._execute_simple_trade(p_idx)
            elif action == ActionType.PASS_TURN:
                self._advance_to_next_active_player()

    def _move_player(self, p_idx: int, roll: int):
        """Move player forward, process GO bonus, Jail tile, or rent payment."""
        prev_pos = self.player_pos[p_idx]
        new_pos = (prev_pos + roll) % NUM_TILES
        self.player_pos[p_idx] = new_pos

        # Pass GO bonus
        if new_pos < prev_pos:
            self.player_cash[p_idx] += 200.0

        # Special Tile: Go to Jail (Tile 30)
        if new_pos == 30:
            self.player_pos[p_idx] = 10
            self.player_in_jail[p_idx] = 1
            self._advance_to_next_active_player()
            return

        # Special Tile: Income Tax (4) or Luxury Tax (38)
        if new_pos == 4:
            self.player_cash[p_idx] -= 200.0
        elif new_pos == 38:
            self.player_cash[p_idx] -= 100.0

        # Rent handling if owned by opponent
        owner = self.property_owner[new_pos]
        if owner != -1 and owner != p_idx and self.property_mortgaged[new_pos] == 0:
            rent = self._calculate_rent(new_pos, owner, self.last_dice_roll)
            self.player_cash[p_idx] -= rent
            self.player_cash[owner] += rent

        # Transition to next phase
        if self.property_owner[new_pos] == -1 and self.tile_prices[new_pos] > 0 and self.player_cash[p_idx] >= self.tile_prices[new_pos]:
            self.current_phase = TurnPhase.BUY_OR_PASS
        else:
            self.current_phase = TurnPhase.MANAGE_OR_END

    def _calculate_rent(self, tile: int, owner: int, roll: int) -> float:
        """Calculate accurate rent based on houses, color monopoly, railroads, or utilities."""
        g_id = TILE_TO_GROUP.get(tile, -1)

        # Railroads
        if g_id == GROUP_RAILROAD:
            rr_tiles = COLOR_GROUPS[GROUP_RAILROAD]
            count = sum(1 for t in rr_tiles if self.property_owner[t] == owner and self.property_mortgaged[t] == 0)
            return 25.0 * (2 ** (count - 1)) if count > 0 else 25.0

        # Utilities
        if g_id == GROUP_UTILITY:
            u_tiles = COLOR_GROUPS[GROUP_UTILITY]
            count = sum(1 for t in u_tiles if self.property_owner[t] == owner and self.property_mortgaged[t] == 0)
            multiplier = 10.0 if count == 2 else 4.0
            return float(roll) * multiplier

        # Standard street properties
        houses = self.property_houses[tile]
        if tile in RENT_TABLE:
            base_rent = float(RENT_TABLE[tile][0])
            if houses == 0:
                # Check for complete monopoly (doubles unimproved rent)
                if self._player_owns_group(owner, g_id):
                    return base_rent * 2.0
                return base_rent
            else:
                return float(RENT_TABLE[tile][houses])

        return 20.0

    def _player_owns_group(self, p_idx: int, g_id: int) -> bool:
        """Check if player owns all tiles in a group with none mortgaged."""
        if g_id not in COLOR_GROUPS:
            return False
        tiles = COLOR_GROUPS[g_id]
        return all(self.property_owner[t] == p_idx and self.property_mortgaged[t] == 0 for t in tiles)

    def _execute_build_house(self, p_idx: int):
        """Build house on the lowest-level property of an eligible monopoly."""
        for g_id in range(8):  # Street groups 0..7
            if self._player_owns_group(p_idx, g_id):
                tiles = COLOR_GROUPS[g_id]
                min_houses = min(self.property_houses[t] for t in tiles)
                if min_houses < 5:
                    target_tile = [t for t in tiles if self.property_houses[t] == min_houses][0]
                    cost = self.house_costs[target_tile]
                    if self.player_cash[p_idx] >= cost:
                        self.player_cash[p_idx] -= cost
                        self.property_houses[target_tile] += 1
                        return

    def _execute_mortgage(self, p_idx: int):
        """Mortgage an unmortgaged property that has 0 houses to gain liquidity."""
        owned = [t for t in range(NUM_TILES) if self.property_owner[t] == p_idx and self.property_mortgaged[t] == 0]
        for t in owned:
            g_id = TILE_TO_GROUP.get(t, -1)
            # Cannot mortgage if any property in group has houses
            if g_id in COLOR_GROUPS and any(self.property_houses[ot] > 0 for ot in COLOR_GROUPS[g_id]):
                continue
            self.property_mortgaged[t] = 1
            self.player_cash[p_idx] += self.tile_prices[t] * 0.5
            return

    def _execute_unmortgage(self, p_idx: int):
        """Unmortgage a property by paying mortgage value + 10% fee."""
        mortgaged = [t for t in range(NUM_TILES) if self.property_owner[t] == p_idx and self.property_mortgaged[t] == 1]
        for t in mortgaged:
            cost = self.tile_prices[t] * 0.55
            if self.player_cash[p_idx] >= cost:
                self.player_cash[p_idx] -= cost
                self.property_mortgaged[t] = 0
                return

    def _execute_simple_trade(self, p_idx: int):
        """Transfer an isolated single property to highest cash opponent for fair price."""
        isolated = [
            t for t in range(NUM_TILES)
            if self.property_owner[t] == p_idx
            and not self._player_owns_group(p_idx, TILE_TO_GROUP.get(t, -1))
            and self.property_houses[t] == 0
        ]
        if not isolated:
            return
        trade_tile = isolated[0]
        price = self.tile_prices[trade_tile] * 1.2
        opponents = [i for i in range(4) if i != p_idx and self.possible_agents[i] in self.agents]
        opponents.sort(key=lambda x: self.player_cash[x], reverse=True)
        for opp in opponents:
            if self.player_cash[opp] >= price:
                self.player_cash[opp] -= price
                self.player_cash[p_idx] += price
                self.property_owner[trade_tile] = opp
                return

    def _advance_to_next_active_player(self):
        """Advance turn count, reset phase to ROLL and advance active player index."""
        self.turn_count += 1
        self.current_phase = TurnPhase.ROLL
        if not self.agents:
            return
        for _ in range(4):
            self.current_agent_idx = (self.current_agent_idx + 1) % 4
            if self.possible_agents[self.current_agent_idx] in self.agents:
                break

    def _get_action_mask(self, agent: str) -> np.ndarray:
        """Returns 1 for legal actions and 0 for illegal actions based on current micro-phase."""
        mask = np.zeros(self.num_actions, dtype=np.int8)
        p_idx = self.possible_agents.index(agent)

        if p_idx != self.current_agent_idx:
            # Inactive player: can only pass in parallel step
            mask[ActionType.PASS_TURN] = 1
            return mask

        # Active player turn logic by phase
        if self.current_phase == TurnPhase.ROLL:
            mask[ActionType.ROLL_DICE] = 1
            mask[ActionType.PASS_TURN] = 1

        elif self.current_phase == TurnPhase.BUY_OR_PASS:
            pos = self.player_pos[p_idx]
            if self.property_owner[pos] == -1 and self.tile_prices[pos] > 0 and self.player_cash[p_idx] >= self.tile_prices[pos]:
                mask[ActionType.BUY_PROPERTY] = 1
            mask[ActionType.PASS_TURN] = 1

        elif self.current_phase == TurnPhase.MANAGE_OR_END:
            mask[ActionType.PASS_TURN] = 1

            # Build House Check
            for g_id in range(8):
                if self._player_owns_group(p_idx, g_id):
                    tiles = COLOR_GROUPS[g_id]
                    if min(self.property_houses[t] for t in tiles) < 5:
                        cost = self.house_costs[tiles[0]]
                        if self.player_cash[p_idx] >= cost:
                            mask[ActionType.BUILD_HOUSE] = 1
                            break

            # Mortgage Check (has unmortgaged property without houses)
            for t in range(NUM_TILES):
                if self.property_owner[t] == p_idx and self.property_mortgaged[t] == 0:
                    g_id = TILE_TO_GROUP.get(t, -1)
                    if g_id not in COLOR_GROUPS or all(self.property_houses[ot] == 0 for ot in COLOR_GROUPS[g_id]):
                        mask[ActionType.MORTGAGE] = 1
                        break

            # Unmortgage Check
            for t in range(NUM_TILES):
                if self.property_owner[t] == p_idx and self.property_mortgaged[t] == 1:
                    if self.player_cash[p_idx] >= self.tile_prices[t] * 0.55:
                        mask[ActionType.UNMORTGAGE] = 1
                        break

            # Propose Trade Check
            mask[ActionType.PROPOSE_TRADE] = 1

        return mask

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """Constructs observation vector and action mask."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Board Properties (indices 0..119)
        obs[0:40] = (self.property_owner + 1.0) / 4.0  # normalized owner (-1..3 -> 0..1)
        obs[40:80] = self.property_houses / 5.0
        obs[80:120] = self.property_mortgaged

        # 2. Player Stats (indices 120..135)
        for i in range(4):
            base = 120 + i * 4
            obs[base] = self.player_cash[i] / 1000.0
            obs[base + 1] = self.player_pos[i] / 40.0
            obs[base + 2] = float(self.player_in_jail[i]) / 3.0
            obs[base + 3] = self.calculate_net_worth(i) / 1000.0

        # 3. Global & Phase State (indices 136..143)
        obs[136] = self.turn_count / float(self.max_turns)
        obs[137 + self.current_agent_idx] = 1.0
        obs[141 + int(self.current_phase)] = 1.0

        return {
            "observation": obs,
            "action_mask": self._get_action_mask(agent),
        }

    def render(self) -> Optional[str]:
        """Renders text state representation of the board with ANSI colors."""
        if self.render_mode in ("ansi", "human"):
            phase_name = self.current_phase.name
            header = (
                f"{ANSI_CYAN}{ANSI_BOLD}\n=== Turn {self.turn_count:03d} | Active: "
                f"{self.possible_agents[self.current_agent_idx]} [{phase_name}] ==={ANSI_RESET}"
            )
            lines = [header]
            for i, p in enumerate(self.possible_agents):
                is_active = p in self.agents
                pos = self.player_pos[i]
                t_name = TILE_NAMES[pos]
                nw = self.calculate_net_worth(i)
                cash = self.player_cash[i]

                if not is_active:
                    color = ANSI_RED
                    suffix = " (Bankrupt)"
                elif cash < 0:
                    color = ANSI_RED
                    suffix = ""
                else:
                    color = ANSI_GREEN
                    suffix = ""

                lines.append(
                    f"  {color}{p}{ANSI_RESET}: "
                    f"Cash={color}${cash:6.0f}{ANSI_RESET} | "
                    f"NetWorth=${nw:6.0f} | Pos={pos:2d} ({t_name}){suffix}"
                )
            rendered = "\n".join(lines)
            if self.render_mode == "human":
                print(rendered)
            return rendered
        return None

    def close(self):
        """Cleanup environment resources."""
        pass
