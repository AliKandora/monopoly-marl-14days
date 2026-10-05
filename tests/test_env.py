import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
from src.envs.monopoly_env import MonopolyEnv, ActionType, TurnPhase
from src.envs.board_constants import COLOR_GROUPS, TILE_PRICES, RENT_TABLE, NUM_TILES
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.utils.reward_wrappers import compute_shaped_rewards, RewardConfig
from src.utils.feature_extraction import extract_global_state, extract_player_cash


def test_env_initialization():
    env = MonopolyEnv(max_turns=50)
    assert len(env.possible_agents) == 4
    assert "player_0" in env.possible_agents
    assert len(env.action_spaces) == 4
    assert len(env.observation_spaces) == 4
    assert env.obs_dim == 144
    assert len(TILE_PRICES) == NUM_TILES


def test_env_reset():
    env = MonopolyEnv(max_turns=50)
    obs, infos = env.reset(seed=42)

    assert len(obs) == 4
    for agent in env.possible_agents:
        assert "observation" in obs[agent]
        assert "action_mask" in obs[agent]
        assert obs[agent]["observation"].shape == (144,)
        assert obs[agent]["action_mask"].shape == (len(ActionType),)
        assert infos[agent]["phase"] == int(TurnPhase.ROLL)


def test_micro_phase_transitions():
    env = MonopolyEnv(max_turns=50)
    env.reset(seed=10)

    # Initial phase is ROLL
    assert env.current_phase == TurnPhase.ROLL
    p0 = "player_0"
    mask = env._get_action_mask(p0)
    assert mask[ActionType.ROLL_DICE] == 1

    # Step ROLL_DICE
    actions = {a: ActionType.PASS_TURN for a in env.agents}
    actions[p0] = ActionType.ROLL_DICE
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Should transition to either BUY_OR_PASS (if unowned property with cash) or MANAGE_OR_END
    assert env.current_phase in (TurnPhase.BUY_OR_PASS, TurnPhase.MANAGE_OR_END)


def test_board_monopoly_and_rent():
    env = MonopolyEnv(max_turns=50)
    env.reset(seed=42)

    # Give player_0 Brown group: Badstraße (1) and Turmstraße (3)
    env.property_owner[1] = 0
    env.property_owner[3] = 0

    assert env._player_owns_group(0, 0) is True
    assert env._player_owns_group(1, 0) is False

    # Base unimproved rent for Turmstraße is 4; with monopoly it is 8
    rent = env._calculate_rent(3, owner=0, roll=7)
    assert rent == 8.0

    # Build 1 house on Badstraße
    env.property_houses[1] = 1
    rent_badstrasse = env._calculate_rent(1, owner=0, roll=7)
    # Rent for Badstraße with 1 house is 10
    assert rent_badstrasse == 10.0


def test_net_worth_calculation():
    env = MonopolyEnv(max_turns=50)
    env.reset(seed=42)

    initial_nw = env.calculate_net_worth(0)
    assert initial_nw == 1500.0

    # Add property (Parkstraße 37, price 350)
    env.property_owner[37] = 0
    env.player_cash[0] -= 350.0
    # Net worth remains 1500 (Cash 1150 + Property 350)
    assert env.calculate_net_worth(0) == 1500.0


def test_simulation_run_with_agents():
    env = MonopolyEnv(max_turns=60)
    obs, infos = env.reset(seed=99)

    agents = {
        "player_0": HeuristicAgent("player_0"),
        "player_1": RandomAgent("player_1", seed=1),
        "player_2": RandomAgent("player_2", seed=2),
        "player_3": RandomAgent("player_3", seed=3),
    }

    for step_i in range(50):
        if not env.agents:
            break
        actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        for a in env.agents:
            assert isinstance(rewards[a], float)
            assert not np.isnan(rewards[a])


def test_ctde_global_state_extractor():
    env = MonopolyEnv(max_turns=50)
    env.reset(seed=42)
    s_global = extract_global_state(env)
    assert s_global.shape == (146,)
    assert not np.isnan(s_global).any()


def test_pettingzoo_api_conformance():
    from pettingzoo.test import api_test
    from pettingzoo.utils.conversions import parallel_to_aec_wrapper

    env = parallel_to_aec_wrapper(MonopolyEnv(max_turns=20))
    api_test(env, num_cycles=20)


if __name__ == "__main__":
    print("Running tests in test_env.py...")
    test_env_initialization()
    test_env_reset()
    test_micro_phase_transitions()
    test_board_monopoly_and_rent()
    test_net_worth_calculation()
    test_simulation_run_with_agents()
    test_ctde_global_state_extractor()
    test_pettingzoo_api_conformance()
    print(">>> ALL 8 UNIT TESTS PASSED SUCCESSFULLY! <<<")

