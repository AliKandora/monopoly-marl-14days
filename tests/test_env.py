"""Unit tests for Monopoly PettingZoo Environment and Baseline Agents."""

import pytest
from src.envs.monopoly_env import MonopolyEnv, ActionType
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent


def test_env_initialization():
    env = MonopolyEnv(max_turns=50)
    assert len(env.possible_agents) == 4
    assert "player_0" in env.possible_agents
    assert len(env.action_spaces) == 4
    assert len(env.observation_spaces) == 4


def test_env_reset():
    env = MonopolyEnv(max_turns=50)
    obs, infos = env.reset(seed=42)

    assert len(obs) == 4
    for agent in env.possible_agents:
        assert "observation" in obs[agent]
        assert "action_mask" in obs[agent]
        assert obs[agent]["observation"].shape == (env.obs_dim,)
        assert obs[agent]["action_mask"].shape == (env.num_actions,)


def test_random_agent_execution():
    env = MonopolyEnv(max_turns=30)
    obs, infos = env.reset(seed=123)

    agents = {agent: RandomAgent(agent, seed=123) for agent in env.possible_agents}

    for _ in range(15):
        if not env.agents:
            break
        actions = {
            agent: agents[agent].select_action(obs[agent])
            for agent in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Check reward and termination types
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)


def test_heuristic_agent_execution():
    env = MonopolyEnv(max_turns=30)
    obs, infos = env.reset(seed=42)

    agents = {agent: HeuristicAgent(agent) for agent in env.possible_agents}

    for _ in range(15):
        if not env.agents:
            break
        actions = {
            agent: agents[agent].select_action(obs[agent])
            for agent in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(rewards) >= 1
