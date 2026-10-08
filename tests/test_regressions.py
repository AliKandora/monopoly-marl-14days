import numpy as np
import pytest
import torch
from pettingzoo.test import parallel_api_test
from src.envs.monopoly_env import MonopolyEnv, ActionType, TurnPhase
from src.agents.rollout_buffer import MultiAgentRolloutBuffer
from src.agents.ippo_agent import IPPOAgent
from src.agents.mappo_agent import MAPPOAgent
from src.utils.evaluation import episode_winners, wilson_interval
from scripts.run_tournament import create_agent


def actions(env, action=ActionType.PASS_TURN):
    result = {a: int(ActionType.PASS_TURN) for a in env.agents}
    result[env.possible_agents[env.current_agent_idx]] = int(action)
    return result


def test_parallel_api():
    parallel_api_test(MonopolyEnv(max_turns=8), num_cycles=100)


def test_seed_is_local_and_reproducible():
    a, b = MonopolyEnv(), MonopolyEnv()
    a.reset(seed=17)
    b.reset(seed=17)
    for _ in range(20):
        active = a.possible_agents[a.current_agent_idx]
        choice = np.flatnonzero(a._get_action_mask(active))[0]
        a.step(actions(a, choice))
        np.random.seed(999)
        b.step(actions(b, choice))
        np.testing.assert_array_equal(a.player_pos, b.player_pos)
        np.testing.assert_array_equal(a.player_cash, b.player_cash)


@pytest.mark.parametrize('invalid', [-1, 7, 999, 1.5, 'roll'])
def test_invalid_actions_rejected(invalid):
    env = MonopolyEnv()
    env.reset()
    data = actions(env)
    data['player_0'] = invalid
    with pytest.raises(ValueError):
        env.step(data)


def test_missing_actions_rejected():
    env = MonopolyEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step({})


def test_management_cannot_deadlock():
    env = MonopolyEnv(max_turns=1)
    env.reset()
    env.current_phase = TurnPhase.MANAGE_OR_END
    env.property_owner[1] = 0
    for i in range(100):
        if not env.agents:
            break
        choice = ActionType.MORTGAGE if env.property_mortgaged[1] == 0 else ActionType.UNMORTGAGE
        env.step(actions(env, choice))
    assert not env.agents
    assert i <= env.max_management_steps
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_trade_disabled():
    env = MonopolyEnv()
    env.reset()
    env.current_phase = TurnPhase.MANAGE_OR_END
    env.property_owner[1] = 0
    assert env._get_action_mask('player_0')[ActionType.PROPOSE_TRADE] == 0
    env._execute_simple_trade(0)
    assert env.property_owner[1] == 0


def test_bankruptcy_releases_all_assets():
    env = MonopolyEnv()
    env.reset()
    env.property_owner[[1,3]] = 1
    env.property_houses[[1,3]] = 2
    env.player_cash[1] = -1
    _, _, term, _, _ = env.step(actions(env, ActionType.ROLL_DICE))
    assert term['player_1']
    assert 'player_1' not in env.agents
    assert np.all(env.property_owner[[1,3]] == -1)
    assert np.all(env.property_houses[[1,3]] == 0)


def test_gae_does_not_cross_reset():
    b = MultiAgentRolloutBuffer(3, num_agents=1, obs_dim=1, num_actions=1, global_dim=1, gamma=1, gae_lambda=1)
    b.rewards[:,0] = [1,2,100]
    b.values[:,0] = [10,20,30]
    b.dones[:,0] = [0,1,0]
    b.compute_gae(np.array([40]), np.array([0]))
    np.testing.assert_allclose(b.returns[:,0], [3,2,140])


def test_dead_agents_not_in_minibatches():
    b = MultiAgentRolloutBuffer(1, num_agents=2, obs_dim=1, num_actions=1, global_dim=1)
    b.insert(np.array([[1],[99]]), np.ones((2,1)), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(1), valid=np.array([True,False]))
    b.compute_gae(np.zeros(2),np.zeros(2))
    batches = list(b.get_generator(10))
    assert batches[0][0].shape == (1,1)
    assert batches[0][0][0,0] == 1


def test_missing_checkpoint_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_agent('IPPO','player_0',ippo_path=str(tmp_path/'absent.pt'))


@pytest.mark.parametrize('agent_type', [IPPOAgent,MAPPOAgent])
def test_deterministic_action_and_checkpoint_roundtrip(agent_type,tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    a = agent_type('player_0')
    env = MonopolyEnv()
    obs,_ = env.reset(seed=1)
    selected = [a.select_action(obs['player_0'], deterministic=True) for _ in range(10)]
    assert len(set(selected)) == 1
    assert obs['player_0']['action_mask'][selected[0]] == 1
    path=tmp_path/'model.pt'
    a.save(str(path))
    b=agent_type('player_0')
    b.load(str(path))
    assert b.select_action(obs['player_0'],deterministic=True) == selected[0]


def test_ties_not_assigned_to_first_seat():
    env=MonopolyEnv()
    env.reset()
    assert len(episode_winners(env)) == 4
    env.player_cash[0]=-1
    assert 'player_0' not in episode_winners(env)
    lo,hi=wilson_interval(10,20)
    assert 0 < lo < .5 < hi < 1


@pytest.mark.parametrize('name',['ippo','mappo'])
def test_training_smoke(name,tmp_path):
    from scripts.train_ippo import train_ippo
    from scripts.train_mappo import train_mappo
    path=tmp_path/(name+'.pt')
    (train_ippo if name=='ippo' else train_mappo)(total_timesteps=16,num_steps=8,batch_size=8,update_epochs=1,save_path=str(path))
    checkpoint=torch.load(path,weights_only=True)
    assert checkpoint['metadata']['env_steps'] == 16
    assert all(torch.isfinite(t).all() for t in checkpoint['actor_state_dict'].values())


def test_heuristic_keeps_cash_reserve_after_purchase():
    from src.agents.heuristic_agent import HeuristicAgent
    env = MonopolyEnv()
    env.reset()
    env.current_phase = TurnPhase.BUY_OR_PASS
    env.player_pos[0] = 39
    env.player_cash[0] = 500
    agent = HeuristicAgent('player_0', cash_safety_margin=150)
    assert agent.select_action(env._get_obs('player_0')) == ActionType.PASS_TURN
    env.player_cash[0] = 600
    assert agent.select_action(env._get_obs('player_0')) == ActionType.BUY_PROPERTY
