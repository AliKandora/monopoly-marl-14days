"""Seeded four-way evaluation; missing trained checkpoints are an error."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.envs.monopoly_env import MonopolyEnv
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ippo_agent import IPPOAgent
from src.agents.mappo_agent import MAPPOAgent
from src.visualization.plots import plot_win_rates
from src.utils.evaluation import episode_winners, wilson_interval


def create_agent(name, agent_id, ippo_path='checkpoints/ippo_best.pt', mappo_path='checkpoints/mappo_best.pt', seed=42):
    if name == 'Random':
        return RandomAgent(agent_id, seed=seed)
    if name == 'Heuristic':
        return HeuristicAgent(agent_id)
    if name not in ('IPPO', 'MAPPO'):
        raise ValueError(f'Unknown agent type: {name}')
    path = Path(ippo_path if name == 'IPPO' else mappo_path)
    if not path.is_file():
        raise FileNotFoundError(f'{name} checkpoint missing: {path}. Train the model first.')
    agent = (IPPOAgent if name == 'IPPO' else MAPPOAgent)(agent_id)
    agent.load(str(path))
    return agent


def run_grand_tournament(games_per_matchup=20, max_turns=150, results_dir='results', ippo_path='checkpoints/ippo_best.pt', mappo_path='checkpoints/mappo_best.pt', seed=42):
    if games_per_matchup < 1:
        raise ValueError('games must be positive')
    checkpoints = {}
    for name, filename in [('IPPO', ippo_path), ('MAPPO', mappo_path)]:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f'{name} checkpoint missing: {filename}. No untrained fallback is allowed.')
        checkpoints[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    torch.set_num_threads(1)
    names = ['Random', 'Heuristic', 'IPPO', 'MAPPO']
    wins = dict.fromkeys(names, 0)
    games = []
    env = MonopolyEnv(max_turns=max_turns)
    for g in range(games_per_matchup):
        game_seed = seed + g
        np.random.seed(game_seed)
        torch.manual_seed(game_seed)
        roster = [names[(i + g) % 4] for i in range(4)]
        agents = {f'player_{i}': create_agent(name, f'player_{i}', ippo_path, mappo_path, game_seed * 4 + i) for i, name in enumerate(roster)}
        obs, _ = env.reset(seed=game_seed)
        steps = 0
        while env.agents:
            obs, _, _, _, _ = env.step({a: agents[a].select_action(obs[a]) for a in env.agents})
            steps += 1
        winners = episode_winners(env)
        if len(winners) == 1:
            wins[roster[env.possible_agents.index(winners[0])]] += 1
        games.append({'seed': game_seed, 'roster': roster, 'winners': winners, 'tie': len(winners) != 1, 'steps': steps, 'turns': env.turn_count, 'net_worth': {a: env.calculate_net_worth(i) for i, a in enumerate(env.possible_agents)}})
    env.close()
    rates = {name: wins[name] / games_per_matchup * 100 for name in names}
    report = {'version': '1.0', 'seed': seed, 'max_turns': max_turns, 'games': games, 'checkpoints': checkpoints, 'wins': wins, 'ties': sum(g['tie'] for g in games), 'win_rates': rates, 'confidence_95': {name: wilson_interval(wins[name], games_per_matchup) for name in names}, 'seat_balance': 'Exact only when game count is a multiple of four', 'winner_rule': 'Surviving highest net worth; ties recorded, not assigned to first seat', 'note': 'Functional evaluation is not evidence of convergence or superiority.'}
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tournament.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = ['# Monopoly-MARL: Reproducible Tournament', '', f'Seed: {seed}; games: {games_per_matchup}; max turns: {max_turns}.', '', '| Agent | Wins | Win rate | 95% Wilson interval |', '|---|---:|---:|---:|']
    for name in names:
        lo, hi = report['confidence_95'][name]
        rows.append(f'| {name} | {wins[name]} | {rates[name]:.1f}% | {lo*100:.1f}–{hi*100:.1f}% |')
    rows += ['', f"Ties: {report['ties']}. Small samples do not establish algorithm superiority.", '', 'Checkpoint hashes and individual games: [tournament.json](tournament.json).', '', '![Win rates](win_rates.png)']
    (output / 'tournament_summary.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    plot_win_rates(rates, save_path=str(output / 'win_rates.png'))
    print(json.dumps({'win_rates': rates, 'ties': report['ties'], 'output': str(output)}, indent=2))
    return rates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--max-turns', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--ippo-path', default='checkpoints/ippo_best.pt')
    parser.add_argument('--mappo-path', default='checkpoints/mappo_best.pt')
    args = parser.parse_args()
    run_grand_tournament(args.games, args.max_turns, args.results_dir, args.ippo_path, args.mappo_path, args.seed)
