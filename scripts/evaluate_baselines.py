"""Tournament Evaluation Script: HeuristicAgent vs RandomAgents."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.monopoly_env import MonopolyEnv
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent


def run_tournament(num_games: int = 100, max_turns: int = 200):
    print(f"\n=======================================================")
    print(f" Monopoly MARL Tournament: 1 HeuristicAgent vs 3 Random")
    print(f" Total Games: {num_games} | Max Turns per Game: {max_turns}")
    print(f"=======================================================\n")

    wins: Dict[str, int] = {"Heuristic": 0, "Random": 0, "Draw/Timeout": 0}
    heuristic_turns_to_win: List[int] = []
    final_net_worths: List[float] = []

    env = MonopolyEnv(max_turns=max_turns)

    for game_idx in range(1, num_games + 1):
        # Rotate heuristic seat (0..3) to eliminate position bias
        heuristic_seat = (game_idx - 1) % 4
        heuristic_id = f"player_{heuristic_seat}"

        agents = {}
        for i in range(4):
            p_id = f"player_{i}"
            if p_id == heuristic_id:
                agents[p_id] = HeuristicAgent(p_id)
            else:
                agents[p_id] = RandomAgent(p_id, seed=game_idx * 100 + i)

        obs, _ = env.reset(seed=game_idx)
        step_count = 0

        while env.agents:
            actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

        # Determine winner: last survivor or player with highest net worth
        active_survivors = [a for a in env.possible_agents if not terminations.get(a, False)]
        if len(active_survivors) == 1:
            winner_id = active_survivors[0]
        else:
            # Fallback for truncation timeout: highest net worth
            net_worths = {a: env.calculate_net_worth(env.possible_agents.index(a)) for a in env.possible_agents}
            winner_id = max(net_worths, key=net_worths.get)

        if winner_id == heuristic_id:
            wins["Heuristic"] += 1
            heuristic_turns_to_win.append(step_count)
        else:
            wins["Random"] += 1

        h_nw = env.calculate_net_worth(heuristic_seat)
        final_net_worths.append(h_nw)

        if game_idx % 25 == 0 or game_idx == num_games:
            rate = (wins["Heuristic"] / game_idx) * 100.0
            print(f"Game {game_idx:3d}/{num_games:3d} | Heuristic Wins: {wins['Heuristic']:3d} ({rate:5.1f}%) | Random Wins: {wins['Random']:3d}")

    win_rate = (wins["Heuristic"] / num_games) * 100.0
    avg_turns = np.mean(heuristic_turns_to_win) if heuristic_turns_to_win else 0.0
    avg_nw = np.mean(final_net_worths)

    print("\n---------------- FINAL TOURNAMENT REPORT ----------------")
    print(f"Heuristic Win Rate:       {win_rate:6.1f} %")
    print(f"Random Bots Win Rate:      {(100.0 - win_rate):6.1f} %")
    print(f"Avg Turns until Victory:   {avg_turns:6.1f} turns")
    print(f"Avg Heuristic Net Worth:  ${avg_nw:6.0f}")
    print("---------------------------------------------------------\n")

    return win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Monopoly Baselines")
    parser.add_argument("--games", type=int, default=50, help="Number of games to simulate")
    parser.add_argument("--max_turns", type=int, default=150, help="Max turns per game")
    args = parser.parse_args()

    run_tournament(num_games=args.games, max_turns=args.max_turns)
