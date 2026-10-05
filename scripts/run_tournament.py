"""Grand Tournament Harness: 4-Way Evaluation between Random, Heuristic, IPPO, and MAPPO."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.monopoly_env import MonopolyEnv
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ippo_agent import IPPOAgent
from src.agents.mappo_agent import MAPPOAgent
from src.visualization.plots import plot_win_rates


def create_agent(name: str, agent_id: str, ippo_path: str = "checkpoints/ippo_best.pt", mappo_path: str = "checkpoints/mappo_best.pt"):
    """Factory creating agent instance by name."""
    if name == "Random":
        return RandomAgent(agent_id)
    elif name == "Heuristic":
        return HeuristicAgent(agent_id)
    elif name == "IPPO":
        agent = IPPOAgent(agent_id)
        if os.path.exists(ippo_path):
            agent.load(ippo_path)
        return agent
    elif name == "MAPPO":
        agent = MAPPOAgent(agent_id)
        if os.path.exists(mappo_path):
            agent.load(mappo_path)
        return agent
    else:
        raise ValueError(f"Unknown agent type: {name}")


def run_grand_tournament(
    games_per_matchup: int = 10,
    max_turns: int = 150,
    results_dir: str = "results",
    ippo_path: str = "checkpoints/ippo_best.pt",
    mappo_path: str = "checkpoints/mappo_best.pt",
):
    os.makedirs(results_dir, exist_ok=True)
    agent_names = ["Random", "Heuristic", "IPPO", "MAPPO"]
    win_counts: Dict[str, int] = {name: 0 for name in agent_names}
    total_games_played: Dict[str, int] = {name: 0 for name in agent_names}

    print("\n=======================================================")
    print("      MONOPOLY-MARL: GRAND 4-WAY TOURNAMENT")
    print(f" Agents: {', '.join(agent_names)}")
    print(f" Games per Config: {games_per_matchup} | Max Turns: {max_turns}")
    print("=======================================================\n")

    env = MonopolyEnv(max_turns=max_turns)

    # 4 players in every game: 1 of each agent type!
    # Rotate seats across games to eliminate seat advantage
    for g in range(1, games_per_matchup + 1):
        # Permute assignment
        roll = (g - 1) % 4
        roster = [agent_names[(i + roll) % 4] for i in range(4)]

        agents = {
            f"player_{i}": create_agent(roster[i], f"player_{i}", ippo_path, mappo_path)
            for i in range(4)
        }

        obs, _ = env.reset(seed=g * 50)
        turns = 0

        while env.agents:
            actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            turns += 1

        # Determine winner
        active_survivors = [a for a in env.possible_agents if not terminations.get(a, False)]
        if len(active_survivors) == 1:
            winning_agent_id = active_survivors[0]
        else:
            # Highest net worth
            nws = {a: env.calculate_net_worth(env.possible_agents.index(a)) for a in env.possible_agents}
            winning_agent_id = max(nws, key=nws.get)

        winner_idx = env.possible_agents.index(winning_agent_id)
        winner_name = roster[winner_idx]

        win_counts[winner_name] += 1
        for name in roster:
            total_games_played[name] += 1

        print(f"Game {g:2d}/{games_per_matchup:2d} | Winner: {winner_name:<10s} (Seat: {winning_agent_id}) in {turns} steps")

    # Calculate win rates
    win_rates = {
        name: (win_counts[name] / max(1, total_games_played[name])) * 100.0
        for name in agent_names
    }

    # Save Plot
    plot_path = os.path.join(results_dir, "win_rates.png")
    plot_win_rates(win_rates, save_path=plot_path)

    # Generate Markdown Summary
    summary_path = os.path.join(results_dir, "tournament_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Monopoly-MARL: Grand Tournament Summary\n\n")
        f.write("| Agent | Games Played | Wins | Win Rate (%) |\n")
        f.write("|---|---|---|---|\n")
        for name in sorted(agent_names, key=lambda x: win_rates[x], reverse=True):
            f.write(f"| **{name}** | {total_games_played[name]} | {win_counts[name]} | {win_rates[name]:.1f}% |\n")
        f.write("\n![Win Rates](win_rates.png)\n")

    print("\n---------------- FINAL TOURNAMENT REPORT ----------------")
    for name in sorted(agent_names, key=lambda x: win_rates[x], reverse=True):
        print(f"{name:<12s}: {win_counts[name]:2d}/{total_games_played[name]:2d} Wins ({win_rates[name]:5.1f}%)")
    print(f"Summary written to: {summary_path}")
    print(f"Plot saved to:       {plot_path}")
    print("---------------------------------------------------------\n")

    return win_rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monopoly MARL Grand Tournament")
    parser.add_argument("--games", type=int, default=20, help="Number of games to simulate")
    parser.add_argument("--max-turns", type=int, default=150, help="Max turns per game")
    parser.add_argument("--ippo-path", type=str, default="checkpoints/ippo_best.pt", help="Path to IPPO checkpoint")
    parser.add_argument("--mappo-path", type=str, default="checkpoints/mappo_best.pt", help="Path to MAPPO checkpoint")
    args = parser.parse_args()

    run_grand_tournament(
        games_per_matchup=args.games,
        max_turns=args.max_turns,
        ippo_path=args.ippo_path,
        mappo_path=args.mappo_path,
    )
