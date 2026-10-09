"""Interactive Terminal Demo for Monopoly-MARL."""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.envs.monopoly_env import MonopolyEnv, ActionType, TurnPhase
from src.envs.board_constants import TILE_NAMES
from src.utils.evaluation import episode_winners
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"


def run_demo(delay: float = 0.1, max_steps: int = 100):
    print(f"{C_BOLD}{C_CYAN}================================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}         MONOPOLY-MARL: INTERACTIVE TERMINAL SIMULATION          {C_RESET}")
    print(f"{C_BOLD}{C_CYAN}================================================================={C_RESET}\n")

    env = MonopolyEnv(render_mode="human", max_turns=max_steps)
    obs, _ = env.reset(seed=42)

    # 1 Smart Heuristic Bot vs 3 Random Bots
    agents = {
        "player_0": HeuristicAgent("player_0"),
        "player_1": RandomAgent("player_1", seed=101),
        "player_2": RandomAgent("player_2", seed=102),
        "player_3": RandomAgent("player_3", seed=103),
    }

    step_num = 0

    while env.agents and step_num < max_steps:
        step_num += 1
        curr_p = env.possible_agents[env.current_agent_idx]
        curr_phase = env.current_phase.name

        actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
        chosen_action = ActionType(actions[curr_p]).name if curr_p in actions else "NONE"

        print(f"\n{C_BOLD}--- Step {step_num:03d} | {curr_p} [{curr_phase}] chooses: {C_GREEN}{chosen_action}{C_RESET} ---")

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Print current standings
        for i, p in enumerate(env.possible_agents):
            pos = env.player_pos[i]
            cash = env.player_cash[i]
            nw = env.calculate_net_worth(i)
            t_name = TILE_NAMES[pos]
            status = f"{C_RED}[BANKRUPT]{C_RESET}" if p not in env.agents else f"{C_GREEN}[ACTIVE]{C_RESET}"
            reward_str = f"Rwd: {rewards.get(p, 0.0):+5.2f}" if p in env.agents else "Rwd:   ---"
            print(f"  {p} {status}: Cash=${cash:5.0f} | NetWorth=${nw:5.0f} | Pos={pos:2d} ({t_name[:16]:16s}) | {reward_str}")

        if delay > 0:
            time.sleep(delay)

    finished = not env.agents
    label = "EPISODE FINISHED" if finished else "DEMO PREVIEW LIMIT REACHED"
    print(f"\n{C_BOLD}{C_CYAN}================ {label} ================{C_RESET}")
    leaders = episode_winners(env)
    print(f"{'Final leaders' if finished else 'Current leaders (not final winners)'}: {leaders}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monopoly Terminal Live Demo")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps in seconds")
    parser.add_argument("--steps", type=int, default=60, help="Max steps to demonstrate")
    args = parser.parse_args()

    run_demo(delay=args.delay, max_steps=args.steps)
