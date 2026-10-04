"""Monopoly MARL Environment Performance Benchmark Script."""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.monopoly_env import MonopolyEnv
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent


def run_benchmark(num_steps: int = 10000, use_heuristic: bool = False):
    print(f"=== Monopoly Environment FPS Benchmark ===")
    print(f"Target steps: {num_steps:,} | Agent type: {'Heuristic' if use_heuristic else 'Random'}")

    env = MonopolyEnv(max_turns=200)
    obs, _ = env.reset(seed=42)

    if use_heuristic:
        agents = {a: HeuristicAgent(a) for a in env.possible_agents}
    else:
        agents = {a: RandomAgent(a, seed=42) for a in env.possible_agents}

    episodes = 0
    total_actions = 0

    start_time = time.perf_counter()

    for step_i in range(num_steps):
        if not env.agents:
            obs, _ = env.reset()
            episodes += 1

        actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_actions += len(actions)

    elapsed = time.perf_counter() - start_time
    fps = num_steps / elapsed
    actions_per_sec = total_actions / elapsed

    print(f"--- Benchmark Results ---")
    print(f"Total time:       {elapsed:.3f} s")
    print(f"Environment FPS:  {fps:,.1f} steps/s")
    print(f"Agent Action FPS: {actions_per_sec:,.1f} decisions/s")
    print(f"Episodes played:  {episodes}")

    if fps >= 500:
        print("[SUCCESS] Environment exceeds the Day 07 target of >500 Steps/sec!")
    else:
        print(f"[NOTE] FPS ({fps:.1f}) below 500. Consider multi-environment vectorization.")

    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monopoly MARL Environment Benchmark")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to run")
    parser.add_argument("--heuristic", action="store_true", help="Use HeuristicAgent instead of RandomAgent")
    args = parser.parse_args()

    run_benchmark(num_steps=args.steps, use_heuristic=args.heuristic)
