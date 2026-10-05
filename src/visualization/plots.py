"""Visualization Utilities for Win-Rates, Net-Worth Curves and Training Metrics."""

from __future__ import annotations
import os
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import numpy as np


def plot_win_rates(win_rates: Dict[str, float], save_path: str = "results/win_rates.png"):
    """Plot bar chart comparing agent win rates."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    agents = list(win_rates.keys())
    rates = [win_rates[a] for a in agents]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    bars = plt.bar(agents, rates, color=colors[:len(agents)], width=0.55)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.title("Monopoly MARL: Tournament Win Rates", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_net_worth_trajectories(history: Dict[str, List[float]], save_path: str = "results/net_worth_curves.png"):
    """Plot net worth trajectory of all 4 players across an episode."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = {"player_0": "#2980b9", "player_1": "#27ae60", "player_2": "#c0392b", "player_3": "#d35400"}
    for player, values in history.items():
        plt.plot(values, label=player, color=colors.get(player, None), linewidth=2)

    plt.xlabel("Turn Step", fontsize=12)
    plt.ylabel("Net Worth ($)", fontsize=12)
    plt.title("Player Net Worth Dynamics Over Episode", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
