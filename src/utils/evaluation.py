"""Honest evaluation helpers for the simplified finite-horizon environment."""
import math
import numpy as np


def episode_winners(env):
    """Return surviving net-worth leaders, retaining ties instead of seat bias."""
    survivors = [a for i, a in enumerate(env.possible_agents) if env.player_cash[i] >= 0]
    if not survivors:
        return []
    worth = {a: env.calculate_net_worth(env.possible_agents.index(a)) for a in survivors}
    best = max(worth.values())
    return [a for a in survivors if np.isclose(worth[a], best, rtol=0, atol=1e-5)]


def wilson_interval(wins, games, z=1.959963984540054):
    """Approximate 95% binomial interval; does not measure training-seed variance."""
    if games < 1 or not 0 <= wins <= games:
        raise ValueError('Require games > 0 and 0 <= wins <= games')
    p = wins / games
    d = 1 + z*z/games
    center = (p + z*z/(2*games)) / d
    radius = z * math.sqrt(p*(1-p)/games + z*z/(4*games*games)) / d
    return max(0., center-radius), min(1., center+radius)
