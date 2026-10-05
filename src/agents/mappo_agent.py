"""Multi-Agent PPO (MAPPO) Agent implementation with Centralized Critic."""

from __future__ import annotations
from typing import Any, Dict
import torch
import numpy as np

from src.agents.networks import MaskedActor, CentralizedCritic


class MAPPOAgent:
    """Agent using Multi-Agent PPO with Centralized Critic (CTDE)."""

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 144,
        global_dim: int = 146,
        num_actions: int = 7,
        num_agents: int = 4,
        hidden_dim: int = 128,
        device: str = "cpu",
        shared_actor: MaskedActor | None = None,
        shared_critic: CentralizedCritic | None = None,
    ):
        self.agent_id = agent_id
        self.device = torch.device(device)

        self.actor = shared_actor if shared_actor is not None else MaskedActor(obs_dim, num_actions, hidden_dim).to(self.device)
        self.critic = shared_critic if shared_critic is not None else CentralizedCritic(global_dim, num_agents, hidden_dim).to(self.device)

    def select_action(self, observation: Dict[str, Any], deterministic: bool = False) -> int:
        """Decentralized action selection using local observation only."""
        obs = torch.as_tensor(observation["observation"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(observation["action_mask"], dtype=torch.int8, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = self.actor(obs, mask)
        return int(action.item())

    def reset(self):
        pass

    def save(self, path: str):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
