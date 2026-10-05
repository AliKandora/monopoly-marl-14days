"""Multi-Agent Rollout Buffer with Generalized Advantage Estimation (GAE)."""

from __future__ import annotations
from typing import Generator, Tuple
import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """Rollout buffer collecting trajectories across multiple agents in parallel."""

    def __init__(
        self,
        buffer_size: int,
        num_agents: int = 4,
        obs_dim: int = 144,
        num_actions: int = 7,
        global_dim: int = 146,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.global_dim = global_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.reset()

    def reset(self):
        """Reset storage buffers."""
        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.masks = np.zeros((self.buffer_size, self.num_agents, self.num_actions), dtype=np.int8)
        self.actions = np.zeros((self.buffer_size, self.num_agents), dtype=np.int64)
        self.logprobs = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.global_states = np.zeros((self.buffer_size, self.global_dim), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)

        self.step_idx = 0

    def insert(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        global_state: np.ndarray,
    ):
        """Store step transition in buffer."""
        idx = self.step_idx
        self.obs[idx] = obs
        self.masks[idx] = masks
        self.actions[idx] = actions
        self.logprobs[idx] = logprobs
        self.rewards[idx] = rewards
        self.values[idx] = values
        self.dones[idx] = dones
        self.global_states[idx] = global_state
        self.step_idx += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """Compute Generalized Advantage Estimation across all agents."""
        last_adv = np.zeros(self.num_agents, dtype=np.float32)
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_val = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            self.advantages[t] = last_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv

        self.returns = self.advantages + self.values

    def get_generator(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Flatten buffer over (buffer_size * num_agents) and yield minibatches."""
        total_samples = self.buffer_size * self.num_agents
        indices = np.random.permutation(total_samples)

        # Flatten agent dimension
        flat_obs = self.obs.reshape(-1, self.obs_dim)
        flat_masks = self.masks.reshape(-1, self.num_actions)
        flat_actions = self.actions.reshape(-1)
        flat_logprobs = self.logprobs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        # Global states tiled for each agent
        tiled_global = np.repeat(self.global_states, self.num_agents, axis=0)

        # One-hot agent identities
        agent_ids = np.tile(np.eye(self.num_agents, dtype=np.float32), (self.buffer_size, 1))

        # Standardize advantages
        flat_adv_mean = np.mean(flat_advantages)
        flat_adv_std = np.std(flat_advantages) + 1e-8
        flat_advantages = (flat_advantages - flat_adv_mean) / flat_adv_std

        for start_i in range(0, total_samples, batch_size):
            end_i = min(start_i + batch_size, total_samples)
            batch_idx = indices[start_i:end_i]

            yield (
                torch.as_tensor(flat_obs[batch_idx], device=self.device),
                torch.as_tensor(flat_masks[batch_idx], device=self.device),
                torch.as_tensor(flat_actions[batch_idx], device=self.device),
                torch.as_tensor(flat_logprobs[batch_idx], device=self.device),
                torch.as_tensor(flat_advantages[batch_idx], device=self.device),
                torch.as_tensor(flat_returns[batch_idx], device=self.device),
                torch.as_tensor(flat_values[batch_idx], device=self.device),
                torch.as_tensor(tiled_global[batch_idx], device=self.device),
                torch.as_tensor(agent_ids[batch_idx], device=self.device),
            )
