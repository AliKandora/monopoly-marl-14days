"""Training script for Independent PPO (IPPO) with Parameter Sharing."""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.monopoly_env import MonopolyEnv
from src.agents.networks import MaskedActor, DecentralizedCritic
from src.agents.rollout_buffer import MultiAgentRolloutBuffer
from src.utils.feature_extraction import extract_global_state


def train_ippo(
    total_timesteps: int = 20000,
    num_steps: int = 128,
    batch_size: int = 64,
    update_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    save_path: str = "checkpoints/ippo_best.pt",
    device: str = "cpu",
    seed: int = 42,
):
    print("=======================================================")
    print(f" Starting IPPO Multi-Agent Training on {device.upper()}")
    print(f" Total Steps: {total_timesteps:,} | Rollout Buffer: {num_steps} | Batch: {batch_size}")
    print("=======================================================\n")

    if min(total_timesteps, num_steps, batch_size, update_epochs) < 1:
        raise ValueError("Training sizes must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    device_obj = torch.device(device)

    env = MonopolyEnv(max_turns=150)
    obs_dim = env.obs_dim
    num_actions = env.num_actions

    actor = MaskedActor(obs_dim, num_actions).to(device_obj)
    critic = DecentralizedCritic(obs_dim).to(device_obj)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr, eps=1e-5)

    buffer = MultiAgentRolloutBuffer(
        buffer_size=num_steps,
        num_agents=4,
        obs_dim=obs_dim,
        num_actions=num_actions,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    global_step = 0
    start_time = time.time()

    obs_dict, _ = env.reset(seed=seed)

    while global_step < total_timesteps:
        buffer.reset()

        for step in range(num_steps):
            global_step += 1  # one environment micro-step

            step_obs = np.zeros((4, obs_dim), dtype=np.float32)
            step_masks = np.zeros((4, num_actions), dtype=np.int8)
            step_actions = np.zeros(4, dtype=np.int64)
            step_logprobs = np.zeros(4, dtype=np.float32)
            step_values = np.zeros(4, dtype=np.float32)
            step_rewards = np.zeros(4, dtype=np.float32)
            step_dones = np.zeros(4, dtype=np.float32)

            valid = np.array([a in env.agents for a in env.possible_agents])
            # Get actions for all agents
            for i, agent in enumerate(env.possible_agents):
                if agent in env.agents:
                    step_obs[i] = obs_dict[agent]["observation"]
                    step_masks[i] = obs_dict[agent]["action_mask"]
                else:
                    # Agent is bankrupt, default mask
                    step_masks[i, 2] = 1  # pass
                    step_dones[i] = 1.0

            obs_tensor = torch.as_tensor(step_obs, device=device_obj)
            mask_tensor = torch.as_tensor(step_masks, device=device_obj)

            with torch.no_grad():
                actions, logprobs, _ = actor(obs_tensor, mask_tensor)
                values = critic(obs_tensor)

            step_actions = actions.cpu().numpy()
            step_logprobs = logprobs.cpu().numpy()
            step_values = values.cpu().numpy()

            action_dict = {
                agent: int(step_actions[i])
                for i, agent in enumerate(env.possible_agents)
                if agent in env.agents
            }

            s_global = extract_global_state(env)
            next_obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

            for i, agent in enumerate(env.possible_agents):
                step_rewards[i] = rewards.get(agent, 0.0)
                if terminations.get(agent, False) or truncations.get(agent, False):
                    step_dones[i] = 1.0

            buffer.insert(
                obs=step_obs,
                masks=step_masks,
                actions=step_actions,
                logprobs=step_logprobs,
                rewards=step_rewards,
                values=step_values,
                dones=step_dones,
                global_state=s_global,
                valid=valid,
            )

            obs_dict = next_obs_dict
            if not env.agents:
                obs_dict, _ = env.reset()

        # Compute GAE
        with torch.no_grad():
            next_obs = np.zeros((4, obs_dim), dtype=np.float32)
            for i, agent in enumerate(env.possible_agents):
                if agent in env.agents:
                    next_obs[i] = obs_dict[agent]["observation"]
            last_obs_tensor = torch.as_tensor(next_obs, device=device_obj)
            last_values = critic(last_obs_tensor).cpu().numpy()

        buffer.compute_gae(last_values=last_values, last_dones=step_dones)

        # Optimize PPO Loss
        pg_losses, v_losses, ent_losses = [], [], []

        for epoch in range(update_epochs):
            for batch in buffer.get_generator(batch_size):
                b_obs, b_masks, b_actions, b_logprobs, b_advantages, b_returns, b_values, _, _ = batch

                new_logprob, entropy = actor.evaluate(b_obs, b_masks, b_actions)
                new_value = critic(b_obs)

                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()

                # Policy Loss
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss with clipping
                v_loss_unclipped = (new_value - b_returns) ** 2
                v_clipped = b_values + torch.clamp(new_value - b_values, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step:6d}/{total_timesteps} | FPS: {fps:4d} | PG Loss: {np.mean(pg_losses):.4f} | VF Loss: {np.mean(v_losses):.4f} | Entropy: {np.mean(ent_losses):.4f}")

    # Save Checkpoint
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "metadata": {"seed": seed, "env_steps": global_step, "algorithm": "IPPO", "version": "1.0", "horizon": "finite episodic; truncation treated as terminal"},
    }, save_path)
    print(f"\n[SUCCESS] IPPO training finished! Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPPO on Monopoly MARL")
    parser.add_argument("--steps", type=int, default=20000, help="Total timesteps to train")
    parser.add_argument("--save-path", type=str, default="checkpoints/ippo_best.pt", help="Path to save checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ippo(total_timesteps=args.steps, save_path=args.save_path, seed=args.seed)
