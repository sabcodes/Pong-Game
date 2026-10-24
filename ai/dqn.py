"""Phase 3: Deep Q-Network in plain PyTorch.

Uses an experience replay buffer (randomized mini-batches of (s, a, r, s', done))
and a target network synced from the policy network every `target_sync` steps.
Usage:  python dqn.py --steps 200000
"""
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from pong_env import PongEnv


class QNetwork(nn.Module):
    def __init__(self, obs_size=6, n_actions=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train(total_steps=200_000, batch_size=64, gamma=0.99, lr=1e-3, buffer_size=100_000,
          learning_starts=1_000, target_sync=1_000, eps_start=1.0, eps_end=0.05, eps_fraction=0.3,
          log_every=10_000):
    env = PongEnv()
    policy_net, target_net = QNetwork(), QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    obs, _ = env.reset()
    episode_return, returns = 0.0, []
    for step in range(1, total_steps + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / (eps_fraction * total_steps))
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(policy_net(torch.as_tensor(obs).unsqueeze(0)).argmax())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, terminated)
        obs, episode_return = next_obs, episode_return + reward
        if terminated or truncated:
            returns.append(episode_return)
            obs, _ = env.reset()
            episode_return = 0.0

        if len(buffer) >= learning_starts:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                td_target = rewards + gamma * (1 - dones) * target_net(next_states).max(1).values
            loss = nn.functional.smooth_l1_loss(q_values, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % target_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % log_every == 0:
            recent = f"{np.mean(returns[-10:]):8.2f}" if returns else "     n/a"
            print(f"step {step:7d}  avg return (last 10 eps) {recent}  epsilon {epsilon:.3f}")
    return policy_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--out", default="models/dqn_pong.pt")
    args = parser.parse_args()
    net = train(args.steps)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")
