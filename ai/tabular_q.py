"""Phase 2: tabular Q-learning prototype, used to validate the environment.

The continuous observation is binned into a grid so it can index a Q-table.
Usage:  python tabular_q.py --episodes 3000
"""
import argparse

import numpy as np

from pong_env import PongEnv

# Bins per observation dimension: ball_x, ball_y, ball_vx (direction), ball_vy, paddle_y.
# The opponent's paddle position is left out to keep the table small.
BINS = (12, 12, 2, 6, 12)
LOWS = np.array([0, 0, -1, -1, 0], dtype=np.float32)
HIGHS = np.array([1, 1, 1, 1, 1], dtype=np.float32)


def discretize(obs):
    ratios = (np.asarray(obs)[:len(BINS)] - LOWS) / (HIGHS - LOWS)
    idx = (ratios * np.array(BINS)).astype(int)
    return tuple(np.clip(idx, 0, np.array(BINS) - 1))


class TabularQAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95):
        self.q_table = np.zeros(state_size + (action_size,))
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        return int(np.argmax(self.q_table[state]))  # Exploit

    def learn(self, state, action, reward, next_state, done=False):
        best_next_action = np.argmax(self.q_table[next_state])
        future = 0.0 if done else self.gamma * self.q_table[next_state][best_next_action]
        td_target = reward + future
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error


def train(episodes, max_steps=2000, epsilon_min=0.05, epsilon_decay=0.998, log_every=100):
    env = PongEnv(max_steps=max_steps)
    agent = TabularQAgent(BINS, env.action_space.n)
    returns = []
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state = discretize(obs)
        total, done = 0.0, False
        while not done:
            action = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(obs)
            agent.learn(state, action, reward, next_state, done=terminated)
            state, total, done = next_state, total + reward, terminated or truncated
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        returns.append(total)
        if episode % log_every == 0:
            print(f"episode {episode:5d}  avg return {np.mean(returns[-log_every:]):8.2f}  epsilon {agent.epsilon:.3f}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--out", default="models/q_table.npy")
    args = parser.parse_args()
    trained = train(args.episodes)
    np.save(args.out, trained.q_table)
    print(f"saved {args.out}")
