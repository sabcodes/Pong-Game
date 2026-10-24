"""Gymnasium wrapper around the headless Pong core.

The agent controls the left paddle. The right paddle is the hardcoded CPU from game.html
by default, or any opponent from opponents.py (self-play passes a sampler instead).

  Action space:      Discrete(3)  0 = Up, 1 = Down, 2 = Stay
  Observation space: Box(6,)      [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y] (normalized)
  Reward:            +1 ball hit, -10 scored on, +10 agent scores (hit reward and a per-frame
                     time cost are configurable: the medium and hard levels use 0 and a small cost)
  Termination:       either player reaches target_score
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from opponents import CpuOpponent
from pong_core import PongGame

REWARD_HIT = 1.0
REWARD_MISSED = -10.0
REWARD_SCORED = 10.0


class PongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, target_score=11, max_steps=5000, render_mode=None, opponent=None, opponent_sampler=None,
                 ball_speed=1.0, hit_reward=REWARD_HIT, time_penalty=0.0):
        """opponent: fixed right-paddle opponent (default: classic CPU).
        opponent_sampler: called on every reset to pick a new opponent (used for self-play).
        hit_reward: reward for returning the ball. With it, long rallies pay, so the agent has
            little reason to go for winning shots.
        time_penalty: subtracted every frame, so finishing points sooner pays."""
        super().__init__()
        self.hit_reward = hit_reward
        self.time_penalty = time_penalty
        self.opponent = opponent or CpuOpponent()
        self.opponent_sampler = opponent_sampler
        self.target_score = target_score
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.game = PongGame(ball_speed=ball_speed)
        self._viewer = None
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.rng = self.np_random
        self.game.reset()
        self._steps = 0
        if self.opponent_sampler is not None:
            self.opponent = self.opponent_sampler()
        return self.game.observation(), {}

    def step(self, action):
        events = self.game.step(int(action), self.opponent.act(self.game))
        self._steps += 1

        reward = -self.time_penalty
        if events["left_hit"]:
            reward += self.hit_reward
        if events["right_scored"]:
            reward += REWARD_MISSED
        if events["left_scored"]:
            reward += REWARD_SCORED

        terminated = max(self.game.left_score, self.game.right_score) >= self.target_score
        truncated = not terminated and self._steps >= self.max_steps
        info = {"left_score": self.game.left_score, "right_score": self.game.right_score,
                "opponent": self.opponent.name}

        if self.render_mode == "human":
            self.render()
        return self.game.observation(), reward, terminated, truncated, info

    def render(self):
        if self._viewer is None:
            from viewer import PongViewer  # pygame is only imported when actually rendering
            self._viewer = PongViewer()
        self._viewer.draw(self.game)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
