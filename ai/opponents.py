"""Opponents for the right paddle, used for training and evaluation.

  CpuOpponent          the classic CPU from game.html (slow, returns the ball almost straight)
  SharpshooterOpponent predicts where the ball lands, gets there fast and hits it off-centre
                       for sharp angles, roughly how a good human plays
  PolicyOpponent       a trained network (a past copy of the agent, for self-play)

Each opponent's act(game) returns the right paddle's action for game.step(), or None
to let the classic CPU move it.
"""
import numpy as np
import torch.nn as nn

from pong_core import HEIGHT, PADDLE_HEIGHT, STAY, WIDTH

SHARPSHOOTER_SPEED = 14
SHARPSHOOTER_MAX_OFFSET = 40  # how far from the paddle centre it aims to hit, in pixels


class CpuOpponent:
    name = "cpu"

    def act(self, game):
        return None


class SharpshooterOpponent:
    name = "sharpshooter"

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.offset = 0.0
        self.react_x = 0.0
        self.tracking_vx = 0.0

    def act(self, game):
        if game.ball_vx > 0:
            if self.tracking_vx <= 0:  # new incoming shot: pick a reaction point and where to hit it
                self.offset = self.rng.uniform(-SHARPSHOOTER_MAX_OFFSET, SHARPSHOOTER_MAX_OFFSET)
                self.react_x = self.rng.uniform(WIDTH * 0.25, WIDTH * 0.55)
            target = predict_intercept_y(game) - self.offset if game.ball_x >= self.react_x else None
        else:
            target = HEIGHT / 2
        self.tracking_vx = game.ball_vx

        if target is not None:
            center = game.right_y + PADDLE_HEIGHT / 2
            game.right_y += np.clip(target - center, -SHARPSHOOTER_SPEED, SHARPSHOOTER_SPEED)
            game.right_y = float(np.clip(game.right_y, 0, HEIGHT - PADDLE_HEIGHT))
        return STAY


def predict_intercept_y(game):
    """Ball y on the frame it passes the right edge, including wall bounces."""
    frames = np.floor((WIDTH - game.ball_x) / game.ball_vx) + 1
    y = (game.ball_y + game.ball_vy * frames) % (2 * HEIGHT)
    return 2 * HEIGHT - y if y > HEIGHT else y


class NumpyPolicy:
    """The PPO actor network evaluated with numpy: much faster than SB3's predict() per step."""

    def __init__(self, layers):
        self.layers = layers  # list of (weights, bias)
        self.input_size = layers[0][0].shape[1]

    @classmethod
    def from_sb3(cls, policy):
        linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [policy.action_net]
        return cls([(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in linears])

    def act(self, obs):
        x = obs[:self.input_size]  # older agents were trained without the opponent's paddle position
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return int(np.argmax(x))


class PolicyOpponent:
    def __init__(self, policy, name="self"):
        self.policy = policy
        self.name = name

    def act(self, game):
        return self.policy.act(game.observation("right"))
