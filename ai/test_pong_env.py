"""Checks from the plan's integration workflow (steps 1 and 2). Run: pytest"""
import sys

import numpy as np
from gymnasium.utils.env_checker import check_env

from pong_core import HEIGHT, PADDLE_HEIGHT, WIDTH, PongGame
from opponents import CpuOpponent, SharpshooterOpponent, predict_intercept_y
from pong_env import REWARD_HIT, REWARD_MISSED, REWARD_SCORED, PongEnv


def test_core_runs_10000_steps_without_a_window():
    game = PongGame(rng=np.random.default_rng(0))
    for _ in range(10_000):
        game.step(np.random.randint(3))
    assert "pygame" not in sys.modules


def test_env_passes_gymnasium_checker():
    check_env(PongEnv())


def test_step_returns_gymnasium_tuple():
    env = PongEnv()
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (6,) and obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_rewards_for_hit_miss_and_score():
    env = PongEnv()
    env.reset(seed=0)
    g = env.game

    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 5, HEIGHT / 2, -10, 0
    g.left_y = HEIGHT / 2 - PADDLE_HEIGHT / 2
    assert env.step(2)[1] == REWARD_HIT

    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 5, 10, -10, 0
    g.left_y = HEIGHT - PADDLE_HEIGHT
    assert env.step(2)[1] == REWARD_MISSED

    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = WIDTH - 5, 10, 10, 0
    g.right_y = HEIGHT - PADDLE_HEIGHT
    assert env.step(2)[1] == REWARD_SCORED


def test_terminates_at_target_score():
    env = PongEnv(target_score=3)
    env.reset(seed=0)
    env.game.right_score = 2
    g = env.game
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 5, 10, -10, 0
    g.left_y = HEIGHT - PADDLE_HEIGHT
    assert env.step(2)[2] is True


def test_observation_is_mirrored_for_right_side():
    game = PongGame(rng=np.random.default_rng(0))
    game.ball_x, game.ball_vx = 200, 10
    game.left_y, game.right_y = 0, HEIGHT - PADDLE_HEIGHT
    left, right = game.observation("left"), game.observation("right")
    assert np.isclose(left[0], 0.25) and np.isclose(right[0], 0.75)
    assert left[2] == 1 and right[2] == -1
    assert (left[4], left[5]) == (0, 1) and (right[4], right[5]) == (1, 0)


def test_sharpshooter_predicts_bounce_and_returns_the_ball():
    game = PongGame(rng=np.random.default_rng(0))
    game.ball_x, game.ball_y, game.ball_vx, game.ball_vy = 400, 300, 10, 15
    assert np.isclose(predict_intercept_y(game), 285)  # 41 frames: 300 + 615 = 915, bounces off the bottom to 285

    env = PongEnv(opponent=SharpshooterOpponent(np.random.default_rng(0)))
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, info = env.step(2)
        if terminated:
            break
    assert info["left_score"] == 0  # a still paddle never scores against it


def test_opponent_sampler_is_called_on_reset():
    calls = []
    env = PongEnv(opponent_sampler=lambda: calls.append(1) or CpuOpponent())
    env.reset(seed=0)
    env.reset(seed=1)
    assert len(calls) == 2


def test_ball_speed_scales_ball_but_not_observation_range():
    fast = PongGame(rng=np.random.default_rng(0), ball_speed=1.3)
    assert abs(fast.ball_vx) == 13
    assert abs(fast.observation()[2]) == 1


def test_hit_reward_and_time_penalty_are_configurable():
    env = PongEnv(hit_reward=0.0, time_penalty=0.002)
    env.reset(seed=0)
    g = env.game
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 5, HEIGHT / 2, -10, 0
    g.left_y = HEIGHT / 2 - PADDLE_HEIGHT / 2
    assert np.isclose(env.step(2)[1], -0.002)
