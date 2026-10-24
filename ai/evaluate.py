"""Measure how strong agents are: each one plays games against the classic CPU,
the sharpshooter, and every other agent given.

  python evaluate.py models/ppo_pong_agent_v1.zip models/ppo_pong_agent.zip --games 20
"""
import argparse
import itertools
import os

import numpy as np
from stable_baselines3 import PPO

from opponents import CpuOpponent, NumpyPolicy, PolicyOpponent, SharpshooterOpponent
from pong_core import PongGame


def play_match(policy, opponent, games, target_score=11, max_steps=5000, seed=0):
    """Agent on the left vs opponent on the right. Returns (wins, avg points for, avg points against)."""
    rng = np.random.default_rng(seed)
    wins, scored, conceded = 0, 0, 0
    for _ in range(games):
        game = PongGame(rng=rng)
        if hasattr(opponent, "rng"):
            opponent.rng = rng
        for _ in range(max_steps):
            game.step(policy.act(game.observation("left")), opponent.act(game))
            if max(game.left_score, game.right_score) >= target_score:
                break
        wins += game.left_score > game.right_score
        scored += game.left_score
        conceded += game.right_score
    return wins, scored / games, conceded / games


def load_policy(path):
    return NumpyPolicy.from_sb3(PPO.load(path, device="cpu").policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()

    policies = {os.path.basename(p).removesuffix(".zip"): load_policy(p) for p in args.models}
    for name, policy in policies.items():
        for opponent in (CpuOpponent(), SharpshooterOpponent()):
            wins, scored, conceded = play_match(policy, opponent, args.games)
            print(f"{name:24s} vs {opponent.name:24s} won {wins:2d}/{args.games}  "
                  f"avg score {scored:4.1f} - {conceded:4.1f}")
    for (a, pa), (b, pb) in itertools.permutations(policies.items(), 2):
        wins, scored, conceded = play_match(pa, PolicyOpponent(pb, b), args.games)
        print(f"{a:24s} vs {b:24s} won {wins:2d}/{args.games}  avg score {scored:4.1f} - {conceded:4.1f}")
