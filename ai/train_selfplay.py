"""Train a stronger agent with self-play.

Every game the agent gets a random opponent from a pool:
  - the classic CPU and the sharpshooter (fixed difficulty, so it never forgets the basics)
  - earlier agents passed with --baseline
  - frozen copies of itself saved during training, which get harder as it improves
Playing only the latest copy can make the agent chase its own tail, so older copies stay in the pool.

  # easy (rewarded +1 per hit)
  python train_selfplay.py --steps 10000000 --out models/ppo_pong_easy
  # hard: paid only for winning points, at 1.3x ball speed, starting from easy and checked against it
  # (the game uses the last checkpoint, hard_final, which beat the "best" one 30/30)
  python train_selfplay.py --steps 10000000 --ball-speed 1.3 --hit-reward 0 --time-penalty 0.002 \
      --gamma 0.995 --init-from models/ppo_pong_easy.zip \
      --baseline models/ppo_pong_easy.zip models/ppo_pong_agent_v1.zip \
      --eval-against models/ppo_pong_easy.zip --out models/ppo_pong_hard
  # medium: the same at 1.15x, starting from hard
  python train_selfplay.py --steps 10000000 --ball-speed 1.15 --hit-reward 0 --time-penalty 0.002 \
      --gamma 0.995 --init-from models/ppo_pong_hard.zip \
      --baseline models/ppo_pong_easy.zip models/ppo_pong_hard.zip models/ppo_pong_agent_v1.zip \
      --eval-against models/ppo_pong_easy.zip --out models/ppo_pong_medium

Saves the checkpoint that did best in the periodic evaluation to <out>.zip, and the last one to <out>_final.zip.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate import load_policy, play_match
from opponents import CpuOpponent, NumpyPolicy, PolicyOpponent, SharpshooterOpponent
from pong_env import REWARD_HIT, PongEnv


class OpponentPool:
    def __init__(self, max_snapshots=8, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_snapshots = max_snapshots
        self.fixed = [CpuOpponent(), SharpshooterOpponent(self.rng)]
        self.baselines = []
        self.snapshots = []

    def add(self, opponent):
        self.snapshots = (self.snapshots + [opponent])[-self.max_snapshots:]

    def sample(self):
        # 20% classic CPU, 30% sharpshooter, 50% an earlier agent or past copy of itself (once there are any)
        roll = self.rng.random()
        others = self.baselines + self.snapshots
        if roll < 0.2:
            return self.fixed[0]
        if roll < 0.5 or not others:
            return self.fixed[1]
        return others[self.rng.integers(len(others))]


class SelfPlayCallback(BaseCallback):
    """Adds snapshots to the pool and keeps the best checkpoint.

    Best = most wins (then best score difference) against eval_opponent if given,
    otherwise against the sharpshooter."""

    def __init__(self, pool, snapshot_every, eval_every, eval_games, best_path, ball_speed, eval_opponent=None):
        super().__init__()
        self.pool = pool
        self.snapshot_every = snapshot_every
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.best_path = best_path
        self.ball_speed = ball_speed
        self.eval_opponent = eval_opponent
        self.best = None
        self.next_snapshot = snapshot_every
        self.next_eval = eval_every

    def _on_step(self):
        if self.num_timesteps >= self.next_snapshot:
            self.next_snapshot += self.snapshot_every
            policy = NumpyPolicy.from_sb3(self.model.policy)
            self.pool.add(PolicyOpponent(policy, name=f"self@{self.num_timesteps}"))

        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            policy = NumpyPolicy.from_sb3(self.model.policy)
            results = {}
            opponents = [SharpshooterOpponent(), CpuOpponent()] + ([self.eval_opponent] if self.eval_opponent else [])
            for opponent in opponents:
                results[opponent.name] = play_match(policy, opponent, self.eval_games, seed=123,
                                                    ball_speed=self.ball_speed)
            print(f"[eval @ {self.num_timesteps:,}] " + " | ".join(
                f"vs {name}: won {w}/{self.eval_games}, avg {s:.1f} - {c:.1f}" for name, (w, s, c) in results.items()),
                flush=True)
            w, s, c = results[self.eval_opponent.name if self.eval_opponent else "sharpshooter"]
            score = (w, s - c)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(self.best_path)
                print(f"  new best, saved {self.best_path}.zip", flush=True)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--out", default="models/ppo_pong_selfplay")
    parser.add_argument("--ball-speed", type=float, default=1.0)
    parser.add_argument("--hit-reward", type=float, default=REWARD_HIT)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="higher looks further ahead: a winning shot only pays off ~80 frames later")
    parser.add_argument("--init-from", help="start from this agent's weights instead of from scratch")
    parser.add_argument("--baseline", nargs="*", default=[], help="earlier agents to add to the opponent pool")
    parser.add_argument("--eval-against", help="pick the best checkpoint by results against this agent")
    args = parser.parse_args()

    pool = OpponentPool()
    for path in args.baseline:
        pool.baselines.append(PolicyOpponent(load_policy(path), name=os.path.basename(path)))
    eval_opponent = (PolicyOpponent(load_policy(args.eval_against), name=os.path.basename(args.eval_against))
                     if args.eval_against else None)

    def make_env():
        return Monitor(PongEnv(opponent_sampler=pool.sample, ball_speed=args.ball_speed,
                               hit_reward=args.hit_reward, time_penalty=args.time_penalty))

    env = DummyVecEnv([make_env for _ in range(args.envs)])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=256,
                ent_coef=0.01, gamma=args.gamma, device="cpu")
    if args.init_from:
        # copy the weights only, so the new reward settings and hyperparameters apply
        model.policy.load_state_dict(PPO.load(args.init_from, device="cpu").policy.state_dict())
    callback = SelfPlayCallback(pool, snapshot_every=100_000, eval_every=250_000, eval_games=10,
                                best_path=args.out, ball_speed=args.ball_speed, eval_opponent=eval_opponent)
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(args.out + "_final")
    print(f"done. best checkpoint: {args.out}.zip, last: {args.out}_final.zip")
