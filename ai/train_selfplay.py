"""Train a stronger agent with self-play.

Every game the agent gets a random opponent from a pool:
  - the classic CPU and the sharpshooter (fixed difficulty, so it never forgets the basics)
  - frozen copies of itself saved during training, which get harder as it improves
Playing only the latest copy can make the agent chase its own tail, so older copies stay in the pool.

  python train_selfplay.py --steps 3000000
Writes models/ppo_pong_agent.zip (the checkpoint that did best against the sharpshooter).
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
from pong_env import PongEnv


class OpponentPool:
    def __init__(self, max_snapshots=8, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_snapshots = max_snapshots
        self.fixed = [CpuOpponent(), SharpshooterOpponent(self.rng)]
        self.snapshots = []

    def add(self, opponent):
        self.snapshots = (self.snapshots + [opponent])[-self.max_snapshots:]

    def sample(self):
        # 20% classic CPU, 30% sharpshooter, 50% a past copy of the agent (once there are any)
        roll = self.rng.random()
        if roll < 0.2:
            return self.fixed[0]
        if roll < 0.5 or not self.snapshots:
            return self.fixed[1]
        return self.snapshots[self.rng.integers(len(self.snapshots))]


class SelfPlayCallback(BaseCallback):
    def __init__(self, pool, snapshot_every, eval_every, eval_games, best_path):
        super().__init__()
        self.pool = pool
        self.snapshot_every = snapshot_every
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.best_path = best_path
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
            wins, scored, conceded = play_match(policy, SharpshooterOpponent(), self.eval_games, seed=123)
            cpu_wins, _, _ = play_match(policy, CpuOpponent(), self.eval_games, seed=123)
            print(f"[eval @ {self.num_timesteps:,}] vs sharpshooter: won {wins}/{self.eval_games}, "
                  f"avg score {scored:.1f} - {conceded:.1f} | vs cpu: won {cpu_wins}/{self.eval_games}", flush=True)
            score = (wins, scored - conceded)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(self.best_path)
                print(f"  new best, saved {self.best_path}.zip", flush=True)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--out", default="models/ppo_pong_agent")
    parser.add_argument("--baseline", default="models/ppo_pong_agent_v1.zip",
                        help="an earlier agent to include in the opponent pool, if it exists")
    args = parser.parse_args()

    pool = OpponentPool()
    if os.path.exists(args.baseline):
        pool.add(PolicyOpponent(load_policy(args.baseline), name="v1"))

    env = DummyVecEnv([lambda: Monitor(PongEnv(opponent_sampler=pool.sample)) for _ in range(args.envs)])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=256,
                ent_coef=0.01, device="cpu")
    callback = SelfPlayCallback(pool, snapshot_every=100_000, eval_every=250_000, eval_games=10,
                                best_path=args.out)
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(args.out + "_final")
    print(f"done. best checkpoint: {args.out}.zip, last: {args.out}_final.zip")
