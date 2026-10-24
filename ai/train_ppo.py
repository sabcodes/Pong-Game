"""Phase 4: train the production agent with PPO (Stable Baselines 3), headlessly.

Usage:  python train_ppo.py --steps 500000
Then:   python play.py --model models/ppo_pong_agent_cpu_only.zip   (watch it in Pygame)
For the stronger agent the game uses, see train_selfplay.py.
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pong_env import PongEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--out", default="models/ppo_pong_agent_cpu_only",
                        help="kept separate from models/ppo_pong_agent.zip, the self-play agent the game uses")
    args = parser.parse_args()

    # 1. Initialize custom environment (Monitor logs ep_rew_mean so reward progress is visible)
    env = DummyVecEnv([lambda: Monitor(PongEnv())])
    # 2. Instantiate the PPO agent (Multi-Layer Perceptron Policy)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    # 3. Train the model headlessly
    model.learn(total_timesteps=args.steps)
    # 4. Save the optimized weights
    model.save(args.out)
    print(f"saved {args.out}.zip")
