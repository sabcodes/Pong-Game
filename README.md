# Pong-Game
All the code (both JavaScript and HTML5) for the game is in game.html. To run it, download the repo and open game.html in a web browser. Move the mouse to control the left paddle.

Press **A** to switch the right paddle between the classic CPU and a **neural network AI** trained with reinforcement learning.

## The AI (`ai/`)
The AI is trained in Python with PyTorch, Gymnasium and Stable Baselines 3, following a four-phase plan:

| Phase | File | What it does |
|---|---|---|
| 1. Environment | `pong_core.py`, `pong_env.py` | The game rules with no rendering, wrapped as a `gymnasium.Env` that runs headless at thousands of steps per second |
| 2. Tabular Q-learning | `tabular_q.py` | Bins the board into a grid and learns a Q-table, to validate the environment |
| 3. Deep Q-Network | `dqn.py` | PyTorch DQN with an experience replay buffer and a target network synced every 1000 steps |
| 4. PPO | `train_ppo.py` | The production agent, trained with Stable Baselines 3 PPO |

Environment spec: actions are `Discrete(3)` (0 = up, 1 = down, 2 = stay); observations are `Box(5,)` = `[ball_x, ball_y, ball_vx, ball_vy, paddle_y]` (normalized); rewards are +1 for hitting the ball, -10 for being scored on and +10 for scoring; an episode ends when either side reaches 11. The agent plays the left paddle against the hardcoded CPU from game.html.

### Setup
```bash
cd ai
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
```

### Train and play
```bash
../.venv/bin/python train_ppo.py        # ~10 min on a laptop CPU, writes models/ppo_pong_agent.zip
../.venv/bin/python export_policy.py    # writes models/ppo_pong_agent.js for game.html
../.venv/bin/python play.py             # watch the agent vs the CPU in Pygame
../.venv/bin/python play.py --human     # play against the agent in Pygame
../.venv/bin/python play.py --headless --games 20   # win rate vs the CPU
../.venv/bin/pytest                     # environment checks
```
The trained agent is committed in `ai/models/`, so game.html works without training.

Note on the game rules: the ball now leaves the paddle at an angle depending on where it hits (both in game.html and the Python core). Without this the ball's vertical speed never changes and the CPU can never be beaten.
