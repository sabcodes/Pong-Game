# Pong-Game
All the code (both JavaScript and HTML5) for the game is in game.html. To run it, download the repo and open game.html in a web browser. Move the mouse to control the left paddle, and press **P** or **Space** to pause.

Press **A** to switch the right paddle between the classic CPU and a **neural network AI** trained with reinforcement learning.

## The AI (`ai/`)
The AI is trained in Python with PyTorch, Gymnasium and Stable Baselines 3, following a four-phase plan:

| Phase | File | What it does |
|---|---|---|
| 1. Environment | `pong_core.py`, `pong_env.py` | The game rules with no rendering, wrapped as a `gymnasium.Env` that runs headless at thousands of steps per second |
| 2. Tabular Q-learning | `tabular_q.py` | Bins the board into a grid and learns a Q-table, to validate the environment |
| 3. Deep Q-Network | `dqn.py` | PyTorch DQN with an experience replay buffer and a target network synced every 1000 steps |
| 4. PPO | `train_ppo.py` | The first agent, trained with Stable Baselines 3 PPO against the classic CPU |
| 5. Self-play | `train_selfplay.py` | The agent in the game: PPO trained against a pool of opponents, including frozen copies of itself |

Environment spec: actions are `Discrete(3)` (0 = up, 1 = down, 2 = stay); observations are `Box(6,)` = `[ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y]` (normalized); rewards are +1 for hitting the ball, -10 for being scored on and +10 for scoring; an episode ends when either side reaches 11. The agent plays the left paddle. By default the right paddle is the hardcoded CPU from game.html; `opponents.py` has the other opponents.

### Self-play
The PPO agent trained only against the classic CPU (now kept as `models/ppo_pong_agent_v1.zip`) was easy for people to beat. The CPU returns the ball almost straight, so the agent never learned to handle angled shots. `train_selfplay.py` gives the agent a random opponent every game:
- 20%: the classic CPU
- 30%: the *sharpshooter*, which predicts where the ball will land, gets there fast and hits it off-centre to make sharp angles
- 50%: a frozen copy of the agent from earlier in training (a new copy is added every 100k steps; the last 8 are kept)

The agent also sees the opponent's paddle position, so it can learn to aim away from it.

Results from `evaluate.py` (30 games each; games stop at 5,000 frames, so a score below 11 means the game ran out of time):

| Agent | vs classic CPU | vs sharpshooter (points let in) | vs v1 |
|---|---|---|---|
| v1 (CPU only, 500k steps) | won 23/30, 10.4 – 7.4 | 9.9 | — |
| Self-play, 10M steps (in the game) | won 30/30, 11.0 – 0.2 | 0.1 | won 30/30, 10.0 – 0.7 |

Nothing has scored on the sharpshooter yet, so it measures defence only.

### Setup
```bash
cd ai
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
```

### Train and play
```bash
../.venv/bin/python train_selfplay.py --steps 10000000   # ~14 min on a laptop CPU, writes models/ppo_pong_agent.zip
../.venv/bin/python evaluate.py models/ppo_pong_agent_v1.zip models/ppo_pong_agent.zip   # compare agents
../.venv/bin/python export_policy.py    # writes models/ppo_pong_agent.js for game.html
../.venv/bin/python play.py             # watch the agent vs the CPU in Pygame
../.venv/bin/python play.py --human     # play against the agent in Pygame
../.venv/bin/python play.py --headless --games 20   # win rate vs the CPU
../.venv/bin/pytest                     # environment checks
```
The trained agent is committed in `ai/models/`, so game.html works without training.

Notes on the game rules (the same in game.html and the Python core):
- The ball leaves the paddle at an angle depending on where it hits. Without this the ball's vertical speed never changes and the CPU can never be beaten.
- The ball bounces off the top and bottom walls before the paddle check, so a ball in a corner can't slip past a paddle.
