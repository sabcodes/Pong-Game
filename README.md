# Pong-Game
All the code (both JavaScript and HTML5) for the game is in game.html. To run it, download the repo and open game.html in a web browser. Move the mouse to control the left paddle, and press **P** or **Space** to pause.

Press **1**, **2** or **3** to play a **neural network AI** trained with reinforcement learning on easy, medium or hard. Medium and hard also speed up the ball. Press **A** to switch between the AI and the classic CPU.

## The AI (`ai/`)
The AI is trained in Python with PyTorch, Gymnasium and Stable Baselines 3, following a four-phase plan:

| Phase | File | What it does |
|---|---|---|
| 1. Environment | `pong_core.py`, `pong_env.py` | The game rules with no rendering, wrapped as a `gymnasium.Env` that runs headless at thousands of steps per second |
| 2. Tabular Q-learning | `tabular_q.py` | Bins the board into a grid and learns a Q-table, to validate the environment |
| 3. Deep Q-Network | `dqn.py` | PyTorch DQN with an experience replay buffer and a target network synced every 1000 steps |
| 4. PPO | `train_ppo.py` | The first agent, trained with Stable Baselines 3 PPO against the classic CPU |
| 5. Self-play | `train_selfplay.py` | The agents in the game: PPO trained against a pool of opponents, including frozen copies of itself |

Environment spec: actions are `Discrete(3)` (0 = up, 1 = down, 2 = stay); observations are `Box(6,)` = `[ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y]` (normalized); rewards are +1 for hitting the ball, -10 for being scored on and +10 for scoring (the hit reward and an optional per-frame time cost can be changed); an episode ends when either side reaches 11. The agent plays the left paddle. By default the right paddle is the hardcoded CPU from game.html; `opponents.py` has the other opponents.

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
| Self-play, 10M steps (now the easy level) | won 30/30, 11.0 – 0.2 | 0.1 | won 30/30, 10.0 – 0.7 |

Nothing has scored on the sharpshooter yet, so it measures defence only.

### Difficulty levels
Levels are defined in `levels.py`:

| Level | Ball speed | Training |
|---|---|---|
| Easy | normal | self-play with +1 per hit |
| Medium | 1.15× | hard's AI retrained at 1.15× |
| Hard | 1.3× | no hit reward, a 0.002 per-frame time cost and gamma 0.995, so it is paid only for winning points and finishing them sooner |

The +1 hit reward pays the easy AI to keep rallies going, so it defends well but rarely goes for a winning shot. Medium and hard start from the easy AI's weights and are rewarded only for winning points.

Results (30 games each, at each level's ball speed):
- **Medium beats easy at 1.15×:** 21/30 when it plays the left side; easy wins 2/30 when the sides are swapped.
- **Hard beats easy at 1.3×:** 23/30. At 1.3× hard's AI and medium's AI are about even, so hard is harder mainly because of its faster ball.
- **Speed matters:** each AI only has its edge at the speed it trained at. The same reward change at normal speed didn't beat easy (easy won 22/30), because at normal speed a good defender can reach every ball.

Scores in these games are low, because most games between two AIs run to the 5,000-frame limit. A "win" usually means leading when time ran out.

### Setup
```bash
cd ai
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
```

### Train and play
```bash
# each run is ~15 min on a laptop CPU; see train_selfplay.py for the exact medium and hard commands
../.venv/bin/python train_selfplay.py --steps 10000000 --out models/ppo_pong_easy
../.venv/bin/python evaluate.py models/ppo_pong_easy.zip models/ppo_pong_medium.zip --ball-speed 1.15   # compare agents
../.venv/bin/python export_policy.py    # writes models/ppo_pong_<level>.js for game.html (then bump ?v= in game.html)
../.venv/bin/python play.py             # watch the easy agent vs the CPU in Pygame
../.venv/bin/python play.py --human --level hard   # play against the hard agent in Pygame
../.venv/bin/python play.py --headless --games 20   # win rate vs the CPU
../.venv/bin/pytest                     # environment checks
```
The trained agents are committed in `ai/models/`, so game.html works without training.

Notes on the game rules (the same in game.html and the Python core):
- The ball leaves the paddle at an angle depending on where it hits. Without this the ball's vertical speed never changes and the CPU can never be beaten.
- The ball bounces off the top and bottom walls before the paddle check, so a ball in a corner can't slip past a paddle.
