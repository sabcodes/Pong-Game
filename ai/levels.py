"""The AI difficulty levels: which trained agent each uses and how fast the ball goes.

  easy    self-play, rewarded for every hit, so it defends well but rarely attacks
  hard    self-play with no hit reward and a small time cost, so it goes for winning shots,
          trained and played with the ball 1.3x faster
  medium  hard's AI retrained with the ball 1.15x faster

The same reward change at normal speed didn't beat easy: the ball is slow enough that a
good defender reaches everything. Each AI only has its edge at the speed it trained at.
"""
LEVELS = {
    "easy": {"model": "models/ppo_pong_easy.zip", "ball_speed": 1.0},
    "medium": {"model": "models/ppo_pong_medium.zip", "ball_speed": 1.15},
    "hard": {"model": "models/ppo_pong_hard.zip", "ball_speed": 1.3},
}
