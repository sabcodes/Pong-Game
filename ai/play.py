"""Phase 4 validation: load the trained PPO agent into the visual Pygame loop.

  python play.py                 # easy PPO agent (left) vs hardcoded CPU (right)
  python play.py --human --level hard    # you (mouse, left) vs the hard agent (right), at its ball speed
  python play.py --games 20 --headless   # just report win rate vs the CPU
"""
import argparse

import pygame
from stable_baselines3 import PPO

from levels import LEVELS
from pong_core import HEIGHT, PADDLE_HEIGHT, PongGame


def agent_action(model, game, side):
    action, _ = model.predict(game.observation(side), deterministic=True)
    return int(action)


def play_vs_cpu(model, viewer, ball_speed, target_score=11):
    game = PongGame(ball_speed=ball_speed)
    while max(game.left_score, game.right_score) < target_score:
        game.step(agent_action(model, game, "left"))
        if viewer:
            viewer.draw(game, "PPO agent", "Hardcoded CPU")
            if pygame.event.peek(pygame.QUIT):
                return None
    return game.left_score, game.right_score


def play_vs_human(model, viewer, ball_speed, target_score=11):
    game = PongGame(ball_speed=ball_speed)
    while max(game.left_score, game.right_score) < target_score:
        if pygame.event.peek(pygame.QUIT):
            return
        mouse_y = pygame.mouse.get_pos()[1]
        game.left_y = min(max(mouse_y - PADDLE_HEIGHT / 2, 0), HEIGHT - PADDLE_HEIGHT)
        game.step(left_action=2, right_action=agent_action(model, game, "right"))
        viewer.draw(game, "You (mouse)", "PPO agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=LEVELS, default="easy")
    parser.add_argument("--model", help="play a different trained agent at the level's ball speed")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    model = PPO.load(args.model or LEVELS[args.level]["model"])
    ball_speed = LEVELS[args.level]["ball_speed"]
    viewer = None
    if not args.headless:
        from viewer import PongViewer
        viewer = PongViewer()

    if args.human:
        play_vs_human(model, viewer, ball_speed)
    else:
        wins = 0
        for i in range(args.games):
            result = play_vs_cpu(model, viewer, ball_speed)
            if result is None:
                break
            wins += result[0] > result[1]
            print(f"game {i + 1}: agent {result[0]} - {result[1]} CPU")
        print(f"agent won {wins}/{args.games}")
    if viewer:
        viewer.close()
