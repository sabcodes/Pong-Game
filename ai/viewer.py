"""Pygame renderer. Only draws a PongGame's state; it never updates it."""
import pygame

from pong_core import HEIGHT, PADDLE_HEIGHT, PADDLE_THICKNESS, WIDTH


class PongViewer:
    def __init__(self, fps=30, caption="Pong AI"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.small_font = pygame.font.SysFont(None, 24)
        self.fps = fps

    def draw(self, game, left_label="", right_label=""):
        pygame.event.pump()
        self.screen.fill((0, 0, 0))
        for y in range(0, HEIGHT, 40):
            pygame.draw.rect(self.screen, (60, 60, 60), (WIDTH // 2 - 1, y, 2, 20))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, game.left_y, PADDLE_THICKNESS, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (WIDTH - PADDLE_THICKNESS, game.right_y, PADDLE_THICKNESS, PADDLE_HEIGHT))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(game.ball_x), int(game.ball_y)), 10)

        self.screen.blit(self.font.render(str(game.left_score), True, (255, 255, 255)), (100, 60))
        self.screen.blit(self.font.render(str(game.right_score), True, (255, 255, 255)), (WIDTH - 120, 60))
        self.screen.blit(self.small_font.render(left_label, True, (150, 150, 150)), (20, HEIGHT - 30))
        label = self.small_font.render(right_label, True, (150, 150, 150))
        self.screen.blit(label, (WIDTH - 20 - label.get_width(), HEIGHT - 30))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
