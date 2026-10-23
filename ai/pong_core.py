"""Headless Pong game logic, decoupled from any rendering.

This is a direct port of the rules in game.html (same canvas size, speeds and
paddle sizes), so an agent trained here behaves the same in the browser.
No pygame import here: the core can run millions of steps without a window.
"""
import numpy as np

WIDTH = 800
HEIGHT = 600
PADDLE_HEIGHT = 100
PADDLE_THICKNESS = 10
BALL_SPEED_X = 10
BALL_SPEED_Y = 4
SPIN_FACTOR = 0.35        # how much the hit position on the paddle angles the ball
MAX_BALL_SPEED_Y = PADDLE_HEIGHT / 2 * SPIN_FACTOR
AGENT_PADDLE_SPEED = 12   # pixels the agent paddle moves per step
CPU_PADDLE_SPEED = 6      # same as computerMovement() in game.html
CPU_DEADZONE = 35

UP, DOWN, STAY = 0, 1, 2


class PongGame:
    """Left paddle = agent (or human), right paddle = hardcoded CPU.

    step() advances one frame and returns a dict of events so callers
    (the Gym env, the Pygame viewer) can react without re-deriving them.
    """

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.left_score = 0
        self.right_score = 0
        self.left_y = (HEIGHT - PADDLE_HEIGHT) / 2
        self.right_y = (HEIGHT - PADDLE_HEIGHT) / 2
        self.serve(direction=self.rng.choice([-1, 1]))

    def serve(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = self.rng.uniform(HEIGHT * 0.25, HEIGHT * 0.75)
        self.ball_vx = BALL_SPEED_X * direction
        self.ball_vy = BALL_SPEED_Y * self.rng.choice([-1, 1])

    def move_left_paddle(self, action):
        if action == UP:
            self.left_y -= AGENT_PADDLE_SPEED
        elif action == DOWN:
            self.left_y += AGENT_PADDLE_SPEED
        self.left_y = float(np.clip(self.left_y, 0, HEIGHT - PADDLE_HEIGHT))

    def cpu_move_right_paddle(self):
        center = self.right_y + PADDLE_HEIGHT / 2
        if center < self.ball_y - CPU_DEADZONE:
            self.right_y += CPU_PADDLE_SPEED
        elif center > self.ball_y + CPU_DEADZONE:
            self.right_y -= CPU_PADDLE_SPEED
        self.right_y = float(np.clip(self.right_y, 0, HEIGHT - PADDLE_HEIGHT))

    def _bounce_off(self, paddle_y):
        self.ball_vx = -self.ball_vx
        delta = self.ball_y - (paddle_y + PADDLE_HEIGHT / 2)
        self.ball_vy = delta * SPIN_FACTOR

    def step(self, left_action, right_action=None):
        """Advance one frame. right_action=None means the CPU drives the right paddle."""
        events = {"left_hit": False, "right_hit": False, "left_scored": False, "right_scored": False}

        self.move_left_paddle(left_action)
        if right_action is None:
            self.cpu_move_right_paddle()
        else:
            self.right_y += {UP: -AGENT_PADDLE_SPEED, DOWN: AGENT_PADDLE_SPEED}.get(right_action, 0)
            self.right_y = float(np.clip(self.right_y, 0, HEIGHT - PADDLE_HEIGHT))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x < 0:
            if self.left_y < self.ball_y < self.left_y + PADDLE_HEIGHT:
                self._bounce_off(self.left_y)
                self.ball_x = 0
                events["left_hit"] = True
            else:
                self.right_score += 1
                events["right_scored"] = True
                self.serve(direction=1)
        elif self.ball_x > WIDTH:
            if self.right_y < self.ball_y < self.right_y + PADDLE_HEIGHT:
                self._bounce_off(self.right_y)
                self.ball_x = WIDTH
                events["right_hit"] = True
            else:
                self.left_score += 1
                events["left_scored"] = True
                self.serve(direction=-1)

        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.ball_vy = -self.ball_vy
        elif self.ball_y > HEIGHT:
            self.ball_y = 2 * HEIGHT - self.ball_y
            self.ball_vy = -self.ball_vy

        return events

    def observation(self, side="left"):
        """Normalized [ball_x, ball_y, ball_vx, ball_vy, paddle_y] from one paddle's point of view.

        For the right paddle the x axis is mirrored, so one trained agent can play either side.
        """
        if side == "left":
            bx, vx, py = self.ball_x, self.ball_vx, self.left_y
        else:
            bx, vx, py = WIDTH - self.ball_x, -self.ball_vx, self.right_y
        return np.array([
            bx / WIDTH,
            self.ball_y / HEIGHT,
            vx / BALL_SPEED_X,
            self.ball_vy / MAX_BALL_SPEED_Y,
            py / (HEIGHT - PADDLE_HEIGHT),
        ], dtype=np.float32)
