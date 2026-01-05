import pygame
import time
from stable_baselines3 import PPO
from pacman_env import PacmanGridEnv

GRID = 7
CELL = 80
SIZE = GRID * CELL

WHITE = (240,240,240)
BLACK = (0,0,0)
BLUE = (70,120,255)
GREEN = (80,220,120)
RED = (220,70,70)
YELLOW = (255,200,50)
GRAY = (160,160,160)

pygame.init()
screen = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption("Pacman-Style RL Demo (PPO)")
clock = pygame.time.Clock()

env = PacmanGridEnv()
model = PPO.load("ppo_pacman")

obs, _ = env.reset()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)

    screen.fill(WHITE)

    # Grid
    for i in range(GRID):
        for j in range(GRID):
            pygame.draw.rect(screen, GRAY,
                pygame.Rect(j*CELL, i*CELL, CELL, CELL), 1)

    # Obstacles
    for (x,y) in env.obstacles:
        pygame.draw.rect(screen, BLACK,
            pygame.Rect(y*CELL, x*CELL, CELL, CELL))

    # Goal
    gx, gy = env.goal_pos
    pygame.draw.circle(screen, GREEN,
        (gy*CELL+CELL//2, gx*CELL+CELL//2), CELL//3)

    # Ghost
    gx, gy = env.ghost_pos
    pygame.draw.circle(screen, RED,
        (gy*CELL+CELL//2, gx*CELL+CELL//2), CELL//3)

    # Agent (Pacman)
    ax, ay = env.agent_pos
    pygame.draw.circle(screen, YELLOW,
        (ay*CELL+CELL//2, ax*CELL+CELL//2), CELL//3)

    pygame.display.flip()
    clock.tick(7)
    time.sleep(0.08)

    if terminated or truncated:
        time.sleep(0.6)
        obs, _ = env.reset()

pygame.quit()
