import pygame
import time
from stable_baselines3 import PPO

from gridworld_shaped_env import GridWorldShapedEnv

# Grid settings
GRID_SIZE = 5
CELL_SIZE = 90
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
BLUE = (60, 120, 255)
GREEN = (60, 200, 100)
RED = (220, 60, 60)
GRAY = (200, 200, 200)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("GridWorld – PPO Agent (Shaped Reward)")
clock = pygame.time.Clock()

env = GridWorldShapedEnv()
model = PPO.load("ppo_gridworld_shaped")

obs, _ = env.reset()
running = True

font = pygame.font.SysFont(None, 24)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # PPO policy: keep stochastic for visual interest
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)

    # Draw background
    screen.fill(WHITE)

    # Draw grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)

    # Obstacles
    for (x, y) in env.obstacles:
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)

    # Goal
    gx, gy = env.goal_pos
    pygame.draw.rect(
        screen, GREEN,
        pygame.Rect(gy * CELL_SIZE, gx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Agent
    ax, ay = env.agent_pos
    pygame.draw.rect(
        screen, BLUE,
        pygame.Rect(ay * CELL_SIZE, ax * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Status text
    text = font.render(f"Reward: {reward:.2f}", True, BLACK)
    screen.blit(text, (5, 5))

    pygame.display.flip()
    clock.tick(6)  # slower = easier to watch
    time.sleep(0.08)

    if terminated or truncated:
        time.sleep(0.4)
        obs, _ = env.reset()

pygame.quit()
