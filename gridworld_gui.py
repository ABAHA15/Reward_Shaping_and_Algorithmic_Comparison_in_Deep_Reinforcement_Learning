import pygame
import time
from stable_baselines3 import DQN

from gridworld_shaped_env import GridWorldShapedEnv

# Grid settings
GRID_SIZE = 5
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
GREEN = (50, 200, 50)
RED = (200, 50, 50)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("GridWorld – Shaped Reward Agent")
clock = pygame.time.Clock()

# Load environment and trained model
env = GridWorldShapedEnv()
model = DQN.load("dqn_gridworld_shaped")

obs, _ = env.reset()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Agent action
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)

    # Draw grid
    screen.fill(WHITE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Draw obstacles
    for (x, y) in env.obstacles:
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)

    # Draw goal
    gx, gy = env.goal_pos
    goal_rect = pygame.Rect(gy * CELL_SIZE, gx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, goal_rect)

    # Draw agent
    ax, ay = env.agent_pos
    agent_rect = pygame.Rect(ay * CELL_SIZE, ax * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, BLUE, agent_rect)

    pygame.display.flip()
    clock.tick(5)
    time.sleep(0.1)

    if terminated or truncated:
        obs, _ = env.reset()

pygame.quit()
