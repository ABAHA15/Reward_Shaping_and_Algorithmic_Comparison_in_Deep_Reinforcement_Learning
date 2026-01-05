import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Grid settings
        self.grid_size = 5
        self.max_steps = 100

        # Fixed positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = {(1, 1), (2, 2), (3, 1)}

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation: agent_x, agent_y, goal_x, goal_y
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1],
             self.goal_pos[0], self.goal_pos[1]],
            dtype=np.float32
        )

    def step(self, action):
        self.steps += 1

        # Compute new position
        new_pos = self.agent_pos.copy()

        if action == 0:   # up
            new_pos[0] -= 1
        elif action == 1: # down
            new_pos[0] += 1
        elif action == 2: # left
            new_pos[1] -= 1
        elif action == 3: # right
            new_pos[1] += 1

        # Check boundaries
        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size):
            self.agent_pos = new_pos

        reward = 0.0
        terminated = False
        truncated = False

        # Check obstacle
        if tuple(self.agent_pos) in self.obstacles:
            reward = -1.0

        # Check goal
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10.0
            terminated = True

        # Step limit
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
