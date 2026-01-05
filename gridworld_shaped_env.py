import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldShapedEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.grid_size = 5
        self.max_steps = 100

        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = {(1, 1), (2, 2), (3, 1)}

        self.action_space = spaces.Discrete(4)
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
        self.prev_distance = self._distance_to_goal()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1],
             self.goal_pos[0], self.goal_pos[1]],
            dtype=np.float32
        )

    def _distance_to_goal(self):
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

    def step(self, action):
        self.steps += 1

        new_pos = self.agent_pos.copy()

        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] -= 1
        elif action == 3:
            new_pos[1] += 1

        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size):
            self.agent_pos = new_pos

        reward = -0.1  # step penalty
        terminated = False
        truncated = False

        # Obstacle penalty
        if tuple(self.agent_pos) in self.obstacles:
            reward -= 1.0

        # Distance-based shaping
        current_distance = self._distance_to_goal()
        reward += 0.05 * (self.prev_distance - current_distance)
        self.prev_distance = current_distance

        # Goal reward
        if tuple(self.agent_pos) == self.goal_pos:
            reward += 10.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
