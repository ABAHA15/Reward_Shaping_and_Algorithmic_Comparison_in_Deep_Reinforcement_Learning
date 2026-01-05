import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PacmanGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.grid_size = 7
        self.max_steps = 200

        self.start_pos = (0, 0)
        self.goal_pos = (6, 6)

        self.obstacles = {(2,2), (2,3), (3,2), (4,4), (1,5)}

        self.action_space = spaces.Discrete(4)

        # agent_x, agent_y, ghost_x, ghost_y, goal_x, goal_y
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(6,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.ghost_pos = [random.randint(3,6), random.randint(0,3)]
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.ghost_pos[0], self.ghost_pos[1],
            self.goal_pos[0], self.goal_pos[1]
        ], dtype=np.float32)

    def _move(self, pos, action):
        new = pos.copy()
        if action == 0: new[0] -= 1
        elif action == 1: new[0] += 1
        elif action == 2: new[1] -= 1
        elif action == 3: new[1] += 1

        if (0 <= new[0] < self.grid_size and
            0 <= new[1] < self.grid_size and
            tuple(new) not in self.obstacles):
            return new
        return pos

    def step(self, action):
        self.steps += 1

        # Agent move
        self.agent_pos = self._move(self.agent_pos, action)

        # Ghost move (stochastic)
        ghost_action = random.choice([0,1,2,3])
        self.ghost_pos = self._move(self.ghost_pos, ghost_action)

        reward = -0.05
        terminated = False
        truncated = False

        # Distance shaping
        dist_before = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        dist_after = dist_before
        reward += 0.03 * (dist_before - dist_after)

        # Ghost collision
        if self.agent_pos == self.ghost_pos:
            reward = -10
            terminated = True

        # Goal reached
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 15
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
