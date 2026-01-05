import gymnasium as gym
import numpy as np

class RewardShapedCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cart_pos, cart_vel, pole_angle, pole_vel = obs

        # Reward shaping terms
        shaped_reward = reward
        shaped_reward -= 0.05 * abs(pole_angle)
        shaped_reward -= 0.01 * abs(cart_pos)

        return obs, shaped_reward, terminated, truncated, info
