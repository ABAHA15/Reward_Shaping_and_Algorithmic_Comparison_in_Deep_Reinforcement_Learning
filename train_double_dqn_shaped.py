import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from cartpole_shaped_env import RewardShapedCartPole

# Environment
base_env = gym.make("CartPole-v1")
env = RewardShapedCartPole(base_env)
env = Monitor(env)

# Double DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=50000)

# Save
model.save("double_dqn_cartpole_shaped")

# Plot
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Double DQN – Shaped Reward (CartPole)")
plt.show()

env.close()
