from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from gridworld_env import GridWorldEnv

# Create environment
env = GridWorldEnv()
env = Monitor(env)

# DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=5000,
    learning_starts=500,
    batch_size=32,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=30000)

# Save model
model.save("dqn_gridworld_sparse")

# Plot rewards
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("GridWorld – Sparse Reward")
plt.show()

env.close()
