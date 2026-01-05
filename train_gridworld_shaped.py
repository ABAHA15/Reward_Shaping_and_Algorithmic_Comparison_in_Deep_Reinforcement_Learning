from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from gridworld_shaped_env import GridWorldShapedEnv

# Create environment
env = GridWorldShapedEnv()
env = Monitor(env)

# DQN model (same settings)
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
model.save("dqn_gridworld_shaped")

# Plot rewards
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("GridWorld – Shaped Reward")
plt.show()

env.close()
