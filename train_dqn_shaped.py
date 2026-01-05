import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from cartpole_shaped_env import RewardShapedCartPole

# Create environment
base_env = gym.make("CartPole-v1")
env = RewardShapedCartPole(base_env)
env = Monitor(env)

# Initialize DQN model (same hyperparameters as baseline)
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

# Train model
model.learn(total_timesteps=50000)

# Save model
model.save("dqn_cartpole_shaped")

# Plot rewards
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Shaped Reward Training")
plt.show()

env.close()
