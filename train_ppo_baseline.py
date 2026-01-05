import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Environment
env = gym.make("CartPole-v1")
env = Monitor(env)

# PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=50000)

# Save
model.save("ppo_cartpole_baseline")

# Plot
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("PPO – Baseline Reward (CartPole)")
plt.show()

env.close()
