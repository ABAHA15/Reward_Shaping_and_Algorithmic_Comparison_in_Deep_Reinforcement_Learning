import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Environment
env = gym.make("CartPole-v1")
env = Monitor(env)

# A2C model
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=50000)

# Save
model.save("a2c_cartpole_baseline")

# Plot
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("A2C – Baseline Reward (CartPole)")
plt.show()

env.close()
