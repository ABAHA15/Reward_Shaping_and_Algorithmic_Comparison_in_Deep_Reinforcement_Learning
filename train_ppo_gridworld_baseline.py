from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from gridworld_env import GridWorldEnv

# Environment
env = GridWorldEnv()
env = Monitor(env)

# PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # encourages exploration
    verbose=1
)

# Train
model.learn(total_timesteps=40000)

# Save model
model.save("ppo_gridworld_baseline")

# Plot learning curve
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("PPO – GridWorld (Baseline Reward)")
plt.show()

env.close()
