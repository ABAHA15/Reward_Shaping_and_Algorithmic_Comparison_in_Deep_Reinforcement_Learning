from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from gridworld_shaped_env import GridWorldShapedEnv

# Environment
env = GridWorldShapedEnv()
env = Monitor(env)

# PPO model (slightly more exploratory)
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,   # MORE exploration = nicer demo
    verbose=1
)

# Train
model.learn(total_timesteps=40000)

# Save model
model.save("ppo_gridworld_shaped")

# Plot rewards
rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("PPO – GridWorld (Shaped Reward)")
plt.show()

env.close()
