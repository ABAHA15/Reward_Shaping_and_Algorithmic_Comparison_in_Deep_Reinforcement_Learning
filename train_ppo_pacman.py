from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from pacman_env import PacmanGridEnv

env = PacmanGridEnv()
env = Monitor(env)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,   # key for varied behavior
    verbose=1
)

model.learn(total_timesteps=80000)

model.save("ppo_pacman")

rewards = env.get_episode_rewards()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("PPO – Pacman Style GridWorld")
plt.show()

env.close()
