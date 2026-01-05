import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# Environment
env = gym.make("CartPole-v1")

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99
num_episodes = 1000
episode_rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        obs_tensor = torch.FloatTensor(obs)
        probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_rewards.append(sum(rewards))

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {episode_rewards[-1]}")

env.close()

plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("REINFORCE – Baseline Reward (CartPole)")
plt.show()
