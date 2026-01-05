from gridworld_env import GridWorldEnv

env = GridWorldEnv()
obs, _ = env.reset()
print("Initial observation:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print("Obs:", obs, "Reward:", reward)
    if terminated or truncated:
        break
