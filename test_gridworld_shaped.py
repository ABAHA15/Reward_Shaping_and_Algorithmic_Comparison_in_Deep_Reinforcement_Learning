from gridworld_shaped_env import GridWorldShapedEnv

env = GridWorldShapedEnv()
obs, _ = env.reset()
print("Initial obs:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print("Obs:", obs, "Reward:", round(reward, 2))
    if terminated or truncated:
        break
