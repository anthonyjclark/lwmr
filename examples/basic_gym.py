import gymnasium as gym
import lwmr  # noqa: F401

SEED = 47
STEPS = 1000

env = gym.make("lwmr/Lwmr-v0", render_mode="human")

observation, info = env.reset(seed=SEED)

for _ in range(STEPS):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
