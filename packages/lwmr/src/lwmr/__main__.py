"""
This file is mainly used to test the installation.
"""

import gymnasium as gym

import lwmr  # noqa: F401 <-- register the environment

env = gym.make("lwmr/Lwmr-v0", quiet=True)
env.reset()
env.step(env.action_space.sample())
env.close()

print("Lwmr package was able to reset, step, and close.")
