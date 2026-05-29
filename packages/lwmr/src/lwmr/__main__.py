"""
This file is mainly used to test the installation.
"""

from dataclasses import dataclass

import gymnasium as gym
import tyro

import lwmr  # noqa: F401 <-- register the environment


@dataclass
class Args:
    quiet: bool = False
    device: str = "cpu"


args = tyro.cli(Args)

env = gym.make("lwmr/Lwmr-v0", quiet=args.quiet, device=args.device)
env.reset()
env.step(env.action_space.sample())
env.close()

print("Lwmr package was able to reset, step, and close.")
