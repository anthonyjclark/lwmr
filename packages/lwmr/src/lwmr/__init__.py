from gymnasium.envs.registration import register

from .envs.plane import LwmrPlaneEnv
from .robot import LwmrRobot

register(
    id="lwmr/Lwmr-v0",
    entry_point="lwmr.envs:LwmrPlaneEnv",
)

__all__ = ["LwmrPlaneEnv", "LwmrRobot"]
