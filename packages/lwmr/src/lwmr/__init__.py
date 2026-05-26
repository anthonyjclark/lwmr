from gymnasium.envs.registration import register

from .envs.plane import LwmrPlaneEnv
from .robot import LwmrRobotConfig, add_lwmr_robot

# TODO: review environment names
register(
    id="lwmr/Lwmr-v0",
    entry_point="lwmr.envs:LwmrPlaneEnv",
)

__all__ = ["LwmrPlaneEnv", "add_lwmr_robot", "LwmrRobotConfig"]
