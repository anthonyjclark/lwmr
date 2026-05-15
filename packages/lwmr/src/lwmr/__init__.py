from gymnasium.envs.registration import register

from .envs.plane import LwmrPlaneEnv
from .robot import add_lwmr_robot

register(
    id="lwmr/Lwmr-v0",
    entry_point="lwmr.envs:LwmrPlaneEnv",
)

__all__ = ["LwmrPlaneEnv", "add_lwmr_robot"]
