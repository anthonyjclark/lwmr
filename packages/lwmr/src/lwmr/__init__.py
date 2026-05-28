from gymnasium.envs.registration import register

from .envs.plane import LwmrPlaneEnv
from .robot import LwmrRobotConfig, add_lwmr_robot
from .utils import control_sequence_generator

# TODO: review environment names
register(
    id="lwmr/Lwmr-v0",
    entry_point="lwmr.envs:LwmrPlaneEnv",
)

__all__ = ["LwmrPlaneEnv", "add_lwmr_robot", "LwmrRobotConfig", "control_sequence_generator"]
