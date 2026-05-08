from typing import Any

import gymnasium as gym
from gymnasium.core import RenderFrame

from lwmr.robot import LwmrRobot

# TODO: figure out correct observation type
ObsType = float  # gym.spaces.Space
InfoType = dict[str, Any]
OptType = dict[str, Any]


class LwmrPlaneEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.lwmr = LwmrRobot()

    def step(self, action) -> tuple[ObsType, float, bool, bool, InfoType]: ...

    def reset(
        self, *, seed: int | None = None, options: OptType | None = None
    ) -> tuple[ObsType, InfoType]: ...

    def render(self, mode="human") -> RenderFrame | list[RenderFrame] | None: ...
