from collections.abc import Generator
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    """Example using Lwmr environment."""

    quiet: bool = False
    seed: int = 47
    steps: int = 80
    device: str = "cuda"
    max_steps: int | None = None


MotorCommand = tuple[float, float, float, float]


def looping_control_sequence_generator(
    lo: float, mid: float, hi: float, steps_per_command: int | list[int] = 40
) -> Generator[MotorCommand, None, None]:

    if isinstance(steps_per_command, int):
        steps_per_command = [steps_per_command] * 5

    control_sequence = [
        (steps_per_command[0], (hi, hi, hi, hi)),  # forward
        (steps_per_command[1], (mid, hi, mid, hi)),  # veer left
        (steps_per_command[2], (hi, hi, hi, hi)),  # forward
        (steps_per_command[3], (hi, mid, hi, mid)),  # veer right
        (steps_per_command[4], (lo, lo, lo, lo)),  # stop
    ]

    control_iter = iter(control_sequence)
    current_control_steps, current_control = next(control_iter)
    while True:
        yield current_control

        current_control_steps -= 1
        if current_control_steps <= 0:
            try:
                current_control_steps, current_control = next(control_iter)
            except StopIteration:
                control_iter = iter(control_sequence)
                current_control_steps, current_control = next(control_iter)


def main(args: Args) -> None:
    # NOTE: moving imports here to better handle quiet mode and speed up imports on help
    import gymnasium as gym
    import lwmr  # noqa: F401
    import warp as wp
    from tqdm.auto import trange

    if args.quiet:
        wp.config.quiet = True

    env = gym.make(
        "lwmr/Lwmr-v0",
        render_mode="viser",
        max_episode_steps=args.max_steps,
        quiet=args.quiet,
        device=args.device,
    )

    control_sequence = looping_control_sequence_generator(lo=0.0, mid=4.0, hi=12.0)

    observation, info = env.reset(seed=args.seed)
    # TODO: might still need to limit total number of steps (how is max_episode_steps handled in vectorised envs? or single environments)
    for step in trange(args.steps):
        action = next(control_sequence)

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
