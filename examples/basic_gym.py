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


def main(args: Args) -> None:
    # NOTE: moving imports here to better handle quiet mode and speed up imports on help
    import gymnasium as gym
    import lwmr  # noqa: F401
    import warp as wp
    from tqdm.auto import tqdm, trange

    if args.quiet:
        wp.config.quiet = True

    env = gym.make(
        "lwmr/Lwmr-v0",
        render_mode="viser",
        max_episode_steps=args.max_steps,
        quiet=args.quiet,
        device=args.device,
    )

    # TODO: consider vectorised environments
    # vec_env = gym.make_vec(..., n_envs=...)

    hih = 12.0
    mid = 4.0
    off = 0.0
    control_sequence = [
        (40, [hih, hih, hih, hih]),  # forward
        (40, [mid, hih, mid, hih]),  # veer left
        (40, [hih, hih, hih, hih]),  # forward
        (40, [hih, mid, hih, mid]),  # veer right
        (40, [off, off, off, off]),  # stop
    ]
    control_iter = iter(control_sequence)
    current_control_steps, current_control = next(control_iter)
    # tqdm.write(f"==>> Switching to next control: {current_control} for {current_control_steps} steps")

    observation, info = env.reset(seed=args.seed)
    # TODO: might still need to limit total number of steps (how is max_episode_steps handled in vectorised envs? or single environments)
    for step in trange(args.steps):
        # action = env.action_space.sample()
        action = current_control

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            observation, info = env.reset()
            break

        current_control_steps -= 1
        if current_control_steps <= 0:
            try:
                current_control_steps, current_control = next(control_iter)
            except StopIteration:
                current_control_steps = 20

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
