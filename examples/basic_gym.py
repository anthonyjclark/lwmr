from dataclasses import dataclass


def main(seed: int, steps: int, quiet: bool) -> None:
    # NOTE: moving imports here to better handle quiet mode and speed up imports on help
    import gymnasium as gym
    import lwmr  # noqa: F401

    env = gym.make("lwmr/Lwmr-v0", render_mode="viser", max_episode_steps=steps, quiet=quiet)
    # TODO: consider vectorised environments
    # vec_env = gym.make_vec(..., n_envs=...)

    observation, info = env.reset(seed=seed)
    # TODO: might still need to limit total number of steps (how is max_episode_steps handled in vectorised envs? or single environments)
    for step in tqdm.trange(steps):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation[:3])
        env.render()

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()


@dataclass
class Args:
    """Example using Lwmr environment."""

    quiet: bool = False
    seed: int = 47
    steps: int = 80


if __name__ == "__main__":
    import tqdm
    import tyro
    import warp as wp

    args = tyro.cli(Args)

    if args.quiet:
        wp.config.quiet = True

    main(seed=args.seed, steps=args.steps, quiet=args.quiet)
