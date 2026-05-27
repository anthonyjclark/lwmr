import gymnasium as gym
import lwmr  # noqa: F401  # registers "lwmr/Lwmr-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    print("Testing PPO on lwmr/Lwmr-v0...")

    env_kwargs = {"quiet": True, "render_mode": "none"}
    env = make_vec_env("lwmr/Lwmr-v0", n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # env = gym.make("lwmr/Lwmr-v0", quiet=True)
    model = PPO("MlpPolicy", env, n_steps=512, verbose=1, device="cpu")
    # model.learn(total_timesteps=16_000, progress_bar=True)
    # model.learn(total_timesteps=64_000, progress_bar=True)
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("ppo_cartpole")
    del model  # remove to demonstrate saving and loading

    env = gym.make("lwmr/Lwmr-v0", quiet=True, render_mode="viser")
    model = PPO.load("ppo_cartpole", device="cpu")

    obs = env.reset()
    obs = obs[0]
    # while True:
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        # action = [5.0] * 4
        obs, reward, dones, info, blah = env.step(action)
        env.render()
        print(reward, action)

    env.close()
