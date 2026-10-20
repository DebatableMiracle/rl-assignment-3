"""--videos : one representative evaluation video per method (seed 0) from the SAVED final policy (no retraining).
The stochastic policy is rolled out exactly like in evaluation (sample u, clip for the env).  Needs a MuJoCo renderer:
on a headless box set  MUJOCO_GL=egl  (or osmesa)  before running."""
import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from policies import make_policy
from train import ENV_ID, SIGMA, inspect_env


def make_videos(runs_dir, outdir, seed=0, episode_seed=12345):
    from train import METHODS
    os.makedirs(outdir, exist_ok=True)
    info = inspect_env()
    low, high = info["action_low"][0], info["action_high"][0]
    for m in METHODS:
        path = os.path.join(runs_dir, f"{m}_seed{seed}_params.pt")
        if not os.path.exists(path):
            print(f"skip {m}: {path} not found"); continue
        ck = torch.load(path, map_location="cpu")
        policy = make_policy(ck["kind"], seed, info["obs_dim"], info["act_dim"])
        policy.load_state_dict(ck["state_dict"])
        env = gym.make(ENV_ID, render_mode="rgb_array")
        gen = torch.Generator().manual_seed(777)
        obs, _ = env.reset(seed=episode_seed)
        frames, ret, done = [env.render()], 0.0, False
        while not done:
            with torch.no_grad():
                u = policy.mean(obs[None]) + SIGMA * torch.randn((1, info["act_dim"]), generator=gen)
            obs, r, te, tr, _ = env.step(u.clamp(low, high).numpy()[0])
            ret += r; done = te or tr; frames.append(env.render())
        fps = env.metadata.get("render_fps", 25); env.close()
        base = os.path.join(outdir, f"{m}_seed{seed}")
        try:
            imageio.mimsave(base + ".mp4", frames, fps=fps); out = base + ".mp4"
        except Exception as e:                                   # no ffmpeg -> gif fallback
            imageio.mimsave(base + ".gif", frames[::2], fps=fps // 2); out = base + ".gif"
            print("  (mp4 failed:", e, ") wrote gif instead")
        print(f"{m}: return {ret:.0f} over {len(frames) - 1} steps -> {out}")
