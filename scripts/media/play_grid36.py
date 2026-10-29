"""Open a trained policy in a Viser viewer with multiple vectorized environments."""

from __future__ import annotations

from pathlib import Path

import torch
import tyro
import viser
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


class _DictPolicy:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 36,
    device: str = "cuda:0",
    port: int = 8081,
    seed: int = 0,
) -> None:
    files = [str(path) for path in discover_npz(npz_dir, names=clips)]
    if not files:
        raise ValueError(f"No motion NPZ files found in {npz_dir}")

    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=seed,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=True,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    server = viser.ViserServer(port=port, label=f"SuperMimic {num_envs} envs")
    print(
        f"[INFO] {len(files)} motion clips, {num_envs} parallel environments; "
        f"viewer: http://127.0.0.1:{port}",
        flush=True,
    )
    try:
        ViserPlayViewer(env, _DictPolicy(policy), viser_server=server).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

# Maintainer: Jianglin Yan | 2026-10-28
