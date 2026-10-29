# Attribution and provenance

Prepared by **Jianglin Yan**, 2026-10-28.

## Implementation source

The core implementation was copied from the independent reproduction project [`CuboiLeo/SuperMimic-Reproduced`](https://github.com/CuboiLeo/SuperMimic-Reproduced), pinned locally at commit `ba6105349069bf040f1fdad60873d13138d505a2` (2026-10-26). It is not official code from the SuperMimic paper authors. The source repository identifies **Yuxi Liu** as the copyright holder and licenses its code under MIT; the original `LICENSE` is retained unmodified.

This repository adds a local project wrapper and reproduction documentation. Its `docs/EXPERIMENTS.md` is inherited from the source project and records that project's reported experiments. Those records are not results produced by Jianglin Yan.

## Paper

Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, and K. Sreenath, “SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model,” CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480). This repository is an independent reproduction effort and is not affiliated with the paper authors.

## Third-party dependencies and datasets

mjlab, MuJoCo, MuJoCo Warp, PyTorch, Warp, and other dependencies retain their own licenses; see `pyproject.toml` and upstream project notices. LAFAN1 and BONES-SEED data, processed motion files, model checkpoints, and generated demos are not bundled here. Their separate licenses and terms apply. See [`docs/local/SOURCES_AND_DATA.md`](docs/local/SOURCES_AND_DATA.md).
