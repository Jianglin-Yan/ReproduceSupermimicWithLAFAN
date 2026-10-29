# SuperMimic reproduction with LAFAN1

**Maintainer: Jianglin Yan · 2026-10-28**

This workspace contains a runnable SuperMimic reproduction pipeline for G1 motion tracking: data conversion, a differentiable dynamics model, policy training through imagined rollouts, evaluation, and visualization. It is intended for internal research and learning.

The repository contains source and selected notes only. Datasets, converted motion files, checkpoints, training logs, caches, PDFs, and generated videos are excluded. Dataset terms apply to any data obtained separately. Consolidated source and license notices are in [ATTRIBUTION.md](ATTRIBUTION.md).

## Current 4090 run

The first full-scale LAFAN1 run on the remote RTX 4090 completed its three-hour budget. It is a working end-to-end run, but not a full reproduction of the paper's reported score.

| Item | Result |
|---|---|
| Data | 40 G1-retargeted LAFAN1 clips, 2.45 hours, converted from 30 to 50 fps |
| Training | 4096 environments; 17,650 iterations; 3.01 hours; about 1.7 iterations/s |
| Final training return | about 0.670 |
| Full-clip evaluation | mean normalized return 0.5669; step-weighted 0.5709; 6 failed clips |
| Paper LAFAN1 reference | 0.82 on one H100 after 1.2 hours |

The exact launch line was not preserved in shell history. The run configuration and logs were inspected, and an equivalent command plus metric details are recorded in [the run report](docs/local/RUN_LAFAN1_20261027.md). Treat the score gap as an open reproduction issue.

## Environment

The project is locked to Python 3.12 and uses `uv.lock` for pinned dependencies. On the Ubuntu 22.04 RTX 4090 host, the project environment is isolated from system Python and CUDA packages.

```bash
uv sync --locked --no-dev
```

If `PYTHONPATH` points at another Python installation, prefix project commands with `env -u PYTHONPATH`.

## Prepare data

LAFAN1 and BONES-SEED data are not included. Check the source terms before downloading or redistributing either dataset. The data preparation commands are:

```bash
uv run python scripts/data/csv_to_npz_batch.py --help
uv run python scripts/data/filter_bones_seed.py --help
uv run python scripts/data/build_library.py --help
```

See [data sources and preparation notes](docs/local/SOURCES_AND_DATA.md).

## Train and evaluate

```bash
env -u PYTHONPATH uv run sm-train \
  --run-name lafan1 \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 \
  --policy-trainer.batch-size 4096 \
  --schedule.max-hours 3

uv run python scripts/eval_benchmark.py --help
uv run python scripts/play.py --help
```

Training data and outputs go under ignored runtime directories. Do not add them to Git.

## View 36 environments

After training, the helper can open 36 vectorized environments in one Viser viewer. Select motion names with `--clips` or let the motion library sample from the supplied NPZ directory.

```bash
env -u PYTHONPATH uv run python scripts/media/play_grid36.py \
  --ckpt logs/lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --num-envs 36 \
  --port 8081
```

The viewer is a post-training policy replay, not a 36-environment training visualization. To create a montage from captured per-clip MP4 tiles, use `scripts/media/make_montage.py --help`.

For an offline MP4 montage from a chosen set of clips, first record rollouts, render one MP4 per clip, then tile the results. Replace the example clip names with the clips you want to show:

```bash
env -u PYTHONPATH uv run python scripts/media/record_rollout.py \
  --ckpt logs/lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --clips walk1_subject1 dance1_subject1 \
  --out media/rollouts.npz

MUJOCO_GL=egl env -u PYTHONPATH uv run python scripts/media/render_rollout.py \
  media/rollouts.npz media/tiles --fps 10 --seconds 12

uv run python scripts/media/make_montage.py \
  media/tiles media/supermimic_montage.mp4 --cols 6 --duration 12 --fps 10
```

This creates a grid MP4 from separately rendered deterministic policy rollouts. These media and trajectory files are ignored by Git.

## Project map

```text
src/supermimic/       environment, data pipeline, FK, world model, policy, training
scripts/              data preparation, training helpers, evaluation, replay, rendering
scripts/media/        rollout rendering and multi-environment viewer utilities
tests/                focused numerical and environment checks
docs/                 method notes and experiment records
docs/local/           local hardware, data, run, and reproduction notes
```

## Reproduction notes

- [Paper settings and open gaps](docs/local/PAPER_SPEC.md)
- [Run record from the RTX 4090](docs/local/RUN_LAFAN1_20261027.md)
- [Reproduction roadmap](docs/local/REPRODUCTION_ROADMAP.md)
- [Data sources and preparation](docs/local/SOURCES_AND_DATA.md)
- [Initial RTX 4090 environment snapshot](docs/local/REMOTE_4090_BASELINE_2026-10-27.md)
- [Historical experiment log](docs/EXPERIMENTS.md)

<!-- Maintainer: Jianglin Yan | 2026-10-28 -->
