# SuperMimic reproduction roadmap

> Planning snapshot assembled on 2026-10-27. It records the initial preparation plan and may not reflect the current 4090 environment or experiment status. Use the current README and merged experiment records for live status.

Goal: prepare this workspace locally, then perform training only on the separate Ubuntu machine with one 24 GB RTX 4090. This roadmap deliberately separates a functioning pipeline from a paper-faithful result.

## Stage 0 — freeze the target and the code

1. Use the local `2610.28480v1.pdf` as the target paper version; record its SHA-256 and keep the PDF unchanged.
2. Review `PAPER_SPEC.md` against Appendix A, D, E, and F in the PDF: observations, reward, termination, domain randomization, policy/world-model/critic, dataset filtering, and evaluator.
3. Inspect `code/SuperMimic-Reproduced/README.md`, `pyproject.toml`, scripts, and data pipeline. Pin a git commit (do not train from a moving `main`).
4. Record every discrepancy between the code and paper before running it. The most visible current gap is LAFAN1 paper-observation return, which the candidate reports below 0.82.
5. Keep a separate branch/config for any changes. Report them as modified reproduction, not paper-faithful reproduction.

**Exit condition:** an experiment sheet defines exact source commit, data revision, paper target, changed assumptions, metrics, and output paths.

## Stage 1 — inspect the 4090 host without changing its system

Before installing or downloading anything, collect:

```bash
hostname
lsb_release -a
nvidia-smi
df -h
free -h
python3 --version
command -v uv || true
```

Also record GPU model/VRAM, driver and CUDA compatibility, Ubuntu release, free disk, RAM, and whether another process is using the GPU. Keep the research/code workspace and all outputs under a dedicated project directory. Do not upgrade Ubuntu, the NVIDIA driver, system Python, or CUDA as an exploratory fix. Prefer the pinned project environment (`uv`/venv); any system package changes must be separately approved by the user.

**Decision gate:** if disk/RAM are tight, prepare LAFAN1 only first. Plan additional free disk for raw archives, processed motion files, cache, checkpoints, and logs; measure actual dataset size before a full download.

## Stage 2 — environment and import/simulation smoke check

1. Create the project's isolated Python environment using its pinned lock file and the documented CUDA-enabled PyTorch/Warp dependencies.
2. Check `torch.cuda.is_available()`, visible device name/VRAM, MuJoCo-Warp device selection, and a one-environment G1 simulation.
3. Run the repository's fast unit checks and FK/simulation consistency checks. Do not run the full suite unless required for the chosen commit; mark each check as pass/fail with logs.
4. Do not start a large training run until the simulator can step, reset, load one motion, and emit finite state/reward values.

**Exit condition:** pinned environment is reproducible, one GPU is selected, and a small simulator smoke check is clean.

## Stage 3 — LAFAN1 end-to-end pilot

1. Download the G1-retargeted LAFAN1 subset and produce a manifest/checksum.
2. Convert and visually/numerically inspect representative motions: standing, locomotion, turn, jump, and fall/get-up if present.
3. Run the paper-observation configuration at reduced scale first (e.g. 256–512 environments and proportional small batches), for a brief pilot only. Record peak VRAM, throughput, loss finiteness, and checkpoints.
4. Increase environment and batch sizes gradually (for example 512 → 1024 → 2048 → 4096) only while VRAM has headroom and rollouts remain stable. The exact safe ceiling is measured on this 4090; don't assume it from 5090/H100 reports.
5. Run the full LAFAN1 training budget and Appendix-F-style evaluator. Record normalized return, failures, wall time, seed, and compute details. Use at least three seeds if the goal includes a defensible scientific comparison.
6. Compare against the paper's LAFAN1 target (0.82 after 1.2 h on H100) and the selected implementation's current reported result. Investigate observation/reward/evaluator issues before adding extra inputs or changing the objective.

**Exit condition:** end-to-end LAFAN1 run and evaluator complete without hidden deviations; resource curve establishes the largest stable configuration.

## Stage 4 — BONES-SEED data and full-scale pilot

1. Review and accept the dataset license; verify remote free disk before download. Hugging Face currently lists the G1 archive as 23.5 GB; estimated converted uncompressed NPZ data is about 85 GB and the compact library about 6.7 GB. With only 160 GB free, do not extract the complete archive while also retaining all converted outputs without a storage plan.
2. Download the G1 archive and checksum it. Before extracting all files, choose either enough free disk to hold archive + extracted CSVs + converted outputs, or stage extraction/conversion and delete each raw batch after its converted outputs are verified. The candidate conversion script currently expects directory inputs, so staged processing may require a small wrapper or controlled batches.
3. Apply the paper's two-pass filtering (name filter and feasibility filter) as implemented/audited; save the exact manifest and rejection reasons.
4. Convert to the expected 50 Hz data format and verify final clip count, total duration, joint ranges, missing/NaN frames, and representative rollouts.
5. Build the compact motion library and measure peak RAM/VRAM. A candidate README reports a roughly 6.7 GB compact library on GPU; the 24 GB card has to fit this plus simulation, replay, and differentiable rollout activations. Treat full-data fit as plausible but unproven until measured.
6. Start a small full-data pilot, check GPU memory over several update cycles and checkpoint recovery, then scale. If it OOMs, reduce library residency/batch/environment sizes using the candidate's supported settings and document each change.

**Exit condition:** full filtered dataset is reproducibly built and a sustained training pilot fits memory without dropping clips silently.

## Stage 5 — full training and paper-protocol evaluation

1. Run the paper's 4096 environments, H=32, 4096 imagined rollouts, 1000-step/environment buffer, and other Table A.5 settings if the 4090 fits. Otherwise use the largest measured stable configuration and record it.
2. Save checkpoints and logs to local disk on the 4090 host; verify resume before committing to a long run.
3. Run Appendix F evaluation with deterministic policy, no observation noise or domain randomization, one complete rollout per clip from its first frame.
4. Report: normalized return, failure count, exact motion count/duration, code commit, configuration, seed, GPU model, wall time, peak VRAM, and any filter/evaluation changes.
5. Compare with paper targets: `>0.90` and 25/129,514 failures after 3 H100 hours; zero failures after 9.8 hours on 2 H100s. Do not equate raw wall time on 4090 with H100 results.

## Practical schedule and current priority

1. **Now, local preparation:** validate documents, audit/freeze code, write remote setup and data manifests. No system changes on the development machine.
2. **First 4090 session:** read-only hardware/storage inspection; then isolated environment and one-environment simulation check.
3. **First experiment:** LAFAN1 reduced-scale pilot, then full LAFAN1 paper-observation run.
4. **Only after that:** full BONES-SEED processing, VRAM pilot, and full benchmark run.

One RTX 4090 24 GB is a reasonable target for the pilot and may be sufficient for a tuned single-GPU full-data run. The paper's H100 timings and other implementations' multi-GPU results do not establish that the paper defaults will fit or match their speed on this card.

## Experiment record template

For every run record: run ID; date/time; repository URL and commit; paper version; config diff; dataset revision/license and manifest hash; Ubuntu/driver/CUDA/Python/torch/mjlab versions; GPU and peak VRAM; random seed; wall time; throughput; training/evaluation metrics; checkpoint path; and deviations or failures.

<!-- Project maintainer: Jianglin Yan; local copy assembled 2026-10-28. See ATTRIBUTION.md for upstream authorship and license. -->
