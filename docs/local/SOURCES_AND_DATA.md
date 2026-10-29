# Code and data sources

Checked 2026-10-27. The paper authors' project page links the paper, video, and demo; no author-maintained training repository was found. Treat all code implementations below as independent reimplementations and audit before training.

## Implementation candidates

### Primary candidate: CuboiLeo/SuperMimic-Reproduced

<https://github.com/CuboiLeo/SuperMimic-Reproduced>

- Implements the training loop, world model, differentiable kinematics/reward terms, motion data pipeline, multi-clip sampling, and Appendix-F evaluator.
- Local shallow clone is pinned at commit `ba6105349069bf040f1fdad60873d13138d505a2` (2026-10-26); `uv.lock` is included. Preserve this commit for the first reproduction run instead of following future `main` changes.
- README currently reports full BONES-SEED training on 4× RTX 5090, and a paper-observation LAFAN1 score below the paper's figure result. This makes it a useful code base, not proof that its implementation reproduces all paper numbers.
- Requirements in current `pyproject.toml` specify Python 3.12, a pinned `mjlab` git commit, MuJoCo/MuJoCo-Warp 3.16, PyTorch CUDA 13.0 index and Warp. Pin the exact source commit before dependency installation.
- The older notes in `.scratch/supermimic/repro_readme.md` are not authoritative for current repository results; compare them against the cloned repository's current README, source, and commit.

### Secondary comparison: Shimingyi/allmimic

<https://github.com/Shimingyi/allmimic>

- Reports two RTX 4090s training/evaluating the paper's 306 difficult-motion subset.
- It explicitly changes action parameterization, policy latent size, exploration-data mixture, and world-model loss weighting. Use it as a practical 4090 comparison or debugging reference, not as the paper-faithful baseline.

### Not preferred as baseline: zita-ch/supermimic_vibed

<https://github.com/zita-ch/supermimic_vibed>

- Reports quick LAFAN1 results, but describes itself as a fast, unreviewed implementation and uses a different Isaac Lab-based path. It may inform troubleshooting, but should not define the paper baseline.

## Datasets

### LAFAN1 G1 retargets

- Dataset: <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>
- The implementation's data scripts determine exact file selection, conversion, resampling, and train/eval split. Audit those before downloading/processing.
- Observe the dataset's stated license and any restrictions on redistribution.

### BONES-SEED G1 trajectories

- Dataset: <https://huggingface.co/datasets/bones-studio/seed>
- Contains 142,220 motions, including Unitree G1 CSV trajectories; the paper filters this to 129,514 clips / 264.56 hours.
- The dataset uses a dedicated BONES-SEED license. Review and accept its current terms on Hugging Face before downloading. Keep raw data and derived motion files private unless the license explicitly permits redistribution.
- The paper's exact filtering is part of the experiment. Preserve a manifest of included/excluded filenames, filter reason, frame count, and total duration.

## Existing browser-demo assets

The previously collected files under `.scratch/supermimic/` (`worker.mjs`, `weights.bin`, `robot.bin`, HTML/JS) belong to the web demo/runtime. They are not the training source or a usable training checkpoint for the candidate code. The paper PDF in this workspace is SHA-256 `fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`.

## Data preparation rules

1. Download only the chosen robot representation and only after checking available disk space on the 4090 host.
2. Preserve the original archive and its checksum; write conversion outputs to a separate directory.
3. Convert to the control frequency and coordinate convention expected by the chosen implementation; verify units, joint order, root frame, quaternion convention, and clip length.
4. Record dataset revision/date, download command, license, preprocessing code commit, and filtered manifest.
5. Start with LAFAN1 before the full BONES-SEED data. Do not download the full corpus until the code audit and remote storage check are complete.

## Disk budget for the 4090 host

The BONES-SEED Hugging Face file listing currently reports `g1.tar.gz` at **23.5 GB** (gated access). The candidate converter writes uncompressed `np.savez` files containing joint position/velocity and 30-body positions, quaternions, and linear/angular velocities. At the paper's 264.56 h and 50 Hz, these arrays are approximately **85 GB** (`47.62M frames × 448 float32 values × 4 bytes`), before small per-file overhead. Its final compact fp16 library is approximately **6.7 GB**. These are calculated/observed implementation sizes, not a guaranteed peak-disk figure.

With 160 GB free, a full run is **possible only if intermediate copies are managed**. Keeping the compressed archive, all extracted CSVs, all converted NPZs, and the compact library simultaneously is likely to exceed 160 GB or leave unsafe headroom. The current converter expects CSV files in a directory and emits uncompressed NPZs, so before the full-data download, either reserve roughly 200+ GB free or implement and verify staged extraction/conversion that deletes each raw CSV batch after its converted outputs are confirmed. Keep enough headroom for the Python/CUDA environment, caches, checkpoints, logs, and temporary files. LAFAN1 is small and is suitable for the first launch on a 160 GB disk.

<!-- Project maintainer: Jianglin Yan; local copy assembled 2026-10-28. See ATTRIBUTION.md for upstream authorship and license. -->
