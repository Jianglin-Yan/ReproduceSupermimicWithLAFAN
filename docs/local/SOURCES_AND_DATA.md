# Data sources and preparation

Checked 2026-10-27. This note describes datasets needed to run the included pipeline; no dataset files are bundled in this repository.

## LAFAN1 G1 retargets

- Dataset page: <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>
- The first RTX 4090 run used the G1-retargeted subset: 40 CSV clips at 30 fps, totaling approximately 2.45 hours.
- Respect the dataset's current license and access terms. Do not redistribute raw or converted data unless the terms allow it.
- Convert and inspect the motion data using `scripts/data/csv_to_npz_batch.py`; verify units, joint order, root frame, quaternion convention, frame rate, and clip lengths before training.

## BONES-SEED G1 trajectories

- Dataset page: <https://huggingface.co/datasets/bones-studio/seed>
- The dataset includes 142,220 motions; the target paper filters the G1 portion to 129,514 clips / 264.56 hours.
- The dataset has separate terms. Review and accept the current terms before downloading. Keep raw and derived data private unless redistribution is explicitly allowed.
- Preserve an exact manifest of included and excluded files, filter reasons, frame counts, and total duration.

## Processing sequence

1. Download only the robot subset needed for the current experiment and record the revision and checksums.
2. Preserve raw data separately from converted output.
3. Convert to the control frequency and coordinate convention expected by the environment.
4. Check joint ranges, missing or non-finite frames, and representative clips in simulation.
5. Record the exact data manifest and preprocessing command alongside each run.

## Disk planning for BONES-SEED

The G1 source archive was listed at about 23.5 GB. At the target 264.56 hours and 50 Hz, the converted uncompressed NPZ arrays are approximately 85 GB, before per-file overhead; the compact library is about 6.7 GB. These are observed/calculated sizes and not a guaranteed peak-disk figure.

With 160 GB free, processing the complete corpus requires managing intermediate files. Retaining the archive, all extracted CSVs, all converted NPZs, and the compact library at once can exhaust that space. Reserve roughly 200 GB or more, or use staged extraction/conversion and verify each converted batch before removing its raw temporary files. Retain headroom for the environment, caches, checkpoints, and logs.

<!-- Maintainer: Jianglin Yan | 2026-10-28 -->
