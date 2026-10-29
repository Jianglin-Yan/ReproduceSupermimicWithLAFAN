# Remote RTX 4090 host: read-only baseline

> Historical snapshot only. It predates later environment setup and training activity; it does not describe the machine's current state.

Inspected over SSH on 2026-10-27 at the user's request. This records machine state only; no files, packages, services, or processes were changed. Credentials and network address are intentionally not recorded.

## Hardware and operating system

- Ubuntu 22.04.5 LTS, kernel `6.8.0-138-generic`.
- Intel Core i9-14900K, 24 cores / 32 logical CPUs.
- 62 GiB RAM; `free -h` showed 37 GiB available at inspection time.
- NVIDIA GeForce RTX 4090, 24,564 MiB VRAM; NVIDIA driver 580.178.04; `nvidia-smi` reports CUDA 13.0 compatibility.
- NVIDIA GPU utilization was 15% at inspection time. About 12,459 MiB VRAM was occupied, leaving about 12.1 GiB free.
- Root filesystem: 938 GiB total, 780 GiB used, 110 GiB available (88% full). No separate data mount appeared in `df -hT`.

## Existing activity and software

- Existing processes from other projects were using the GPU (one Python process had about 11.4 GiB VRAM and 100% CPU; another interactive process had about 0.5 GiB VRAM). They were left untouched. Do not start a competing run until the owner confirms the GPU is available.
- System Python is 3.10.12 and has no `pip` module; `torch` is not installed there. `uv` and `nvcc` were not found on the inspected shell's PATH.
- Conda is installed with multiple existing environments. No SuperMimic project environment was established during this read-only inspection.
- This project requires Python 3.12 and uses pinned dependencies. Use a project-local environment; do not replace system Python or alter CUDA/driver packages.

## Readiness assessment

- **Hardware:** suitable for a single-GPU LAFAN1 run and likely useful for full-data training after a memory pilot.
- **Immediate blocker:** current GPU jobs leave less than half of the 4090 VRAM free. Nothing was stopped.
- **Disk:** 110 GiB free is enough to begin with LAFAN1 and prepare an isolated environment, but is tight for full BONES-SEED plus conversion intermediates and software caches. First run LAFAN1; before the full corpus, arrange more space or implement staged extraction/conversion with verified cleanup.
- **Next step:** after the user confirms that the existing GPU jobs may be paused or have completed, create the SuperMimic environment inside the project directory and perform a small simulation/training smoke run. No system-level changes are planned.

<!-- Maintainer: Jianglin Yan | 2026-10-28 -->
