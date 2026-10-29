# Attribution and licensing

This file consolidates the source and license notices for this internal reproduction workspace. Prepared by **Jianglin Yan** on **2026-10-28**.

## Code provenance

The core implementation in `src/`, `scripts/`, and `tests/` was copied from the independent reproduction project [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), commit `ba6105349069bf040f1fdad60873d13138d505a2` (2026-10-26). It is not the official implementation from the paper authors. The upstream project identifies **Yuxi Liu** as copyright holder. `docs/EXPERIMENTS.md` is the historical experiment log from that same project; its runs and metrics are not Jianglin Yan's. The local LAFAN1 run report and small viewer utilities were added for this workspace.

The paper being reproduced is Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, and K. Sreenath, “SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model,” CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480). This workspace is an independent reproduction and is not affiliated with the paper authors.

## Consolidated MIT License

The upstream MIT notice is reproduced in full here; it was consolidated from the standalone `LICENSE` file so the repository has one place for code provenance and license notices.

```text
MIT License

Copyright (c) 2026 Yuxi Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other components

Third-party libraries listed in `pyproject.toml`, including mjlab, MuJoCo, MuJoCo Warp, PyTorch, and Warp, retain their own terms. LAFAN1 and BONES-SEED data are not included; their dataset terms apply separately. No datasets, checkpoints, logs, paper PDFs, or generated media are included in this repository.

<!-- Maintainer: Jianglin Yan | 2026-10-28 -->
