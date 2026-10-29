# Paper specification notes

Source: Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, arXiv:2610.28480v1 (2026-10-20). The local PDF is `../2610.28480v1.pdf`.

## Method in one loop

The policy collects transitions in MuJoCo Warp into a replay buffer. A learned differentiable world model is fit on short replay windows. The policy and critic are then updated through imagined model rollouts using backpropagation through time. Simulator gradients are not required; the learned model is updated again on the next iteration.

## Key reported settings (Appendix D.2, Table A.5)

| Component | Paper setting |
|---|---|
| World-model autoregressive window | 8 steps |
| World-model optimizer / learning rate | Adam / `2e-4` for 1 GPU |
| World-model batch | 10 batches × 2048 windows per update |
| Imagination horizon | 32 steps |
| Imagined rollout batch | 4096 |
| Policy optimizer | Adam, betas `(0.9, 0.95)` |
| Critic | Adam; GAE `lambda=0.95`; 5 epochs × 4 minibatches |
| Discount | `gamma=0.99` |
| Simulator environments | 4096 per GPU |
| Replay buffer | 1000 steps per environment |
| Update-to-data ratio | 0.4 |
| Training schedule | 320,000 iterations; cosine LR decay to `0.03×` over first 250,000 iterations |

These are target-paper settings, not guaranteed safe defaults for a 24 GB RTX 4090. Reduce batch/environment sizes only after recording the deviation and first measuring baseline memory use.

## Data and paper targets

- **LAFAN1 controlled comparison:** 40 G1-retargeted clips, about 2.45 hours; paper reports normalized evaluation return 0.82 after 1.2 hours on one H100.
- **BONES-SEED full benchmark:** 129,514 filtered clips, 264.56 hours; paper reports return above 0.90 and 25 failed motions after 3 hours on one H100, and zero failed motions after 9.8 hours on two H100 GPUs.
- **Hardware in the paper:** NVIDIA H100 SXM5, 80 GB. Paper wall times cannot be directly promised on a 24 GB RTX 4090.
- **Metric:** Appendix F deterministic full-clip evaluation, starting from each clip's first frame, without noise or domain randomization. Report normalized return and failure count, with the precise filter and benchmark set version.

## Reproduction gaps to track

The current third-party candidate reports paper-observation LAFAN1 scores around `0.56–0.59` on one RTX 5090 (paper: `0.82`), and full BONES-SEED results on 4× RTX 5090 of `0.930` with `145 / 129,491` failures (paper: `>0.90`, `25 / 129,514` on one H100). These are not equivalent results. First identify implementation, dataset filtering, environment, evaluator, and checkpoint differences; do not hide these gaps by changing observations/rewards and labeling the run paper-faithful.

## Primary references

- Paper: <https://arxiv.org/abs/2610.28480>
- Project page: <https://supermimic.github.io/>
- Candidate implementation and its own experiment details: <https://github.com/CuboiLeo/SuperMimic-Reproduced>

<!-- Project maintainer: Jianglin Yan; local copy assembled 2026-10-28. See ATTRIBUTION.md for upstream authorship and license. -->
