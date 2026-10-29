"""Merge sharded eval_benchmark JSON outputs into one summary (full-set benchmark run on several GPUs).

  uv run python scripts/merge_benchmark_shards.py --shards logs/e8/benchmark_full_shard_*.json --out logs/e8/benchmark_full.json
"""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from supermimic.eval.benchmark import ClipResult, save_results, summarize


def main(shards: list[Path], out: Path | None = None) -> None:
    results: list[ClipResult] = []
    for p in shards:
        results += [ClipResult(**c) for c in json.load(open(p))["clips"]]
    s = summarize(results)
    print(f"{s['clips']} clips | failed {s['failed']} ({100*s['failed']/max(s['clips'],1):.3f}%) | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}")
    if "terms" in s:
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    failed = sorted([r for r in results if r.failed], key=lambda r: r.name)
    print(f"failed clips ({len(failed)}):")
    for r in failed:
        print(f"  {r.name:40s} step {r.fail_step}/{r.frames}")
    if out:
        save_results(results, out)
        print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)

# Project maintainer: Jianglin Yan; local copy assembled 2026-10-28. See ATTRIBUTION.md for upstream authorship and license.
