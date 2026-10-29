"""Combine per-motion MP4 tiles into one grid video using ffmpeg."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import tyro


def main(
    tile_dir: Path,
    out: Path,
    pattern: str = "*.mp4",
    cols: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
    duration: float = 12.0,
    fps: int = 10,
    crf: int = 20,
) -> None:
    out = out.expanduser().resolve()
    files = sorted(
        path.resolve()
        for path in tile_dir.expanduser().glob(pattern)
        if path.resolve() != out
    )
    if not files:
        raise ValueError(f"No input videos matched {pattern!r} in {tile_dir}")
    if cols < 1 or duration <= 0 or fps < 1:
        raise ValueError("cols, fps, and duration must be positive")

    rows = math.ceil(len(files) / cols)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "1"]
    for path in files:
        cmd.extend(["-i", str(path)])

    filters = []
    for idx in range(len(files)):
        filters.append(
            f"[{idx}:v]fps={fps},scale={tile_width}:{tile_height}:"
            f"force_original_aspect_ratio=decrease,pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2,"
            f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},"
            f"setpts=PTS-STARTPTS[v{idx}]"
        )
    layout = "|".join(f"{(idx % cols) * tile_width}_{(idx // cols) * tile_height}" for idx in range(len(files)))
    inputs = "".join(f"[v{idx}]" for idx in range(len(files)))
    filters.append(f"{inputs}xstack=inputs={len(files)}:layout={layout}:fill=black:shortest=1[out]")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    )
    subprocess.run(cmd, check=True)
    print(f"Wrote {out} ({len(files)} tiles, {cols}×{rows}, {duration:g}s)")


if __name__ == "__main__":
    tyro.cli(main)

# Maintainer: Jianglin Yan | 2026-10-28
