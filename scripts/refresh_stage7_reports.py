#!/usr/bin/env python3
"""Refresh Stage 7 reports for existing processed video directories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_video_dirs(root: Path) -> list[Path]:
    """Return video dirs that already have a stage7 checkpoint."""
    video_dirs: list[Path] = []
    pattern = "**/analysis/debug/checkpoints/stage7_report.json"
    for cp in sorted(root.glob(pattern)):
        video_dir = cp.parents[3]
        if (video_dir / "audio.wav").exists() and list(video_dir.glob("*.mp4")):
            video_dirs.append(video_dir)
    return video_dirs


def refresh_stage7(video_dir: Path, allow_fallbacks: bool, dry_run: bool) -> int:
    """Delete stage7 checkpoint and rerun pipeline until stage 7."""
    checkpoint = video_dir / "analysis" / "debug" / "checkpoints" / "stage7_report.json"
    if dry_run:
        print(f"[dry-run] {video_dir}")
        return 0

    checkpoint.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "run.py",
        "--video-dir",
        str(video_dir),
        "--stop-after",
        "stage7_report",
    ]
    if allow_fallbacks:
        cmd.extend(["--allow-fallbacks", "--skip-quality-gate"])

    print(f"\nRefreshing: {video_dir}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh existing stage7 reports.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root path to scan for analysis/debug/checkpoints/stage7_report.json",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Run refresh in non-strict mode (recommended for legacy outputs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target directories only.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = find_video_dirs(root)
    if not targets:
        print("No stage7 checkpoints found.")
        return

    print(f"Found {len(targets)} video directories to refresh under {root}")
    failed = 0
    for video_dir in targets:
        code = refresh_stage7(video_dir, args.allow_fallbacks, args.dry_run)
        failed += int(code != 0)

    if failed > 0:
        print(f"\nCompleted with {failed} failures.")
        sys.exit(1)
    print("\nRefresh complete.")


if __name__ == "__main__":
    main()
