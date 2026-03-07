"""
Stage 3: Scene Detection with PySceneDetect.

Detects visual scene boundaries using content-aware detection.
These boundaries feed into the semantic chunking stage (Stage 4)
as structural markers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_scenes"


def run(ctx: VideoContext) -> VideoContext:
    """Detect scene boundaries in the video."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.scene_boundaries = cached
            logger.info("Stage 3: skipped (checkpoint exists)")
            return ctx

    logger.info(f"Stage 3: Detecting scenes in {ctx.video_path.name} ...")
    t0 = time.time()

    boundaries = _detect_scenes(ctx.video_path)
    ctx.scene_boundaries = boundaries

    out_path = ctx.debug_dir / "scene_boundaries.json"
    out_path.write_text(json.dumps({
        "boundaries": boundaries,
        "total_scenes": len(boundaries) + 1,
    }, indent=2))

    elapsed = time.time() - t0
    logger.info(
        f"Stage 3: done in {elapsed:.1f}s — "
        f"{len(boundaries)} scene boundaries detected"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _detect_scenes(video_path: Path) -> list[float]:
    """
    Run PySceneDetect content-aware detection.
    Uses downscaling and an optimized pipeline for drastic speedups.

    Returns list of scene boundary timestamps in seconds.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        raise ImportError(
            "scenedetect not installed. Run: pip install scenedetect[opencv]"
        )

    logger.info("  Running optimized ContentDetector (downscaled) ...")

    # Open the video using the modern v0.6+ API
    video = open_video(str(video_path))
    scene_manager = SceneManager()

    # Add ContentDetector algorithm.
    scene_manager.add_detector(ContentDetector(
        threshold=27.0,
        min_scene_len=15,
    ))

    # Improve processing speed by downscaling before processing.
    # A typical video is 1080p, setting downscale_factor to 4 speeds up histogram comparisons significantly.
    scene_manager.downscale = 4

    # Perform scene detection.
    scene_manager.detect_scenes(video=video)

    # Obtain list of detected scenes.
    scene_list = scene_manager.get_scene_list()

    # Convert FrameTimecode pairs to boundary seconds
    boundaries = []
    for scene_start, _ in scene_list:
        t = scene_start.get_seconds()
        if t > 0:
            boundaries.append(round(t, 3))

    return sorted(set(boundaries))


def _load_cached(ctx: VideoContext) -> list[float] | None:
    for path in (
        ctx.debug_dir / "scene_boundaries.json",
        ctx.analysis_dir / "scene_boundaries.json",  # legacy path
    ):
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("boundaries", [])
    return None
