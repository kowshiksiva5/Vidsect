"""
Pipeline orchestrator — runs all stages in sequence with checkpointing.

Each stage is idempotent: if a checkpoint exists, it's skipped.
This allows the pipeline to resume from where it left off after a crash.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from video_chunker.config import PipelineConfig, VideoContext
from video_chunker.stages import (
    stage0_denoise,
    stage1_vtt_cleanup,
    stage1b_transcribe,
    stage1c_diarize,
    stage2_merge,
    stage3_scenes,
    stage4_chunking,
    stage5_identity,
    stage6_per_chunk,
    stage7_report,
)

logger = logging.getLogger(__name__)


# Pipeline stages in execution order
STAGES = [
    ("Stage 0: Audio Denoising",      stage0_denoise),
    ("Stage 1: VTT Cleanup",          stage1_vtt_cleanup),
    ("Stage 1b: Transcription",       stage1b_transcribe),
    ("Stage 1c: Diarization",         stage1c_diarize),
    ("Stage 2: Merge",                stage2_merge),
    ("Stage 3: Scene Detection",      stage3_scenes),
    ("Stage 4: Semantic Chunking",    stage4_chunking),
    ("Stage 5: Person Identity",      stage5_identity),
    ("Stage 6: Per-Chunk Processing", stage6_per_chunk),
    ("Stage 7: Report Assembly",      stage7_report),
]


def run_pipeline(
    video_dir: Path,
    config: PipelineConfig | None = None,
    skip_stages: set[str] | None = None,
    stop_after: str | None = None,
) -> VideoContext:
    """
    Run the full pipeline on a video directory.

    Args:
        video_dir: Path to directory with video.mp4, audio.wav, etc.
        config: Pipeline config (defaults to PipelineConfig())
        skip_stages: Set of stage module names to skip (e.g. {"stage0_denoise"})
        stop_after: Stop after this stage completes (e.g. "stage2_merge")

    Returns:
        VideoContext with all outputs populated
    """
    config = config or PipelineConfig()
    skip_stages = skip_stages or set()

    ctx = VideoContext.from_video_dir(video_dir, config)

    logger.info("=" * 60)
    logger.info(f"Pipeline starting: {video_dir}")
    logger.info(f"  Video: {ctx.video_path.name}")
    logger.info(f"  Audio: {ctx.audio_path.name}")
    logger.info(f"  VTT:   {ctx.vtt_path.name if ctx.vtt_path else 'None'}")
    logger.info(f"  Output: {ctx.analysis_dir}")
    logger.info("=" * 60)

    t0 = time.time()

    for stage_name, stage_module in STAGES:
        module_name = stage_module.__name__.split(".")[-1]

        if module_name in skip_stages:
            logger.info(f"\n{'='*40}")
            logger.info(f"SKIPPING {stage_name} (user requested)")
            logger.info(f"{'='*40}")
            continue

        logger.info(f"\n{'='*40}")
        logger.info(f"RUNNING {stage_name}")
        logger.info(f"{'='*40}")

        stage_t0 = time.time()
        try:
            ctx = stage_module.run(ctx)
        except Exception as e:
            logger.error(f"FAILED {stage_name}: {e}", exc_info=True)
            _save_error(ctx, module_name, str(e))
            raise

        stage_elapsed = time.time() - stage_t0
        logger.info(f"{stage_name}: completed in {stage_elapsed:.1f}s")

        if stop_after and module_name == stop_after:
            logger.info(f"\nStopping after {stage_name} (stop_after={stop_after})")
            break

    total_elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete: {total_elapsed:.1f}s total")
    logger.info(f"Output directory: {ctx.analysis_dir}")
    logger.info(f"{'='*60}")

    # Clear stale error artifact from previous failed runs.
    err_path = ctx.debug_dir / "pipeline_error.json"
    if err_path.exists():
        err_path.unlink()

    # Write pipeline summary
    _write_summary(ctx, total_elapsed)

    return ctx


def _save_error(ctx: VideoContext, stage_name: str, error_msg: str) -> None:
    """Save error info for debugging."""
    err_path = ctx.debug_dir / "pipeline_error.json"
    err_path.write_text(json.dumps({
        "failed_stage": stage_name,
        "error": error_msg,
    }, indent=2))


def _write_summary(ctx: VideoContext, elapsed: float) -> None:
    """Write a pipeline execution summary."""
    summary = {
        "video_dir": str(ctx.video_dir),
        "elapsed_seconds": round(elapsed, 1),
        "stages_completed": [],
    }

    # Check which checkpoints exist
    cp_dir = ctx.debug_dir / "checkpoints"
    if cp_dir.exists():
        for cp_file in sorted(cp_dir.glob("*.json")):
            summary["stages_completed"].append(cp_file.stem)

    # Chunk summary
    if ctx.chunks:
        summary["chunks"] = {
            "total": len(ctx.chunks),
            "types": {},
        }
        for c in ctx.chunks:
            t = c.get("type", "unknown")
            summary["chunks"]["types"][t] = summary["chunks"]["types"].get(t, 0) + 1

    # Transcript summary
    if ctx.merged_transcript:
        summary["transcript"] = {
            "total_segments": len(ctx.merged_transcript),
            "speakers": list(set(
                s["speaker"] for s in ctx.merged_transcript
                if s.get("speaker")
            )),
        }

    out_path = ctx.debug_dir / "pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
