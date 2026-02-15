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

from config import PipelineConfig, VideoContext
from stages import (
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

    if config.run_quality_gate:
        _run_quality_gate(ctx, stop_after)

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


def _run_quality_gate(ctx: VideoContext, stop_after: str | None) -> None:
    """Validate output quality and write debug/quality_gate.json."""
    skip_reason = _quality_gate_skip_reason(ctx, stop_after)
    if skip_reason:
        logger.info("Quality gate skipped (%s).", skip_reason)
        return

    _, metrics, failures, warnings = _evaluate_report_quality(ctx)
    fallbacks, fallback_failures = _evaluate_fallback_quality(ctx)
    failures.extend(fallback_failures)

    retranscription_hist = _collect_retranscription_histogram(
        ctx.debug_dir / "chunks"
    )
    quality = {
        "status": "pass" if not failures else "fail",
        "strict_mode": ctx.config.strict_production_checks,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "fallbacks": fallbacks,
        "retranscription_status_histogram": retranscription_hist,
    }
    (ctx.debug_dir / "quality_gate.json").write_text(
        json.dumps(quality, indent=2)
    )

    if failures:
        logger.error("Quality gate failed: %s", ", ".join(failures))
        if ctx.config.strict_production_checks:
            raise RuntimeError(
                "Quality gate failed in strict mode: " + ", ".join(failures)
            )
    else:
        logger.info("Quality gate passed.")
    if warnings:
        logger.warning("Quality gate warnings: %s", ", ".join(warnings))


def _quality_gate_skip_reason(
    ctx: VideoContext,
    stop_after: str | None,
) -> str | None:
    """Return a skip reason when quality gate should not run."""
    if stop_after and stop_after != "stage7_report":
        return "pipeline stopped before stage 7"
    stage7_cp = ctx.debug_dir / "checkpoints" / "stage7_report.json"
    if not stage7_cp.exists():
        return "stage7 report checkpoint not found"
    return None


def _evaluate_report_quality(
    ctx: VideoContext,
) -> tuple[dict, dict, list[str], list[str]]:
    """Compute report-level quality metrics, failures, and warnings."""
    report_path = ctx.analysis_dir / "video_report.json"
    failures: list[str] = []
    warnings: list[str] = []
    report_data: dict = {}

    if not report_path.exists():
        failures.append("video_report_missing")
        return report_data, _empty_quality_metrics(), failures, warnings

    report_data = json.loads(report_path.read_text())
    chunks = report_data.get("chunks", [])
    metrics = _build_report_metrics(report_data, chunks)

    if report_data and not chunks:
        failures.append("no_chunks_in_report")
    if metrics["unknown_chunk_count"] > 0:
        failures.append(f"unknown_chunks:{metrics['unknown_chunk_count']}")
    if metrics["missing_summary_count"] > 0:
        failures.append(f"missing_chunk_summaries:{metrics['missing_summary_count']}")
    if metrics["missing_location_count"] > 0:
        warnings.append(f"chunks_missing_locations:{metrics['missing_location_count']}")
    if metrics["speaker_mapping_count"] == 0:
        warnings.append("speaker_mapping_empty")
    if metrics["face_gallery_count"] == 0:
        warnings.append("face_gallery_empty")

    return report_data, metrics, failures, warnings


def _build_report_metrics(report_data: dict, chunks: list[dict]) -> dict:
    """Build structured metrics from consolidated report content."""
    unknown_chunk_count = sum(
        1 for c in chunks if str(c.get("chunk_type", "")).lower() in {"", "unknown"}
    )
    missing_summary_count = sum(
        1 for c in chunks if not str(c.get("summary_text") or "").strip()
    )
    missing_location_count = sum(
        1 for c in chunks if not c.get("location_mentions")
    )
    return {
        "chunk_count": len(chunks),
        "unknown_chunk_count": unknown_chunk_count,
        "missing_summary_count": missing_summary_count,
        "missing_location_count": missing_location_count,
        "speaker_mapping_count": len(report_data.get("speaker_mapping", {})),
        "face_gallery_count": len(report_data.get("face_gallery", {})),
    }


def _empty_quality_metrics() -> dict:
    """Metrics shape for missing/invalid report cases."""
    return {
        "chunk_count": 0,
        "unknown_chunk_count": 0,
        "missing_summary_count": 0,
        "missing_location_count": 0,
        "speaker_mapping_count": 0,
        "face_gallery_count": 0,
    }


def _evaluate_fallback_quality(
    ctx: VideoContext,
) -> tuple[dict[str, bool], list[str]]:
    """Read fallback metadata and return failures for strict criteria."""
    failures: list[str] = []
    diar_meta = _load_debug_json(ctx.debug_dir / "diarization_meta.json")
    chunk_meta = _load_debug_json(ctx.debug_dir / "chunking_meta.json")
    fallbacks = {
        "diarization_fallback_used": bool(diar_meta.get("fallback_used", False)),
        "chunking_fallback_used": bool(chunk_meta.get("fallback_used", False)),
    }
    if fallbacks["diarization_fallback_used"]:
        failures.append("diarization_fallback_used")
    if fallbacks["chunking_fallback_used"]:
        failures.append("chunking_fallback_used")
    return fallbacks, failures


def _load_debug_json(path: Path) -> dict:
    """Load a debug JSON file, returning {} when missing/invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _collect_retranscription_histogram(chunks_dir: Path) -> dict[str, int]:
    """Aggregate chunk retranscription statuses from summary.json files."""
    hist: dict[str, int] = {}
    if not chunks_dir.exists():
        return hist
    for summary_file in chunks_dir.glob("*/summary.json"):
        data = _load_debug_json(summary_file)
        status = (
            data.get("vtt_quality", {}).get("retranscription_status")
            or "unknown"
        )
        hist[status] = hist.get(status, 0) + 1
    return hist
