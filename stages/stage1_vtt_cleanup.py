"""
Stage 1: VTT Cleanup.

Parse YouTube auto-captions, deduplicate rolling-window text,
strip [Music] tags, and produce a clean cue list with quality scores.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from config import VideoContext
from utils.vtt_parser import VTTCue, parse_vtt, assess_vtt_quality

logger = logging.getLogger(__name__)

STAGE_NAME = "stage1_vtt_cleanup"


def run(ctx: VideoContext) -> VideoContext:
    """Run VTT cleanup. Skips if no VTT available."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.vtt_cleaned = cached
            logger.info("Stage 1: skipped (checkpoint exists)")
            return ctx

    if ctx.vtt_path is None or not ctx.vtt_path.exists():
        logger.info("Stage 1: no VTT file — pipeline will rely on Whisper only")
        ctx.vtt_cleaned = []
        ctx.save_checkpoint(STAGE_NAME)
        return ctx

    logger.info(f"Stage 1: Parsing VTT from {ctx.vtt_path} ...")
    t0 = time.time()

    cues = parse_vtt(ctx.vtt_path)
    ctx.vtt_cleaned = [_cue_to_dict(c) for c in cues]

    # Save cleaned VTT
    out_path = ctx.debug_dir / "vtt_cleaned.json"
    out_path.write_text(json.dumps(ctx.vtt_cleaned, indent=2))

    # Overall quality
    if cues:
        overall_q = assess_vtt_quality(cues, cues[0].start, cues[-1].end)
        quality_path = ctx.debug_dir / "vtt_quality.json"
        quality_path.write_text(json.dumps({
            "score": overall_q.score,
            "total_cues": overall_q.total_cues,
            "text_cues": overall_q.text_cues,
            "music_cues": overall_q.music_cues,
            "empty_gaps": overall_q.empty_gaps,
            "avg_words_per_cue": overall_q.avg_words_per_cue,
            "issues": overall_q.issues,
        }, indent=2))
        logger.info(
            f"  VTT quality: {overall_q.score:.2f} | "
            f"{overall_q.text_cues} text cues, "
            f"{overall_q.music_cues} music cues, "
            f"avg {overall_q.avg_words_per_cue} words/cue"
        )

    elapsed = time.time() - t0
    logger.info(
        f"Stage 1: done in {elapsed:.1f}s — "
        f"{len(cues)} cues extracted from VTT"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _cue_to_dict(cue: VTTCue) -> dict:
    return {
        "start": cue.start,
        "end": cue.end,
        "text": cue.text,
        "is_music": cue.is_music,
    }


def _load_cached(ctx: VideoContext) -> list[dict] | None:
    for path in (
        ctx.debug_dir / "vtt_cleaned.json",
        ctx.analysis_dir / "vtt_cleaned.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None
