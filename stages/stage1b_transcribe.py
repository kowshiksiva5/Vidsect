"""
Stage 1b: Transcription with mlx-whisper.

Runs whisper on the full cleaned audio to produce word-level timestamps.
This is the SINGLE full transcription pass — per-chunk processing only
slices this transcript, with selective re-transcription for bad segments.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage1b_transcribe"


def run(ctx: VideoContext) -> VideoContext:
    """Run full-audio transcription with mlx-whisper."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.whisper_transcript = cached
            logger.info("Stage 1b: skipped (checkpoint exists)")
            return ctx

    audio_path = ctx.audio_clean_path or ctx.audio_path
    logger.info(f"Stage 1b: Transcribing with mlx-whisper from {audio_path.name} ...")
    t0 = time.time()

    segments = _transcribe(
        audio_path=audio_path,
        model=ctx.config.whisper_model,
        language=ctx.config.whisper_language,
    )

    ctx.whisper_transcript = segments
    out_path = ctx.debug_dir / "whisper_transcript.json"
    out_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    total_words = sum(len(s.get("words", [])) for s in segments)
    logger.info(
        f"Stage 1b: done in {elapsed:.1f}s — "
        f"{len(segments)} segments, {total_words} words"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _transcribe(
    audio_path: Path,
    model: str,
    language: str,
) -> list[dict]:
    """
    Run mlx-whisper on the full audio file.

    Returns list of segments with word-level timestamps:
    [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "What will happen today",
        "words": [
          {"word": "What", "start": 0.0, "end": 0.3},
          {"word": "will", "start": 0.3, "end": 0.5},
          ...
        ]
      },
      ...
    ]
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ImportError(
            "mlx-whisper import failed. Install/repair dependencies with:\n"
            "  pip install mlx-whisper\n"
            "If already installed, check numpy/numba compatibility."
        ) from exc

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        fp16=True,
    )

    segments = []
    for seg in result.get("segments", []):
        segment_data = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        if "words" in seg:
            segment_data["words"] = [
                {
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                }
                for w in seg["words"]
            ]
        segments.append(segment_data)

    return segments


def _load_cached(ctx: VideoContext) -> list[dict] | None:
    for path in (
        ctx.debug_dir / "whisper_transcript.json",
        ctx.analysis_dir / "whisper_transcript.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None
