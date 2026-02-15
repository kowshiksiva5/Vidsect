"""
Stage 1c: Speaker Diarization with pyannote.

Uses pyannote community-1 pipeline (exclusive mode) on cleaned audio
to produce speaker segments: who speaks when, with no overlapping assignments.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from video_chunker.config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage1c_diarize"


def run(ctx: VideoContext) -> VideoContext:
    """Run speaker diarization."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.diarization_segments = cached
            logger.info("Stage 1c: skipped (checkpoint exists)")
            return ctx

    audio_path = ctx.audio_clean_path or ctx.audio_path
    logger.info(f"Stage 1c: Diarizing with pyannote from {audio_path.name} ...")
    t0 = time.time()

    try:
        segments = _diarize(
            audio_path=audio_path,
            pipeline_name=ctx.config.pyannote_pipeline,
            min_speakers=ctx.config.min_speakers,
            max_speakers=ctx.config.max_speakers,
        )
    except Exception as exc:
        logger.warning(
            "Stage 1c fallback: diarization unavailable (%s). "
            "Using single-speaker timeline.",
            exc,
        )
        segments = _fallback_diarization(ctx)

    ctx.diarization_segments = segments
    out_path = ctx.debug_dir / "diarization.json"
    out_path.write_text(json.dumps(segments, indent=2))

    # Log summary
    speakers = set(s["speaker"] for s in segments)
    elapsed = time.time() - t0
    logger.info(
        f"Stage 1c: done in {elapsed:.1f}s — "
        f"{len(segments)} segments, {len(speakers)} speakers detected"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _diarize(
    audio_path: Path,
    pipeline_name: str,
    min_speakers: int,
    max_speakers: int,
) -> list[dict]:
    """
    Run pyannote speaker diarization.

    Returns list of segments:
    [
      {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"},
      {"start": 5.2, "end": 12.8, "speaker": "SPEAKER_01"},
      ...
    ]
    """
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set. Export your HuggingFace token:\n"
            "  export HF_TOKEN='your_token'\n"
            "Then accept model terms at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        raise ImportError(
            "pyannote.audio not installed. Run: pip install pyannote.audio torch"
        )

    # Load pipeline
    logger.info(f"  Loading pyannote pipeline: {pipeline_name}")
    try:
        pipeline = Pipeline.from_pretrained(pipeline_name, token=hf_token)
    except TypeError:
        # Backward compatibility with older pyannote versions.
        pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=hf_token)

    # Use MPS (Metal) if available on Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)

    logger.info(f"  Running diarization on {device} ...")
    diarization = pipeline(
        str(audio_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # pyannote API compatibility:
    # - older versions return Annotation directly
    # - newer versions return DiarizeOutput with .speaker_diarization
    if hasattr(diarization, "itertracks"):
        annotation = diarization
    elif hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    else:
        raise TypeError(
            f"Unsupported diarization output type: {type(diarization)!r}"
        )

    # Convert to segments list
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    return segments


def _load_cached(ctx: VideoContext) -> list[dict] | None:
    for path in (
        ctx.debug_dir / "diarization.json",
        ctx.analysis_dir / "diarization.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None


def _fallback_diarization(ctx: VideoContext) -> list[dict]:
    """Fallback when pyannote is unavailable: treat entire stream as one speaker."""
    if ctx.whisper_transcript:
        start = float(ctx.whisper_transcript[0]["start"])
        end = float(ctx.whisper_transcript[-1]["end"])
    else:
        end = _probe_audio_duration(ctx.audio_clean_path or ctx.audio_path)
        start = 0.0

    if end <= start:
        end = start + 1.0

    return [{
        "start": round(start, 3),
        "end": round(end, 3),
        "speaker": "SPEAKER_00",
    }]


def _probe_audio_duration(audio_path: Path) -> float:
    """Read duration in seconds via ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 1.0
