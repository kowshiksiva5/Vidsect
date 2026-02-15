"""
Stage 2: Merge Transcript + Speaker Diarization.

Aligns Whisper word-level timestamps with pyannote speaker segments
to produce a merged transcript where each word/segment has a speaker label.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from video_chunker.config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2_merge"


def run(ctx: VideoContext) -> VideoContext:
    """Merge whisper transcript with diarization segments."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.merged_transcript = cached
            logger.info("Stage 2: skipped (checkpoint exists)")
            return ctx

    if not ctx.whisper_transcript:
        raise ValueError("Stage 2 requires whisper_transcript (stage 1b)")
    if not ctx.diarization_segments:
        raise ValueError("Stage 2 requires diarization_segments (stage 1c)")

    logger.info("Stage 2: Merging transcript + speaker segments ...")
    t0 = time.time()

    merged = _merge(ctx.whisper_transcript, ctx.diarization_segments)

    ctx.merged_transcript = merged
    out_path = ctx.debug_dir / "merged_transcript.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    # Stats
    speakers = set(s["speaker"] for s in merged if s.get("speaker"))
    elapsed = time.time() - t0
    logger.info(
        f"Stage 2: done in {elapsed:.1f}s — "
        f"{len(merged)} merged segments, "
        f"{len(speakers)} speakers"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _merge(
    whisper_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """
    Merge whisper transcript segments with speaker labels.

    Strategy: For each whisper segment, find the dominant speaker
    (the pyannote speaker with the most temporal overlap).
    Also propagate word-level speaker labels via word timestamps.
    """
    # Pre-sort diarization segments by start time
    dia_sorted = sorted(diarization_segments, key=lambda s: s["start"])

    merged: list[dict] = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find dominant speaker for this segment
        speaker = _find_dominant_speaker(seg_start, seg_end, dia_sorted)

        merged_seg = {
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
            "speaker": speaker,
        }

        # Word-level speaker attribution (if words available)
        if "words" in seg:
            merged_words = []
            for w in seg["words"]:
                word_speaker = _find_dominant_speaker(
                    w["start"], w["end"], dia_sorted
                )
                merged_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": word_speaker or speaker,
                })
            merged_seg["words"] = merged_words

        merged.append(merged_seg)

    return merged


def _find_dominant_speaker(
    start: float,
    end: float,
    dia_segments: list[dict],
) -> str | None:
    """
    Find the speaker with the most overlap in [start, end].

    Uses binary search for efficiency with long diarization lists.
    """
    if not dia_segments or start >= end:
        return None

    speaker_overlap: dict[str, float] = {}

    for seg in dia_segments:
        # Early exit: past our window
        if seg["start"] >= end:
            break
        # Skip segments that don't overlap
        if seg["end"] <= start:
            continue

        overlap_start = max(start, seg["start"])
        overlap_end = min(end, seg["end"])
        overlap = overlap_end - overlap_start

        if overlap > 0:
            speaker = seg["speaker"]
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

    if not speaker_overlap:
        return None

    return max(speaker_overlap, key=speaker_overlap.get)


def slice_transcript(
    merged_transcript: list[dict],
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    """
    Slice the merged transcript for a specific chunk time window.

    Used by Stage 6 per-chunk processing.  This is the primary method —
    no re-transcription needed unless quality is bad.
    """
    sliced: list[dict] = []

    for seg in merged_transcript:
        if seg["end"] <= start_sec:
            continue
        if seg["start"] >= end_sec:
            break

        # Clip segment to chunk boundaries
        clipped_start = max(seg["start"], start_sec)
        clipped_end = min(seg["end"], end_sec)

        clipped_seg = {
            "start": clipped_start,
            "end": clipped_end,
            "text": seg["text"],
            "speaker": seg.get("speaker"),
        }

        # Clip words if present
        if "words" in seg:
            clipped_words = [
                w for w in seg["words"]
                if w["start"] >= start_sec and w["end"] <= end_sec
            ]
            if clipped_words:
                # Rebuild text from clipped words
                clipped_seg["text"] = " ".join(w["word"] for w in clipped_words)
                clipped_seg["words"] = clipped_words

        if clipped_seg["text"].strip():
            sliced.append(clipped_seg)

    return sliced


def _load_cached(ctx: VideoContext) -> list[dict] | None:
    for path in (
        ctx.debug_dir / "merged_transcript.json",
        ctx.analysis_dir / "merged_transcript.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None
