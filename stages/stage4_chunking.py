"""
Stage 4: Semantic Chunking with LLM.

Uses a local LLM (via Ollama) to identify natural chunk boundaries
in the merged transcript, guided by scene boundaries and speaker changes.
Live-stream aware: recognizes rides, waiting, encounters, stream events.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage4_chunking"

# System prompt for the chunking LLM
CHUNKING_SYSTEM_PROMPT = """You are an expert at segmenting Twitch hitchhiking livestream transcripts into meaningful chunks.

You will receive a merged transcript with speaker labels and scene boundary timestamps.
Your job is to identify the natural narrative segments.

CHUNK TYPES for a hitchhiking livestream:
- **ride**: Getting into a car, riding with someone, conversation with driver
- **waiting**: Standing on highway ramp, walking to a spot, thumbing for rides
- **encounter**: Brief interaction — gas station worker, stranger, convenience store
- **food_stop**: Eating at restaurant, fast food, snacking
- **rest_stop**: Camping, sleeping, resting
- **stream_event**: Raid, significant donation, sub train, break screen
- **walking**: Walking along road, through town
- **intro**: Stream starting, setting up, recap
- **outro**: Stream ending, sign-off

OUTPUT FORMAT — respond with ONLY a JSON array:
[
  {
    "chunk_id": 1,
    "start_sec": 0.0,
    "end_sec": 1200.0,
    "type": "intro",
    "title": "Stream Start - Setting up in Arvin",
    "reason": "Stream intro with setup recap before heading out"
  },
  ...
]

RULES:
1. Chunks should be 2-45 minutes (120-2700 seconds)
2. Prefer breaking at speaker changes or scene boundaries
3. A single ride with one driver = one chunk (even if 30+ min)
4. Brief encounters (<2 min) can be part of a larger waiting/walking chunk
5. Use descriptive titles that mention people/locations if mentioned
6. Every second of the transcript MUST be covered (no gaps)"""


def run(ctx: VideoContext) -> VideoContext:
    """Run semantic chunking."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.chunks = cached
            meta = _load_meta(ctx) or {}
            fallback_used = bool(meta.get("fallback_used", False))
            ctx.runtime_flags["chunking_fallback_used"] = fallback_used
            _raise_if_strict_fallback(
                ctx,
                fallback_used,
                meta.get("fallback_reason"),
            )
            logger.info("Stage 4: skipped (checkpoint exists)")
            return ctx

    if not ctx.merged_transcript:
        raise ValueError("Stage 4 requires merged_transcript (stage 2)")

    logger.info("Stage 4: Semantic chunking with LLM ...")
    t0 = time.time()

    chunks, fallback_used = _chunk_with_llm(
        merged_transcript=ctx.merged_transcript,
        scene_boundaries=ctx.scene_boundaries or [],
        model=ctx.config.chunking_model,
        min_duration=ctx.config.min_chunk_duration_sec,
        max_duration=ctx.config.max_chunk_duration_sec,
    )

    ctx.chunks = chunks
    out_path = ctx.debug_dir / "chunk_manifest.json"
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    _save_meta(
        ctx=ctx,
        data={
            "fallback_used": fallback_used,
            "fallback_reason": (
                "chunking_llm_unavailable_or_unparseable" if fallback_used else None
            ),
            "chunk_count": len(chunks),
        },
    )
    ctx.runtime_flags["chunking_fallback_used"] = fallback_used
    _raise_if_strict_fallback(ctx, fallback_used, None)

    elapsed = time.time() - t0
    logger.info(
        f"Stage 4: done in {elapsed:.1f}s — "
        f"{len(chunks)} chunks identified"
    )
    for c in chunks:
        dur = c["end_sec"] - c["start_sec"]
        logger.info(
            f"  Chunk {c['chunk_id']:02d}: "
            f"{dur / 60:.1f}min [{c['type']}] {c['title']}"
        )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _chunk_with_llm(
    merged_transcript: list[dict],
    scene_boundaries: list[float],
    model: str,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict], bool]:
    """
    Send transcript to Ollama LLM for semantic chunking.

    Handles long transcripts by windowing (LLMs have context limits).
    """
    # Determine actual duration from transcript
    actual_end = merged_transcript[-1]["end"] if merged_transcript else 3600.0

    # Format transcript for the LLM
    transcript_text = _format_transcript_for_llm(
        merged_transcript, scene_boundaries
    )

    # For long transcripts, we need to window
    # Qwen3-30B has ~32k context — each whisper segment ≈ 20 tokens
    # So we can fit ~1500 segments per call
    max_segments_per_call = 1200

    if len(merged_transcript) <= max_segments_per_call:
        chunks, fallback_used = _single_pass_chunk(
            transcript_text,
            model,
            fallback_start=0.0,
            fallback_end=actual_end,
        )
    else:
        # Multi-pass for very long transcripts
        chunks, fallback_used = _multi_pass_chunk(
            merged_transcript, scene_boundaries, model,
            max_segments_per_call, min_duration, max_duration, actual_end,
        )

    # Post-parse validation: enforce continuity and duration constraints
    chunks = _validate_chunks(chunks, actual_end, min_duration, max_duration)

    return chunks, fallback_used


def _format_transcript_for_llm(
    segments: list[dict],
    scene_boundaries: list[float],
) -> str:
    """Format segments + scene boundaries for the LLM prompt."""
    lines: list[str] = []

    # Insert scene boundary markers
    scene_set = set(round(b, 0) for b in scene_boundaries)
    current_speaker = None

    for seg in segments:
        t = seg["start"]
        # Check for nearby scene boundary
        for delta in range(-1, 2):
            if round(t, 0) + delta in scene_set:
                lines.append(f"--- SCENE CHANGE at {_format_time(t)} ---")
                break

        speaker = seg.get("speaker", "???")
        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n[{speaker}] ({_format_time(t)}):")

        lines.append(f"  {seg['text']}")

    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def _single_pass_chunk(
    transcript_text: str,
    model: str,
    fallback_start: float,
    fallback_end: float,
) -> tuple[list[dict], bool]:
    """Send entire transcript to LLM in one pass."""
    prompt = (
        f"{CHUNKING_SYSTEM_PROMPT}\n\n"
        f"## TRANSCRIPT\n\n{transcript_text}\n\n"
        f"## OUTPUT\nRespond with ONLY the JSON array of chunks:"
    )

    try:
        import ollama

        logger.info(f"  Calling {model} for chunking ...")
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 4096},
        )
        text = response["message"]["content"]
        chunks = _parse_chunks_json(text, fallback_start, fallback_end)
        fallback_used = _contains_fallback_chunk(chunks)
        return chunks, fallback_used
    except Exception as exc:
        logger.warning("  LLM chunking unavailable (%s); using fallback chunk.", exc)
        return _fallback_chunks(fallback_start, fallback_end), True


def _multi_pass_chunk(
    segments: list[dict],
    scene_boundaries: list[float],
    model: str,
    window_size: int,
    min_duration: float,
    max_duration: float,
    actual_end: float,
) -> tuple[list[dict], bool]:
    """Chunk long transcripts in overlapping windows."""
    all_chunks: list[dict] = []
    chunk_id_offset = 0
    fallback_used = False

    for i in range(0, len(segments), window_size - 100):  # 100-segment overlap
        window = segments[i:i + window_size]
        if not window:
            break

        window_start = window[0]["start"]
        window_end = window[-1]["end"]

        # Filter scene boundaries for this window
        window_scenes = [
            b for b in scene_boundaries
            if window_start <= b <= window_end
        ]

        text = _format_transcript_for_llm(window, window_scenes)
        chunks, chunk_fallback = _single_pass_chunk(
            text,
            model,
            fallback_start=window_start,
            fallback_end=window_end,
        )
        fallback_used = fallback_used or chunk_fallback

        # Re-number and add
        for c in chunks:
            c["chunk_id"] = chunk_id_offset + c["chunk_id"]
        all_chunks.extend(chunks)
        chunk_id_offset += len(chunks)

    # Deduplicate overlapping chunks
    return _deduplicate_chunks(all_chunks), fallback_used


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove overlapping chunks from multi-pass windowing."""
    if not chunks:
        return []

    sorted_chunks = sorted(chunks, key=lambda c: c["start_sec"])
    deduped = [sorted_chunks[0]]

    for chunk in sorted_chunks[1:]:
        prev = deduped[-1]
        # If this chunk starts within the previous one, skip it
        if chunk["start_sec"] < prev["end_sec"] - 30:
            continue
        # Adjust start to avoid gap
        if chunk["start_sec"] > prev["end_sec"]:
            prev["end_sec"] = chunk["start_sec"]
        deduped.append(chunk)

    # Renumber
    for i, c in enumerate(deduped, 1):
        c["chunk_id"] = i

    return deduped


def _parse_chunks_json(
    llm_output: str,
    fallback_start: float,
    fallback_end: float,
) -> list[dict]:
    """Extract JSON array from LLM response (may have markdown fencing)."""
    # Try to find JSON array in the response
    text = llm_output.strip()

    # Remove markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Find the JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            chunks = json.loads(match.group(0))
            if isinstance(chunks, list):
                return chunks
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM chunking output, using fallback")
    return _fallback_chunks(fallback_start, fallback_end)


def _fallback_chunks(fallback_start: float, fallback_end: float) -> list[dict]:
    """Create one fallback chunk spanning the current transcript window."""
    logger.warning("Using fallback single-chunk strategy")
    return [{
        "chunk_id": 1,
        "start_sec": fallback_start,
        "end_sec": fallback_end,
        "type": "unknown",
        "title": "Full segment (LLM chunking failed)",
        "reason": "Fallback — LLM output was not parseable",
    }]


def _contains_fallback_chunk(chunks: list[dict]) -> bool:
    """Detect whether the returned chunk set came from fallback logic."""
    for chunk in chunks:
        reason = str(chunk.get("reason", "")).lower()
        title = str(chunk.get("title", "")).lower()
        if "fallback" in reason or "llm chunking failed" in title:
            return True
    return False


def _validate_chunks(
    chunks: list[dict],
    actual_end: float,
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    """
    Post-parse validation: enforce full coverage and duration constraints.

    Fixes:
    - Gaps between chunks (extends previous chunk's end)
    - Last chunk not reaching actual_end (extends it)
    - Chunks shorter than min_duration (merges with neighbor)
    - Chunks longer than max_duration (logged as warning only — semantic
      boundaries take priority over arbitrary splitting)
    """
    if not chunks:
        return _fallback_chunks(0.0, actual_end)

    # Clamp chunk bounds to [0, actual_end] and drop invalid ranges.
    clamped: list[dict] = []
    for chunk in chunks:
        start = max(0.0, min(float(chunk.get("start_sec", 0.0)), actual_end))
        end = max(0.0, min(float(chunk.get("end_sec", 0.0)), actual_end))
        if end <= start:
            continue
        c = dict(chunk)
        c["start_sec"] = start
        c["end_sec"] = end
        c.setdefault("type", "unknown")
        c.setdefault("title", "Untitled chunk")
        clamped.append(c)

    if not clamped:
        return _fallback_chunks(0.0, actual_end)

    # Sort by start time
    chunks = sorted(clamped, key=lambda c: c["start_sec"])

    # Fix continuity: close gaps and overlaps
    for i in range(1, len(chunks)):
        prev_end = chunks[i - 1]["end_sec"]
        curr_start = chunks[i]["start_sec"]
        gap = curr_start - prev_end

        if abs(gap) > 1.0:
            # Close the gap by extending previous chunk
            midpoint = prev_end + gap / 2
            chunks[i - 1]["end_sec"] = midpoint
            chunks[i]["start_sec"] = midpoint

    # Ensure first chunk starts at 0
    if chunks[0]["start_sec"] > 1.0:
        chunks[0]["start_sec"] = 0.0

    # Ensure last chunk extends to actual end
    if chunks[-1]["end_sec"] < actual_end - 1.0:
        chunks[-1]["end_sec"] = actual_end

    # Merge too-short chunks into their neighbors
    merged = []
    for chunk in chunks:
        duration = chunk["end_sec"] - chunk["start_sec"]
        if duration < min_duration and merged:
            # Merge into the previous chunk
            prev = merged[-1]
            prev["end_sec"] = chunk["end_sec"]
            prev["title"] = f"{prev['title']} + {chunk.get('title', '')}"
            logger.info(
                f"  ⚠ Merged short chunk ({duration:.0f}s) into previous"
            )
        else:
            merged.append(chunk)

    # Warn about overly long chunks (but don't split — semantic > arbitrary)
    for chunk in merged:
        duration = chunk["end_sec"] - chunk["start_sec"]
        if duration > max_duration:
            logger.warning(
                f"  ⚠ Chunk {chunk['chunk_id']} is {duration / 60:.1f}min "
                f"(exceeds {max_duration / 60:.0f}min max)"
            )

    # Renumber after merging
    for i, c in enumerate(merged, 1):
        c["chunk_id"] = i

    return merged


def _load_cached(ctx: VideoContext) -> list[dict] | None:
    for path in (
        ctx.debug_dir / "chunk_manifest.json",
        ctx.analysis_dir / "chunk_manifest.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None


def _save_meta(ctx: VideoContext, data: dict) -> None:
    """Persist chunking execution metadata."""
    meta_path = ctx.debug_dir / "chunking_meta.json"
    meta_path.write_text(json.dumps(data, indent=2))


def _load_meta(ctx: VideoContext) -> dict | None:
    """Load chunking metadata if present."""
    meta_path = ctx.debug_dir / "chunking_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def _raise_if_strict_fallback(
    ctx: VideoContext,
    fallback_used: bool,
    fallback_reason: str | None,
) -> None:
    """Fail fast in strict mode if chunking fallback was used."""
    if not fallback_used:
        return
    if not (
        ctx.config.strict_production_checks
        and ctx.config.fail_on_chunking_fallback
    ):
        return
    reason = fallback_reason or "LLM unavailable or output not parseable"
    raise RuntimeError(
        "Strict mode: chunking fallback was used. "
        f"Fix local LLM/model and rerun. Reason: {reason}"
    )
