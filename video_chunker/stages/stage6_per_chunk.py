"""
Stage 6: Per-Chunk Processing.

For each chunk identified in Stage 4:
  a. Slice the merged transcript by chunk boundaries
  b. Assess VTT quality for this region
  c. Selective Whisper re-transcription if quality is low
  d. Generate LLM summary via Ollama (parallelized)
  e. Persist per-chunk output files
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from video_chunker.config import VideoContext
from video_chunker.stages.stage2_merge import slice_transcript
from video_chunker.utils.vtt_parser import VTTCue, assess_vtt_quality

logger = logging.getLogger(__name__)

STAGE_NAME = "stage6_per_chunk"


CHUNK_SUMMARY_PROMPT = """You are summarizing a chunk of a Twitch hitchhiking livestream.

## Chunk Metadata
- Type: {chunk_type}
- Duration: {duration_min:.1f} minutes
- Speakers: {speakers}

## Transcript
{transcript}

## Task
Write a detailed summary (3-8 sentences) of this chunk. Include:
- What is happening (hitchhiking, waiting, riding, etc.)
- Key events, conversations, and interactions
- People mentioned or present (with names if known)
- Locations or landmarks mentioned
- Mood/atmosphere of the segment

Be specific and factual. Reference actual dialogue when relevant."""


def run(ctx: VideoContext) -> VideoContext:
    """Process each chunk: slice transcript, assess quality, summarize."""
    if ctx.has_checkpoint(STAGE_NAME):
        logger.info("Stage 6: skipped (checkpoint exists)")
        return ctx

    if not ctx.chunks:
        raise ValueError("Stage 6 requires chunks (stage 4)")
    if not ctx.merged_transcript:
        raise ValueError("Stage 6 requires merged_transcript (stage 2)")

    logger.info(f"Stage 6: Processing {len(ctx.chunks)} chunks ...")
    t0 = time.time()

    chunks_dir = ctx.debug_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # Build VTT cues list for quality assessment
    vtt_cues = _build_vtt_cues(ctx.vtt_cleaned or [])

    # Phase 1: Slice transcripts + assess quality (fast, sequential)
    chunk_inputs: list[dict] = []
    for chunk in ctx.chunks:
        prepared = _prepare_chunk(
            chunk=chunk,
            merged_transcript=ctx.merged_transcript,
            vtt_cues=vtt_cues,
            chunks_dir=chunks_dir,
            ctx=ctx,
        )
        chunk_inputs.append(prepared)

    # Phase 2: Generate LLM summaries in parallel
    workers = ctx.config.parallel_llm_workers
    logger.info(
        f"  Generating LLM summaries for {len(chunk_inputs)} chunks "
        f"with {workers} parallel workers ..."
    )
    _generate_summaries_parallel(chunk_inputs, ctx, workers)

    elapsed = time.time() - t0
    logger.info(
        f"Stage 6: done in {elapsed:.1f}s — {len(ctx.chunks)} chunks processed"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _build_vtt_cues(vtt_cleaned: list[dict]) -> list[VTTCue]:
    """Convert cached VTT dicts back to VTTCue objects for quality assessment."""
    return [
        VTTCue(
            start=c["start"],
            end=c["end"],
            text=c.get("text", ""),
            raw_text=c.get("text", ""),
            is_music=c.get("is_music", False),
        )
        for c in vtt_cleaned
    ]


def _prepare_chunk(
    chunk: dict,
    merged_transcript: list[dict],
    vtt_cues: list[VTTCue],
    chunks_dir: Path,
    ctx: VideoContext,
) -> dict:
    """
    Prepare a single chunk: slice transcript, assess quality,
    optionally re-transcribe, save transcript + people files.

    Returns a dict with everything needed for LLM summary generation.
    """
    chunk_id = chunk["chunk_id"]
    start_sec = chunk["start_sec"]
    end_sec = chunk["end_sec"]
    chunk_type = chunk.get("type", "unknown")
    chunk_title = chunk.get("title", f"Chunk {chunk_id}")
    duration_min = (end_sec - start_sec) / 60

    chunk_dir = chunks_dir / f"chunk_{chunk_id:03d}_{_slugify(chunk_type)}"
    chunk_dir.mkdir(exist_ok=True)

    logger.info(
        f"  Chunk {chunk_id:02d}: {_format_time(start_sec)} → "
        f"{_format_time(end_sec)} ({duration_min:.1f}min) [{chunk_type}]"
    )

    # 1. Slice the merged transcript
    sliced = slice_transcript(merged_transcript, start_sec, end_sec)
    transcript_source = "slice_from_merged"

    # 2. Assess VTT quality for this region
    quality = assess_vtt_quality(vtt_cues, start_sec, end_sec)
    sliced_word_count = sum(len(s.get("text", "").split()) for s in sliced)
    sparse_slice = sliced_word_count < 20
    needs_retranscribe = (
        quality.score < ctx.config.retranscribe_if_quality_below and sparse_slice
    )

    quality_info = {
        "score": quality.score,
        "issues": quality.issues,
        "needs_retranscription": needs_retranscribe,
        "text_cues": quality.text_cues,
        "music_cues": quality.music_cues,
        "sliced_word_count": sliced_word_count,
        "sparse_slice": sparse_slice,
    }

    # 3. Selective Whisper re-transcription for low-quality chunks
    if needs_retranscribe:
        logger.info(
            f"    ⚠ Low VTT quality ({quality.score:.2f}) — "
            f"re-transcribing [{', '.join(quality.issues)}]"
        )
        reference_with_speakers = sliced
        retranscribed = _selective_retranscribe(
            audio_path=ctx.audio_clean_path or ctx.audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            whisper_model=ctx.config.whisper_model,
            whisper_language=ctx.config.whisper_language,
        )
        if retranscribed:
            sliced = _assign_speakers_from_reference(
                retranscribed,
                reference_with_speakers,
            )
            transcript_source = "selective_whisper_retranscription"
            quality_info["retranscription_status"] = "completed"
            logger.info(f"    ✓ Re-transcription done: {len(sliced)} segments")
        else:
            quality_info["retranscription_status"] = "failed_using_sliced"
            logger.warning(
                "    ✗ Re-transcription failed; using sliced transcript"
            )
    else:
        if (
            quality.score < ctx.config.retranscribe_if_quality_below
            and not sparse_slice
        ):
            logger.info(
                "    ✓ Re-transcription skipped: low VTT quality (%.2f) but "
                "sliced transcript is dense (%d words)",
                quality.score,
                sliced_word_count,
            )
            quality_info["retranscription_status"] = "skipped_sliced_dense"
        else:
            logger.info(f"    ✓ VTT quality OK ({quality.score:.2f})")

    # Persist transcript
    transcript_path = chunk_dir / "transcript.json"
    transcript_path.write_text(json.dumps({
        "chunk_id": chunk_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "source": transcript_source,
        "segments": sliced,
    }, indent=2, ensure_ascii=False))

    # Speaker and text stats
    full_text = " ".join(s["text"] for s in sliced if s.get("text"))
    speakers_in_chunk = sorted(set(
        s["speaker"] for s in sliced if s.get("speaker")
    ))
    word_count = len(full_text.split())

    # People file
    people_data = {
        "chunk_id": chunk_id,
        "speakers": speakers_in_chunk,
        "speaker_segments": _per_speaker_segments(sliced),
    }
    if ctx.speaker_mapping:
        people_data["identified_speakers"] = {
            spk: ctx.speaker_mapping.get(spk, {"display_name": spk})
            for spk in speakers_in_chunk
        }
    people_path = chunk_dir / "people.json"
    people_path.write_text(json.dumps(people_data, indent=2))

    # Emotions placeholder
    emotions_path = chunk_dir / "emotions.json"
    emotions_path.write_text(json.dumps({
        "chunk_id": chunk_id,
        "status": "not_implemented",
        "note": "HSEmotion integration pending",
    }, indent=2))

    return {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "chunk_title": chunk_title,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_min": duration_min,
        "sliced": sliced,
        "speakers_in_chunk": speakers_in_chunk,
        "word_count": word_count,
        "quality_info": quality_info,
        "chunk_dir": chunk_dir,
        "full_text": full_text,
    }


def _generate_summaries_parallel(
    chunk_inputs: list[dict],
    ctx: VideoContext,
    max_workers: int,
) -> None:
    """Generate and persist LLM summaries for all chunks using a thread pool."""

    def _summarize_and_save(ci: dict) -> None:
        llm_summary = _generate_llm_summary(
            sliced=ci["sliced"],
            chunk_type=ci["chunk_type"],
            duration_min=ci["duration_min"],
            speakers=ci["speakers_in_chunk"],
            ctx=ctx,
        )

        summary = {
            "chunk_id": ci["chunk_id"],
            "type": ci["chunk_type"],
            "title": ci["chunk_title"],
            "start_sec": ci["start_sec"],
            "end_sec": ci["end_sec"],
            "duration_min": round(ci["duration_min"], 1),
            "word_count": ci["word_count"],
            "speakers": ci["speakers_in_chunk"],
            "vtt_quality": ci["quality_info"],
            "llm_summary": llm_summary,
            "text_preview": (
                ci["full_text"][:500]
                + ("..." if len(ci["full_text"]) > 500 else "")
            ),
        }

        summary_path = ci["chunk_dir"] / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

    # Use ThreadPoolExecutor (Ollama calls are I/O-bound)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_summarize_and_save, ci): ci["chunk_id"]
            for ci in chunk_inputs
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.warning(f"    Chunk {cid} summary failed: {exc}")


# ---------------------------------------------------------------------------
# Selective re-transcription
# ---------------------------------------------------------------------------

def _selective_retranscribe(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    whisper_model: str,
    whisper_language: str,
) -> list[dict] | None:
    """
    Re-transcribe a specific time range using mlx-whisper.

    Extracts the audio segment with ffmpeg, then transcribes it.
    Returns segments list or None on failure.
    """
    import subprocess
    import tempfile

    try:
        import mlx_whisper
    except ImportError:
        logger.warning("    mlx-whisper not available for re-transcription")
        return None

    duration = end_sec - start_sec
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(audio_path),
            "-t", str(duration),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                f"    ffmpeg segment extraction failed: "
                f"{result.stderr[-200:]}"
            )
            return None

        whisper_result = mlx_whisper.transcribe(
            str(tmp_path),
            path_or_hf_repo=whisper_model,
            language=whisper_language,
            word_timestamps=True,
            fp16=True,
        )

        # Re-base timestamps to full video time
        segments = []
        for seg in whisper_result.get("segments", []):
            segment_data = {
                "start": seg["start"] + start_sec,
                "end": seg["end"] + start_sec,
                "text": seg["text"].strip(),
                "speaker": None,
            }
            if "words" in seg:
                segment_data["words"] = [
                    {
                        "word": w["word"].strip(),
                        "start": w["start"] + start_sec,
                        "end": w["end"] + start_sec,
                    }
                    for w in seg["words"]
                ]
            segments.append(segment_data)

        return segments

    except Exception as e:
        logger.warning(f"    Re-transcription failed: {e}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _assign_speakers_from_reference(
    retranscribed: list[dict],
    reference: list[dict],
) -> list[dict]:
    """Project speaker labels from reference transcript onto retranscribed."""
    if not reference:
        return retranscribed

    assigned: list[dict] = []
    for seg in retranscribed:
        speaker = _dominant_speaker(seg["start"], seg["end"], reference)
        seg_out = {**seg, "speaker": speaker}
        if "words" in seg_out:
            seg_out["words"] = [
                {**w, "speaker": speaker} for w in seg_out["words"]
            ]
        assigned.append(seg_out)
    return assigned


def _dominant_speaker(
    start: float, end: float, reference: list[dict],
) -> str | None:
    """Pick the speaker with highest overlap against reference segments."""
    if end <= start:
        return None

    overlaps: dict[str, float] = {}
    for seg in reference:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        overlap_start = max(start, seg["start"])
        overlap_end = min(end, seg["end"])
        overlap = overlap_end - overlap_start
        if overlap > 0:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

    if not overlaps:
        return None
    return max(overlaps, key=overlaps.get)


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

def _generate_llm_summary(
    sliced: list[dict],
    chunk_type: str,
    duration_min: float,
    speakers: list[str],
    ctx: VideoContext,
) -> str | None:
    """Generate an LLM summary for the chunk using Ollama."""
    if not sliced:
        return None

    try:
        import ollama
    except ImportError:
        logger.info("    ollama not installed — skipping LLM summary")
        return None

    # Format transcript for the prompt
    transcript_lines: list[str] = []
    for seg in sliced:
        spk = seg.get("speaker", "???")
        if ctx.speaker_mapping and spk in ctx.speaker_mapping:
            display = ctx.speaker_mapping[spk].get("display_name", spk)
        else:
            display = spk
        transcript_lines.append(f"[{display}] {seg.get('text', '')}")

    transcript_text = "\n".join(transcript_lines)

    # Truncate very long transcripts to fit context window
    max_chars = 12000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n... (truncated)"

    prompt = CHUNK_SUMMARY_PROMPT.format(
        chunk_type=chunk_type,
        duration_min=duration_min,
        speakers=", ".join(speakers) if speakers else "Unknown",
        transcript=transcript_text,
    )

    try:
        response = ollama.chat(
            model=ctx.config.chunking_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 512},
        )
        summary_text = response["message"]["content"].strip()
        logger.info(
            f"    ✓ Chunk {chunk_type} summary: {len(summary_text)} chars"
        )
        return summary_text
    except Exception as e:
        logger.warning(f"    LLM summary failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_speaker_segments(sliced: list[dict]) -> dict:
    """Calculate per-speaker stats from sliced transcript."""
    speaker_stats: dict[str, dict] = {}
    for seg in sliced:
        spk = seg.get("speaker")
        if not spk:
            continue
        if spk not in speaker_stats:
            speaker_stats[spk] = {
                "total_duration": 0.0,
                "word_count": 0,
                "first_appearance": seg["start"],
                "last_appearance": seg["end"],
            }
        stats = speaker_stats[spk]
        stats["total_duration"] += seg["end"] - seg["start"]
        stats["word_count"] += len(seg.get("text", "").split())
        stats["last_appearance"] = max(stats["last_appearance"], seg["end"])

    for stats in speaker_stats.values():
        stats["total_duration"] = round(stats["total_duration"], 1)

    return speaker_stats


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30]


def _format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
