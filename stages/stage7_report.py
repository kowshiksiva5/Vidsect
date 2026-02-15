"""
Stage 7: Final Report Assembly.

Aggregates all per-chunk outputs into a unified video analysis report.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage7_report"


def run(ctx: VideoContext) -> VideoContext:
    """Assemble the final video analysis report."""
    if ctx.has_checkpoint(STAGE_NAME):
        logger.info("Stage 7: skipped (checkpoint exists)")
        return ctx

    logger.info("Stage 7: Assembling final report ...")
    t0 = time.time()

    report = _assemble_report(ctx)
    consolidated = _assemble_consolidated_chunk_json(ctx)

    out_path = ctx.analysis_dir / "video_report.json"
    # Keep only one top-level JSON: video_report.json (rich consolidated report).
    out_path.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False))

    # Save debug variants under analysis/debug.
    legacy_path = ctx.debug_dir / "video_report_legacy.json"
    legacy_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    consolidated_path = ctx.debug_dir / "final_chunk_consolidated.json"
    consolidated_path.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False)
    )

    # Keep top-level analysis directory clean for consumption.
    _organize_top_level(ctx)

    elapsed = time.time() - t0
    logger.info(
        f"Stage 7: done in {elapsed:.1f}s → "
        f"{out_path.name}, {legacy_path.relative_to(ctx.analysis_dir)}"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _assemble_report(ctx: VideoContext) -> dict:
    """Build the unified report from all pipeline outputs."""
    report: dict = {
        "video_dir": str(ctx.video_dir),
        "video_file": ctx.video_path.name if ctx.video_path else None,
    }

    # Metadata
    if ctx.metadata_path and ctx.metadata_path.exists():
        meta = json.loads(ctx.metadata_path.read_text())
        report["metadata"] = {
            "title": meta.get("title", meta.get("fulltitle")),
            "upload_date": meta.get("upload_date"),
            "duration_sec": meta.get("duration"),
            "channel": meta.get("channel"),
        }

    # Transcript stats
    if ctx.merged_transcript:
        speakers = list(set(
            s["speaker"] for s in ctx.merged_transcript if s.get("speaker")
        ))
        total_words = sum(
            len(s.get("text", "").split()) for s in ctx.merged_transcript
        )
        report["transcript"] = {
            "total_segments": len(ctx.merged_transcript),
            "total_words": total_words,
            "speakers": speakers,
        }

    # Scene info
    if ctx.scene_boundaries:
        report["scenes"] = {
            "total_boundaries": len(ctx.scene_boundaries),
            "total_scenes": len(ctx.scene_boundaries) + 1,
        }

    # Identity pipeline
    if ctx.speaker_mapping:
        report["identities"] = {
            "speaker_mapping": ctx.speaker_mapping,
            "total_persons": len(ctx.person_identities or {}),
        }

    # Chunk summaries
    if ctx.chunks:
        report["chunks"] = []
        chunks_dir = ctx.debug_dir / "chunks"
        for chunk in ctx.chunks:
            chunk_summary = {
                "chunk_id": chunk["chunk_id"],
                "type": chunk.get("type"),
                "title": chunk.get("title"),
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "duration_min": round(
                    (chunk["end_sec"] - chunk["start_sec"]) / 60, 1
                ),
            }

            # Load per-chunk summary if available
            slug = _slugify(chunk.get("type", "unknown"))
            chunk_dir = chunks_dir / f"chunk_{chunk['chunk_id']:03d}_{slug}"
            summary_file = chunk_dir / "summary.json"
            if summary_file.exists():
                per_chunk = json.loads(summary_file.read_text())
                chunk_summary["word_count"] = per_chunk.get("word_count")
                chunk_summary["speakers"] = per_chunk.get("speakers")
                chunk_summary["vtt_quality_score"] = per_chunk.get(
                    "vtt_quality", {}
                ).get("score")
                chunk_summary["llm_summary"] = per_chunk.get("llm_summary")

            report["chunks"].append(chunk_summary)

    return report


def _assemble_consolidated_chunk_json(ctx: VideoContext) -> dict:
    """Single JSON with chunk timeline, summaries, speakers, faces, and text."""
    metadata_title = None
    if ctx.metadata_path and ctx.metadata_path.exists():
        meta = json.loads(ctx.metadata_path.read_text())
        metadata_title = meta.get("title") or meta.get("fulltitle")

    face_gallery = ctx.face_gallery or {}
    chunks_dir = ctx.debug_dir / "chunks"
    chunks_out: list[dict] = []

    if ctx.chunks:
        for chunk in ctx.chunks:
            chunk_id = chunk["chunk_id"]
            slug = _slugify(chunk.get("type", "unknown"))
            chunk_dir = chunks_dir / f"chunk_{chunk_id:03d}_{slug}"
            summary = _load_json(chunk_dir / "summary.json")
            transcript = _load_json(chunk_dir / "transcript.json")
            segments = transcript.get("segments", [])
            summary_text = summary.get("llm_summary")
            full_text = " ".join(s.get("text", "") for s in segments).strip()
            speaker_data = _build_speaker_breakdown(
                segments=segments,
                speaker_mapping=ctx.speaker_mapping or {},
                face_gallery=face_gallery,
            )
            locations = _extract_location_mentions(
                title=chunk.get("title", ""),
                summary_text=summary_text or "",
                transcript_text=full_text,
                metadata_title=metadata_title or "",
            )

            chunk_entry = {
                "chunk_id": chunk_id,
                "chunk_type": chunk.get("type"),
                "chunk_title": chunk.get("title"),
                "timeline": {
                    "start_sec": chunk["start_sec"],
                    "end_sec": chunk["end_sec"],
                    "start_hms": _format_hms(chunk["start_sec"]),
                    "end_hms": _format_hms(chunk["end_sec"]),
                    "duration_sec": round(
                        float(chunk["end_sec"]) - float(chunk["start_sec"]), 3
                    ),
                },
                "summary_text": summary_text,
                "who_spoke_what_summary": [
                    {
                        "speaker_id": s["speaker_id"],
                        "display_name": s["display_name"],
                        "total_words": s["total_words"],
                        "total_duration_sec": s["total_duration_sec"],
                        "sample_lines": [u["text"] for u in s["utterances"][:3]],
                    }
                    for s in speaker_data
                ],
                "speakers_with_faces": speaker_data,
                "chunk_transcription": {
                    "full_text": full_text,
                    "segments": segments,
                },
                "location_mentions": locations,
                "source_files": {
                    "chunk_dir": str(chunk_dir.relative_to(ctx.analysis_dir)),
                    "summary_json": str(
                        (chunk_dir / "summary.json").relative_to(ctx.analysis_dir)
                    ),
                    "people_json": str(
                        (chunk_dir / "people.json").relative_to(ctx.analysis_dir)
                    ),
                    "transcript_json": str(
                        (chunk_dir / "transcript.json").relative_to(ctx.analysis_dir)
                    ),
                },
            }
            chunks_out.append(chunk_entry)

    return {
        "video_dir": str(ctx.video_dir),
        "video_file": ctx.video_path.name if ctx.video_path else None,
        "metadata_title": metadata_title,
        "speaker_mapping": ctx.speaker_mapping or {},
        "face_gallery": face_gallery,
        "chunks": chunks_out,
    }


def _build_speaker_breakdown(
    segments: list[dict],
    speaker_mapping: dict,
    face_gallery: dict,
) -> list[dict]:
    """Aggregate chunk transcript into who-spoke-what with face links."""
    per_speaker: dict[str, dict] = {}

    for seg in segments:
        speaker_id = seg.get("speaker")
        if not speaker_id:
            continue

        info = speaker_mapping.get(speaker_id, {})
        person_id = info.get("person_id")
        face_info = face_gallery.get(person_id, {}) if person_id else {}

        entry = per_speaker.setdefault(speaker_id, {
            "speaker_id": speaker_id,
            "display_name": info.get("display_name", speaker_id),
            "person_id": person_id,
            "mapping_confidence": info.get("confidence"),
            "face_images": face_info.get("face_images", []),
            "primary_face_image": face_info.get("primary_face_image"),
            "total_words": 0,
            "total_duration_sec": 0.0,
            "utterances": [],
        })

        text = seg.get("text", "").strip()
        entry["total_words"] += len(text.split())
        entry["total_duration_sec"] += max(
            0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        )
        entry["utterances"].append({
            "start_sec": seg.get("start"),
            "end_sec": seg.get("end"),
            "text": text,
        })

    out = list(per_speaker.values())
    for item in out:
        item["total_duration_sec"] = round(item["total_duration_sec"], 3)
    out.sort(key=lambda x: x["speaker_id"])
    return out


def _extract_location_mentions(
    title: str,
    summary_text: str,
    transcript_text: str,
    metadata_title: str,
) -> list[str]:
    """Best-effort location mentions from title/summary/transcript text."""
    texts = " ".join([title, summary_text, transcript_text[:2000], metadata_title])
    mentions: set[str] = set()
    trailing_words = {
        "in", "at", "near", "from", "to", "around", "towards", "into", "on",
    }

    patterns = [
        r"\b(?:in|at|near|from|to|around|towards)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"\b([A-Z][a-z]+,\s*(?:California|Texas|Arizona|Nevada|New York|Florida))\b",
    ]
    for pat in patterns:
        for match in re.findall(pat, texts):
            m = match.strip()
            words = m.split()
            while words and words[-1].lower() in trailing_words:
                words.pop()
            m = " ".join(words).strip(", ")
            if len(m) < 3:
                continue
            if m.lower() in {"stream", "ride", "chunk"}:
                continue
            if re.search(r"[^A-Za-z,\s-]", m):
                continue
            mentions.add(m)

    return sorted(mentions)


def _format_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _organize_top_level(ctx: VideoContext) -> None:
    """
    Keep only consumer-facing assets at analysis/ top-level.

    Allowed at top level:
      - video_report.json
      - faces/
      - debug/
    Everything else is moved into analysis/debug/top_level_archive/.
    """
    import shutil

    whitelist = {"video_report.json", "faces", ctx.config.debug_dir_name}
    archive_dir = ctx.debug_dir / "top_level_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in ctx.analysis_dir.iterdir():
        if item.name in whitelist:
            continue
        target = archive_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        item.rename(target)


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30]
