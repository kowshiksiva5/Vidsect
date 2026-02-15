#!/usr/bin/env python3
"""
Prepare a 1-hour sample from a full video for pipeline testing.

Creates a self-contained sample directory with:
  - 1-hour .mp4 clip
  - Matching audio.wav (16kHz mono)
  - Sliced transcript.vtt
  - Copied metadata.json (with updated paths)

Usage:
    python prepare_sample.py \
        --source-dir "playlist_downloads/.../01 - ..." \
        --start 0 --duration 3600 \
        --output-dir sample_1h
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_vtt_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, rest = parts
        s_parts = rest.split(".")
        s = float(s_parts[0])
        ms = float(s_parts[1]) / 1000 if len(s_parts) > 1 else 0.0
        return int(h) * 3600 + int(m) * 60 + s + ms
    elif len(parts) == 2:
        m, rest = parts
        s_parts = rest.split(".")
        s = float(s_parts[0])
        ms = float(s_parts[1]) / 1000 if len(s_parts) > 1 else 0.0
        return int(m) * 60 + s + ms
    raise ValueError(f"Bad timestamp: {ts}")


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to 'HH:MM:SS.mmm'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def slice_vtt(
    vtt_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> int:
    """
    Extract VTT cues within [start_sec, end_sec) and re-base timestamps.

    Returns the number of cues written.
    """
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = ["WEBVTT", ""]

    cue_count = 0
    i = 0

    # Skip header
    while i < len(lines) and not re.match(r"\d{2}:\d{2}:", lines[i]):
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        # Match timestamp line
        match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
            line,
        )
        if match:
            cue_start = parse_vtt_timestamp(match.group(1))
            cue_end = parse_vtt_timestamp(match.group(2))

            # Collect cue text
            i += 1
            text_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            # Check overlap with our window
            if cue_start >= end_sec:
                break
            if cue_end > start_sec and cue_start < end_sec:
                # Re-base timestamps relative to start
                new_start = max(0.0, cue_start - start_sec)
                new_end = min(end_sec - start_sec, cue_end - start_sec)
                out_lines.append(
                    f"{format_vtt_timestamp(new_start)} --> "
                    f"{format_vtt_timestamp(new_end)}"
                )
                out_lines.extend(text_lines)
                out_lines.append("")
                cue_count += 1
        i += 1

    output_path.write_text("\n".join(out_lines), encoding="utf-8")
    return cue_count


def extract_video_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    """Extract a clip with ffmpeg (fast seek + copy codec where possible)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    print(f"  Extracting video clip: {duration_sec}s from {start_sec}s ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg stderr: {result.stderr[-500:]}")
        raise RuntimeError("ffmpeg video extraction failed")
    print(f"  ✓ Video clip: {output_path.stat().st_size / 1e6:.1f} MB")


def extract_audio(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    """Extract audio as 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    print(f"  Extracting audio (16kHz mono WAV) ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    print(f"  ✓ Audio: {output_path.stat().st_size / 1e6:.1f} MB")


def prepare_sample(
    source_dir: Path,
    output_dir: Path,
    start_sec: float = 0,
    duration_sec: float = 3600,
) -> Path:
    """
    Create a self-contained sample directory for pipeline testing.

    Returns the output directory path.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_sec = start_sec + duration_sec

    # 1. Find source video
    video_files = list(source_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 in {source_dir}")
    src_video = video_files[0]
    print(f"\nSource: {src_video.name}")
    print(f"Window: {start_sec}s → {end_sec}s ({duration_sec / 60:.0f} min)")
    print(f"Output: {output_dir}\n")

    # 2. Extract video clip
    out_video = output_dir / "sample.mp4"
    extract_video_clip(src_video, out_video, start_sec, duration_sec)

    # 3. Extract audio
    out_audio = output_dir / "audio.wav"
    extract_audio(src_video, out_audio, start_sec, duration_sec)

    # 4. Slice VTT
    src_vtt = source_dir / "transcript.vtt"
    if src_vtt.exists():
        out_vtt = output_dir / "transcript.vtt"
        n_cues = slice_vtt(src_vtt, start_sec, end_sec, out_vtt)
        print(f"  ✓ VTT: {n_cues} cues extracted")
    else:
        print("  ⚠ No transcript.vtt found; pipeline will use Whisper only")

    # 5. Create metadata
    src_meta = source_dir / "metadata.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text())
    else:
        meta = {}

    meta["sample"] = {
        "source_dir": str(source_dir),
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "end_sec": end_sec,
    }
    meta["files"] = {
        "video": str(out_video),
        "audio_wav": str(out_audio),
        "transcript_vtt": str(output_dir / "transcript.vtt"),
    }

    out_meta = output_dir / "metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"  ✓ Metadata written\n")

    print("Sample ready. Run the pipeline with:")
    print(f"  python run.py --video-dir {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 1-hour sample for testing")
    parser.add_argument(
        "--source-dir", required=True,
        help="Path to the full video directory",
    )
    parser.add_argument(
        "--output-dir", default="sample_1h",
        help="Output directory for the sample (default: sample_1h)",
    )
    parser.add_argument(
        "--start", type=float, default=0,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--duration", type=float, default=3600,
        help="Duration in seconds (default: 3600 = 1 hour)",
    )
    args = parser.parse_args()

    prepare_sample(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        start_sec=args.start,
        duration_sec=args.duration,
    )


if __name__ == "__main__":
    main()
