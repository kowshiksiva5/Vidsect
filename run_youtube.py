"""
Top-level CLI: ingest a YouTube URL and run the full pipeline.

This command adds a simple "one URL in, report out" layer:
1) Download video + metadata + subtitles via yt-dlp
2) Prepare standard inputs (video.mp4, audio.wav, transcript.vtt, metadata.json)
3) Run video_chunker pipeline

Usage:
  python run_youtube.py --url "https://www.youtube.com/watch?v=..."
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import PipelineConfig
from orchestrator import run_pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube URL and run the video chunking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot ingest + pipeline run
  python run_youtube.py --url "https://www.youtube.com/watch?v=abc123"

  # Save under a specific output root
  python run_youtube.py --url "..." --output-root playlist_downloads/youtube_jobs

  # Skip denoise and use local chunking model
  python run_youtube.py --url "..." --skip stage0_denoise --chunking-model llama3.2:latest
        """,
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output-root",
        default="playlist_downloads/youtube_jobs",
        help="Root directory where downloaded video folders will be created",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional custom folder name for this run",
    )
    parser.add_argument(
        "--stop-after",
        choices=[
            "stage0_denoise", "stage1_vtt_cleanup",
            "stage1b_transcribe", "stage1c_diarize",
            "stage2_merge", "stage3_scenes",
            "stage4_chunking", "stage5_identity",
            "stage6_per_chunk", "stage7_report",
        ],
        help="Stop after this stage completes",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Stage(s) to skip (can repeat)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument(
        "--denoise-beta",
        type=float,
        default=0.02,
        help="DeepFilterNet aggressiveness",
    )
    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model name/path for MLX",
    )
    parser.add_argument(
        "--chunking-model",
        default="qwen3:30b-a3b-q4_K_M",
        help="Ollama model for semantic chunking",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/prepare inputs, do not run pipeline",
    )
    parser.add_argument(
        "--ytdlp-format",
        default="bv*+ba/b",
        help=(
            "yt-dlp format selector. "
            "Default (bv*+ba/b) downloads highest quality video+audio."
        ),
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    ensure_cli("yt-dlp")
    ensure_cli("ffmpeg")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = fetch_video_metadata(args.url)
    video_dir = build_video_dir(output_root, metadata, args.job_name)
    video_dir.mkdir(parents=True, exist_ok=True)

    logger.info("YouTube ingest target: %s", video_dir)
    download_youtube_assets(
        url=args.url,
        video_dir=video_dir,
        format_selector=args.ytdlp_format,
    )
    prepare_standard_files(video_dir, metadata)

    logger.info("Prepared video directory: %s", video_dir)

    if args.download_only:
        print(f"\n✅ Download complete. Prepared folder: {video_dir}")
        return

    config = PipelineConfig(
        denoise_beta=args.denoise_beta,
        whisper_model=args.whisper_model,
        chunking_model=args.chunking_model,
    )

    ctx = run_pipeline(
        video_dir=video_dir,
        config=config,
        skip_stages=set(args.skip),
        stop_after=args.stop_after,
    )
    print(f"\n✅ Pipeline complete. Output: {ctx.analysis_dir}")


def ensure_cli(name: str) -> None:
    """Ensure required external CLI is available."""
    if shutil.which(name):
        return
    raise RuntimeError(
        f"Required CLI not found: {name}. Install it and rerun."
    )


def fetch_video_metadata(url: str) -> dict:
    """Fetch metadata for naming and metadata.json bootstrap."""
    cmd = ["yt-dlp", "--no-playlist", "--dump-single-json", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("yt-dlp metadata fetch failed, using URL fallback naming")
        return {"webpage_url": url, "title": "youtube_video", "id": "unknown"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Metadata JSON parse failed, using fallback metadata")
        return {"webpage_url": url, "title": "youtube_video", "id": "unknown"}


def build_video_dir(output_root: Path, metadata: dict, job_name: str | None) -> Path:
    """Build deterministic output folder path for one video."""
    if job_name:
        return output_root / sanitize_name(job_name)

    title = str(metadata.get("title") or "youtube_video")
    vid = str(metadata.get("id") or "unknown")
    folder = f"{sanitize_name(title)[:120]} [{sanitize_name(vid)}]"
    return output_root / folder


def sanitize_name(text: str) -> str:
    """Sanitize folder/file names to be filesystem-safe."""
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", text).strip() or "item"


def download_youtube_assets(
    url: str,
    video_dir: Path,
    format_selector: str,
) -> None:
    """Download video, subtitles, and info json into target folder."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*,en",
        "--convert-subs", "vtt",
        "-o", str(video_dir / "video.%(ext)s"),
        url,
    ]
    logger.info("Downloading via yt-dlp (format=%s) ...", format_selector)
    run_checked(cmd, "yt-dlp download failed")


def prepare_standard_files(video_dir: Path, fallback_metadata: dict) -> None:
    """Normalize downloaded files into pipeline-expected filenames."""
    video_path = resolve_video_path(video_dir)
    if video_path.name != "video.mp4":
        target = video_dir / "video.mp4"
        if target.exists():
            target.unlink()
        video_path = video_path.replace(target)

    extract_audio_wav(video_path, video_dir / "audio.wav")
    export_transcript_vtt(video_dir)
    export_metadata_json(video_dir, fallback_metadata)


def resolve_video_path(video_dir: Path) -> Path:
    """Find downloaded mp4 video."""
    candidates = sorted(video_dir.glob("video*.mp4"))
    if not candidates:
        candidates = sorted(video_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No mp4 found in {video_dir}")
    return candidates[0]


def extract_audio_wav(video_path: Path, audio_out: Path) -> None:
    """Extract 16k mono wav for pipeline input."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_out),
    ]
    logger.info("Extracting audio.wav ...")
    run_checked(cmd, "ffmpeg audio extraction failed")


def export_transcript_vtt(video_dir: Path) -> None:
    """Pick best downloaded English VTT and save as transcript.vtt."""
    vtts = list(video_dir.glob("*.vtt"))
    if not vtts:
        logger.warning("No .vtt subtitle found; pipeline will use Whisper transcript")
        return

    best = max(vtts, key=lambda p: subtitle_score(p))
    dst = video_dir / "transcript.vtt"
    if best.resolve() != dst.resolve():
        dst.write_text(best.read_text(encoding="utf-8", errors="ignore"))
    logger.info("Selected transcript.vtt: %s", best.name)


def subtitle_score(path: Path) -> tuple[int, int]:
    """Rank subtitle candidate by language hint and size."""
    name = path.name.lower()
    score = 0
    if ".en." in name or name.endswith(".en.vtt"):
        score += 100
    if "en-us" in name or "en_gb" in name or "en-" in name:
        score += 20
    if "live_chat" in name:
        score -= 100
    size = path.stat().st_size if path.exists() else 0
    return score, size


def export_metadata_json(video_dir: Path, fallback_metadata: dict) -> None:
    """Save metadata.json from yt-dlp info json (or fallback metadata)."""
    info_files = sorted(video_dir.glob("*.info.json"))
    data = fallback_metadata
    if info_files:
        try:
            data = json.loads(info_files[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s; using fallback metadata", info_files[0])

    out = video_dir / "metadata.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run_checked(cmd: list[str], fail_msg: str) -> None:
    """Run subprocess and raise with stderr context on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    tail = (result.stderr or result.stdout or "")[-4000:]
    raise RuntimeError(f"{fail_msg}\n{tail}")


if __name__ == "__main__":
    main()
