"""
CLI entry point for the video chunking pipeline.

Usage:
    # Full pipeline on a video directory
    python run.py --video-dir sample_1h/

    # Stop after a specific stage
    python run.py --video-dir sample_1h/ --stop-after stage2_merge

    # Skip denoising (if audio already clean)
    python run.py --video-dir sample_1h/ --skip stage0_denoise
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import PipelineConfig
from orchestrator import run_pipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with timestamps and colors."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video Chunking Pipeline — analyze livestream videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python run.py --video-dir sample_1h/

  # Test early stages only
  python run.py --video-dir sample_1h/ --stop-after stage2_merge

  # Skip denoising
  python run.py --video-dir sample_1h/ --skip stage0_denoise

  # Debug mode
  python run.py --video-dir sample_1h/ --verbose
        """,
    )
    parser.add_argument(
        "--video-dir", required=True,
        help="Path to video directory (with .mp4, audio.wav, etc.)",
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
        "--skip", action="append", default=[],
        help="Stage(s) to skip (can repeat: --skip stage0_denoise --skip stage3_scenes)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--denoise-beta", type=float, default=0.02,
        help="DeepFilterNet aggressiveness: 0=max, 0.05=gentle (default: 0.02)",
    )
    parser.add_argument(
        "--whisper-model", default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model name/path for MLX",
    )
    parser.add_argument(
        "--chunking-model", default="qwen3:30b-a3b-q4_K_M",
        help="Ollama model for semantic chunking",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help=(
            "Allow fallback outputs (single-speaker diarization / fallback chunking) "
            "instead of failing in strict mode."
        ),
    )
    parser.add_argument(
        "--skip-quality-gate",
        action="store_true",
        help="Skip final quality gate checks and quality_gate.json generation.",
    )
    parser.add_argument(
        "--retranscribe-mode",
        choices=["always", "sparse"],
        default="always",
        help=(
            "Low-quality VTT retranscription policy: "
            "'always' (recommended) or 'sparse' (only sparse low-quality chunks)."
        ),
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = PipelineConfig(
        denoise_beta=args.denoise_beta,
        whisper_model=args.whisper_model,
        chunking_model=args.chunking_model,
        strict_production_checks=not args.allow_fallbacks,
        run_quality_gate=not args.skip_quality_gate,
        retranscribe_low_quality_all_chunks=(args.retranscribe_mode == "always"),
    )

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)

    ctx = run_pipeline(
        video_dir=video_dir,
        config=config,
        skip_stages=set(args.skip),
        stop_after=args.stop_after,
    )

    print(f"\n✅ Pipeline complete. Output: {ctx.analysis_dir}")


if __name__ == "__main__":
    main()
