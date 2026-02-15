"""
Stage 0: Audio Denoising with DeepFilterNet.

Produces audio_clean.wav from audio.wav, removing wind/traffic/engine noise.
All downstream stages (Whisper, pyannote) use the cleaned audio.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage0_denoise"


def run(ctx: VideoContext) -> VideoContext:
    """Run audio denoising. Skips if already done."""
    if ctx.has_checkpoint(STAGE_NAME):
        clean_path = ctx.video_dir / "audio_clean.wav"
        if clean_path.exists():
            ctx.audio_clean_path = clean_path
            logger.info("Stage 0: skipped (checkpoint exists)")
            return ctx

    logger.info("Stage 0: Denoising audio with DeepFilterNet ...")
    t0 = time.time()

    try:
        output_path = _denoise(
            input_wav=ctx.audio_path,
            output_dir=ctx.video_dir,
            beta=ctx.config.denoise_beta,
        )
    except FileNotFoundError:
        logger.warning(
            "  deep-filter CLI not found; using original audio.wav as fallback"
        )
        output_path = ctx.audio_path

    ctx.audio_clean_path = output_path
    elapsed = time.time() - t0
    logger.info(f"Stage 0: done in {elapsed:.1f}s → {output_path}")
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


def _denoise(input_wav: Path, output_dir: Path, beta: float) -> Path:
    """
    Run the deep-filter CLI on a WAV file.

    Uses a temp subdirectory to avoid clobbering the original audio.wav
    when input and output directories are the same.
    """
    final_output = output_dir / "audio_clean.wav"

    # If final output already exists (from a previous partial run), use it
    if final_output.exists() and final_output.stat().st_size > 1000:
        logger.info("  audio_clean.wav already exists; skipping denoising")
        return final_output

    # Capture input size BEFORE any file operations
    orig_size = input_wav.stat().st_size

    # Use a temp subdirectory so DeepFilterNet can't clobber the input
    # (it writes <input_name>.wav to --output-dir)
    tmp_dir = output_dir / "_denoise_tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        "deep-filter",
        str(input_wav),
        "--output-dir", str(tmp_dir),
        "--post-filter-beta", str(beta),
        "--compensate-delay",
    ]

    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"DeepFilterNet failed:\n{result.stderr}")
        raise RuntimeError(
            f"DeepFilterNet exited with code {result.returncode}: "
            f"{result.stderr[-500:]}"
        )

    # Find the DeepFilterNet output file in temp dir
    df_output = tmp_dir / input_wav.name
    if not df_output.exists():
        # DeepFilterNet may use a different naming scheme
        candidates = list(tmp_dir.glob("*.wav"))
        if candidates:
            df_output = candidates[0]
        else:
            raise RuntimeError(
                f"DeepFilterNet produced no output in {tmp_dir}"
            )

    # Move to final location
    import shutil
    shutil.move(str(df_output), str(final_output))

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Sanity check (using captured orig_size — input file is untouched)
    clean_size = final_output.stat().st_size
    ratio = clean_size / orig_size
    logger.info(
        f"  Input: {orig_size / 1e6:.1f} MB, "
        f"Output: {clean_size / 1e6:.1f} MB (ratio: {ratio:.2f})"
    )
    if not (0.3 < ratio < 2.0):
        logger.warning(
            f"⚠ Suspicious size ratio {ratio:.2f} — denoised file may be truncated"
        )

    return final_output
