# Vidsect Video Chunker

Local-first pipeline to ingest YouTube videos and generate structured chunk reports with:

- cleaned transcription with timestamps
- speaker diarization (who spoke when)
- face clustering and speaker-to-face mapping
- per-chunk summaries
- one consolidated JSON report (`video_report.json`)

## Repository Layout

- `run_youtube.py` - top-level YouTube URL -> full report
- `run.py` - run pipeline on an existing local video directory
- `prepare_sample.py` - create a 1-hour sample dataset
- `scripts/refresh_stage7_reports.py` - refresh only Stage 7 for existing outputs
- `stages/` - pipeline stage implementations
- `utils/` - VTT parsing and helper utilities
- `PROJECT_OVERVIEW.md` - high-level architecture and execution flow

## Prerequisites

1. Python 3.10+ (recommended: 3.11 or 3.12)
2. `ffmpeg` installed and available on `PATH`
3. (For YouTube ingest) `yt-dlp` available on `PATH`
4. (For diarization) Hugging Face token with accepted terms for:
   - `pyannote/speaker-diarization-community-1`
5. (For LLM chunking/summaries) Ollama installed and running
6. (Optional) `deep-filter` CLI for Stage 0 denoising

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Run: One YouTube URL (recommended)

```bash
python run_youtube.py \
  --url "https://www.youtube.com/watch?v=PNTCM7cbrsc" \
  --skip stage0_denoise \
  --chunking-model llama3.2:latest
```

Strict production checks are ON by default. The run fails if:

- diarization falls back to single-speaker mode
- chunking falls back to synthetic single-chunk output
- final quality gate fails

By default, YouTube download uses highest quality format selection:

- `--ytdlp-format "bv*+ba/b"` (best video + best audio, then fallback)

Default output root:

- `playlist_downloads/youtube_jobs/<video-title> [<video-id>]/`

## Run: Existing Local Video Folder

Your folder should contain:

- `video.mp4`
- `audio.wav`
- `transcript.vtt` (optional, but recommended)
- `metadata.json` (optional)

Run:

```bash
python run.py --video-dir /path/to/video_dir
```

## Output Structure

Per video, outputs are written to:

- `<video_dir>/analysis/video_report.json` (main consolidated output)
- `<video_dir>/analysis/faces/` (exported face crops grouped by person)
- `<video_dir>/analysis/debug/` (all intermediate artifacts/checkpoints)
- `<video_dir>/analysis/debug/quality_gate.json` (pass/fail quality validation)

## Pipeline Stages

1. `stage0_denoise` - audio denoising (DeepFilterNet)
2. `stage1_vtt_cleanup` - VTT cleaning/deduplication
3. `stage1b_transcribe` - full transcription (`mlx-whisper`)
4. `stage1c_diarize` - speaker diarization (`pyannote`)
5. `stage2_merge` - align transcript + diarization
6. `stage3_scenes` - scene boundaries (`scenedetect`)
7. `stage4_chunking` - semantic chunks (Ollama LLM)
8. `stage5_identity` - face detection/clustering + speaker-face map
9. `stage6_per_chunk` - chunk summaries and chunk-level artifacts
10. `stage7_report` - final consolidated JSON report

## Useful Flags

- `--skip <stage_name>` - skip one or more stages
- `--stop-after <stage_name>` - stop pipeline after a specific stage
- `--verbose` - enable debug logs
- `--download-only` - download YouTube assets and stop before analysis
- `--ytdlp-format "<selector>"` - override yt-dlp quality selector
- `--allow-fallbacks` - disable strict failure on fallback behavior
- `--skip-quality-gate` - skip final quality gate
- `--retranscribe-mode always|sparse` - low-quality retranscription policy

## Refresh Existing Reports

Use this to re-run only Stage 7 for already-processed videos (for example after
location extraction improvements):

```bash
python scripts/refresh_stage7_reports.py --root playlist_downloads --allow-fallbacks
```

## Notes

- In strict mode (default), fallback behavior becomes a hard failure.
- Use `--allow-fallbacks` for legacy/backfill runs where partial output is acceptable.
