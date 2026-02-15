"""Pipeline configuration and data models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class PipelineConfig:
    """Global pipeline config — tune per-video if needed."""

    # --- Audio denoising ---
    denoise_beta: float = 0.02  # 0=aggressive, 0.05=gentle

    # --- Transcription ---
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    whisper_language: str = "en"

    # --- Diarization ---
    pyannote_pipeline: str = "pyannote/speaker-diarization-community-1"
    min_speakers: int = 2
    max_speakers: int = 8

    # --- VTT quality thresholds (for selective re-transcription) ---
    vtt_min_words_per_segment: int = 3
    vtt_max_music_tag_ratio: float = 0.5  # >50% [Music] → bad
    vtt_max_repetition_ratio: float = 0.6  # >60% repeated text → bad

    # --- Face tracking ---
    face_sample_interval_sec: float = 2.0  # Sample every N seconds for detections
    face_det_threshold: float = 0.5
    face_track_distance_threshold: float = 100  # Reserved for bbox-track distance
    face_embedding_distance_threshold: float = 0.5  # 1-cosine similarity

    # --- Clustering ---
    dbscan_eps: float = 0.65
    dbscan_min_samples: int = 3

    # --- Chunking ---
    chunking_model: str = "qwen3:30b-a3b-q4_K_M"
    min_chunk_duration_sec: float = 120.0   # 2 min
    max_chunk_duration_sec: float = 2700.0  # 45 min

    # --- Per-chunk processing ---
    parallel_llm_workers: int = 4  # Concurrent Ollama calls for chunk summaries

    # --- Selective Whisper re-transcription ---
    retranscribe_if_quality_below: float = 0.4  # 0-1 quality score
    retranscribe_low_quality_all_chunks: bool = True

    # --- Reliability / quality gates ---
    strict_production_checks: bool = True
    run_quality_gate: bool = True
    fail_on_diarization_fallback: bool = True
    fail_on_chunking_fallback: bool = True

    # --- Adaptive face sampling ---
    face_sample_adaptive_threshold_sec: float = 3600.0  # Double interval above this

    # --- Output ---
    output_dir_name: str = "analysis"
    debug_dir_name: str = "debug"

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> PipelineConfig:
        data = json.loads(path.read_text())
        return cls(**data)


@dataclass
class VideoContext:
    """
    Everything known about a single video being processed.

    Populated incrementally as stages complete.  Each stage reads what
    it needs and writes its outputs back here.
    """

    # --- Inputs (populated at start) ---
    video_dir: Path                        # Folder with video/audio/metadata
    video_path: Path = field(default=None)
    audio_path: Path = field(default=None)
    vtt_path: Path | None = field(default=None)
    metadata_path: Path | None = field(default=None)

    # --- Stage 0 outputs ---
    audio_clean_path: Path | None = field(default=None)

    # --- Stage 1 outputs ---
    vtt_cleaned: list[dict] | None = field(default=None)  # cleaned cues

    # --- Stage 1b outputs ---
    whisper_transcript: list[dict] | None = field(default=None)

    # --- Stage 1c outputs ---
    diarization_segments: list[dict] | None = field(default=None)

    # --- Stage 2 outputs ---
    merged_transcript: list[dict] | None = field(default=None)

    # --- Stage 3 outputs ---
    scene_boundaries: list[float] | None = field(default=None)

    # --- Stage 4 outputs ---
    chunks: list[dict] | None = field(default=None)

    # --- Stage 5 outputs ---
    face_gallery: dict | None = field(default=None)
    person_identities: dict | None = field(default=None)
    speaker_mapping: dict | None = field(default=None)

    # --- Runtime state ---
    runtime_flags: dict[str, Any] = field(default_factory=dict)

    # --- Config ---
    config: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def analysis_dir(self) -> Path:
        """Output directory for all analysis artifacts."""
        d = self.video_dir / self.config.output_dir_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def debug_dir(self) -> Path:
        """Debug/intermediate artifacts directory under analysis/."""
        d = self.analysis_dir / self.config.debug_dir_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def from_video_dir(
        cls,
        video_dir: Path,
        config: PipelineConfig | None = None,
    ) -> VideoContext:
        """Build a VideoContext from a standard video directory."""
        cfg = config or PipelineConfig()
        video_dir = Path(video_dir)

        # Find video file
        video_files = list(video_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"No .mp4 found in {video_dir}")
        video_path = video_files[0]

        # Audio
        audio_path = video_dir / "audio.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"No audio.wav in {video_dir}")

        # VTT (optional — YouTube videos have it, local recordings may not)
        vtt_path = video_dir / "transcript.vtt"
        vtt_path = vtt_path if vtt_path.exists() else None

        # Metadata
        meta_path = video_dir / "metadata.json"
        meta_path = meta_path if meta_path.exists() else None

        return cls(
            video_dir=video_dir,
            video_path=video_path,
            audio_path=audio_path,
            vtt_path=vtt_path,
            metadata_path=meta_path,
            config=cfg,
        )

    def save_checkpoint(self, stage_name: str) -> Path:
        """Save a checkpoint JSON for pipeline resumption."""
        cp_dir = self.debug_dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        cp_path = cp_dir / f"{stage_name}.json"
        cp_path.write_text(json.dumps({
            "stage": stage_name,
            "status": "complete",
            "video_dir": str(self.video_dir),
        }, indent=2))
        return cp_path

    def has_checkpoint(self, stage_name: str) -> bool:
        # New layout: analysis/debug/checkpoints
        cp_new = self.debug_dir / "checkpoints" / f"{stage_name}.json"
        if cp_new.exists():
            return True
        # Backward compatibility with older layout
        cp_old = self.analysis_dir / "checkpoints" / f"{stage_name}.json"
        return cp_old.exists()
