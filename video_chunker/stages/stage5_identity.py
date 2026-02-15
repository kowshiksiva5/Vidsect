"""
Stage 5: Person Identity Pipeline.

Detects faces, tracks them continuously across frames with Norfair,
clusters track embeddings via DBSCAN into person identities,
maps speakers to face clusters with 1-second bin fusion,
and infers names from transcript.

Sub-stages:
  5a. Face detection + continuous Norfair tracking (InsightFace + Norfair)
  5b. Face clustering (DBSCAN on track-level mean embeddings)
  5c. Speaker-face mapping (1-second bin_fusion_exclusive)
  5d. Name inference (regex from transcript)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from video_chunker.config import VideoContext

logger = logging.getLogger(__name__)

STAGE_NAME = "stage5_identity"

# Speaker-face bin fusion scoring constants
AMBIGUOUS_WEIGHT = 0.3
MIN_CONFIDENCE = 0.35
MIN_MARGIN = 0.15


def run(ctx: VideoContext) -> VideoContext:
    """Run the person identity pipeline."""
    if ctx.has_checkpoint(STAGE_NAME):
        cached = _load_cached(ctx)
        if cached is not None:
            ctx.face_gallery = cached.get("face_gallery")
            ctx.person_identities = cached.get("person_identities")
            ctx.speaker_mapping = cached.get("speaker_mapping")
            logger.info("Stage 5: skipped (checkpoint exists)")
            return ctx

    if not ctx.merged_transcript:
        raise ValueError("Stage 5 requires merged_transcript (stage 2)")
    if not ctx.diarization_segments:
        raise ValueError("Stage 5 requires diarization_segments (stage 1c)")

    logger.info("Stage 5: Running Person Identity Pipeline ...")
    t0 = time.time()

    # 5a: Face detection + continuous Norfair tracking
    face_tracks = _build_face_tracks(ctx)

    # 5b: DBSCAN clustering on track-level mean embeddings
    persons = _cluster_face_tracks(face_tracks, ctx)

    # 5c: 1-second bin fusion speaker-face mapping
    video_duration = _get_video_duration(ctx.video_path)
    speaker_mapping = _build_speaker_face_fusion(
        speaker_segments=ctx.diarization_segments,
        persons=persons,
        video_duration=video_duration,
    )

    # 5d: Name inference from transcript
    speaker_mapping = _infer_names(
        speaker_mapping=speaker_mapping,
        transcript=ctx.merged_transcript,
    )

    # Export representative face crops for each person ID.
    face_exports = _export_face_gallery_images(
        video_path=ctx.video_path,
        persons=persons,
        faces_root=ctx.analysis_dir / "faces",
    )

    # Build face gallery for output
    face_gallery = _build_face_gallery(persons, face_exports)

    # Build person identities dict
    person_identities = _build_person_identities(persons)

    # Store results
    ctx.face_gallery = face_gallery
    ctx.person_identities = person_identities
    ctx.speaker_mapping = speaker_mapping

    # Persist
    out_path = ctx.debug_dir / "identity_pipeline.json"
    out_path.write_text(json.dumps({
        "face_gallery": face_gallery,
        "person_identities": person_identities,
        "speaker_mapping": speaker_mapping,
    }, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    logger.info(
        f"Stage 5: done in {elapsed:.1f}s — "
        f"{len(person_identities)} identities, "
        f"{len(speaker_mapping)} speaker mappings"
    )
    ctx.save_checkpoint(STAGE_NAME)
    return ctx


# ---------------------------------------------------------------------------
# 5a: Face detection + continuous Norfair tracking
# ---------------------------------------------------------------------------

def _build_face_tracks(ctx: VideoContext) -> list[dict]:
    """
    Detect faces with InsightFace, link them into continuous tracks via Norfair.

    Uses OpenCV VideoCapture for batch frame reading (single video open).
    Norfair maintains tracker state across frames, linking detections into
    continuous face tracks based on bbox-center distance.

    Returns a list of face tracks, each with:
      - track_id: int
      - detections: [{timestamp, bbox, det_score}]
      - mean_embedding: list[float] (512-dim, averaged over track)
      - start_sec, end_sec: float
      - detection_count: int
    """
    logger.info("  5a: Face detection + Norfair tracking ...")

    import warnings

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.warning(
            "  insightface not installed — skipping face detection. "
            "Run: pip install insightface onnxruntime"
        )
        return []

    try:
        import norfair
    except ImportError:
        logger.warning(
            "  norfair not installed — falling back to embedding-only tracker. "
            "Run: pip install norfair"
        )
        return _fallback_embedding_tracker(ctx)

    try:
        import cv2
    except ImportError:
        logger.warning(
            "  cv2 not installed — cannot extract frames. "
            "Run: pip install opencv-python"
        )
        return []

    import numpy as np

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"insightface\.utils\.face_align",
    )

    # Initialize InsightFace with best available provider
    providers = _get_onnx_providers()
    logger.info(f"  InsightFace providers: {providers}")

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Initialize Norfair tracker
    tracker = norfair.Tracker(
        distance_function="euclidean",
        distance_threshold=ctx.config.face_track_distance_threshold,
        hit_counter_max=15,      # Keep track alive for 15 missed frames
        initialization_delay=2,  # Require 2 consecutive detections
    )

    # Open video
    cap = cv2.VideoCapture(str(ctx.video_path))
    if not cap.isOpened():
        logger.error(f"  Failed to open video: {ctx.video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # Compute frame sampling interval
    sample_interval = ctx.config.face_sample_interval_sec
    threshold = ctx.config.face_sample_adaptive_threshold_sec
    if duration_sec > threshold:
        sample_interval *= 2
        logger.info(
            f"  Adaptive sampling: {duration_sec:.0f}s video → "
            f"interval {sample_interval:.1f}s"
        )

    frame_interval = max(1, int(fps * sample_interval))
    expected_frames = total_frames // frame_interval
    logger.info(
        f"  Video: {duration_sec:.0f}s, {fps:.1f}fps, "
        f"sampling every {frame_interval} frames "
        f"(~{expected_frames} samples) ..."
    )

    det_threshold = ctx.config.face_det_threshold

    # Process frames sequentially with Norfair tracking
    raw_tracks: dict[int, list[dict]] = {}  # track_id → detections
    track_embeddings: dict[int, list] = {}  # track_id → [embeddings]
    frame_idx = 0
    sampled_count = 0

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        sampled_count += 1

        # Resize for detection consistency + speed
        frame_resized = cv2.resize(frame, (640, 360))

        faces = app.get(frame_resized)

        # Convert InsightFace detections to Norfair format
        norfair_dets = []
        face_extras = {}  # id(det) → metadata dict

        for face in faces:
            if face.det_score < det_threshold:
                continue

            bbox = face.bbox.astype(int)
            center = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            )
            det = norfair.Detection(points=center.reshape(1, 2))

            # Store metadata keyed by detection object id
            face_extras[id(det)] = {
                "embedding": face.embedding,
                "det_score": float(face.det_score),
                "bbox": bbox.tolist(),
                "timestamp": timestamp,
            }
            norfair_dets.append(det)

        # Update Norfair tracker — links detections to existing tracks
        tracked_objects = tracker.update(detections=norfair_dets)

        for obj in tracked_objects:
            last_det = obj.last_detection
            if last_det is None:
                continue

            extra = face_extras.get(id(last_det))
            if extra is None:
                continue

            tid = obj.id
            if tid not in raw_tracks:
                raw_tracks[tid] = []
                track_embeddings[tid] = []

            raw_tracks[tid].append({
                "timestamp": extra["timestamp"],
                "bbox": extra["bbox"],
                "det_score": extra["det_score"],
            })
            track_embeddings[tid].append(extra["embedding"])

        if sampled_count % 120 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"    Processed {sampled_count}/{expected_frames} frames "
                f"({elapsed:.1f}s, "
                f"{len(raw_tracks)} tracks so far)"
            )

        frame_idx += 1

    cap.release()

    elapsed = time.time() - t_start
    logger.info(
        f"    Detection+tracking done: {sampled_count} frames, "
        f"{len(raw_tracks)} raw tracks in {elapsed:.1f}s"
    )

    # Build final track objects with mean embeddings
    min_detections = 3  # Filter very short tracks (noise)
    tracks = []
    for tid, dets in raw_tracks.items():
        if len(dets) < min_detections:
            continue

        embeddings = np.stack(track_embeddings[tid])
        mean_emb = embeddings.mean(axis=0)

        tracks.append({
            "track_id": tid,
            "detections": dets,
            "mean_embedding": mean_emb.tolist(),
            "start_sec": min(d["timestamp"] for d in dets),
            "end_sec": max(d["timestamp"] for d in dets),
            "detection_count": len(dets),
        })

    logger.info(
        f"    {len(tracks)} tracks after filtering "
        f"(>= {min_detections} detections)"
    )
    return tracks


# ---------------------------------------------------------------------------
# 5b: Face clustering (DBSCAN on track-level mean embeddings)
# ---------------------------------------------------------------------------

def _cluster_face_tracks(
    tracks: list[dict],
    ctx: VideoContext,
) -> dict[str, dict]:
    """
    Cluster face tracks by identity using DBSCAN on mean embeddings.

    Each track already represents a continuous presence of one person,
    so min_samples=1 is appropriate — a single track is sufficient
    to consider a person real.

    Returns {person_id: {tracks, detection_count, first_seen, last_seen}}.
    """
    logger.info("  5b: DBSCAN face clustering ...")

    if not tracks:
        logger.info("    No tracks to cluster")
        return {}

    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning(
            "  sklearn not installed — using 1:1 track→person mapping. "
            "Run: pip install scikit-learn"
        )
        return _fallback_cluster(tracks)

    embeddings = np.stack([t["mean_embedding"] for t in tracks])

    clustering = DBSCAN(
        eps=ctx.config.dbscan_eps,
        min_samples=1,  # Each track is already vetted
        metric="cosine",
    )
    labels = clustering.fit_predict(embeddings)

    persons: dict[str, dict] = {}
    noise_count = 0

    for label, track in zip(labels, tracks):
        if label == -1:
            noise_count += 1
            continue
        pid = f"person_{label:03d}"
        if pid not in persons:
            persons[pid] = {
                "tracks": [],
                "detection_count": 0,
                "first_seen": float("inf"),
                "last_seen": 0.0,
            }
        persons[pid]["tracks"].append(track)
        persons[pid]["detection_count"] += track["detection_count"]
        persons[pid]["first_seen"] = min(
            persons[pid]["first_seen"], track["start_sec"]
        )
        persons[pid]["last_seen"] = max(
            persons[pid]["last_seen"], track["end_sec"]
        )

    logger.info(
        f"    {len(persons)} persons from {len(tracks)} tracks "
        f"({noise_count} noise tracks discarded)"
    )
    return persons


def _fallback_cluster(tracks: list[dict]) -> dict[str, dict]:
    """Fallback 1:1 track→person mapping when sklearn is unavailable."""
    persons: dict[str, dict] = {}
    for i, track in enumerate(tracks):
        pid = f"person_{i:03d}"
        persons[pid] = {
            "tracks": [track],
            "detection_count": track["detection_count"],
            "first_seen": track["start_sec"],
            "last_seen": track["end_sec"],
        }
    return persons


# ---------------------------------------------------------------------------
# 5c: Speaker-face mapping (1-second bin fusion)
# ---------------------------------------------------------------------------

def _build_speaker_face_fusion(
    speaker_segments: list[dict],
    persons: dict[str, dict],
    video_duration: float,
) -> dict:
    """
    1-second bin speaker-face fusion.

    For each second of video:
      1. Determine which speaker is active (at most one)
      2. Determine which person IDs are visible (0, 1, or many)
      3. Classify: EXCLUSIVE (1 face) / AMBIGUOUS (many) / OFF_CAMERA (0)
      4. Accumulate evidence for (speaker, person) pairs

    Exclusive bins count 1.0, ambiguous bins count AMBIGUOUS_WEIGHT (0.3).
    """
    logger.info("  5c: Speaker-face mapping (1s bin_fusion_exclusive) ...")

    total_seconds = int(video_duration)
    if total_seconds == 0:
        return _empty_speaker_mapping(speaker_segments)

    # Step 1: Build per-second speaker label array
    speaker_at: list[str | None] = [None] * total_seconds
    for seg in speaker_segments:
        for t in range(int(seg["start"]), min(int(seg["end"]), total_seconds)):
            speaker_at[t] = seg["speaker"]

    # Step 2: Build per-second active person IDs
    # A person is "active" at second t if any detection is within ±1s
    tracks_at: list[set[str]] = [set() for _ in range(total_seconds)]
    for pid, person in persons.items():
        for track in person["tracks"]:
            for det in track["detections"]:
                t = int(det["timestamp"])
                for dt in range(max(0, t - 1), min(total_seconds, t + 2)):
                    tracks_at[dt].add(pid)

    # Step 3: Classify each second and accumulate bin evidence
    # evidence[(speaker, person)] = {exclusive: count, ambiguous: count}
    evidence: dict[tuple[str, str], dict[str, int]] = {}
    speaker_stats: dict[str, dict[str, int]] = {}

    for t in range(total_seconds):
        spk = speaker_at[t]
        if spk is None:
            continue  # SILENT bin

        active = tracks_at[t]

        if spk not in speaker_stats:
            speaker_stats[spk] = {
                "total": 0, "exclusive": 0, "ambiguous": 0, "off_camera": 0,
            }
        speaker_stats[spk]["total"] += 1

        if len(active) == 0:
            # OFF_CAMERA: speaker but no face visible
            speaker_stats[spk]["off_camera"] += 1

        elif len(active) == 1:
            # EXCLUSIVE: one face, one speaker → strong evidence
            pid = next(iter(active))
            key = (spk, pid)
            if key not in evidence:
                evidence[key] = {"exclusive": 0, "ambiguous": 0}
            evidence[key]["exclusive"] += 1
            speaker_stats[spk]["exclusive"] += 1

        else:
            # AMBIGUOUS: multiple faces, can't tell who is speaking
            for pid in active:
                key = (spk, pid)
                if key not in evidence:
                    evidence[key] = {"exclusive": 0, "ambiguous": 0}
                evidence[key]["ambiguous"] += 1
            speaker_stats[spk]["ambiguous"] += 1

    # Step 4: Score each speaker → best person mapping
    mapping: dict[str, dict] = {}
    unique_speakers = set(speaker_at) - {None}

    for spk in unique_speakers:
        stats = speaker_stats.get(spk, {})
        total_sec = stats.get("total", 0)
        if total_sec < 10:
            mapping[spk] = {
                "person_id": None,
                "display_name": spk,
                "confidence": 0.0,
                "method": "insufficient_data",
            }
            continue

        # Score each candidate person for this speaker
        candidates: list[tuple[str, float, dict]] = []
        for (s, pid), ev in evidence.items():
            if s != spk:
                continue
            score = (
                ev["exclusive"] + ev["ambiguous"] * AMBIGUOUS_WEIGHT
            ) / total_sec
            candidates.append((pid, score, ev))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            off_pct = stats["off_camera"] / total_sec
            mapping[spk] = {
                "person_id": None,
                "display_name": spk,
                "confidence": 0.0,
                "method": "off_camera_speaker",
                "off_camera_pct": round(off_pct, 3),
            }
            continue

        best_pid, best_score, best_ev = candidates[0]
        runner_up_score = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = best_score - runner_up_score

        if best_score >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
            mapping[spk] = {
                "person_id": best_pid,
                "display_name": best_pid,
                "confidence": round(best_score, 3),
                "margin": round(margin, 3),
                "exclusive_seconds": best_ev["exclusive"],
                "ambiguous_seconds": best_ev["ambiguous"],
                "off_camera_pct": round(
                    stats["off_camera"] / total_sec, 3
                ),
                "method": "bin_fusion_exclusive",
            }
        else:
            mapping[spk] = {
                "person_id": best_pid,
                "display_name": spk,
                "confidence": round(best_score, 3),
                "margin": round(margin, 3),
                "method": "NEEDS_HUMAN_VERIFICATION",
                "reason": f"score={best_score:.2f}, margin={margin:.2f}",
            }

    matched = sum(
        1 for m in mapping.values()
        if m.get("method") == "bin_fusion_exclusive"
    )
    logger.info(
        f"    Mapped {matched}/{len(mapping)} speakers to faces "
        f"(bin_fusion_exclusive)"
    )
    return mapping


def _empty_speaker_mapping(speaker_segments: list[dict]) -> dict:
    """Return empty mapping when no video duration is available."""
    speakers = set(s["speaker"] for s in speaker_segments)
    return {
        spk: {
            "person_id": None,
            "display_name": spk,
            "confidence": 0.0,
            "method": "no_video_duration",
        }
        for spk in speakers
    }


# ---------------------------------------------------------------------------
# 5d: Name inference
# ---------------------------------------------------------------------------

def _infer_names(
    speaker_mapping: dict,
    transcript: list[dict],
) -> dict:
    """
    Infer person names from transcript using regex patterns.

    Looks for self-introductions: "I'm Mike", "my name is Sarah", etc.
    """
    import re
    logger.info("  5d: Name inference from transcript ...")

    name_patterns = [
        r"(?:I'm|I am|my name is|they call me|call me)\s+([A-Z][a-z]+)",
        r"(?:thanks|thank you),?\s+([A-Z][a-z]+)",
        r"(?:nice to meet you),?\s+([A-Z][a-z]+)",
    ]

    for seg in transcript:
        spk = seg.get("speaker")
        if not spk or spk not in speaker_mapping:
            continue

        text = seg.get("text", "")
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                current = speaker_mapping[spk]
                if current["confidence"] < 0.8:
                    current["display_name"] = name
                    current["confidence"] = max(current["confidence"], 0.5)
                    logger.info(f"    Found name '{name}' for {spk}")
                break

    return speaker_mapping


# ---------------------------------------------------------------------------
# Fallback tracker (when Norfair is not installed)
# ---------------------------------------------------------------------------

def _fallback_embedding_tracker(ctx: VideoContext) -> list[dict]:
    """
    Fallback: detect faces + group by embedding similarity.

    Used when Norfair is not installed. No spatial tracking, no motion model.
    Quality will be lower than Norfair-based tracking.
    """
    logger.info("  5a (fallback): Embedding-only face grouping ...")

    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.warning("  Missing dependencies for fallback tracker")
        return []

    import warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"insightface\.utils\.face_align",
    )

    providers = _get_onnx_providers()
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    frame_times = _get_sample_frame_times(ctx)
    if not frame_times:
        return []

    # Batch extract + detect via OpenCV
    cap = cv2.VideoCapture(str(ctx.video_path))
    if not cap.isOpened():
        return []

    det_threshold = ctx.config.face_det_threshold
    track_threshold = ctx.config.face_embedding_distance_threshold
    detections: list[dict] = []

    for i, t in enumerate(frame_times, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_resized = cv2.resize(frame, (640, 360))
        faces = app.get(frame_resized)
        for face in faces:
            if face.det_score < det_threshold:
                continue
            detections.append({
                "timestamp": t,
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score),
            })
        if i % 120 == 0:
            logger.info(f"    Processed {i}/{len(frame_times)} frames")

    cap.release()
    logger.info(f"    {len(detections)} detections from {len(frame_times)} frames")

    # Group by embedding similarity (incremental mean)
    tracks: list[dict] = []
    for det in sorted(detections, key=lambda d: d["timestamp"]):
        emb = np.array(det["embedding"])
        matched = False
        for track in tracks:
            track_mean = track["_emb_sum"] / track["_emb_count"]
            distance = 1.0 - float(
                np.dot(emb, track_mean)
                / (np.linalg.norm(emb) * np.linalg.norm(track_mean) + 1e-9)
            )
            if distance <= track_threshold:
                track["detections"].append({
                    "timestamp": det["timestamp"],
                    "bbox": det["bbox"],
                    "det_score": det["det_score"],
                })
                track["_emb_sum"] += emb
                track["_emb_count"] += 1
                matched = True
                break

        if not matched:
            tracks.append({
                "track_id": len(tracks),
                "detections": [{
                    "timestamp": det["timestamp"],
                    "bbox": det["bbox"],
                    "det_score": det["det_score"],
                }],
                "_emb_sum": emb.copy(),
                "_emb_count": 1,
            })

    # Finalize
    result = []
    for track in tracks:
        if len(track["detections"]) < 3:
            continue
        mean_emb = (track["_emb_sum"] / track["_emb_count"]).tolist()
        dets = track["detections"]
        result.append({
            "track_id": track["track_id"],
            "detections": dets,
            "mean_embedding": mean_emb,
            "start_sec": min(d["timestamp"] for d in dets),
            "end_sec": max(d["timestamp"] for d in dets),
            "detection_count": len(dets),
        })

    logger.info(f"    {len(result)} tracks after filtering")
    return result


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_onnx_providers() -> list[str]:
    """Pick the best ONNX Runtime providers available on this machine."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    preferred_order = [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in preferred_order if p in available]
    return providers or ["CPUExecutionProvider"]


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        logger.warning("  Could not determine video duration via ffprobe")
        return 0.0


def _get_sample_frame_times(ctx: VideoContext) -> list[float]:
    """Get frame sample times, doubling interval for long videos."""
    duration = _get_video_duration(ctx.video_path)
    if duration <= 0:
        return []

    interval = ctx.config.face_sample_interval_sec
    threshold = ctx.config.face_sample_adaptive_threshold_sec
    if duration > threshold:
        interval *= 2
        logger.info(
            f"  Adaptive sampling: {duration:.0f}s video → "
            f"interval {interval:.1f}s"
        )

    times: list[float] = []
    t = 0.0
    while t < duration:
        times.append(round(t, 3))
        t += interval
    return times


def _build_face_gallery(
    persons: dict[str, dict],
    face_exports: dict[str, dict] | None = None,
) -> dict:
    """Build serializable face gallery from persons dict."""
    face_exports = face_exports or {}
    gallery = {}
    for pid, person in persons.items():
        export = face_exports.get(pid, {})
        gallery[pid] = {
            "num_tracks": len(person["tracks"]),
            "detection_count": person["detection_count"],
            "first_seen": person["first_seen"],
            "last_seen": person["last_seen"],
            "face_images": export.get("image_paths", []),
            "primary_face_image": export.get("primary_image"),
        }
    return gallery


def _build_person_identities(persons: dict[str, dict]) -> dict:
    """Build person identity metadata."""
    identities = {}
    for pid, person in persons.items():
        identities[pid] = {
            "display_name": f"Person {pid.split('_')[1]}",
            "confidence": 0.0,
            "source": "dbscan_cluster",
            "num_tracks": len(person["tracks"]),
            "detection_count": person["detection_count"],
            "first_seen": person["first_seen"],
            "last_seen": person["last_seen"],
        }
    return identities


def _export_face_gallery_images(
    video_path: Path,
    persons: dict[str, dict],
    faces_root: Path,
    max_images_per_person: int = 5,
) -> dict[str, dict]:
    """
    Save representative face crops under analysis/faces/<person_id>/.

    Returns:
      {person_id: {"image_paths": [...], "primary_image": "..."}}
    """
    try:
        import cv2
    except ImportError:
        logger.warning("  cv2 not installed — skipping face image export")
        return {}

    if not persons:
        return {}

    faces_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  Could not open video for face export: %s", video_path)
        return {}

    exports: dict[str, dict] = {}
    try:
        for pid, person in persons.items():
            detections = _collect_person_detections(person)
            selected = _select_representative_detections(
                detections,
                max_images_per_person,
            )
            if not selected:
                continue

            person_dir = faces_root / pid
            person_dir.mkdir(parents=True, exist_ok=True)
            image_paths: list[str] = []

            for i, det in enumerate(selected, start=1):
                crop = _extract_face_crop(
                    cap=cap,
                    timestamp=float(det["timestamp"]),
                    bbox=det["bbox"],
                )
                if crop is None:
                    continue
                out_path = person_dir / f"face_{i:02d}.jpg"
                if cv2.imwrite(str(out_path), crop):
                    rel = out_path.relative_to(faces_root.parent)
                    image_paths.append(str(rel))

            if image_paths:
                exports[pid] = {
                    "image_paths": image_paths,
                    "primary_image": image_paths[0],
                }
    finally:
        cap.release()

    total = sum(len(v["image_paths"]) for v in exports.values())
    logger.info("  Exported %d face crops across %d persons", total, len(exports))
    return exports


def _collect_person_detections(person: dict) -> list[dict]:
    """Flatten detections from all tracks for one clustered person."""
    detections: list[dict] = []
    for track in person.get("tracks", []):
        for det in track.get("detections", []):
            if "timestamp" not in det or "bbox" not in det:
                continue
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            detections.append({
                "timestamp": float(det["timestamp"]),
                "bbox": det["bbox"],
                "det_score": float(det.get("det_score", 0.0)),
                "area": area,
            })
    return detections


def _select_representative_detections(
    detections: list[dict],
    max_count: int,
) -> list[dict]:
    """Pick high-confidence detections with temporal spread."""
    if not detections or max_count <= 0:
        return []

    ranked = sorted(
        detections,
        key=lambda d: (
            -(d.get("area", 0.0) * max(d.get("det_score", 0.0), 0.01)),
            -d.get("area", 0.0),
            -d.get("det_score", 0.0),
            d["timestamp"],
        ),
    )

    selected: list[dict] = []
    min_gap_sec = 8.0
    for det in ranked:
        if all(abs(det["timestamp"] - s["timestamp"]) >= min_gap_sec for s in selected):
            selected.append(det)
        if len(selected) >= max_count:
            break

    if len(selected) < max_count:
        seen = {
            (round(d["timestamp"], 3), tuple(int(v) for v in d["bbox"]))
            for d in selected
        }
        for det in ranked:
            key = (round(det["timestamp"], 3), tuple(int(v) for v in det["bbox"]))
            if key in seen:
                continue
            selected.append(det)
            seen.add(key)
            if len(selected) >= max_count:
                break

    return sorted(selected, key=lambda d: d["timestamp"])


def _extract_face_crop(cap, timestamp: float, bbox: list[float]):
    """Extract a padded face crop from video at timestamp."""
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    det_w, det_h = 640.0, 360.0
    sx = w / det_w
    sy = h / det_h

    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = int(round(x1 * sx))
    y1 = int(round(y1 * sy))
    x2 = int(round(x2 * sx))
    y2 = int(round(y2 * sy))
    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.25)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)


def _load_cached(ctx: VideoContext) -> dict | None:
    """Load cached identity pipeline results."""
    for path in (
        ctx.debug_dir / "identity_pipeline.json",
        ctx.analysis_dir / "identity_pipeline.json",  # legacy path
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None
