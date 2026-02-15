"""
VTT parser and quality assessment.

Handles YouTube auto-caption VTT files which have a distinctive rolling
format: each new phrase appends to the previous cue text, causing heavy
duplication.  This module deduplicates, cleans, and scores quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VTTCue:
    """A single cleaned VTT cue."""

    start: float    # seconds
    end: float      # seconds
    text: str       # cleaned text (no [Music] tags, no duplication)
    raw_text: str   # original text before cleaning
    is_music: bool  # True if this cue was predominantly [Music]


def parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, rest = parts
    elif len(parts) == 2:
        h = "0"
        m, rest = parts
    else:
        raise ValueError(f"Bad timestamp format: {ts}")

    s_parts = rest.split(".")
    s = float(s_parts[0])
    ms = float(s_parts[1]) / 1000 if len(s_parts) > 1 else 0.0
    return int(h) * 3600 + int(m) * 60 + s + ms


def _strip_music_tags(text: str) -> tuple[str, bool]:
    """
    Remove [Music] tags from text.

    Returns (cleaned_text, was_predominantly_music).
    """
    music_pattern = re.compile(r"\[Music\]", re.IGNORECASE)
    matches = music_pattern.findall(text)
    cleaned = music_pattern.sub("", text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    total_words = len(text.split())
    is_music = len(matches) > 0 and (
        not cleaned or len(matches) / max(total_words, 1) > 0.3
    )
    return cleaned, is_music


def _deduplicate_rolling_text(cues: list[dict]) -> list[dict]:
    """
    YouTube auto-captions use expansion chains:

        cue A: "Expect place like this. Early"
        cue B: "Expect place like this. Early hitchhiking is good for multitude of"
        cue C: "hitchhiking is good for multitude of"
        cue D: "hitchhiking is good for multitude of reasons."
        cue E: "reasons."

    Cues B extends A (same prefix + new tail).  Cue C is just B's tail.
    Cue D extends C.  Cue E is D's tail.

    Strategy:
    1. Walk through cues.  If cue N+1's text *contains* cue N's text at
       any position (prefix or suffix overlap), they're in the same chain.
    2. Keep only the LAST (longest) cue in each chain.
    3. From those survivors, extract genuinely new text by removing overlap
       with the previous survivor.
    """
    if not cues:
        return []

    # Step 1: Find chain-final cues (cues not extended by the next cue)
    chain_finals: list[dict] = []

    for i, cue in enumerate(cues):
        text = cue["text"].strip()
        if not text:
            continue

        # Check if the NEXT cue extends (contains) this one
        is_extended = False
        if i + 1 < len(cues):
            next_text = cues[i + 1]["text"].strip()
            if _is_contained_in(text, next_text):
                is_extended = True

        if not is_extended:
            chain_finals.append(cue)

    # Step 2: From chain-final cues, extract only genuinely new text
    deduped: list[dict] = []
    prev_text = ""

    for cue in chain_finals:
        text = cue["text"].strip()
        if not text:
            continue

        new_text = _extract_new_text(prev_text, text)
        if new_text.strip():
            deduped.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": new_text.strip(),
                "raw_text": cue["text"],
            })

        prev_text = text

    return deduped


def _is_contained_in(shorter: str, longer: str) -> bool:
    """
    Check if shorter's text is a component of longer (prefix, suffix,
    or substantial word overlap).  Handles the YouTube rolling pattern
    where the next cue either starts with or ends with the current text.
    """
    s = shorter.strip().lower()
    l = longer.strip().lower()

    if not s or not l:
        return False
    if len(s) > len(l):
        return False

    # Direct containment (most common case)
    if s in l:
        return True

    # Word-level overlap check: if >60% of shorter's words appear as
    # a contiguous subsequence in longer, treat as contained
    s_words = s.split()
    l_words = l.split()
    threshold = max(2, int(len(s_words) * 0.6))

    for start in range(len(l_words)):
        match_len = 0
        for j, sw in enumerate(s_words):
            if start + j < len(l_words) and l_words[start + j] == sw:
                match_len += 1
            else:
                break
        if match_len >= threshold:
            return True

    return False


def _extract_new_text(prev: str, current: str) -> str:
    """
    Given previous chain-final cue text and current chain-final cue text,
    return only the genuinely new portion.
    """
    if not prev:
        return current

    prev_clean = prev.strip().lower()
    curr_clean = current.strip().lower()

    # If current starts with prev, grab only the suffix
    if curr_clean.startswith(prev_clean):
        new_part = current[len(prev):].strip()
        return new_part if new_part else current

    # Word-level: find longest suffix of prev that matches a prefix of current
    prev_words = prev.split()
    curr_words = current.split()

    for overlap_len in range(min(len(prev_words), len(curr_words)), 0, -1):
        prev_suffix = [w.lower() for w in prev_words[-overlap_len:]]
        curr_prefix = [w.lower() for w in curr_words[:overlap_len]]
        if prev_suffix == curr_prefix:
            new_words = curr_words[overlap_len:]
            return " ".join(new_words) if new_words else current

    # No overlap — entire current is new
    return current


def parse_vtt(vtt_path: Path) -> list[VTTCue]:
    """
    Parse a VTT file into cleaned, deduplicated cues.

    Steps:
    1. Parse raw VTT into timestamp + text pairs
    2. Deduplicate YouTube rolling-window text
    3. Strip [Music] tags
    4. Merge tiny consecutive cues
    """
    raw_cues = _parse_raw_vtt(vtt_path)
    deduped = _deduplicate_rolling_text(raw_cues)

    cleaned: list[VTTCue] = []
    for cue_dict in deduped:
        text, is_music = _strip_music_tags(cue_dict["text"])
        if text or is_music:  # Keep music markers for quality assessment
            cleaned.append(VTTCue(
                start=cue_dict["start"],
                end=cue_dict["end"],
                text=text,
                raw_text=cue_dict.get("raw_text", cue_dict["text"]),
                is_music=is_music,
            ))

    return cleaned


def _parse_raw_vtt(vtt_path: Path) -> list[dict]:
    """Parse raw VTT into list of {start, end, text}."""
    content = vtt_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    cues: list[dict] = []

    i = 0
    # Skip header
    while i < len(lines):
        if re.match(r"\d{2}:\d{2}:", lines[i]):
            break
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
            line,
        )
        if ts_match:
            start = parse_timestamp(ts_match.group(1))
            end = parse_timestamp(ts_match.group(2))
            i += 1
            text_parts: list[str] = []
            while i < len(lines) and lines[i].strip():
                text_parts.append(lines[i].strip())
                i += 1
            cues.append({
                "start": start,
                "end": end,
                "text": " ".join(text_parts),
            })
        else:
            i += 1

    return cues


@dataclass
class VTTQuality:
    """Quality assessment of a VTT region."""

    score: float          # 0.0 (terrible) to 1.0 (great)
    total_cues: int
    text_cues: int        # cues with actual text
    music_cues: int       # cues that are predominantly [Music]
    empty_gaps: int       # gaps > 30s with no cues
    avg_words_per_cue: float
    issues: list[str]


def assess_vtt_quality(
    cues: list[VTTCue],
    start_sec: float,
    end_sec: float,
) -> VTTQuality:
    """
    Assess VTT quality for a time region.

    Used to decide if selective Whisper re-transcription is needed.
    A score < 0.4 triggers re-transcription.
    """
    region_cues = [c for c in cues if c.start < end_sec and c.end > start_sec]
    total = len(region_cues)

    if total == 0:
        return VTTQuality(
            score=0.0,
            total_cues=0,
            text_cues=0,
            music_cues=0,
            empty_gaps=1,
            avg_words_per_cue=0,
            issues=["no_cues_in_region"],
        )

    text_cues = [c for c in region_cues if c.text and not c.is_music]
    music_cues = [c for c in region_cues if c.is_music]
    issues: list[str] = []

    # Words per cue
    word_counts = [len(c.text.split()) for c in text_cues if c.text]
    avg_words = sum(word_counts) / max(len(word_counts), 1)

    # Check for gaps > 30s
    sorted_cues = sorted(region_cues, key=lambda c: c.start)
    empty_gaps = 0
    prev_end = start_sec
    for cue in sorted_cues:
        if cue.start - prev_end > 30.0:
            empty_gaps += 1
        prev_end = cue.end
    if end_sec - prev_end > 30.0:
        empty_gaps += 1

    # Check for repetition
    texts = [c.text.lower().strip() for c in text_cues if c.text]
    unique_ratio = len(set(texts)) / max(len(texts), 1)

    # Score calculation
    score = 1.0

    # Few text cues relative to duration
    expected_cues = (end_sec - start_sec) / 5.0  # expect ~1 cue per 5s
    cue_density = len(text_cues) / max(expected_cues, 1)
    if cue_density < 0.3:
        score -= 0.3
        issues.append("low_cue_density")

    # Too much music
    music_ratio = len(music_cues) / max(total, 1)
    if music_ratio > 0.5:
        score -= 0.3
        issues.append("high_music_ratio")

    # Empty gaps
    if empty_gaps > 0:
        score -= 0.1 * min(empty_gaps, 3)
        issues.append(f"{empty_gaps}_empty_gaps")

    # Repetitive text
    if unique_ratio < 0.5:
        score -= 0.2
        issues.append("repetitive_text")

    # Very short cues
    if avg_words < 3:
        score -= 0.1
        issues.append("very_short_cues")

    score = max(0.0, min(1.0, score))

    return VTTQuality(
        score=score,
        total_cues=total,
        text_cues=len(text_cues),
        music_cues=len(music_cues),
        empty_gaps=empty_gaps,
        avg_words_per_cue=round(avg_words, 1),
        issues=issues,
    )
