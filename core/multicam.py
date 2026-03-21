"""Multi-camera angle detection via transcript similarity.

Compares transcripts pairwise using 5-gram overlap with a sliding window
to handle different start/end times and minor Whisper transcription differences.

Multi-cam means two cameras filming THE SAME interview — transcripts should be
nearly identical since the same words were spoken, just captured from different
angles/mics. With Whisper there will be minor differences but core sentences match.
"""

from __future__ import annotations

import os
import re
import string
from typing import Callable

from core.schemas import MultiCamGroup, MultiCamMatch, VideoResult

# Common English stopwords to exclude from n-grams
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "that", "this", "was", "are", "be", "has", "had",
    "do", "did", "will", "would", "could", "should", "may", "might",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "my", "your", "his", "its", "our", "their", "so", "if", "not", "no",
    "just", "like", "um", "uh", "yeah", "yes", "ok", "okay", "oh",
    "well", "then", "than", "very", "really", "got", "get",
})

# Minimum word count to consider a transcript for matching
MIN_WORDS = 25

# N-gram size — 5-grams are specific enough to avoid false positives
# from common interview phrases while still matching the same interview
# captured from different mics/angles
NGRAM_SIZE = 5

# Minimum number of matched n-grams required (not just percentage).
# Prevents high-percentage matches from short clips with few coincidental hits.
MIN_MATCHED_NGRAMS = 15


def _normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _build_ngrams(words: list[str], n: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Build set of word n-grams from a word list."""
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def compute_overlap(words_a: list[str], words_b: list[str]) -> tuple[float, int, int, int]:
    """
    Compute n-gram overlap between two word lists.

    Returns (score, matched_count, total_a, total_b).
    Score is containment ratio: intersection / min(len_a, len_b).
    This handles one clip being shorter than the other.
    """
    ngrams_a = _build_ngrams(words_a)
    ngrams_b = _build_ngrams(words_b)

    if not ngrams_a or not ngrams_b:
        return 0.0, 0, len(ngrams_a), len(ngrams_b)

    intersection = ngrams_a & ngrams_b
    matched = len(intersection)
    smaller = min(len(ngrams_a), len(ngrams_b))

    return matched / smaller, matched, len(ngrams_a), len(ngrams_b)


def sliding_window_overlap(words_a: list[str], words_b: list[str],
                           step: int = 10) -> tuple[float, int, int, int]:
    """
    Slide the shorter transcript across the longer one to find the best
    overlap region. Handles clips with different start/end times.

    Uses step=10 for fine-grained alignment.

    Returns (best_score, matched_count, total_a, total_b).
    """
    # Ensure a is the longer one
    swapped = False
    if len(words_a) < len(words_b):
        words_a, words_b = words_b, words_a
        swapped = True

    # If they're similar length, just do direct comparison
    ratio = len(words_b) / len(words_a) if len(words_a) > 0 else 0
    if ratio > 0.7:
        return compute_overlap(words_a, words_b)

    # Slide the shorter one across the longer one
    window_size = len(words_b)
    best_score = 0.0
    best_matched = 0
    ngrams_b = _build_ngrams(words_b)
    total_b = len(ngrams_b)

    if not ngrams_b:
        return 0.0, 0, 0, 0

    for offset in range(0, max(1, len(words_a) - window_size // 2), step):
        chunk = words_a[offset:offset + window_size]
        ngrams_chunk = _build_ngrams(chunk)
        if not ngrams_chunk:
            continue

        intersection = ngrams_chunk & ngrams_b
        matched = len(intersection)
        smaller = min(len(ngrams_chunk), total_b)
        score = matched / smaller if smaller > 0 else 0.0

        if score > best_score:
            best_score = score
            best_matched = matched

    total_a = len(_build_ngrams(words_a))
    if swapped:
        return best_score, best_matched, total_b, total_a
    return best_score, best_matched, total_a, total_b


def _union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Simple union-find to group connected indices."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Only return groups with 2+ members
    return [members for members in groups.values() if len(members) > 1]


def find_multicam_groups(
    results: list[VideoResult],
    threshold: float = 0.35,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[MultiCamGroup]:
    """
    Compare all transcript pairs to find multi-camera angle groups.

    Multi-cam = same interview filmed from different angles. Transcripts are
    nearly identical. Uses 5-gram overlap so only genuinely matching content
    scores high.

    Args:
        results: List of completed VideoResults
        threshold: Minimum n-gram overlap score (0-1) to consider a match
        log: Logging callback
        progress: Progress callback (current, total)

    Returns:
        List of MultiCamGroups
    """
    _log = log or (lambda m: None)

    # Filter candidates: must have speech transcripts with enough words
    candidates: list[tuple[int, VideoResult, list[str]]] = []
    for i, r in enumerate(results):
        if not r.transcript or not r.transcript.strip():
            continue
        # Skip clips with no usable audio
        if r.audio_check and not r.audio_check.has_audio:
            continue
        words = _normalize_text(r.transcript)
        if len(words) < MIN_WORDS:
            continue
        candidates.append((i, r, words))

    _log(f"[multicam] {len(candidates)} clips with usable transcripts "
         f"(out of {len(results)} total)")

    if len(candidates) < 2:
        _log("[multicam] Need at least 2 clips with transcripts to compare")
        return []

    # Pairwise comparison
    total_comparisons = len(candidates) * (len(candidates) - 1) // 2
    _log(f"[multicam] Running {total_comparisons} pairwise comparisons "
         f"(using {NGRAM_SIZE}-grams, threshold={threshold}, "
         f"min_matched={MIN_MATCHED_NGRAMS})...")

    matches: list[MultiCamMatch] = []
    edges: list[tuple[int, int]] = []  # indices into candidates list
    comp_count = 0

    # Diagnostic threshold — log pairs that score above this even if they
    # don't meet the match threshold, so users can see near-misses.
    DIAGNOSTIC_THRESHOLD = 0.2

    for i in range(len(candidates)):
        idx_a, res_a, words_a = candidates[i]
        for j in range(i + 1, len(candidates)):
            idx_b, res_b, words_b = candidates[j]
            comp_count += 1

            if progress and comp_count % 10 == 0:
                progress(comp_count, total_comparisons)

            # Pre-filter: skip if durations differ by more than 3x
            dur_a = res_a.duration_seconds or 1
            dur_b = res_b.duration_seconds or 1
            dur_ratio = max(dur_a, dur_b) / max(min(dur_a, dur_b), 1)
            if dur_ratio > 3.0:
                continue

            score, matched, total_a, total_b = sliding_window_overlap(
                words_a, words_b
            )

            name_a = os.path.basename(res_a.video_path)
            name_b = os.path.basename(res_b.video_path)

            # Diagnostic logging for near-misses
            if score >= DIAGNOSTIC_THRESHOLD and score < threshold:
                _log(f"[multicam] Near-miss: {name_a} <-> {name_b} "
                     f"({score:.0%} overlap, {matched} {NGRAM_SIZE}-grams) "
                     f"— below threshold {threshold}")
            elif score >= DIAGNOSTIC_THRESHOLD and matched < MIN_MATCHED_NGRAMS:
                _log(f"[multicam] Near-miss: {name_a} <-> {name_b} "
                     f"({score:.0%} overlap, {matched} {NGRAM_SIZE}-grams) "
                     f"— below minimum matched count {MIN_MATCHED_NGRAMS}")

            if score >= threshold and matched >= MIN_MATCHED_NGRAMS:
                matches.append(MultiCamMatch(
                    video_path_a=res_a.video_path,
                    video_path_b=res_b.video_path,
                    similarity_score=round(score, 3),
                    matched_trigrams=matched,
                    total_trigrams_a=total_a,
                    total_trigrams_b=total_b,
                ))
                edges.append((i, j))
                _log(f"[multicam] Match: {name_a} <-> {name_b} "
                     f"({score:.0%} overlap, {matched} {NGRAM_SIZE}-grams)")

    if progress:
        progress(total_comparisons, total_comparisons)

    if not edges:
        _log("[multicam] No multi-camera groups found")
        return []

    # Group connected matches
    raw_groups = _union_find_groups(len(candidates), edges)

    groups: list[MultiCamGroup] = []
    for group_num, member_indices in enumerate(raw_groups, 1):
        group_id = f"MC_{group_num:03d}"
        clip_paths = [candidates[i][1].video_path for i in member_indices]

        # Collect matches within this group
        group_matches = []
        member_set = set(member_indices)
        for m in matches:
            # Check if both endpoints are in this group
            for i in member_indices:
                for j in member_indices:
                    if (candidates[i][1].video_path == m.video_path_a and
                            candidates[j][1].video_path == m.video_path_b):
                        group_matches.append(m)

        # Deduplicate matches
        seen = set()
        unique_matches = []
        for m in group_matches:
            key = (m.video_path_a, m.video_path_b)
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        groups.append(MultiCamGroup(
            group_id=group_id,
            clip_paths=clip_paths,
            matches=unique_matches,
        ))
        _log(f"[multicam] Group {group_id}: {len(clip_paths)} clips — "
             f"{', '.join(os.path.basename(p) for p in clip_paths)}")

    _log(f"[multicam] Found {len(groups)} multi-camera group(s)")
    return groups
