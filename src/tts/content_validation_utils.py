"""Shared helpers for TTS content validation.

The ASR similarity checks used by the Gemini wrappers rely on
``fuzz.token_set_ratio``, which is deliberately insensitive to word order and
duplicate words. That makes it blind to a common Gemini TTS failure mode where
the model stutters and repeats a sentence (sometimes with slightly different
wording) inside a single response.

``find_unexpected_repetition`` complements those checks by comparing repeated
word n-grams in the ASR transcript against the expected text. A repeat is only
flagged when it is *not* already present in the expected text, so legitimate
repetition (songs, emphatic "нет, нет, нет", names repeated by the speaker)
does not trigger a false positive.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple


def _word_ngrams(words: list, size: int) -> list:
    if size <= 0 or len(words) < size:
        return []
    return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]


def find_unexpected_repetition(
    asr_text: str,
    expected_text: str,
    min_words: int = 4,
    max_extra_occurrences: int = 0,
) -> Optional[Tuple[str, int, int]]:
    """Return the first unexpectedly repeated n-gram, if any.

    Both inputs are expected to be already normalized (lowercased, punctuation
    stripped, whitespace collapsed) with the same normalization function, so
    that tokens compare reliably.

    Args:
        asr_text: Normalized ASR transcript of the synthesized audio.
        expected_text: Normalized text that was requested from the TTS.
        min_words: N-gram length used for repetition detection.
        max_extra_occurrences: How many additional occurrences in ASR (beyond
            the count in the expected text) are tolerated.

    Returns:
        ``(phrase, asr_count, expected_count)`` for the first flagged n-gram,
        or ``None`` when no unexpected repetition is found.
    """
    size = max(1, int(min_words))
    asr_words = asr_text.split()
    expected_words = expected_text.split()

    if len(asr_words) < size * 2:
        return None

    expected_counts = Counter(_word_ngrams(expected_words, size))
    asr_counts = Counter(_word_ngrams(asr_words, size))

    # Report the most frequently repeated n-gram first so the log message
    # names the most informative span. Sort deterministically for stable output.
    for gram in sorted(asr_counts, key=lambda g: (-asr_counts[g], g)):
        asr_count = asr_counts[gram]
        if asr_count < 2:
            continue
        expected_count = expected_counts.get(gram, 0)
        if asr_count > expected_count + max(0, int(max_extra_occurrences)):
            return " ".join(gram), asr_count, expected_count

    return None
