"""Dataclasses returned by :class:`SpeakerSegmentDiarizer`."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueLine:
    """One expected line from the original (pre-TTS) transcript."""

    speaker: str
    text: str


@dataclass(frozen=True)
class SpeakerSegment:
    """One speaker turn located in the audio track, with timings in ms."""

    index: int
    speaker: str
    start_ms: int
    end_ms: int
    asr_text: str
    matched_line_idx: int | None
    matched_text: str | None
    fuzzy_score: float          # 0..100, rapidfuzz.token_set_ratio
    asr_speaker_label: str      # raw AssemblyAI label ("A", "B", ...)
    confidence: float           # 0..1, ASR confidence

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class DiarizationResult:
    """Full diarization output for a single WAV."""

    segments: list[SpeakerSegment]
    speaker_mapping: dict[str, str]
    full_text: str

    def for_speaker(self, speaker: str) -> list[SpeakerSegment]:
        """Return only segments attributed to ``speaker`` (case-sensitive)."""
        return [s for s in self.segments if s.speaker == speaker]
