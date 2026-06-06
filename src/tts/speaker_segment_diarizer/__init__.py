"""Diarize a multi-speaker Gemini TTS audio track against a known transcript."""

from .diarizer import SpeakerSegmentDiarizer
from .models import DialogueLine, DiarizationResult, SpeakerSegment

__all__ = [
    "SpeakerSegmentDiarizer",
    "DialogueLine",
    "DiarizationResult",
    "SpeakerSegment",
]
__version__ = "0.1.0"
