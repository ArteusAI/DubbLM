"""
Transcription package for handling audio transcription and speaker diarization.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TranscriptionFactory",
    "WhisperXTranscriber",
    "PyAnnoteOpenAITranscriber",
    "AssemblyAITranscriber",
    "OpenRouterTranscriber",
]


def __getattr__(name: str) -> Any:
    if name == "TranscriptionFactory":
        from src.transcription.transcription_factory import TranscriptionFactory

        return TranscriptionFactory
    if name == "WhisperXTranscriber":
        from src.transcription.whisperx_transcriber import WhisperXTranscriber

        return WhisperXTranscriber
    if name == "PyAnnoteOpenAITranscriber":
        from src.transcription.pyannote_openai_transcriber import PyAnnoteOpenAITranscriber

        return PyAnnoteOpenAITranscriber
    if name == "AssemblyAITranscriber":
        from src.transcription.assemblyai_transcriber import AssemblyAITranscriber

        return AssemblyAITranscriber
    if name == "OpenRouterTranscriber":
        from src.transcription.openrouter_transcriber import OpenRouterTranscriber

        return OpenRouterTranscriber
    raise AttributeError(name)
