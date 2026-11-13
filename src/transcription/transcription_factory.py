"""
Factory for creating transcription and diarization services.
"""
from typing import Optional, Literal

from src.transcription.transcription_interface import TranscriptionInterface

class TranscriptionFactory:
    """Factory for creating transcription and diarization services."""
    
    @staticmethod
    def create_transcriber(
        transcription_system: Literal["pyannote_openai", "whisperx", "assemblyai", "openrouter"],
        source_language: str,
        device: Optional[str] = None,
        **kwargs
    ) -> TranscriptionInterface:
        """
        Create a transcription service implementation based on the specified system.
        
        Args:
            transcription_system: The transcription system to use
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            **kwargs: Additional parameters for the specific implementation
            
        Returns:
            An initialized transcription service implementation
            
        Raises:
            ValueError: If the specified transcription system is not supported
        """
        cost_tracker = kwargs.pop("cost_tracker", None)
        transcriber_kwargs = dict(kwargs)
        if transcription_system == "whisperx":
            from src.transcription.whisperx_transcriber import WhisperXTranscriber

            return WhisperXTranscriber(
                source_language=source_language,
                device=device,
                **transcriber_kwargs
            )
        elif transcription_system == "pyannote_openai" or transcription_system == "openai":
            from src.transcription.pyannote_openai_transcriber import PyAnnoteOpenAITranscriber

            if cost_tracker is not None:
                transcriber_kwargs["cost_tracker"] = cost_tracker
            return PyAnnoteOpenAITranscriber(
                source_language=source_language,
                device=device,
                **transcriber_kwargs
            )
        elif transcription_system == "assemblyai":
            from src.transcription.assemblyai_transcriber import AssemblyAITranscriber

            if cost_tracker is not None:
                transcriber_kwargs["cost_tracker"] = cost_tracker
            return AssemblyAITranscriber(
                source_language=source_language,
                device=device,
                **transcriber_kwargs
            )
        elif transcription_system == "openrouter":
            from src.transcription.openrouter_transcriber import OpenRouterTranscriber

            if cost_tracker is not None:
                transcriber_kwargs["cost_tracker"] = cost_tracker
            return OpenRouterTranscriber(
                source_language=source_language,
                device=device,
                **transcriber_kwargs
            )
        else:
            raise ValueError(f"Unsupported transcription system: {transcription_system}")
