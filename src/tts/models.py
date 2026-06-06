from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field, validator
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentSynthesisReport:
    """Per-segment telemetry captured by TTS wrappers for the summary report."""

    segment_index: int
    speaker: Optional[str]
    text: str
    requested_model: Optional[str]
    actual_model: Optional[str]
    attempts: int
    used_fallback: bool
    success: bool
    duration_seconds: float
    output_path: Optional[str] = None
    group_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchSynthesisReport:
    """Per-batch telemetry captured by TTS wrappers for the summary report.

    A "batch" here means one scheduling unit handed to the provider (Gemini
    multi-speaker groups, single-speaker consecutive runs, or a solo segment).
    The report records whether the batch-level path succeeded or had to fall
    through to per-segment re-synthesis, so readers see *why* batching either
    paid off or got bypassed.
    """

    batch_index: int
    mode: str  # "multi_speaker" | "single_speaker" | "single_segment"
    segment_indices: List[int]
    speakers: List[str]
    attempts: int
    success: bool
    fallback_segment_count: int
    duration_seconds: float
    reason: str

class TTSSegmentData(BaseModel):
    """Data model for a single text segment to be synthesized."""
    speaker: str = Field(..., description="Identifier for the speaker (e.g., \"SPEAKER_00\").")
    text: str = Field(..., description="The text to be synthesized for this speaker.")
    
    emotion: Optional[str] = Field(None, description="Emotion for this segment (e.g., \"Happy\", \"Sad\").")
    speed: Optional[float] = Field(None, description="Speed factor for this segment (e.g., 1.0 for normal, 1.2 for faster).")
    voice: Optional[str] = Field(None, description="Specific voice name to use for this segment, overriding any global speaker-to-voice mappings.")
    style_prompt: Optional[str] = Field(None, description="Specific style prompt for this segment, overriding any global speaker-to-style_prompt mappings.")
    reference_audio_path: Optional[str] = Field(None, description="Path to a reference audio file for voice cloning for this specific segment/speaker.")
    reference_text: Optional[str] = Field(None, description="Text corresponding to the reference_audio_path, if required by the TTS system.")
    output_path: Optional[str] = Field(None, description="Path to save the synthesized audio for this specific segment.")
    cohesion_with_prev: Optional[str] = Field(
        None,
        description=(
            "Batcher hint from the LLM editor: one of 'tight' | 'normal' | 'loose'. "
            "Controls whether splitting a multi-speaker TTS batch before this segment "
            "hurts dialogue coherence. Unknown or missing values are treated as 'normal'."
        ),
    )
    segment_index: Optional[int] = Field(
        None,
        description="0-based index of this segment in the pipeline (used for telemetry/reports).",
    )
    group_id: Optional[str] = Field(
        None,
        description="Identifier of the speaker group this segment belongs to; None for ungrouped segments.",
    )

    @validator("voice", pre=True)
    def normalize_voice_override(cls, value: Optional[str]) -> Optional[str]:
        """Normalize placeholder/default voice values to 'unset'."""
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized.lower() == "default":
                return None
            return normalized
        return value

    @validator("cohesion_with_prev", pre=True)
    def normalize_cohesion_hint(cls, value: Optional[str]) -> Optional[str]:
        """Clamp cohesion_with_prev to the known vocabulary; None stays None."""
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"tight", "normal", "loose"}:
                return normalized
        return None

    class Config:
        extra = 'allow' # Allow other kwargs to be passed through if a TTS system needs them beyond this model


class VoiceDurationStats(BaseModel):
    """Statistics for voice duration estimation."""
    voice_name: str = Field(..., description="Name of the voice")
    words_per_minute: float = Field(..., description="Average words per minute for this voice")
    characters_per_second: float = Field(..., description="Average characters per second for this voice")
    total_samples: int = Field(default=0, description="Number of samples used to calculate statistics")
    total_words: int = Field(default=0, description="Total words in all samples")
    total_characters: int = Field(default=0, description="Total characters in all samples")
    total_duration_seconds: float = Field(default=0.0, description="Total duration of all samples in seconds")
    
    def update_stats(
        self,
        words: int,
        characters: int,
        duration: float,
        smoothing_alpha: Optional[float] = None
    ) -> None:
        """Update statistics with new sample data.

        Args:
            words: Number of words in the new sample
            characters: Number of characters in the new sample
            duration: Duration of the new sample in seconds
            smoothing_alpha: Optional smoothing factor (0-1). When provided,
                applies exponential smoothing to derived rates instead of a pure average.
        """
        # Store old values for logging
        old_wpm = self.words_per_minute
        old_cps = self.characters_per_second

        self.total_samples += 1
        self.total_words += words
        self.total_characters += characters
        self.total_duration_seconds += duration

        if duration <= 0:
            return

        sample_words_per_minute = (words / duration) * 60 if duration > 0 else 0.0
        sample_characters_per_second = characters / duration if duration > 0 else 0.0

        use_smoothing = smoothing_alpha is not None and smoothing_alpha > 0
        if use_smoothing and self.total_samples > 1:
            alpha = max(0.0, min(1.0, smoothing_alpha))
            # Apply exponential smoothing to derived rates
            self.words_per_minute = (
                (1 - alpha) * self.words_per_minute + alpha * sample_words_per_minute
            )
            self.characters_per_second = (
                (1 - alpha) * self.characters_per_second + alpha * sample_characters_per_second
            )

            logger.debug(
                f"Applied EMA smoothing (alpha={alpha:.2f}): "
                f"sample WPM={sample_words_per_minute:.1f}, CPS={sample_characters_per_second:.1f} -> "
                f"smoothed WPM={self.words_per_minute:.1f}, CPS={self.characters_per_second:.1f}"
            )
        else:
            if self.total_duration_seconds > 0:
                self.words_per_minute = (self.total_words / self.total_duration_seconds) * 60
                self.characters_per_second = self.total_characters / self.total_duration_seconds
            else:
                self.words_per_minute = sample_words_per_minute
                self.characters_per_second = sample_characters_per_second

            logger.debug(
                f"Updated stats using cumulative average: "
                f"sample WPM={sample_words_per_minute:.1f}, CPS={sample_characters_per_second:.1f} -> "
                f"avg WPM={self.words_per_minute:.1f}, CPS={self.characters_per_second:.1f}"
            )


class VoiceDurationDatabase(BaseModel):
    """Database of voice duration statistics."""
    voice_stats: Dict[str, VoiceDurationStats] = Field(default_factory=dict)
    
    def get_or_create_stats(self, voice_name: str) -> VoiceDurationStats:
        """Get existing stats or create new ones for a voice."""
        if voice_name not in self.voice_stats:
            self.voice_stats[voice_name] = VoiceDurationStats(
                voice_name=voice_name,
                words_per_minute=150.0,  # Default fallback
                characters_per_second=12.5  # Default fallback
            )
        return self.voice_stats[voice_name]
    
    def update_voice_stats(
        self,
        voice_name: str,
        words: int,
        characters: int,
        duration: float,
        smoothing_alpha: Optional[float] = None
    ) -> None:
        """Update statistics for a voice with new sample data."""
        stats = self.get_or_create_stats(voice_name)
        stats.update_stats(words, characters, duration, smoothing_alpha=smoothing_alpha)


class DiarizationSegment(BaseModel):
    """Segment detected during diarization of synthesized audio."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds") 
    speaker: str = Field(..., description="Detected speaker ID")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(default=1.0, description="Confidence score for this segment")
    
    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time


class SegmentAlignment(BaseModel):
    """Alignment between original segment and diarized segment."""
    original_segment: TTSSegmentData = Field(..., description="Original segment from input")
    diarized_segment: DiarizationSegment = Field(..., description="Corresponding diarized segment")
    alignment_confidence: float = Field(..., description="Confidence of this alignment")
