from abc import ABC, abstractmethod
import threading
from typing import Optional, Dict, Any, List

from .models import (
    TTSSegmentData,
    SegmentAlignment,
    SegmentSynthesisReport,
    BatchSynthesisReport,
)

class TTSInterface(ABC):
    """
    Abstract base class for text-to-speech systems.
    All TTS implementations should inherit from this class.
    """

    # Per-segment telemetry collected by concrete wrappers for the summary report.
    # Subclasses should append ``SegmentSynthesisReport`` instances via
    # ``_record_segment_report`` and the orchestrator drains them with
    # ``drain_segment_reports()`` between batches.
    segment_reports: List[SegmentSynthesisReport]

    # Per-batch telemetry (multi-speaker / single-speaker / solo) for the
    # summary report. Populated via ``_record_batch_report`` and drained with
    # ``drain_batch_reports()``.
    batch_reports: List[BatchSynthesisReport]

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the TTS system with provider-specific arguments."""
        pass # Specific implementation in derived classes

    def _ensure_segment_report_state(self) -> None:
        """Lazily initialize telemetry buffer + lock for subclasses."""
        if not hasattr(self, "segment_reports") or self.segment_reports is None:
            self.segment_reports = []
        if not hasattr(self, "_segment_reports_lock") or self._segment_reports_lock is None:
            self._segment_reports_lock = threading.Lock()

    def _record_segment_report(self, report: SegmentSynthesisReport) -> None:
        """Thread-safe append of a per-segment telemetry entry."""
        self._ensure_segment_report_state()
        with self._segment_reports_lock:
            self.segment_reports.append(report)

    def drain_segment_reports(self) -> List[SegmentSynthesisReport]:
        """Return and clear collected per-segment telemetry."""
        self._ensure_segment_report_state()
        with self._segment_reports_lock:
            drained = list(self.segment_reports)
            self.segment_reports = []
            return drained

    def _ensure_batch_report_state(self) -> None:
        """Lazily initialize batch telemetry buffer + lock for subclasses."""
        if not hasattr(self, "batch_reports") or self.batch_reports is None:
            self.batch_reports = []
        if not hasattr(self, "_batch_reports_lock") or self._batch_reports_lock is None:
            self._batch_reports_lock = threading.Lock()

    def _record_batch_report(self, report: BatchSynthesisReport) -> None:
        """Thread-safe append of a per-batch telemetry entry."""
        self._ensure_batch_report_state()
        with self._batch_reports_lock:
            self.batch_reports.append(report)

    def drain_batch_reports(self) -> List[BatchSynthesisReport]:
        """Return and clear collected per-batch telemetry."""
        self._ensure_batch_report_state()
        with self._batch_reports_lock:
            drained = list(self.batch_reports)
            self.batch_reports = []
            return drained
    
    @abstractmethod
    def initialize(self) -> None:
        """Perform any necessary setup for the TTS client (e.g., API connections, model loading)."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for a list of text segments.

        Args:
            segments_data: A list of TTSSegmentData objects. Each object contains
                           speaker ID, text, and optional per-segment parameters like
                           emotion, speed, voice override, style_prompt,
                           reference_audio_path, reference_text, and output_path.
            language: Target language code (e.g., "en"). If not provided, uses target_language from init.
            **kwargs: Additional global parameters for the specific TTS system.
            
        Returns:
            List of SegmentAlignment objects mapping original segments to synthesized audio
        """
        pass
    
    @abstractmethod
    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text segment.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (uses target_language from init if not specified)
            
        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS system is available and properly initialized."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice names for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice style prompts for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        """
        Clone a voice using the TTS provider's API.
        
        Args:
            audio_path: Path to reference audio file
            voice_id: Desired ID for the cloned voice
            
        Returns:
            voice_id: The ID of the cloned voice
        """
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources or temporary files used by the TTS system.
        This should be called when the TTS system is no longer needed.
        """
        pass # Specific implementation in derived classes 