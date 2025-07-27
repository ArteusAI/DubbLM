"""Reporter for generating speaker analysis reports."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

from ..debug.performance_tracker import PerformanceTracker
from ..core.log_config import get_logger

logger = get_logger(__name__)


class SpeakerReporter:
    """Generates speaker analysis reports for the Smart Dubbing system."""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        """Initialize the speaker reporter.
        
        Args:
            performance_tracker: Performance tracker instance
        """
        self.performance_tracker = performance_tracker
    
    def create_speaker_report(self, 
                            speaker_audio_paths: Dict[str, str], 
                            transcription: List[Dict],
                            video_path: str) -> Tuple[str, str]:
        """Creates a text report of speakers and copies their voice samples.
        
        Args:
            speaker_audio_paths: Dictionary mapping speaker IDs to their full audio sample paths
            transcription: List of transcription segments (used to list example phrases)
            video_path: Path to the source video
            
        Returns:
            Tuple[str, str]: Path to the speaker report text file and path to the voice samples directory
        """
        report_dir = Path("artifacts/speaker_report")
        samples_dir = report_dir / "voice_samples"
        
        report_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        report_file_path = report_dir / "speaker_report.txt"
        
        # Group transcription by speaker for example phrases
        segments_by_speaker: Dict[str, List[str]] = {}
        for segment in transcription:
            speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment["text"])

        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write("Speaker Diarization Report\n")
            f.write("==========================\n\n")
            f.write(f"Source Video: {video_path}\n")
            f.write(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
            
            f.write("Identified Speakers and Voice Samples:\n")
            f.write("------------------------------------\n")
            
            if not speaker_audio_paths:
                f.write("No speakers were identified or no audio samples could be extracted.\n")
            else:
                for speaker_id, original_sample_path in sorted(speaker_audio_paths.items()):
                    if not os.path.exists(original_sample_path):
                        logger.warning(f"Warning: Original sample path for {speaker_id} not found: {original_sample_path}")
                        f.write(f"Speaker ID: {speaker_id}\n")
                        f.write("  Voice Sample: Error - Original sample file not found.\n")
                        f.write("  Example Phrases: Not available due to sample error.\n\n")
                        continue

                    # Copy the sample file
                    sample_filename = f"{speaker_id}_full_sample.wav"
                    copied_sample_path = samples_dir / sample_filename
                    try:
                        shutil.copy(original_sample_path, copied_sample_path)
                        relative_sample_path = Path(copied_sample_path).relative_to(report_dir.parent)
                        f.write(f"Speaker ID: {speaker_id}\n")
                        f.write(f"  Voice Sample: {relative_sample_path}\n")
                    except Exception as e:
                        logger.error(f"Error copying sample for {speaker_id}: {e}")
                        f.write(f"Speaker ID: {speaker_id}\n")
                        f.write(f"  Voice Sample: Error - Could not copy sample file.\n")

                    # Add a few example phrases
                    example_phrases = segments_by_speaker.get(speaker_id, [])
                    if example_phrases:
                        f.write("  Example Phrases (first few occurrences):\n")
                        for i, phrase in enumerate(example_phrases[:3]):  # Max 3 example phrases
                            f.write(f"    - \"{phrase[:100]}{'...' if len(phrase) > 100 else ''}\"\n")
                    else:
                        f.write("  Example Phrases: No distinct transcribed phrases found for this speaker in the processed segment.\n")
                    f.write("\n")
                    
        logger.debug(f"Speaker report saved to {report_file_path}")
        logger.info(f"Voice samples copied to {samples_dir}")
        return str(report_file_path), str(samples_dir) 