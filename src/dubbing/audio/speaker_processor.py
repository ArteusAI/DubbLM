"""Speaker processing for the Smart Dubbing system."""

import os
import shutil
from typing import Dict, List, Tuple
from pydub import AudioSegment

from ..core.cache_manager import CacheManager
from ..debug.performance_tracker import PerformanceTracker
from ..core.log_config import get_logger

logger = get_logger(__name__)


class SpeakerProcessor:
    """Handles speaker audio extraction and processing for the Smart Dubbing system."""
    
    def __init__(self, cache_manager: CacheManager, performance_tracker: PerformanceTracker):
        """Initialize the speaker processor.
        
        Args:
            cache_manager: Cache manager instance
            performance_tracker: Performance tracker instance
        """
        self.cache_manager = cache_manager
        self.performance_tracker = performance_tracker
    
    def extract_speaker_audio(self, audio_file: str, speakers_rolls: Dict[Tuple[float, float], str]) -> Dict[str, str]:
        """Extract full audio tracks for each speaker, including all their segments.
        
        Args:
            audio_file: Path to the audio file
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            
        Returns:
            Dictionary mapping speaker IDs to their audio file paths
        """
        # Start timing
        self.performance_tracker.start_timing("speaker_audio")

        cache_key = self.cache_manager.generate_cache_key(audio_file, "", "", "")
        step_name = "speaker_audio"
        
        # Check if results are cached
        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading speaker audio from cache...")
            # Copy cached speaker audio files to working directory
            self.cache_manager.copy_cached_files(step_name, cache_key, "*.wav", "artifacts/speakers_audio")
            
            # Load the speaker audio paths dictionary
            speaker_audio_paths = self.cache_manager.load_from_cache(step_name, cache_key)
            # Cache was used, so this process was fast
            self.performance_tracker.end_timing("speaker_audio")
            return speaker_audio_paths
        
        speakers = set(list(speakers_rolls.values()))
        audio = AudioSegment.from_file(audio_file)
        
        # Create a directory for cached speaker audio if needed
        if self.cache_manager.use_cache:
            speaker_cache_dir = self.cache_manager.get_cache_path(step_name) / cache_key
            speaker_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store paths to speaker audio files
        speaker_audio_paths = {}
        
        # For each speaker, extract all segments where they speak
        for speaker in speakers:
            # Get all segments for this speaker
            speaker_segments = [(start, end) for (start, end), spk in speakers_rolls.items() if spk == speaker]
            
            # Sort segments by start time
            speaker_segments.sort(key=lambda x: x[0])
            
            # Combine all segments for this speaker
            speaker_audio = AudioSegment.empty()
            
            for start, end in speaker_segments:
                segment_start_ms = int(start * 1000)
                segment_end_ms = int(end * 1000)
                segment = audio[segment_start_ms:segment_end_ms]
                speaker_audio += segment
            
            # Limit speaker audio to 1 minute (60,000 ms)
            if len(speaker_audio) > 60000:
                speaker_audio = speaker_audio[:60000]
            
            speaker_audio_path = f"artifacts/speakers_audio/{speaker}.wav"
            speaker_audio.export(speaker_audio_path, format="wav")
            speaker_audio_paths[speaker] = speaker_audio_path
            
            # Save to cache if enabled
            if self.cache_manager.use_cache:
                shutil.copy(speaker_audio_path, self.cache_manager.get_cache_path(step_name) / cache_key / f"{speaker}.wav")
            
            logger.debug(f"Extracted complete audio track for speaker {speaker} ({len(speaker_audio)/1000:.2f}s)")
                
        # Save results to cache
        if self.cache_manager.use_cache:
            self.cache_manager.save_to_cache(step_name, cache_key, speaker_audio_paths)
        
        # End timing
        self.performance_tracker.end_timing("speaker_audio")
        
        return speaker_audio_paths
    
    def save_translated_samples(self, segments: List[Dict], audio_file: str, synthesized_audio_dir: str = "artifacts/audio_chunks") -> None:
        """Save translated audio samples for each speaker with both original and translated text.
        
        Args:
            segments: List of transcript segments with translations
            audio_file: Path to the original audio file
            synthesized_audio_dir: Directory containing synthesized audio chunks
        """
        logger.debug("Saving translated samples for each speaker...")
        
        # Create a samples directory
        samples_dir = "artifacts/translated_samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        # Group segments by speaker
        segments_by_speaker = {}
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            
            # Store the segment index for referencing the synthesized audio
            segment["index"] = i
            
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment)
        
        # For each speaker, select up to 3 samples with clear speech
        samples_per_speaker = 3
        min_duration = 1.0  # Minimum segment duration in seconds
        
        # Create a text file to store all transcriptions
        transcription_file = os.path.join(samples_dir, "translated_samples.txt")
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write("Translated Speaker Samples with Original and Translated Text\n")
            f.write("====================================================\n\n")
            
            for speaker, segments in segments_by_speaker.items():
                f.write(f"Speaker: {speaker}\n")
                f.write("-" * 60 + "\n")
                
                # Sort segments by duration (descending)
                sorted_segments = sorted(
                    [s for s in segments if s["end"] - s["start"] >= min_duration], 
                    key=lambda x: x["end"] - x["start"], 
                    reverse=True
                )
                
                # Select samples (up to the specified number)
                samples = sorted_segments[:samples_per_speaker]
                
                if not samples:
                    f.write("No clear samples found for this speaker.\n\n")
                    continue
                
                # Extract and save audio samples
                for i, segment in enumerate(samples):
                    # Get the original segment index to reference the synthesized audio
                    segment_index = segment["index"]
                    
                    # Get the synthesized audio file path
                    synthesized_path = f"{synthesized_audio_dir}/{segment_index}.wav"
                    
                    if not os.path.exists(synthesized_path):
                        f.write(f"Sample {i+1}: Synthesized audio not found\n\n")
                        continue
                    
                    # Save the synthesized audio file
                    sample_filename = f"{speaker}_translated_sample_{i+1}.wav"
                    sample_path = os.path.join(samples_dir, sample_filename)
                    shutil.copy(synthesized_path, sample_path)
                    
                    # Also extract and save the original audio for comparison
                    original_audio = AudioSegment.from_file(audio_file)
                    start_ms = int(segment["start"] * 1000)
                    end_ms = int(segment["end"] * 1000)
                    original_segment_audio = original_audio[start_ms:end_ms]
                    
                    original_sample_filename = f"{speaker}_original_sample_{i+1}.wav"
                    original_sample_path = os.path.join(samples_dir, original_sample_filename)
                    original_segment_audio.export(original_sample_path, format="wav")
                    
                    # Write the comparison information
                    duration = segment["end"] - segment["start"]
                    f.write(f"Sample {i+1} ({duration:.2f}s):\n")
                    f.write(f"Original Audio: {original_sample_filename}\n")
                    f.write(f"Original Text: {segment['text']}\n")
                    f.write(f"Translated Audio: {sample_filename}\n")
                    f.write(f"Translated Text: {segment['translation']}\n\n")
                
                f.write("\n")
        
        logger.info(f"Saved translated samples to {samples_dir}")
        logger.debug(f"Transcription comparison file saved to {transcription_file}") 