"""Audio processing for the Smart Dubbing system."""

import os
import subprocess
import shutil
import json
import re
from statistics import median
from typing import Optional, List, Callable
from pydub import AudioSegment
from audio_separator.separator import Separator
from concurrent.futures import ThreadPoolExecutor, as_completed
from celery.exceptions import SoftTimeLimitExceeded

from ..core.cache_manager import CacheManager
from ..debug.performance_tracker import PerformanceTracker
from ..core.log_config import get_logger
from ..utils.progress_utils import tqdm_progress_callback

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio extraction and processing for the Smart Dubbing system."""
    
    def __init__(self, cache_manager: CacheManager, performance_tracker: PerformanceTracker):
        """Initialize the audio processor.
        
        Args:
            cache_manager: Cache manager instance
            performance_tracker: Performance tracker instance
        """
        self.cache_manager = cache_manager
        self.performance_tracker = performance_tracker
        self.total_duration = None
        
        # Create necessary directories
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Setup required directories for audio processing."""
        directories = ["artifacts/audio", "artifacts/speakers_audio", "artifacts/audio_chunks", "artifacts/su_audio_chunks"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def extract_audio(self, video_path: str, start_time: Optional[float] = None, 
                     duration: Optional[float] = None) -> str:
        """Extract audio from the input video file.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds to begin extraction
            duration: Duration in seconds to extract
            
        Returns:
            Path to the extracted audio file
        """
        # Start timing
        self.performance_tracker.start_timing("extract_audio")
        
        audio_file = "artifacts/audio/source.wav"
        
        # Set the total duration first
        self._determine_video_duration(video_path, start_time, duration)
        
        if start_time is not None or duration is not None:
            # Extract only the specified segment using ffmpeg
            ss_param = f"-ss {start_time}" if start_time is not None else ""
            t_param = f"-t {duration}" if duration is not None else ""
            
            trim_cmd = f'ffmpeg -y {ss_param} -i "{video_path}" {t_param} -y -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_file}'
            subprocess.run(trim_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            start_str = f"from {start_time}s" if start_time is not None else "from beginning"
            duration_str = f"for {duration}s" if duration is not None else "to the end"
            logger.debug(f"Extracted audio segment {start_str} {duration_str} to {audio_file}")
        else:
            # Extract full audio
            audio = AudioSegment.from_file(video_path, format="mp4")
            audio.export(audio_file, format="wav")
            logger.debug(f"Extracted full audio to {audio_file}")
        
        # End timing
        self.performance_tracker.end_timing("extract_audio")
        
        return audio_file
    
    def _determine_video_duration(self, video_path: str, start_time: Optional[float] = None, 
                                 duration: Optional[float] = None) -> None:
        """Determine and store the total duration of the video or segment being processed.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            duration: Duration in seconds
        """
        # If duration is explicitly provided, use it
        if duration is not None:
            self.total_duration = duration
            return
            
        # Otherwise, get the duration from ffmpeg
        try:
            duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
            full_duration = float(subprocess.check_output(duration_cmd, shell=True).decode().strip())
            
            # If we're processing a segment, calculate accordingly
            if start_time is not None:
                if duration is not None:
                    self.total_duration = duration
                else:
                    self.total_duration = full_duration - start_time
            else:
                self.total_duration = full_duration
                
            logger.debug(f"Video duration: {self.total_duration:.2f} seconds ({self.total_duration/60:.2f} minutes)")
        except Exception as e:
            logger.warning(f"Warning: Could not determine video duration: {e}")
            self.total_duration = None
    
    def process_background_audio(
        self,
        audio_file: str,
        voice_denoising: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Process and extract background audio if needed.

        Args:
            audio_file: Path to the audio file
            voice_denoising: Whether to perform voice denoising
            progress_callback: Optional callback(current, total, message) for progress updates
            log_callback: Optional callback(message) for logging to external systems

        Returns:
            Path to the background audio file or None
        """
        def log(message: str):
            logger.info(message)
            if log_callback:
                log_callback(message)
        # Start timing
        self.performance_tracker.start_timing("background_audio")

        if not voice_denoising:
            # No processing needed
            self.performance_tracker.end_timing("background_audio")
            return None

        # Generate cache key
        cache_key = self.cache_manager.generate_cache_key(
            audio_file, "", "", ""  # Empty values for non-transcription cache
        )

        step_name = "background_audio"

        # Check if results are cached (check for WAV file directly)
        output_path = "artifacts/audio/background.wav"
        if self.cache_manager.load_file_from_cache(step_name, cache_key, f"{cache_key}.wav", output_path):
            logger.debug("Loading background audio from cache...")
            if progress_callback:
                progress_callback(1, 1, "Loaded from cache")
            self.performance_tracker.end_timing("background_audio")
            return output_path

        try:
            log("Extracting background audio...")

            # Initialize audio separator
            separator = Separator()
            separator.load_model(model_filename='2_HP-UVR.pth')

            # Separate vocals and background with progress tracking
            with tqdm_progress_callback(progress_callback, "Separating audio"):
                output_file_paths = separator.separate(audio_file)[0]

            log("Audio separation complete")

            # Move the background audio to our audio directory
            background_audio_path = "artifacts/audio/background.wav"
            shutil.move(output_file_paths, background_audio_path)

            # Save to cache
            self.cache_manager.save_file_to_cache(step_name, cache_key, background_audio_path, f"{cache_key}.wav")

            # End timing
            self.performance_tracker.end_timing("background_audio")

            return background_audio_path

        except SoftTimeLimitExceeded:
            # If background audio processing takes too long, skip it and continue
            log("Background audio separation timed out - continuing without background audio")
            logger.warning("Background audio separation exceeded time limit - skipping background audio")
            self.performance_tracker.end_timing("background_audio")
            return None
        except Exception as e:
            # If background audio processing fails for any reason, log and continue without it
            log(f"Background audio separation failed: {str(e)} - continuing without background audio")
            logger.warning(f"Background audio separation failed: {e} - skipping background audio")
            self.performance_tracker.end_timing("background_audio")
            return None
    
    def normalize_audio(self, audio_path: str, mode: str = "gentle") -> str:
        """Apply audio normalization with configurable gentleness.
        
        Args:
            audio_path: Path to the audio file to normalize
            mode: Normalization mode - "ultra_gentle", "gentle", or "standard"
                 Default is "gentle" for quality preservation
        
        Returns:
            Path to the normalized audio file
        """
        # Validate input path
        if not audio_path or not audio_path.strip():
            logger.error("Error: Empty audio path provided to normalize_audio")
            raise ValueError("Audio path cannot be empty")
        
        if not os.path.exists(audio_path):
            logger.error(f"Error: Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Start timing
        self.performance_tracker.start_timing("audio_normalization")
        
        logger.info(f"Applying {mode} audio normalization...")
        normalized_path = f"{os.path.splitext(audio_path)[0]}_{mode}_normalized.wav"
        
        # Validate normalized path
        if not normalized_path or not normalized_path.strip():
            logger.error("Error: Generated normalized path is empty")
            self.performance_tracker.end_timing("audio_normalization")
            return audio_path
        
        # Try to load from cache if cache manager is available
        if self.cache_manager and self.cache_manager.use_cache:
            cache_key = self.cache_manager.generate_cache_key(
                audio_path, mode, "v3", ""
            )
            step_name = "audio_normalization"
            
            cached_file_name = f"{cache_key}_{mode}_normalized.wav"
            if self.cache_manager.load_file_from_cache(step_name, cache_key, cached_file_name, normalized_path):
                logger.debug(f"Loading {mode} normalized audio from cache...")
                self.performance_tracker.end_timing("audio_normalization")
                return normalized_path
        
        # Choose normalization approach based on mode
        if mode == "ultra_gentle":
            # Ultra-gentle: only prevent clipping, no compression
            normalize_cmd = (
                f'ffmpeg -y -i "{audio_path}" '
                f'-af "'
                # Simple peak detection and limiting only when needed
                f'astats=metadata=1:reset=1,'
                f'volume=volume=0dB:eval=peak'
                f'" '
                f'-c:a pcm_s16le -ar 16000 -ac 1 "{normalized_path}"'
            )
            
        elif mode == "gentle":
            # Gentle: soft limiting with minimal compression
            normalize_cmd = (
                f'ffmpeg -y -i "{audio_path}" '
                f'-af "'
                # Soft limiter with gentle settings
                f'alimiter=level_in=1:level_out=1:limit=0.9:attack=2:release=20,'
                # Very gentle compression only for peaks
                f'acompressor=threshold=-6dB:ratio=1.5:attack=1:release=10:makeup=0dB'
                f'" '
                f'-c:a pcm_s16le -ar 16000 -ac 1 "{normalized_path}"'
            )
            
        else:  # mode == "standard"
            # Standard: use the original EBU R128 approach
            return self._normalize_audio_standard(audio_path)
        
        try:
            logger.debug(f"Applying {mode} audio normalization...")
            self._run_ffmpeg_command(normalize_cmd, f"Applying {mode} volume normalization...")
            
            # Verify the output file was created
            if os.path.exists(normalized_path) and os.path.getsize(normalized_path) > 0:
                logger.debug(f"Audio {mode} normalized and saved to {normalized_path}")
            else:
                logger.warning(f"{mode} normalization failed, falling back to original...")
                self.performance_tracker.end_timing("audio_normalization")
                return audio_path
            
            # Save to cache if normalization was successful
            if self.cache_manager and self.cache_manager.use_cache and os.path.exists(normalized_path):
                try:
                    cache_key = self.cache_manager.generate_cache_key(
                        audio_path, mode, "v3", ""
                    )
                    step_name = "audio_normalization"
                    cached_file_name = f"{cache_key}_{mode}_normalized.wav"
                    
                    self.cache_manager.save_file_to_cache(step_name, cache_key, normalized_path, cached_file_name)
                    logger.debug(f"Cached {mode} normalized audio: {cached_file_name}")
                except Exception as cache_error:
                    logger.warning(f"Warning: Failed to cache normalized audio: {cache_error}")
            
            self.performance_tracker.end_timing("audio_normalization")
            return normalized_path
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Warning: {mode} audio normalization failed: {e}")
            self.performance_tracker.end_timing("audio_normalization")
            return audio_path
    
    def _normalize_audio_standard(self, audio_path: str) -> str:
        """Original EBU R128 normalization method (kept for backward compatibility)."""
        normalized_path = f"{os.path.splitext(audio_path)[0]}_standard_normalized.wav"
        
        # 1) Analyse loudness using first pass (collect json stats)
        analyze_cmd = (
            f'ffmpeg -y -i "{audio_path}" '
            f'-af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null -'
        )

        loudness_data = None
        try:
            logger.debug("Running first-pass loudnorm analysis...")
            result = subprocess.run(
                analyze_cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Parse JSON data enclosed in braces { ... }
            match = re.search(r"\{[\s\S]*?\}", result.stderr)
            if match:
                loudness_data = json.loads(match.group())
                logger.debug(f"Loudnorm analysis data: {loudness_data}")
            else:
                logger.warning("Could not parse loudnorm analysis output – falling back to single pass.")
        except subprocess.CalledProcessError as analysis_error:
            logger.warning(f"First pass loudnorm analysis failed: {analysis_error}. Falling back to single-pass normalization.")

        # 2) Build normalization command
        if loudness_data:
            normalize_cmd = (
                f'ffmpeg -y -i "{audio_path}" '
                f'-af loudnorm=I=-16:TP=-1.5:LRA=11:'
                f'measured_I={loudness_data["input_i"]}:'
                f'measured_TP={loudness_data["input_tp"]}:'
                f'measured_LRA={loudness_data["input_lra"]}:'
                f'measured_thresh={loudness_data["input_thresh"]}:'
                f'offset={loudness_data["target_offset"]}:'
                f'linear=true:print_format=summary '
                f'"{normalized_path}"'
            )
            logger.debug("Using two-pass loudnorm for normalization.")
        else:
            # Fallback – single pass
            normalize_cmd = (
                f'ffmpeg -y -i "{audio_path}" -af loudnorm=I=-16:TP=-1.5:LRA=11 '
                f'"{normalized_path}"'
            )

        try:
            self._run_ffmpeg_command(normalize_cmd, "Applying standard volume normalization...")
            logger.debug(f"Audio normalized and saved to {normalized_path}")
            return normalized_path
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Warning: Standard audio normalization failed: {e}")
            return audio_path
    
    def _run_ffmpeg_command(self, command: str, message: str = None) -> None:
        """Run an FFmpeg command with proper error handling and output suppression.
        
        Args:
            command: FFmpeg command to run
            message: Optional message to print before running the command
        """
        # Validate command
        if not command or not command.strip():
            logger.error("Error: Empty or whitespace-only FFmpeg command")
            raise ValueError("FFmpeg command cannot be empty")
            
        if message:
            logger.debug(message)
            
        logger.debug(f"Executing FFmpeg command: {command}")
            
        try:
            # Run the command and capture output
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Only print stderr if it contains error messages that aren't just informational
            stderr = result.stderr
            if stderr and ('error' in stderr.lower() or 'fatal' in stderr.lower()):
                logger.error("FFmpeg errors:")
                for line in stderr.split('\n'):
                    if 'error' in line.lower() or 'fatal' in line.lower():
                        logger.error(f"  {line}")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else "No error details available"
            logger.error(f"FFmpeg command failed: {command}")
            logger.error(f"Error output: {error_output}")
            raise
    
    def get_total_duration(self) -> Optional[float]:
        """Get the total duration of the processed audio/video.
        
        Returns:
            Total duration in seconds or None if not determined
        """
        return self.total_duration
    
    def normalize_segments_to_average(self, segment_files: List[str]) -> None:
        """Softly align segment loudness and gently level dynamics per segment.
        
        The method performs a robust, gentle normalization across segments and
        within each segment:
        - Detects mean loudness of each segment (ffmpeg volumedetect)
        - Uses median loudness as the global target to reduce influence of outliers
        - Applies a capped gain per segment (default: +3 dB / -6 dB)
        - Smooths gain differences between adjacent segments (max step: 2 dB)
        - Adds mild compression and peak limiting within each segment
        - Applies short 50 ms fade-in/out to avoid clicks at boundaries
        
        Original files are overwritten in-place upon success.
        
        Args:
            segment_files: List of paths to segment audio files
        """
        if not segment_files:
            logger.debug("No segments to normalize")
            return
        
        logger.info(f"Normalizing {len(segment_files)} segments to average volume with fading...")
        self.performance_tracker.start_timing("segment_normalization")
        
        # Step 1: Detect volume for all segments
        logger.debug("Step 1/3: Detecting volume levels for all segments...")
        segment_volumes = []
        
        def detect_volume(segment_file: str) -> tuple:
            """Detect volume for a single segment."""
            if not os.path.exists(segment_file):
                logger.warning(f"Segment file not found: {segment_file}")
                return (segment_file, None)
            
            try:
                # Use ffmpeg volumedetect filter to measure volume
                cmd = [
                    'ffmpeg', '-i', segment_file, '-af', 'volumedetect',
                    '-f', 'null', '-'
                ]
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    check=False
                )
                
                # Parse mean volume from stderr
                mean_volume = None
                for line in result.stderr.split('\n'):
                    if 'mean_volume:' in line:
                        # Extract the dB value
                        match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', line)
                        if match:
                            mean_volume = float(match.group(1))
                            break
                
                if mean_volume is None:
                    logger.warning(f"Could not detect volume for {segment_file}")
                    return (segment_file, None)
                
                return (segment_file, mean_volume)
                
            except Exception as e:
                logger.warning(f"Error detecting volume for {segment_file}: {e}")
                return (segment_file, None)
        
        # Detect volumes in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(segment_files))) as executor:
            futures = {executor.submit(detect_volume, sf): sf for sf in segment_files}
            
            for future in as_completed(futures):
                segment_file, volume = future.result()
                if volume is not None:
                    segment_volumes.append((segment_file, volume))
        
        if not segment_volumes:
            logger.warning("Could not detect volume for any segments, skipping normalization")
            self.performance_tracker.end_timing("segment_normalization")
            return
        
        # Step 2: Calculate robust target (median) and build a smoothed gain plan
        volume_map = {sf: vol for sf, vol in segment_volumes}
        ordered_volumes = [volume_map[sf] for sf in segment_files if sf in volume_map]
        if not ordered_volumes:
            logger.warning("No segments with detected volume in provided order; skipping normalization")
            self.performance_tracker.end_timing("segment_normalization")
            return

        target_volume_db = float(median(ordered_volumes))
        logger.info(
            f"Global target (median) volume: {target_volume_db:.2f} dB (from {len(ordered_volumes)} segments)"
        )

        # Gentle caps and smoothing
        max_gain_up_db = 3.0   # do not boost more than +3 dB to avoid raising noise
        max_gain_down_db = 6.0 # allow up to -6 dB attenuation
        max_step_db = 2.0      # limit jump between adjacent segments

        planned_gains: List[tuple[str, float]] = []
        previous_gain: Optional[float] = None
        for sf in segment_files:
            if sf not in volume_map:
                continue
            raw_gain = target_volume_db - volume_map[sf]
            clamped = max(-max_gain_down_db, min(max_gain_up_db, raw_gain))
            if previous_gain is not None:
                # Smooth transitions between segments
                delta = clamped - previous_gain
                if delta > max_step_db:
                    clamped = previous_gain + max_step_db
                elif delta < -max_step_db:
                    clamped = previous_gain - max_step_db
            planned_gains.append((sf, clamped))
            previous_gain = clamped

        # Step 3: Normalize each segment per plan with gentle dynamics and fading
        logger.debug("Step 2/3: Normalizing segments with soft gains, compression, and fades...")

        def normalize_segment(segment_file: str, planned_gain_db: float) -> bool:
            """Normalize a single segment with gentle gain, compression, and fades."""
            try:
                # Get segment duration for fade-out timing
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', segment_file
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                duration = float(probe_result.stdout.strip())
                
                # Calculate fade-out start time (duration - 0.05 seconds, but not less than 0)
                fade_out_start = max(0, duration - 0.05)
                
                # Create temporary output file
                temp_output = f"{segment_file}.temp_normalized.wav"
                
                # Build filter chain: gentle volume, compressor, limiter, and fades
                # Only apply fades if duration is sufficient (> 0.1s for both fades)
                base_chain = (
                    f"volume={planned_gain_db}dB,"
                    # Mild compression for intra-segment leveling
                    f"acompressor=threshold=-18dB:ratio=1.8:attack=10:release=200:knee=4:makeup=0dB,"
                    # Soft peak limiting to avoid overs
                    f"alimiter=limit=0.95:attack=5:release=50"
                )

                if duration > 0.1:
                    filter_chain = (
                        f"{base_chain},"
                        f"afade=t=in:st=0:d=0.05,"
                        f"afade=t=out:st={fade_out_start}:d=0.05"
                    )
                else:
                    filter_chain = base_chain
                
                # Apply normalization and fading
                cmd = [
                    'ffmpeg', '-y', '-i', segment_file,
                    '-af', filter_chain,
                    '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_output
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Replace original file with normalized version
                shutil.move(temp_output, segment_file)
                
                return True
                
            except Exception as e:
                logger.warning(f"Error normalizing segment {segment_file}: {e}")
                # Clean up temp file if it exists
                temp_output = f"{segment_file}.temp_normalized.wav"
                if os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except Exception:
                        pass
                return False
        
        # Normalize segments in parallel
        success_count = 0
        with ThreadPoolExecutor(max_workers=min(4, len(planned_gains))) as executor:
            futures = {
                executor.submit(normalize_segment, seg_file, gain_db): seg_file 
                for seg_file, gain_db in planned_gains
            }
            
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
        
        logger.info(f"Step 3/3: Successfully normalized {success_count}/{len(planned_gains)} segments")
        
        self.performance_tracker.end_timing("segment_normalization")
        logger.info("Segment normalization and fading complete") 