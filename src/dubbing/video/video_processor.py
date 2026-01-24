"""Video processing for the Smart Dubbing system."""

import os
import subprocess
import json
import tempfile
import shutil
from typing import Optional, List, Tuple, Dict, Any

from ..debug.performance_tracker import PerformanceTracker
from ..audio.audio_processor import AudioProcessor
from ..core.log_config import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Handles video processing for the Smart Dubbing system."""

    def __init__(self, performance_tracker: PerformanceTracker, config: Optional[Dict[str, Any]] = None):
        """Initialize the video processor.

        Args:
            performance_tracker: Performance tracker instance
            config: Optional configuration dict (DubbingConfig.config)
        """
        self.performance_tracker = performance_tracker
        self.config = config or {}

        # Extract video processing settings
        self.video_minterpolate_threshold = self.config.get('video_minterpolate_threshold', 0.75)
    
    def _get_video_info(self, video_path: str) -> Dict[str, any]:
        """Get detailed video information including codec, bitrate, and other parameters.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            return {
                'video_codec': video_stream.get('codec_name') if video_stream else None,
                'video_bitrate': video_stream.get('bit_rate') if video_stream else None,
                'video_profile': video_stream.get('profile') if video_stream else None,
                'video_level': video_stream.get('level') if video_stream else None,
                'video_pix_fmt': video_stream.get('pix_fmt') if video_stream else None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'audio_bitrate': audio_stream.get('bit_rate') if audio_stream else None,
                'duration': float(info.get('format', {}).get('duration', 0)),
                'format_name': info.get('format', {}).get('format_name', ''),
                'video_stream': video_stream,
                'audio_stream': audio_stream
            }
        except Exception as e:
            logger.warning(f"Failed to get video info for {video_path}: {e}")
            return {}
    
    def _can_use_stream_copy(self, cuts: List[Tuple[float, float]], keyframes: List[float], 
                           tolerance: float = 0.1) -> bool:
        """Check if we can use stream copy for video cuts based on keyframe alignment.
        
        Args:
            cuts: List of video cuts (start, end) times
            keyframes: List of keyframe timestamps
            tolerance: Tolerance for keyframe alignment in seconds
            
        Returns:
            True if stream copy can be used, False if re-encoding is needed
        """
        if not keyframes:
            logger.debug("No keyframes detected, cannot use stream copy")
            return False
        
        for start, end in cuts:
            # Check if cut start is close to a keyframe
            start_aligned = any(abs(start - kf) <= tolerance for kf in keyframes)
            # End doesn't need to be keyframe-aligned for most cases
            
            if not start_aligned:
                logger.debug(f"Cut at {start:.2f}s not aligned with keyframes, cannot use stream copy")
                return False
        
        logger.debug("All cuts are keyframe-aligned, can use stream copy")
        return True

    def _can_use_stream_copy_for_trim(self, video_path: str, start_time: Optional[float] = None, 
                                    duration: Optional[float] = None, tolerance: float = 0.1) -> bool:
        """Check if video trimming can use stream copy based on keyframe alignment.
        
        Args:
            video_path: Path to the video file
            start_time: Start time for trimming
            duration: Duration for trimming
            tolerance: Tolerance for keyframe alignment in seconds
            
        Returns:
            True if stream copy can be used for trimming, False otherwise
        """
        if start_time is None and duration is None:
            return True  # No trimming needed
        
        try:
            keyframes = self._extract_keyframes(video_path)
            if not keyframes:
                return False
            
            # Check if start time aligns with a keyframe
            if start_time is not None:
                start_aligned = any(abs(start_time - kf) <= tolerance for kf in keyframes)
                if not start_aligned:
                    logger.debug(f"Trim start {start_time:.2f}s not aligned with keyframes")
                    return False
            
            # For end time, we're more flexible since it's easier to handle
            logger.debug("Trim parameters allow stream copy")
            return True
            
        except Exception as e:
            logger.debug(f"Could not check keyframe alignment for trimming: {e}")
            return False
        
    def _get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video file in seconds."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get video duration for {video_path}: {e}")
            return 0.0

    def _run_ffmpeg_with_progress(
        self,
        cmd: List[str],
        total_duration: float,
        progress_callback: Optional[callable] = None,
        operation_name: str = "FFmpeg"
    ) -> subprocess.CompletedProcess:
        """Run FFmpeg command with progress tracking.
        
        Args:
            cmd: FFmpeg command as list of arguments
            total_duration: Expected output duration in seconds for progress calculation
            progress_callback: Optional callback(current, total, message) for progress updates
            operation_name: Name of the operation for logging
            
        Returns:
            CompletedProcess result
        """
        import re
        
        if not progress_callback or total_duration <= 0:
            return subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stderr_lines = []
        last_progress = 0
        time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')
        
        for line in process.stderr:
            stderr_lines.append(line)
            match = time_pattern.search(line)
            if match:
                hours, minutes, seconds = match.groups()
                current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                progress_pct = min(100, int((current_time / total_duration) * 100))
                
                if progress_pct > last_progress:
                    last_progress = progress_pct
                    progress_callback(
                        int(current_time),
                        int(total_duration),
                        f"{operation_name}: {progress_pct}%"
                    )
        
        stdout, _ = process.communicate()
        stderr = ''.join(stderr_lines)
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stdout, stderr
            )
        
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

    def _detect_silence(self, audio_or_video_path: str, min_silence_duration: float) -> List[Tuple[float, float]]:
        """Detect periods of silence in an audio file or video's audio stream."""
        import re
        logger.info(f"Detecting silence periods longer than {min_silence_duration}s...")
        try:
            cmd = [
                'ffmpeg', '-i', audio_or_video_path,
                '-af', f"silencedetect=noise=-30dB:duration={min_silence_duration}",
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            stderr_output = result.stderr
            
            starts = re.findall(r'silence_start: (\d+\.?\d*)', stderr_output)
            ends = re.findall(r'silence_end: (\d+\.?\d*)', stderr_output)
            
            if len(starts) != len(ends):
                logger.warning(f"Mismatch in detected silence start/end times. Starts: {len(starts)}, Ends: {len(ends)}")
                if len(starts) == len(ends) + 1:
                    duration = self._get_video_duration(audio_or_video_path)
                    ends.append(str(duration))
                    logger.debug(f"Appended file duration ({duration}s) as final silence_end.")

            pauses = []
            for i in range(min(len(starts), len(ends))):
                start_time = float(starts[i])
                end_time = float(ends[i])
                if end_time > start_time:
                    pauses.append((start_time, end_time))
            
            logger.info(f"Detected {len(pauses)} silence periods.")
            return pauses

        except Exception as e:
            logger.error(f"Error detecting silence: {e}", exc_info=True)
            return []
    
    def _extract_keyframes(self, video_path: str) -> List[float]:
        """Extract keyframe timestamps from video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of keyframe timestamps in seconds
        """
        try:
            # Use ffprobe to get keyframe information
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_frames', '-select_streams', 'v:0',
                '-show_entries', 'frame=pkt_pts_time,key_frame',
                '-of', 'csv=print_section=0', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            keyframes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            is_keyframe = int(parts[1])
                            if is_keyframe == 1:
                                timestamp = float(parts[0])
                                keyframes.append(timestamp)
                        except (ValueError, IndexError):
                            continue
            
            return sorted(keyframes)
            
        except Exception as e:
            logger.warning(f"Failed to extract keyframes: {e}, assuming no keyframes")
            return []
    
    def _filter_pauses_with_keyframes(self, pauses: List[Tuple[float, float]], 
                                     keyframes: List[float], buffer: float) -> List[Tuple[float, float]]:
        """Filter out pauses that contain keyframes.
        
        Args:
            pauses: List of pause intervals
            keyframes: List of keyframe timestamps
            buffer: Buffer around keyframes to preserve
            
        Returns:
            List of pauses that don't contain keyframes
        """
        removable_pauses = []
        
        for pause_start, pause_end in pauses:
            has_keyframe = False
            
            # Check if any keyframe falls within this pause (with buffer)
            for keyframe_time in keyframes:
                if (pause_start - buffer) <= keyframe_time <= (pause_end + buffer):
                    has_keyframe = True
                    logger.debug(f"Preserving pause {pause_start:.2f}s-{pause_end:.2f}s "
                               f"(keyframe at {keyframe_time:.2f}s)")
                    break
            
            if not has_keyframe:
                removable_pauses.append((pause_start, pause_end))
                logger.debug(f"Marking for removal: pause {pause_start:.2f}s-{pause_end:.2f}s")
        
        return removable_pauses
    
    def _filter_edge_pauses(self, pauses: List[Tuple[float, float]], 
                           video_duration: float, edge_threshold: float = 0.5) -> List[Tuple[float, float]]:
        """Filter out pauses that are at the beginning or end of the video.
        
        Args:
            pauses: List of pause intervals
            video_duration: Total duration of the video in seconds
            edge_threshold: Threshold in seconds to consider a pause as being at the edge
            
        Returns:
            List of pauses that are not at the beginning or end of the video
        """
        filtered_pauses = []
        
        for pause_start, pause_end in pauses:
            # Check if pause is at the beginning of the video
            is_at_beginning = pause_start <= edge_threshold
            
            # Check if pause is at the end of the video
            is_at_end = pause_end >= (video_duration - edge_threshold)
            
            if is_at_beginning:
                logger.debug(f"Preserving pause at video beginning: {pause_start:.2f}s-{pause_end:.2f}s")
            elif is_at_end:
                logger.debug(f"Preserving pause at video end: {pause_start:.2f}s-{pause_end:.2f}s")
            else:
                filtered_pauses.append((pause_start, pause_end))
                logger.debug(f"Allowing pause removal: {pause_start:.2f}s-{pause_end:.2f}s")
        
        return filtered_pauses

    def _format_filter_input_label(self, label: str) -> str:
        """Return a filter_complex input label bracketed exactly once.

        This ensures we can accept either raw stream specs like "1:a:0" or
        already-bracketed labels like "[dub_mixed_with_bg]" without producing
        invalid double-bracketed tokens inside filter graphs.
        """
        if label.startswith("[") and label.endswith("]"):
            return label
        return f"[{label}]"
    
    def _run_ffmpeg_concat(self, input_path: str, cuts_batch: List[Tuple[float, float]], batch_output_path: str, 
                          video_info: Optional[Dict] = None, keyframes: Optional[List[float]] = None,
                          use_two_pass_encoding: bool = False):
        """Helper function to run the ffmpeg concat process for a given list of cuts with quality preservation."""
        try:
            # Get video info if not provided
            if video_info is None:
                video_info = self._get_video_info(input_path)
            
            # Check if we can use stream copy for better quality preservation
            can_use_stream_copy = False
            if keyframes and self._can_use_stream_copy(cuts_batch, keyframes, tolerance=0.2):
                can_use_stream_copy = True
                logger.debug("Using stream copy for lossless video processing")
            else:
                logger.debug("Stream copy not possible, using high-quality re-encoding")
            
            input_parts = []
            for start, end in cuts_batch:
                input_parts.extend(['-ss', str(start), '-t', str(end - start), '-i', input_path])

            filter_streams = []
            for i in range(len(cuts_batch)):
                filter_streams.append(f'[{i}:v:0]')
                filter_streams.append(f'[{i}:a:0]')

            concat_filter = f'{"".join(filter_streams)}concat=n={len(cuts_batch)}:v=1:a=1[outv][outa]'

            cmd = ['ffmpeg', '-y'] + input_parts + [
                '-filter_complex', concat_filter,
                '-map', '[outv]', '-map', '[outa]'
            ]
            
            # Add encoding options based on whether we can use stream copy
            if can_use_stream_copy:
                # Try stream copy first (lossless)
                cmd.extend(['-c:v', 'copy', '-c:a', 'copy'])
                cmd.append(batch_output_path)
                
                logger.debug(f"Executing FFmpeg concat: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            else:
                # Check if we should use two-pass encoding
                original_bitrate = video_info.get('video_bitrate')
                if (use_two_pass_encoding and original_bitrate and 
                    original_bitrate.isdigit() and int(original_bitrate) > 1000000):
                    # Use two-pass encoding for better quality
                    logger.info("Using two-pass encoding for concat operation")
                    self._encode_with_two_pass(cmd, batch_output_path, video_info)
                else:
                    # Use single-pass high-quality re-encoding settings
                    original_codec = video_info.get('video_codec', 'h264')
                    
                    # Video encoding with conservative quality preservation
                    if original_codec in ['h264', 'libx264']:
                        cmd.extend(['-c:v', 'libx264'])
                        
                        # Prefer original bitrate for best quality preservation
                        if original_bitrate and original_bitrate.isdigit() and int(original_bitrate) > 1000000:  # > 1 Mbps
                            cmd.extend(['-b:v', original_bitrate])
                            logger.debug(f"Using original video bitrate: {int(original_bitrate)//1000} kbps")
                        else:
                            # Use CRF 20 for good quality without being too aggressive
                            cmd.extend(['-crf', '20'])
                            logger.debug("Using CRF 20 for high-quality re-encoding")
                        
                        # Preserve original profile and pixel format when valid
                        if video_info.get('video_profile') and video_info['video_profile'] in ['high', 'main', 'baseline']:
                            cmd.extend(['-profile:v', video_info['video_profile']])
                        if video_info.get('video_pix_fmt') and 'yuv420p' in video_info['video_pix_fmt']:
                            cmd.extend(['-pix_fmt', video_info['video_pix_fmt']])
                        
                        cmd.extend(['-preset', 'medium'])  # Balanced quality/speed
                    else:
                        # For other codecs, use conservative settings
                        cmd.extend(['-c:v', 'libx264', '-crf', '20', '-preset', 'medium'])
                    
                    # Audio encoding - preserve original settings when possible
                    original_audio_codec = video_info.get('audio_codec', 'aac')
                    original_audio_bitrate = video_info.get('audio_bitrate')
                    
                    if original_audio_codec == 'aac' and original_audio_bitrate:
                        cmd.extend(['-c:a', 'aac', '-b:a', original_audio_bitrate])
                    elif original_audio_codec in ['mp3', 'libmp3lame'] and original_audio_bitrate:
                        cmd.extend(['-c:a', 'libmp3lame', '-b:a', original_audio_bitrate])
                    else:
                        # High quality AAC fallback
                        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

                    cmd.append(batch_output_path)

                    logger.debug(f"Executing FFmpeg concat: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stderr and ('error' in result.stderr.lower() or 'fatal' in result.stderr.lower()):
                logger.warning(f"FFmpeg warnings during concat operation: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            # If stream copy failed, retry with re-encoding
            if can_use_stream_copy and 'does not support' in str(e.stderr):
                logger.warning("Stream copy failed, retrying with re-encoding")
                return self._run_ffmpeg_concat(input_path, cuts_batch, batch_output_path, video_info, None, False)
            else:
                logger.error(f"FFmpeg command in _run_ffmpeg_concat failed: {e.stderr}")
                raise

    def _apply_video_cuts(self, input_path: str, cuts: List[Tuple[float, float]], output_path: str, 
                         batch_size: int = 50, keyframes: Optional[List[float]] = None,
                         use_two_pass_encoding: bool = False):
        """Apply video cuts using FFmpeg with batching to conserve memory while preserving quality."""
        if not cuts:
            logger.warning("No cuts to apply.")
            return

        # Get video info once for quality preservation
        video_info = self._get_video_info(input_path)
        logger.info(f"Original video: {video_info.get('video_codec', 'unknown')} codec, "
                   f"{int(video_info.get('video_bitrate', 0))//1000 if video_info.get('video_bitrate') and video_info.get('video_bitrate').isdigit() else 'unknown'} kbps")

        # If the number of cuts is small, process directly without batching.
        if len(cuts) <= batch_size:
            logger.info(f"Re-joining {len(cuts)} video segments directly (number is within batch size).")
            self._run_ffmpeg_concat(input_path, cuts, output_path, video_info, keyframes, use_two_pass_encoding)
            return

        # Batch processing for a large number of cuts
        logger.info(f"Re-joining {len(cuts)} video segments. Processing in batches of {batch_size} due to large number.")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_files = []
            num_batches = (len(cuts) + batch_size - 1) // batch_size
            
            # 1. Process cuts in batches, creating intermediate video files
            for i in range(0, len(cuts), batch_size):
                batch_cuts = cuts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                batch_output_path = os.path.join(temp_dir, f"batch_{batch_num}.mp4")
                temp_files.append(batch_output_path)
                
                logger.info(f"Processing batch {batch_num}/{num_batches} with {len(batch_cuts)} cuts...")
                
                try:
                    self._run_ffmpeg_concat(input_path, batch_cuts, batch_output_path, video_info, keyframes, use_two_pass_encoding)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to process batch {batch_num}: {e.stderr}")
                    raise

            # 2. Concatenate the intermediate batch files with quality preservation
            logger.info(f"Concatenating {len(temp_files)} batch files into final output...")
            
            try:
                concat_inputs = []
                for temp_file in temp_files:
                    concat_inputs.extend(['-i', temp_file])

                filter_streams = []
                for i in range(len(temp_files)):
                    filter_streams.append(f'[{i}:v:0]')
                    filter_streams.append(f'[{i}:a:0]')
                
                concat_filter = f'{"".join(filter_streams)}concat=n={len(temp_files)}:v=1:a=1[outv][outa]'
                
                final_cmd = ['ffmpeg', '-y'] + concat_inputs + [
                    '-filter_complex', concat_filter,
                    '-map', '[outv]', '-map', '[outa]'
                ]
                
                # Check if we should use two-pass encoding for final concatenation
                original_bitrate = video_info.get('video_bitrate')
                if (use_two_pass_encoding and original_bitrate and 
                    original_bitrate.isdigit() and int(original_bitrate) > 1000000):
                    # Use two-pass encoding for better quality
                    logger.info("Using two-pass encoding for final concatenation")
                    self._encode_with_two_pass(final_cmd, output_path, video_info)
                else:
                    # Use high-quality settings for final concatenation to preserve quality
                    original_codec = video_info.get('video_codec', 'h264')
                    
                    # Video encoding with conservative quality preservation
                    if original_codec in ['h264', 'libx264']:
                        final_cmd.extend(['-c:v', 'libx264'])
                        
                        # Prefer original bitrate for best quality preservation
                        if original_bitrate and original_bitrate.isdigit() and int(original_bitrate) > 1000000:  # > 1 Mbps
                            final_cmd.extend(['-b:v', original_bitrate])
                            logger.debug(f"Final concat using original video bitrate: {int(original_bitrate)//1000} kbps")
                        else:
                            # Use CRF 20 for good quality balance
                            final_cmd.extend(['-crf', '20'])
                            logger.debug("Final concat using CRF 20 for balanced quality")
                        
                        # Preserve original profile and pixel format when valid
                        if video_info.get('video_profile') and video_info['video_profile'] in ['high', 'main', 'baseline']:
                            final_cmd.extend(['-profile:v', video_info['video_profile']])
                        if video_info.get('video_pix_fmt') and 'yuv420p' in video_info['video_pix_fmt']:
                            final_cmd.extend(['-pix_fmt', video_info['video_pix_fmt']])
                        
                        final_cmd.extend(['-preset', 'medium'])  # Balanced quality/speed
                    else:
                        # For other codecs, use conservative settings
                        final_cmd.extend(['-c:v', 'libx264', '-crf', '20', '-preset', 'medium'])
                    
                    # Audio encoding - preserve original settings when possible
                    original_audio_codec = video_info.get('audio_codec', 'aac')
                    original_audio_bitrate = video_info.get('audio_bitrate')
                    
                    if original_audio_codec == 'aac' and original_audio_bitrate:
                        final_cmd.extend(['-c:a', 'aac', '-b:a', original_audio_bitrate])
                    elif original_audio_codec in ['mp3', 'libmp3lame'] and original_audio_bitrate:
                        final_cmd.extend(['-c:a', 'libmp3lame', '-b:a', original_audio_bitrate])
                    else:
                        # High quality AAC fallback
                        final_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
                    
                    final_cmd.append(output_path)
                    
                    logger.debug(f"Running final concat command: {' '.join(final_cmd)}")
                    result = subprocess.run(final_cmd, capture_output=True, text=True, check=True)
                
                if result.stderr and ('error' in result.stderr.lower() or 'fatal' in result.stderr.lower()):
                    logger.warning(f"FFmpeg warnings during final concatenation: {result.stderr}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to concatenate batch files: {e}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
                raise
        
        logger.info("Batch processing complete and temporary files cleaned up.")
    
    def combine_audio_with_video(self,
                                video_path: str,
                                translated_audio_path: str,
                                background_audio_path: Optional[str] = None,
                                watermark_path: Optional[str] = None,
                                watermark_text: Optional[str] = None,
                                include_original_audio: bool = False,
                                output_file: Optional[str] = None,
                                start_time: Optional[float] = None,
                                duration: Optional[float] = None,
                                keep_original_audio_ranges: Optional[List[Tuple[float, float]]] = None,
                                source_language: str = "en",
                                target_language: str = "es",
                                normalize_audio: bool = True,
                                use_two_pass_encoding: bool = False,
                                pause_removal: str = "disabled",
                                min_pause_duration: float = 3,
                                preserve_pause_duration: float = 1.5,
                                video_speed_min: float = 0.75,
                                video_speed_comfortable: float = 1.25,
                                video_speed_max: float = 1.5,
                                segment_positions: Optional[List[Dict]] = None,
                                keyframe_buffer: float = 0.2,
                                ffmpeg_batch_size: int = 50,
                                dubbed_volume: float = 1.0,
                                background_volume: float = 0.562341,
                                upscale_factor: float = 1.0,
                                upscale_sharpen: bool = True,
                                progress_callback: Optional[callable] = None,
                                log_callback: Optional[callable] = None,
                                segments_with_video_speed: Optional[List[Dict]] = None,
                                video_segment_speed_min: float = 0.75) -> Tuple[str, List[Dict[str, float]]]:
        """Combine the translated audio with the original video, optionally adding a watermark and processing pauses.

        Args:
            video_path: Path to the original video
            translated_audio_path: Path to the translated audio
            background_audio_path: Path to the background audio (optional)
            watermark_path: Path to the watermark image (optional)
            watermark_text: Text to display under the watermark (optional)
            include_original_audio: Whether to include the original audio track
            output_file: Optional path for the output video file
            start_time: Start time in seconds (for trimming)
            duration: Duration in seconds (for trimming)
            keep_original_audio_ranges: Optional list of [start, end] tuples to keep original audio
            source_language: Source language code for metadata
            target_language: Target language code for metadata
            normalize_audio: Whether to normalize audio volume (default: True)
            use_two_pass_encoding: Whether to use two-pass encoding for better quality
            pause_removal: Pause removal mode: 'cut' (remove pauses), 'speedup' (speed up video segments), 'disabled'
            min_pause_duration: Minimum silence duration to consider for processing (seconds)
            preserve_pause_duration: The duration of pause to keep after processing (seconds)
            video_speed_min: Minimum video speed for 'speedup' mode (default: 0.75)
            video_speed_comfortable: Comfortable speedup limit for phase 1 (default: 1.25)
            video_speed_max: Extended speedup limit for phase 2 remainder (default: 1.5)
            segment_positions: List of segment positions for 'speedup' mode
            keyframe_buffer: Buffer around keyframes to preserve (seconds)
            ffmpeg_batch_size: Number of cuts to process in a single ffmpeg command
            dubbed_volume: Gain multiplier for the translated track (e.g., 1.2 for +1.6 dB)
            upscale_factor: Video scale factor (>1.0 to upscale; default 1.0 disables)
            upscale_sharpen: Apply mild sharpening after scaling
            progress_callback: Optional callback(current, total, message) for progress updates
            log_callback: Optional callback(message) for logging to external systems
            segments_with_video_speed: List of segments with 'video_speed_required' for per-segment speed adjustment
            video_segment_speed_min: Minimum video speed for per-segment adjustment (triggers minterpolate)

        Returns:
            Tuple of (Path to the output video file, List of pause adjustments for subtitle timing)
        """
        def log(message: str):
            logger.info(message)
            if log_callback:
                log_callback(message)
        
        # Start timing
        self.performance_tracker.start_timing("video_creation")

        log("Combining audio with video with smart quality preservation...")
        log("Video processing strategy:")
        log("• Stream copy (lossless) when no video filters needed")
        log("• Conservative re-encoding only when necessary")
        log("• Original bitrate preservation when possible")
        
        output_video_path = output_file if output_file else "artifacts/output_video.mp4"
        # Ensure output directory exists
        output_dir = os.path.dirname(output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Audio normalization is now done per-segment before combination
        # Skip whole-track normalization since segments are already normalized
        normalized_translated_audio_path = translated_audio_path
        logger.debug("Using pre-normalized per-segment audio (normalization disabled at track level)")

        # Initialize logo dimensions to default values
        logo_width = 0
        logo_height = 0

        # Check if the files exist before trying to use them
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")
        if not os.path.exists(normalized_translated_audio_path):
            raise FileNotFoundError(f"Normalized translated audio file not found: {normalized_translated_audio_path}")
        if background_audio_path and not os.path.exists(background_audio_path):
            raise FileNotFoundError(f"Background audio file not found: {background_audio_path}")

        # Use list-based approach to build command to avoid quoting issues
        command = ["ffmpeg", "-y"]

        # Validate dubbed volume (must be positive)
        if dubbed_volume <= 0:
            logger.warning(f"Invalid dubbed_volume {dubbed_volume}, defaulting to 1.0")
            dubbed_volume = 1.0

        # Add start time if specified
        if start_time is not None:
            command.extend(["-ss", str(start_time)])

        # Input 0: Original Video
        command.extend(["-i", video_path])
        # Input 1: Normalized Translated Audio
        command.extend(["-i", normalized_translated_audio_path])

        current_ffmpeg_input_idx = 1  # 0 is video, 1 is normalized translated audio

        background_audio_ffmpeg_idx_str = None
        if background_audio_path:
            current_ffmpeg_input_idx += 1
            background_audio_ffmpeg_idx_str = str(current_ffmpeg_input_idx)
            command.extend(["-i", background_audio_path])

        watermark_ffmpeg_idx_str = None
        if watermark_path and os.path.exists(watermark_path):
            current_ffmpeg_input_idx += 1
            watermark_ffmpeg_idx_str = str(current_ffmpeg_input_idx)
            command.extend(["-i", watermark_path])
            
            # Get the watermark image dimensions
            try:
                probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "{watermark_path}"'
                dimensions = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                if dimensions and 'x' in dimensions:
                    logo_width, logo_height = map(int, dimensions.split('x'))
                    logger.debug(f"Logo dimensions: {logo_width}x{logo_height}")
            except Exception as e:
                logger.warning(f"Warning: Could not determine logo dimensions: {e}, using defaults (0x0)")

        # Add duration if specified (applied to inputs)
        if duration is not None:
            command.extend(["-t", str(duration)])

        # --- Start Filter Complex and Mapping Logic ---
        all_filter_complex_parts = []
        video_map_option = "0:v"  # Default to original video stream (Input 0)

        # Optional upscaling and quality enhancement (applied before overlays/text)
        current_video_label = "0:v"
        try:
            if upscale_factor and float(upscale_factor) > 1.0:
                # Clamp factor to a reasonable range
                factor = max(1.0, min(float(upscale_factor), 4.0))
                scale_filter = (
                    f"[{current_video_label}]"
                    f"scale=ceil(iw*{factor}/2)*2:ceil(ih*{factor}/2)*2:flags=lanczos+accurate_rnd+full_chroma_int"
                    f"[vid_scaled]"
                )
                all_filter_complex_parts.append(scale_filter)
                current_video_label = "[vid_scaled]"

                if upscale_sharpen:
                    # Mild sharpening to enhance details post-upscale
                    unsharp_filter = (
                        f"{self._format_filter_input_label(current_video_label)}"
                        f"unsharp=5:5:0.6:5:5:0.0[vid_sharp]"
                    )
                    all_filter_complex_parts.append(unsharp_filter)
                    current_video_label = "[vid_sharp]"

                video_map_option = current_video_label
        except Exception as _:
            # Fail-safe: ignore invalid upscale params
            current_video_label = "0:v"

        # Watermark and text overlay filters (applied to video stream)
        if watermark_path and os.path.exists(watermark_path) or watermark_text:
            margin = 10
            temp_video_input_label = current_video_label if current_video_label != "0:v" else "0:v"

            if watermark_text:
                logger.debug(f"Adding text caption: '{watermark_text}'")
                effective_logo_width_for_text = logo_width if watermark_path and os.path.exists(watermark_path) else 0
                
                drawtext_filter = (
                    f"{self._format_filter_input_label(temp_video_input_label)}drawtext=text='{watermark_text}':"
                    "fontsize=16:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:"
                    f"x=W-{(effective_logo_width_for_text or 0)+190}:y=H-50[withtext]"
                )
                all_filter_complex_parts.append(drawtext_filter)
                temp_video_input_label = "[withtext]"
            
            if watermark_path and os.path.exists(watermark_path) and watermark_ffmpeg_idx_str:
                logger.debug(f"Adding watermark from {watermark_path}")
                overlay_filter = f"{self._format_filter_input_label(temp_video_input_label)}[{watermark_ffmpeg_idx_str}:v]overlay=W-w-{margin}:H-h-{margin}[outv]"
                all_filter_complex_parts.append(overlay_filter)
                video_map_option = "[outv]"
            elif temp_video_input_label == "[withtext]": 
                video_map_option = "[withtext]"

        # Determine the dubbed audio stream before selective mixing
        dubbed_audio_source_stream = "1:a:0"  # Normalized translated audio from Input 1
        processed_dubbed_audio_stream_label = dubbed_audio_source_stream

        # Apply user-specified gain to the translated track (before optional bg mix)
        if abs(dubbed_volume - 1.0) > 1e-6:
            all_filter_complex_parts.append(
                f"[{dubbed_audio_source_stream}]volume={dubbed_volume}[dubbed_vol_adj]"
            )
            processed_dubbed_audio_stream_label = "[dubbed_vol_adj]"

        # Optionally mix with background after speech gain
        if background_audio_path and background_audio_ffmpeg_idx_str:
            # Apply user-specified background volume (default ≈ -5 dB)
            bg_vol = background_volume if background_volume and background_volume > 0 else 0.562341
            all_filter_complex_parts.append(
                f"[{background_audio_ffmpeg_idx_str}:a:0]volume={bg_vol}[bg_audio_reduced]"
            )
            all_filter_complex_parts.append(
                f"{self._format_filter_input_label(processed_dubbed_audio_stream_label)}[bg_audio_reduced]amix=inputs=2:duration=longest[dub_mixed_with_bg]"
            )
            processed_dubbed_audio_stream_label = "[dub_mixed_with_bg]"

        # Main audio track selection logic
        if keep_original_audio_ranges and len(keep_original_audio_ranges) > 0:
            logger.debug(f"Keeping original audio for ranges: {keep_original_audio_ranges}")
            keep_conditions = "+".join([f"between(t,{s},{e})" for s, e in keep_original_audio_ranges])
            
            if not keep_conditions:
                logger.warning("Warning: keep_original_audio_ranges was specified but resulted in empty conditions. Defaulting to full dubbed audio.")
                final_audio_stream_label = processed_dubbed_audio_stream_label
            else:
                # Use conditional volume adjustments instead of aselect to avoid timing issues
                original_volume_expr = f"if({keep_conditions},1,0)"
                dubbed_volume_expr = f"if({keep_conditions},0,1)"

                all_filter_complex_parts.append(
                    f"[0:a:0]volume='{original_volume_expr}':eval=frame[original_conditional]"
                )
                all_filter_complex_parts.append(
                    f"{self._format_filter_input_label(processed_dubbed_audio_stream_label)}volume='{dubbed_volume_expr}':eval=frame[dubbed_conditional]"
                )
                all_filter_complex_parts.append(
                    f"[original_conditional][dubbed_conditional]amix=inputs=2:duration=longest[final_mixed_audio]"
                )
                final_audio_stream_label = "[final_mixed_audio]"
        else:
            final_audio_stream_label = processed_dubbed_audio_stream_label
        
        # Apply final normalization to bring the entire mix to optimal loudness
        all_filter_complex_parts.append(
            f"{self._format_filter_input_label(final_audio_stream_label)}"
            f"loudnorm=I=-14:TP=-1:LRA=11,"
            f"alimiter=limit=0.95:attack=5:release=50[final_normalized]"
        )
        final_audio_stream_label = "[final_normalized]"
        logger.debug("Applied final normalization to bring mix to optimal loudness")
            
        if all_filter_complex_parts:
            command.extend(["-filter_complex", ";".join(all_filter_complex_parts)])

        # Map the correct video stream (either original or filtered)
        command.extend(["-map", video_map_option])

        # Map the final primary audio stream
        command.extend(["-map", final_audio_stream_label])

        if include_original_audio:
            command.extend(["-map", "0:a:0"])

        # Determine if video needs re-encoding (filters/watermarks applied)
        need_video_reencode = video_map_option != "0:v" or any(
            part for part in all_filter_complex_parts if (
                '[outv]' in part or 'drawtext' in part or 'overlay' in part or 'scale=' in part or 'zscale=' in part or 'unsharp' in part
            )
        )
        
        # Check if trimming prevents stream copy due to keyframe alignment
        if (start_time is not None or duration is not None) and not need_video_reencode:
            can_trim_with_stream_copy = self._can_use_stream_copy_for_trim(video_path, start_time, duration)
            if not can_trim_with_stream_copy:
                need_video_reencode = True
                logger.debug("Video trimming requires re-encoding due to keyframe alignment")
        
        # Get video info for quality decisions
        video_info = self._get_video_info(video_path)
        original_bitrate = video_info.get('video_bitrate')
        original_codec = video_info.get('video_codec', 'h264')
        
        logger.debug(f"Original video: {original_codec} codec, "
                    f"{int(original_bitrate)//1000 if original_bitrate and original_bitrate.isdigit() else 'unknown'} kbps")
        
        # Video encoding strategy
        if not need_video_reencode:
            # No video filters applied – use stream copy (LOSSLESS)
            log("No video processing needed - using lossless stream copy")
            command.extend(["-c:v", "copy"])
        else:
            # Video filters applied – need to re-encode with high quality
            log("Video filters detected - using high-quality re-encoding")
            
            # Check if we should use two-pass encoding
            # Disable two-pass encoding for complex filter operations that can cause frame count mismatches
            has_complex_filters = any(part for part in all_filter_complex_parts if 
                                    'overlay' in part or 'concat' in part or 'select' in part)
            
            if has_complex_filters and use_two_pass_encoding:
                logger.info("Disabling two-pass encoding due to complex filter operations")
                logger.info("Will use enhanced single-pass encoding for maximum quality")
                use_two_pass_encoding = False
            
            if (use_two_pass_encoding and original_bitrate and 
                original_bitrate.isdigit() and int(original_bitrate) > 1000000):
                command.extend(["-c:v", "libx264"])  # Placeholder, will be replaced in two-pass
            else:
                # Use high-quality single-pass settings
                command.extend(["-c:v", "libx264"])
                
                # Enhanced quality settings when two-pass is disabled due to complex filters
                if has_complex_filters:
                    logger.info("Using enhanced quality settings for complex filter operations")
                    
                    # For complex filters, use higher bitrate or better CRF to compensate
                    if original_bitrate and original_bitrate.isdigit() and int(original_bitrate) > 1000000:
                        # Use 20% higher bitrate than original for complex operations
                        enhanced_bitrate = str(int(int(original_bitrate) * 1.2))
                        command.extend(["-b:v", enhanced_bitrate])
                        logger.debug(f"Enhanced bitrate for complex filters: {int(enhanced_bitrate)//1000} kbps (+20%)")
                    else:
                        # Use CRF 18 for excellent quality (lower = better)
                        command.extend(["-crf", "18"])
                        logger.debug("Using CRF 18 for maximum quality with complex filters")
                    
                    # Use slower preset for best quality
                    command.extend(["-preset", "slow"])
                    logger.debug("Using 'slow' preset for maximum quality")
                else:
                    # Standard high-quality settings for simple operations
                    if original_bitrate and original_bitrate.isdigit() and int(original_bitrate) > 1000000:
                        command.extend(["-b:v", original_bitrate])
                        logger.debug(f"Preserving original bitrate: {int(original_bitrate)//1000} kbps")
                    else:
                        # Use CRF 19 for very high quality
                        command.extend(["-crf", "19"])
                        logger.debug("Using CRF 19 for high-quality re-encoding")
                    
                    # Use medium preset for good quality/speed balance
                    command.extend(["-preset", "medium"])
                
                # Preserve original settings when possible
                if video_info.get('video_profile') and video_info['video_profile'] in ['high', 'main', 'baseline']:
                    command.extend(["-profile:v", video_info['video_profile']])
                else:
                    command.extend(["-profile:v", "high"])
                    
                if video_info.get('video_pix_fmt') and 'yuv420p' in video_info['video_pix_fmt']:
                    command.extend(["-pix_fmt", video_info['video_pix_fmt']])
                else:
                    command.extend(["-pix_fmt", "yuv420p"])

        # Audio encoding (always needed since audio is being replaced)
        original_audio_codec = video_info.get('audio_codec', 'aac')
        original_audio_bitrate = video_info.get('audio_bitrate')
        
        # Use safer audio settings to avoid AAC encoding issues
        command.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
        # Add error recovery flags
        command.extend(["-err_detect", "ignore_err", "-ignore_unknown"])
        logger.debug("Using safe AAC encoding with error recovery")

        # Set MOV flags
        command.extend(["-movflags", "+faststart"])

        # Set metadata for audio tracks
        command.extend([
            "-metadata:s:a:0", f"language={target_language}",
            "-metadata:s:a:0", "title=Translated Audio",
            "-disposition:a:0", "default"  # Translated audio is default
        ])

        if include_original_audio:
            command.extend([
                "-metadata:s:a:1", f"language={source_language}",
                "-metadata:s:a:1", "title=Original Audio",
                "-disposition:a:1", "0"  # Original audio not default
            ])

        # Use shortest duration of inputs
        command.extend(["-shortest"])

        # Add output file path
        command.extend([output_video_path])

        # Handle pause processing based on mode
        pause_adjustments = []
        temp_files_to_cleanup = []
        
        # Apply per-segment video speed adjustments if segments have video_speed_required
        video_speed_segments = [
            seg for seg in (segments_with_video_speed or []) 
            if seg.get('video_speed_required') is not None
        ]
        
        if video_speed_segments:
            log(f"Applying per-segment video speed adjustments ({len(video_speed_segments)} segments)...")
            
            # Create temporary video with per-segment speed adjustments
            temp_speed_video = "artifacts/temp_per_segment_speed_video.mp4"
            temp_files_to_cleanup.append(temp_speed_video)
            target_audio_duration = 0.0
            try:
                target_audio_duration = self._get_video_duration(normalized_translated_audio_path)
                if background_audio_path:
                    bg_duration = self._get_video_duration(background_audio_path)
                    if bg_duration > target_audio_duration:
                        target_audio_duration = bg_duration
            except Exception as exc:
                logger.warning(f"Unable to determine target audio duration: {exc}")
            
            try:
                _, speed_adjustments = self._apply_per_segment_video_speed(
                    video_path,
                    video_speed_segments,
                    temp_speed_video,
                    video_min_speed=self.video_minterpolate_threshold,
                    use_minterpolate_for_slowdown=True,
                    target_duration=target_audio_duration if target_audio_duration > 0 else None,
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
                
                if os.path.exists(temp_speed_video):
                    # Replace video_path in the command with the speed-adjusted video
                    for i, arg in enumerate(command):
                        if arg == video_path:
                            command[i] = temp_speed_video
                    
                    # Also update the video_path variable for any downstream processing
                    video_path = temp_speed_video
                    
                    # Merge speed adjustments into pause_adjustments if any
                    if speed_adjustments:
                        pause_adjustments.extend(speed_adjustments)
                        logger.info(f"Per-segment video speed adjustments applied: {len(speed_adjustments)} timing changes")
                else:
                    logger.warning("Per-segment speed video was not created, continuing without speed adjustments")
            except Exception as e:
                logger.error(f"Failed to apply per-segment video speed: {e}")
                logger.warning("Continuing without per-segment speed adjustments")
        
        logger.info(f"Pause removal mode: {pause_removal}")
        
        if pause_removal == "speedup":
            logger.info("Applying video speedup mode to sync with dubbed audio...")
            logger.info("Speedup strategy:")
            logger.info("• Calculate speed ratio for each video segment based on dubbed audio")
            logger.info(f"• Speed limits: {video_speed_min}x - {video_speed_comfortable}x (comfortable) - {video_speed_max}x (extended)")
            logger.info("• Two-phase speedup: comfortable speed first, extended for remainder")
            
            if segment_positions:
                # Apply speedup mode
                command, temp_files_to_cleanup, pause_adjustments = self._apply_video_speedup(
                    command, video_path, segment_positions,
                    min_pause_duration, preserve_pause_duration,
                    video_speed_min, video_speed_comfortable, video_speed_max,
                    progress_callback=progress_callback
                )
            else:
                logger.warning("No segment positions provided for speedup mode, skipping")
        
        elif pause_removal == "cut":
            logger.info("Detecting and removing long pauses with smart quality preservation...")
            logger.info("Pause removal strategy:")
            logger.info("• Analyze pauses in final translated audio")
            logger.info("• Preserve keyframes from original video")
            logger.info("• Apply cuts to original video source")
            
            # Create final audio for pause analysis
            final_audio_path = self._create_final_audio_for_pause_analysis(
                normalized_translated_audio_path,
                background_audio_path,
                background_audio_ffmpeg_idx_str,
                keep_original_audio_ranges,
                video_path,
                dubbed_volume,
                background_volume
            )
            
            # Detect pauses in final audio
            silences = self._detect_silence(final_audio_path, min_pause_duration)
            
            if silences:
                # Get keyframes from original video
                keyframes = self._extract_keyframes(video_path)
                
                # Filter pauses that don't contain keyframes
                shortenable_pauses = self._filter_pauses_with_keyframes(
                    silences, keyframes, keyframe_buffer
                )
                
                # Filter out edge pauses
                video_duration = self._get_video_duration(video_path)
                shortenable_pauses = self._filter_edge_pauses(
                    shortenable_pauses, video_duration, edge_threshold=0.5
                )
                
                if shortenable_pauses:
                    logger.info(f"Found {len(shortenable_pauses)} pauses to shorten")
                    
                    # Calculate pause adjustments and cuts
                    removals, pause_adjustments = self._calculate_pause_removals(
                        shortenable_pauses, preserve_pause_duration
                    )
                    
                    if removals:
                        # Calculate video cuts (parts to keep)
                        cuts_to_keep = self._calculate_video_cuts(removals, video_duration)
                        
                        # Modify the ffmpeg command to apply cuts to original video
                        command, temp_files_to_cleanup = self._modify_command_for_cuts(
                            command, video_path, cuts_to_keep, keyframes, 
                            use_two_pass_encoding, video_info,
                            progress_callback=progress_callback
                        )
                        
                        logger.info(f"Will remove {len(removals)} pauses, "
                                  f"total time reduction: {sum(r[1] - r[0] for r in removals):.2f}s")
                else:
                    logger.info("No pauses to shorten after filtering")
            else:
                logger.info("No long pauses detected in final audio")
            
            # Clean up temporary final audio file
            if os.path.exists(final_audio_path):
                os.remove(final_audio_path)

        # Execute the FFmpeg command
        try:
            # Check if we need to use two-pass encoding
            if (use_two_pass_encoding and need_video_reencode and original_bitrate and 
                original_bitrate.isdigit() and int(original_bitrate) > 1000000):
                # Use two-pass encoding for better quality
                logger.info("Using two-pass encoding for combine operation")
                # Remove output path from command and codec settings for two-pass
                base_cmd = command[:-1]  # Remove output path
                
                # Remove all video codec settings
                while "-c:v" in base_cmd:
                    idx = base_cmd.index("-c:v")
                    base_cmd.pop(idx)  # Remove -c:v
                    base_cmd.pop(idx)  # Remove libx264
                
                # Remove audio codec settings to avoid duplicates
                while "-c:a" in base_cmd:
                    idx = base_cmd.index("-c:a")
                    base_cmd.pop(idx)  # Remove -c:a
                    base_cmd.pop(idx)  # Remove aac
                
                # Remove audio bitrate settings
                while "-b:a" in base_cmd:
                    idx = base_cmd.index("-b:a")
                    base_cmd.pop(idx)  # Remove -b:a
                    base_cmd.pop(idx)  # Remove bitrate value
                
                # Remove other video encoding parameters that will be set in two-pass
                params_to_remove = ["-crf", "-b:v", "-profile:v", "-pix_fmt", "-preset"]
                for param in params_to_remove:
                    while param in base_cmd:
                        idx = base_cmd.index(param)
                        base_cmd.pop(idx)  # Remove parameter
                        base_cmd.pop(idx)  # Remove value
                
                # Remove movflags as it will be added in second pass
                while "-movflags" in base_cmd:
                    idx = base_cmd.index("-movflags")
                    base_cmd.pop(idx)  # Remove -movflags
                    base_cmd.pop(idx)  # Remove value
                
                self._encode_with_two_pass(base_cmd, output_video_path, video_info)
            else:
                # Use single-pass encoding
                logger.debug(f"Running FFmpeg command: {' '.join(command)}")
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # Only print stderr if it contains error messages that aren't just informational
            # (Only for single-pass encoding - two-pass encoding handles its own error reporting)
            if not (use_two_pass_encoding and need_video_reencode and original_bitrate and 
                   original_bitrate.isdigit() and int(original_bitrate) > 1000000):
                stderr = result.stderr
                if stderr and ('error' in stderr.lower() or 'fatal' in stderr.lower()):
                    logger.error("FFmpeg errors:")
                    for line in stderr.split('\n'):
                        if 'error' in line.lower() or 'fatal' in line.lower():
                            logger.error(f"  {line}")

            logger.info(f"Output video saved to {output_video_path}")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else "No error details available"
            
            # Check for specific AAC/audio related errors
            if "aac" in error_output.lower() or "audio" in error_output.lower():
                logger.error("Audio processing error detected - this may be due to corrupted audio streams")
                logger.error("Trying to identify problematic audio files...")
                
                # Log the command for debugging
                logger.error("Problem may be with these input files:")
                for i, arg in enumerate(command):
                    if arg == "-i" and i + 1 < len(command):
                        input_file = command[i + 1]
                        logger.error(f"  Input: {input_file}")
                        if os.path.exists(input_file):
                            file_size = os.path.getsize(input_file)
                            logger.error(f"    Size: {file_size} bytes")
                        else:
                            logger.error(f"    File does not exist!")
            
            logger.error(f"FFmpeg command failed with return code {e.returncode}")
            logger.error(f"Error output: {error_output}")
            logger.error(f"Failed command: {' '.join(command)}")
            raise

        # Clean up temporary files created during pause removal
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

        # End timing
        self.performance_tracker.end_timing("video_creation")

        return output_video_path, pause_adjustments 

    def _build_two_pass_command(self, base_cmd: List[str], pass_num: int, 
                               bitrate: str, output_path: str, 
                               video_info: Dict[str, any], 
                               temp_dir: str) -> List[str]:
        """Build FFmpeg command for two-pass encoding.
        
        Args:
            base_cmd: Base FFmpeg command without codec settings
            pass_num: Pass number (1 or 2)
            bitrate: Target bitrate in kbps (e.g., '5000')
            output_path: Final output path (for pass 2)
            video_info: Video information dictionary
            temp_dir: Temporary directory for pass log files
            
        Returns:
            Complete FFmpeg command for the specified pass
        """
        cmd = base_cmd.copy()
        
        # Two-pass encoding settings
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-b:v", f"{bitrate}k"])  # Add 'k' suffix for FFmpeg
        cmd.extend(["-pass", str(pass_num)])
        
        # Use consistent passlogfile naming
        passlogfile_prefix = os.path.join(temp_dir, "ffmpeg2pass")
        cmd.extend(["-passlogfile", passlogfile_prefix])
        
        # Quality settings
        cmd.extend(["-preset", "slow"])  # Use slower preset for better quality
        
        # Preserve original settings when possible
        if video_info.get('video_profile') and video_info['video_profile'] in ['high', 'main', 'baseline']:
            cmd.extend(["-profile:v", video_info['video_profile']])
        else:
            cmd.extend(["-profile:v", "high"])
            
        if video_info.get('video_pix_fmt') and 'yuv420p' in video_info['video_pix_fmt']:
            cmd.extend(["-pix_fmt", video_info['video_pix_fmt']])
        else:
            cmd.extend(["-pix_fmt", "yuv420p"])
        
        if pass_num == 1:
            # First pass: analysis only, no audio processing needed
            cmd.extend(["-an", "-f", "null"])
            if os.name == 'nt':  # Windows
                cmd.append("NUL")
            else:  # Unix/Linux
                cmd.append("/dev/null")
        else:
            # Second pass: final encoding with audio
            original_audio_codec = video_info.get('audio_codec', 'aac')
            original_audio_bitrate = video_info.get('audio_bitrate')
            
            # Use consistent safe audio settings for two-pass encoding
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
            # Add error recovery flags for two-pass as well
            cmd.extend(["-err_detect", "ignore_err", "-ignore_unknown"])
            
            cmd.extend(["-movflags", "+faststart"])
            cmd.append(output_path)
        
        return cmd

    def _encode_with_two_pass(self, base_cmd: List[str], output_path: str, 
                             video_info: Dict[str, any], 
                             use_original_bitrate: bool = True) -> None:
        """Perform two-pass encoding for better quality.
        
        Args:
            base_cmd: Base FFmpeg command without codec/output settings
            output_path: Path for the final output file
            video_info: Video information dictionary
            use_original_bitrate: Whether to use original bitrate or calculate optimal
        """
        # Determine target bitrate
        original_bitrate = video_info.get('video_bitrate')
        if use_original_bitrate and original_bitrate and original_bitrate.isdigit() and int(original_bitrate) > 1000000:
            # Convert from bps to kbps and format as string
            target_bitrate = str(int(original_bitrate) // 1000)
            logger.info(f"Using original bitrate for two-pass encoding: {target_bitrate} kbps")
        else:
            # Calculate reasonable bitrate based on resolution and framerate
            # This is a fallback when original bitrate is not available or too low
            target_bitrate = "5000"  # Conservative default in kbps (no 'k' suffix for internal use)
            logger.info(f"Using fallback bitrate for two-pass encoding: {target_bitrate} kbps")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # First pass
                logger.info("Starting first pass of two-pass encoding...")
                first_pass_cmd = self._build_two_pass_command(
                    base_cmd, 1, target_bitrate, output_path, video_info, temp_dir
                )
                
                logger.debug(f"First pass command: {' '.join(first_pass_cmd)}")
                result = subprocess.run(
                    first_pass_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Check if pass log file was created successfully
                passlogfile = os.path.join(temp_dir, "ffmpeg2pass-0.log")
                passlogfile_alt = os.path.join(temp_dir, "ffmpeg2pass-0.log.mbtree")
                if not (os.path.exists(passlogfile) or os.path.exists(passlogfile_alt)):
                    logger.warning("First pass log file not found, two-pass encoding may fail")
                else:
                    logger.debug("First pass completed successfully, log files created")
                
                # Second pass
                logger.info("Starting second pass of two-pass encoding...")
                second_pass_cmd = self._build_two_pass_command(
                    base_cmd, 2, target_bitrate, output_path, video_info, temp_dir
                )
                
                logger.debug(f"Second pass command: {' '.join(second_pass_cmd)}")
                result = subprocess.run(
                    second_pass_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                logger.info("Two-pass encoding completed successfully")
                
            except subprocess.CalledProcessError as e:
                error_output = e.stderr if e.stderr else "No error details available"
                
                # Check for specific error patterns
                if "2nd pass has more frames than 1st pass" in error_output:
                    logger.error("Frame count mismatch between passes detected")
                    logger.error("This can happen when filter operations affect stream processing")
                    logger.error("Consider disabling two-pass encoding for this content")
                elif "SIGSEGV" in str(e.returncode) or e.returncode == -11:
                    logger.error("FFmpeg segmentation fault detected")
                    logger.error("This might be due to codec/filter incompatibility")
                
                logger.error(f"Two-pass encoding failed: {error_output}")
                logger.error("FFmpeg command that failed:")
                logger.error(f"  {' '.join(second_pass_cmd if 'second_pass_cmd' in locals() else first_pass_cmd)}")
                raise
    
    def _create_final_audio_for_pause_analysis(self,
                                             translated_audio_path: str,
                                             background_audio_path: Optional[str],
                                             background_audio_idx: Optional[str],
                                             keep_original_audio_ranges: Optional[List[Tuple[float, float]]],
                                             video_path: str,
                                             dubbed_volume: float = 1.0,
                                             background_volume: float = 0.562341) -> str:
        """Create the final audio mix for pause analysis.
        
        Args:
            translated_audio_path: Path to translated audio
            background_audio_path: Path to background audio (optional)
            background_audio_idx: Background audio index string
            keep_original_audio_ranges: Ranges to keep original audio
            video_path: Path to original video for original audio
            
        Returns:
            Path to temporary final audio file
        """
        temp_audio_path = "artifacts/audio/temp_final_for_pause_analysis.wav"
        
        try:
            # Build ffmpeg command to create final audio mix
            cmd = ["ffmpeg", "-y"]
            
            # Input 0: Translated audio
            cmd.extend(["-i", translated_audio_path])
            input_count = 1
            
            # Input 1: Background audio (if exists)
            if background_audio_path:
                cmd.extend(["-i", background_audio_path])
                input_count += 1
            
            # Input 2: Original video (for original audio if needed)
            original_audio_input_idx = None
            if keep_original_audio_ranges:
                cmd.extend(["-i", video_path])
                original_audio_input_idx = input_count
                input_count += 1
            
            # Build filter complex
            filter_parts = []
            
            # Start with translated audio
            current_audio_label = "0:a:0"

            # Apply dubbed_volume if not 1.0
            if abs(dubbed_volume - 1.0) > 1e-6:
                filter_parts.append(f"[0:a:0]volume={dubbed_volume}[dubbed_gained]")
                current_audio_label = "[dubbed_gained]"
            
            # Mix with background if needed
            if background_audio_path:
                bg_vol = background_volume if background_volume and background_volume > 0 else 0.562341
                filter_parts.append(f"[1:a:0]volume={bg_vol}[bg_reduced]")
                left = self._format_filter_input_label(current_audio_label)
                filter_parts.append(f"{left}[bg_reduced]amix=inputs=2:duration=longest[mixed_with_bg]")
                current_audio_label = "[mixed_with_bg]"
            
            # Handle original audio ranges if needed
            if keep_original_audio_ranges and original_audio_input_idx is not None:
                keep_conditions = "+".join([f"between(t,{s},{e})" for s, e in keep_original_audio_ranges])
                if keep_conditions:
                    original_volume_expr = f"if({keep_conditions},1,0)"
                    dubbed_volume_expr = f"if({keep_conditions},0,1)"
                    
                    filter_parts.append(f"[{original_audio_input_idx}:a:0]volume='{original_volume_expr}':eval=frame[original_conditional]")
                    filter_parts.append(f"{self._format_filter_input_label(current_audio_label)}volume='{dubbed_volume_expr}':eval=frame[dubbed_conditional]")
                    filter_parts.append("[original_conditional][dubbed_conditional]amix=inputs=2:duration=longest[final_audio]")
                    current_audio_label = "[final_audio]"
            
            # Apply final normalization to bring the entire mix to optimal loudness
            filter_parts.append(
                f"{self._format_filter_input_label(current_audio_label)}"
                f"loudnorm=I=-14:TP=-1:LRA=11,"
                f"alimiter=limit=0.95:attack=5:release=50[final_normalized]"
            )
            current_audio_label = "[final_normalized]"
            
            # Add filter complex if we have filters
            if filter_parts:
                cmd.extend(["-filter_complex", ";".join(filter_parts)])
                cmd.extend(["-map", current_audio_label])
            else:
                cmd.extend(["-map", "0:a:0"])  # Just the translated audio
            
            # Audio encoding settings
            cmd.extend(["-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1"])
            cmd.append(temp_audio_path)
            
            # Execute command
            logger.debug(f"Creating final audio for pause analysis: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return temp_audio_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create final audio for pause analysis: {e.stderr}")
            # Fallback: use translated audio directly
            import shutil
            shutil.copy(translated_audio_path, temp_audio_path)
            return temp_audio_path
    
    def _calculate_pause_removals(self, 
                                shortenable_pauses: List[Tuple[float, float]], 
                                preserve_duration: float) -> Tuple[List[Tuple[float, float]], List[Dict[str, float]]]:
        """Calculate pause removals and adjustments.
        
        Args:
            shortenable_pauses: List of pause intervals that can be shortened
            preserve_duration: Duration to preserve from each pause
            
        Returns:
            Tuple of (removals list, pause adjustments list)
        """
        removals = []
        pause_adjustments = []
        cumulative_time_removed = 0.0
        
        for start, end in sorted(shortenable_pauses):
            duration = end - start
            if duration > preserve_duration:
                removal_start = start + preserve_duration
                time_removed = end - removal_start
                removals.append((removal_start, end))
                
                # Track pause adjustment for subtitle timing
                pause_adjustments.append({
                    'original_start': start,
                    'original_end': end,
                    'time_removed': time_removed,
                    'cumulative_offset': cumulative_time_removed + time_removed
                })
                cumulative_time_removed += time_removed
        
        return removals, pause_adjustments
    
    def _calculate_video_cuts(self, 
                            removals: List[Tuple[float, float]], 
                            video_duration: float) -> List[Tuple[float, float]]:
        """Calculate video cuts (parts to keep) based on removals.
        
        Args:
            removals: List of time ranges to remove
            video_duration: Total video duration
            
        Returns:
            List of video cuts to keep
        """
        cuts_to_keep = []
        current_pos = 0.0
        
        for removal_start, removal_end in removals:
            if current_pos < removal_start:
                cuts_to_keep.append((current_pos, removal_start))
            current_pos = removal_end
        
        if current_pos < video_duration:
            cuts_to_keep.append((current_pos, video_duration))
        
        return cuts_to_keep
    
    def _modify_command_for_cuts(self,
                               original_command: List[str],
                               video_path: str,
                               cuts_to_keep: List[Tuple[float, float]],
                               keyframes: List[float],
                               use_two_pass_encoding: bool,
                               video_info: Dict,
                               progress_callback: Optional[callable] = None) -> Tuple[List[str], List[str]]:
        """Modify FFmpeg command to apply cuts to the original video.
        
        Args:
            original_command: Original FFmpeg command
            video_path: Path to video file
            cuts_to_keep: List of video segments to keep
            keyframes: List of keyframe timestamps
            use_two_pass_encoding: Whether to use two-pass encoding
            video_info: Video information dictionary
            progress_callback: Optional callback(current, total, message) for progress updates
            
        Returns:
            Tuple of (Modified FFmpeg command for video with cuts, List of temporary files to cleanup)
        """
        if len(cuts_to_keep) <= 1:
            # No cuts needed, return original command
            return original_command, []
        
        # Find the output path from the original command
        output_path = original_command[-1]
        
        # Check if we can use stream copy for cuts
        can_use_stream_copy = self._can_use_stream_copy(cuts_to_keep, keyframes, tolerance=0.2)
        
        if can_use_stream_copy:
            logger.info("Using stream copy for lossless pause removal")
            return self._build_stream_copy_cuts_command(
                original_command, video_path, cuts_to_keep, output_path,
                progress_callback=progress_callback
            )
        else:
            logger.info("Using re-encoding for pause removal (keyframe alignment required)")
            return self._build_reencoding_cuts_command(
                original_command, video_path, cuts_to_keep, output_path,
                use_two_pass_encoding, video_info,
                progress_callback=progress_callback
            )
    
    def _build_stream_copy_cuts_command(self,
                                      original_command: List[str],
                                      video_path: str,
                                      cuts_to_keep: List[Tuple[float, float]],
                                      output_path: str,
                                      progress_callback: Optional[callable] = None) -> Tuple[List[str], List[str]]:
        """Build FFmpeg command using stream copy for cuts."""
        # Use the existing _apply_video_cuts method logic but adapted for our case
        # For now, fall back to re-encoding approach since stream copy with audio mixing is complex
        return self._build_reencoding_cuts_command(
            original_command, video_path, cuts_to_keep, output_path, False, {},
            progress_callback=progress_callback
        )
    
    def _build_reencoding_cuts_command(self,
                                     original_command: List[str],
                                     video_path: str,
                                     cuts_to_keep: List[Tuple[float, float]],
                                     output_path: str,
                                     use_two_pass_encoding: bool,
                                     video_info: Dict,
                                     progress_callback: Optional[callable] = None) -> Tuple[List[str], List[str]]:
        """Build FFmpeg command using re-encoding for cuts."""
        logger.info("Applying pause removal to both video and audio")
        
        # Create temporary video with cuts (without audio)
        temp_video_path = "artifacts/temp_video_with_cuts.mp4"
        
        # Create cuts command for video only.
        #
        # IMPORTANT: Use filter trim+setpts on a single input instead of `-ss` before `-i` per segment.
        # Fast input seeking snaps to keyframes/packet boundaries and can introduce small per-cut drift
        # that accumulates over many cuts (visible desync near the end).
        cuts_cmd = ["ffmpeg", "-y", "-i", video_path]

        filter_parts: List[str] = []
        video_labels: List[str] = []
        for i, (start, end) in enumerate(cuts_to_keep):
            start_s = f"{float(start):.6f}"
            end_s = f"{float(end):.6f}"
            filter_parts.append(
                f"[0:v]trim=start={start_s}:end={end_s},setpts=PTS-STARTPTS[v{i}]"
            )
            video_labels.append(f"[v{i}]")

        filter_parts.append(
            f"{''.join(video_labels)}concat=n={len(cuts_to_keep)}:v=1:a=0[outv]"
        )

        cuts_cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cuts_cmd.extend(["-map", "[outv]"])
        cuts_cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])
        cuts_cmd.extend(["-an"])  # No audio
        cuts_cmd.append(temp_video_path)
        
        # Track temporary files for cleanup
        temp_files_to_cleanup = [temp_video_path]
        
        # Calculate total output duration for progress tracking
        total_output_duration = sum(end - start for start, end in cuts_to_keep)
        
        # Execute video cuts command with progress tracking
        try:
            logger.debug(f"Creating video with cuts: {' '.join(cuts_cmd)}")
            self._run_ffmpeg_with_progress(
                cuts_cmd,
                total_output_duration,
                progress_callback,
                "Pauses cut"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create video cuts: {e.stderr}")
            return original_command, []
        
        # Now create cut versions of all audio files used in the original command
        cut_audio_files = {}  # Maps original audio file to cut version
        
        # Find all audio input files in the original command
        i = 0
        while i < len(original_command):
            if original_command[i] == "-i" and i + 1 < len(original_command):
                input_file = original_command[i + 1]
                # Skip the video input
                if input_file != video_path and os.path.exists(input_file):
                    # Check if it's an audio file by trying to probe for audio streams
                    try:
                        probe_cmd = [
                            'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_file
                        ]
                        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                        if 'audio' in result.stdout:
                            # This is an audio file, create cut version
                            cut_audio_path = self._create_cut_audio_file(input_file, cuts_to_keep)
                            if cut_audio_path:
                                cut_audio_files[input_file] = cut_audio_path
                                temp_files_to_cleanup.append(cut_audio_path)
                                logger.debug(f"Created cut audio file: {input_file} -> {cut_audio_path}")
                    except subprocess.CalledProcessError:
                        # Not an audio file or probe failed, skip
                        pass
            i += 1
        
        # Modify the original command to use cut video and cut audio files
        modified_command = []
        for i, arg in enumerate(original_command):
            if arg == video_path and i > 0 and original_command[i-1] == "-i":
                # Replace video path with temp video path
                modified_command.append(temp_video_path)
            elif arg in cut_audio_files:
                # Replace audio file with cut version
                modified_command.append(cut_audio_files[arg])
            else:
                modified_command.append(arg)

        # If the filter graph references [0:a:0] (original audio), but input 0 is now video-without-audio,
        # inject a cut version of the original video's audio as a new input before -filter_complex
        try:
            if "-filter_complex" in modified_command:
                fc_idx = modified_command.index("-filter_complex")
                fc_str = modified_command[fc_idx + 1]

                if "[0:a:0]" in fc_str:
                    # Count inputs before filter_complex to determine new input index
                    inputs_before_fc = sum(1 for j in range(fc_idx) if modified_command[j] == "-i")

                    # Create cut audio from original video audio track
                    cut_orig_audio = self._create_cut_audio_file(video_path, cuts_to_keep)
                    if cut_orig_audio:
                        # Insert new input before -filter_complex
                        modified_command.insert(fc_idx, "-i")
                        modified_command.insert(fc_idx + 1, cut_orig_audio)
                        temp_files_to_cleanup.append(cut_orig_audio)

                        # After insertion, -filter_complex moved by +2
                        fc_token_idx = fc_idx + 2
                        fc_value_idx = fc_token_idx + 1

                        # Replace occurrences of [0:a:0] with the new input index
                        new_input_index = inputs_before_fc  # zero-based
                        fc_str_updated = fc_str.replace("[0:a:0]", f"[{new_input_index}:a:0]")
                        modified_command[fc_value_idx] = fc_str_updated

                        # Also fix any explicit mapping of 0:a:0 if present
                        k = 0
                        while k < len(modified_command) - 1:
                            if modified_command[k] == "-map" and modified_command[k + 1] == "0:a:0":
                                modified_command[k + 1] = f"{new_input_index}:a:0"
                            k += 2 if modified_command[k] == "-map" else 1
                    else:
                        logger.warning("Could not create cut original audio; original ranges may fail.")
        except Exception as e:
            logger.warning(f"Failed to adjust original audio input for filter graph: {e}")
        
        logger.info(f"Applied cuts to video and {len(cut_audio_files)} audio files")
        return modified_command, temp_files_to_cleanup
    
    def _create_cut_audio_file(self, audio_path: str, cuts_to_keep: List[Tuple[float, float]]) -> Optional[str]:
        """Create a cut version of an audio file.
        
        Args:
            audio_path: Path to the original audio file
            cuts_to_keep: List of time segments to keep
            
        Returns:
            Path to the cut audio file, or None if creation failed
        """
        if not cuts_to_keep:
            return None
        
        # Generate cut audio file path (use WAV for better compatibility)
        audio_dir = os.path.dirname(audio_path)
        audio_name = os.path.basename(audio_path)
        name, ext = os.path.splitext(audio_name)
        cut_audio_path = os.path.join(audio_dir, f"{name}_cut.wav")
        
        try:
            # First, probe the audio file to get its properties
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a:0', audio_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(probe_result.stdout)
            
            # Get audio stream properties
            audio_stream = audio_info.get('streams', [{}])[0] if audio_info.get('streams') else {}
            sample_rate = audio_stream.get('sample_rate', '48000')
            channels = min(int(audio_stream.get('channels', '2')), 2)  # Limit to stereo max
            
            logger.debug(f"Source audio: {sample_rate}Hz, {channels} channels")
            
            # Build FFmpeg command to cut audio.
            #
            # IMPORTANT: Use atrim+asetpts on a single input instead of `-ss` before `-i` per segment.
            # Fast seeking on audio can snap to packet boundaries; over many cuts this causes audible drift
            # and desync with the cut video.
            cmd = ["ffmpeg", "-y", "-i", audio_path]

            filter_parts: List[str] = []
            audio_labels: List[str] = []
            for i, (start, end) in enumerate(cuts_to_keep):
                start_s = f"{float(start):.6f}"
                end_s = f"{float(end):.6f}"
                filter_parts.append(
                    f"[0:a]atrim=start={start_s}:end={end_s},asetpts=PTS-STARTPTS[a{i}]"
                )
                audio_labels.append(f"[a{i}]")

            if len(audio_labels) > 1:
                filter_parts.append(
                    f"{''.join(audio_labels)}concat=n={len(audio_labels)}:v=0:a=1[outa]"
                )
                map_label = "[outa]"
            else:
                map_label = audio_labels[0]

            cmd.extend(["-filter_complex", ";".join(filter_parts)])
            cmd.extend(["-map", map_label])
            
            # Audio encoding settings - use PCM WAV for better compatibility
            cmd.extend(["-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels)])
            cmd.append(cut_audio_path)
            
            logger.debug(f"Creating cut audio: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify the created file is valid
            if os.path.exists(cut_audio_path) and os.path.getsize(cut_audio_path) > 0:
                logger.debug(f"Successfully created cut audio file: {cut_audio_path}")
                return cut_audio_path
            else:
                logger.error(f"Created audio file is empty or missing: {cut_audio_path}")
                return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create cut audio file {audio_path}: {e.stderr}")
            return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to probe audio file {audio_path}: {e}")
            return None

    def _calculate_speed_segments(
        self,
        segment_positions: List[Dict],
        video_duration: float,
        min_pause_duration: float,
        preserve_pause_duration: float,
        video_speed_min: float,
        video_speed_comfortable: float,
        video_speed_max: float
    ) -> List[Dict]:
        """Calculate speed factors for each video segment using two-phase speedup.
        
        Two-phase approach:
        1. First phase: speed up segment at comfortable limit
        2. Second phase: if needed, speed up remainder at extended limit
        
        Args:
            segment_positions: List of segment positions with start, end, original_start, original_end
            video_duration: Total video duration
            min_pause_duration: Minimum pause duration to consider for processing
            preserve_pause_duration: Duration of pause to preserve between segments
            video_speed_min: Minimum allowed video speed
            video_speed_comfortable: Comfortable speedup limit (phase 1)
            video_speed_max: Extended speedup limit (phase 2)
            
        Returns:
            List of speed segments with start, end, and speed_factor
        """
        if not segment_positions:
            return []
        
        speed_segments = []
        sorted_positions = sorted(segment_positions, key=lambda x: x.get("original_start", 0))
        
        prev_end = 0.0
        
        for i, pos in enumerate(sorted_positions):
            original_start = pos.get("original_start", pos.get("start", 0))
            original_end = pos.get("original_end", pos.get("end", 0))
            real_start = pos.get("start", original_start)
            real_end = pos.get("end", original_end)
            
            original_duration = original_end - original_start
            real_duration = real_end - real_start
            
            # Handle gap before this segment (pause)
            if original_start > prev_end:
                gap_duration = original_start - prev_end
                
                if gap_duration > min_pause_duration:
                    # This pause is long enough to process
                    target_gap = preserve_pause_duration
                    if gap_duration > target_gap:
                        gap_speed = gap_duration / target_gap
                        gap_speed = min(max(gap_speed, video_speed_min), video_speed_max)
                        
                        speed_segments.append({
                            "start": prev_end,
                            "end": original_start,
                            "speed_factor": gap_speed,
                            "is_pause": True
                        })
                    else:
                        speed_segments.append({
                            "start": prev_end,
                            "end": original_start,
                            "speed_factor": 1.0,
                            "is_pause": True
                        })
                else:
                    speed_segments.append({
                        "start": prev_end,
                        "end": original_start,
                        "speed_factor": 1.0,
                        "is_pause": True
                    })
            
            # Handle the segment itself with two-phase speedup
            if original_duration > 0 and real_duration > 0:
                required_speed = original_duration / real_duration
                speaker = pos.get("speaker", "UNKNOWN")
                
                # Apply speed limits
                required_speed = max(required_speed, video_speed_min)
                
                if required_speed <= 1.0:
                    # Slowdown or no change - use single segment
                    speed_segments.append({
                        "start": original_start,
                        "end": original_end,
                        "speed_factor": max(required_speed, video_speed_min),
                        "is_pause": False,
                        "speaker": speaker
                    })
                elif required_speed <= video_speed_comfortable:
                    # Within comfortable limit - use single segment
                    speed_segments.append({
                        "start": original_start,
                        "end": original_end,
                        "speed_factor": required_speed,
                        "is_pause": False,
                        "speaker": speaker
                    })
                elif required_speed <= video_speed_max:
                    # Exceeds comfortable but within max - use two-phase approach
                    # Split segment: comfortable speed for first part, extended for second
                    phase1_speed = video_speed_comfortable
                    phase2_speed = video_speed_max
                    target_duration = original_duration / required_speed
                    
                    # Calculate split point:
                    # d1/phase1_speed + d2/phase2_speed = target_duration
                    # d1 + d2 = original_duration
                    # Solve: d1 = (target * phase1 * phase2 - original * phase1) / (phase2 - phase1)
                    numerator = target_duration * phase1_speed * phase2_speed - original_duration * phase1_speed
                    denominator = phase2_speed - phase1_speed
                    
                    if denominator > 0.01:  # Avoid division by zero
                        d1 = numerator / denominator
                        d1 = max(0, min(d1, original_duration))  # Clamp to valid range
                        d2 = original_duration - d1
                        
                        if d1 > 0.1:  # Phase 1 segment is meaningful
                            speed_segments.append({
                                "start": original_start,
                                "end": original_start + d1,
                                "speed_factor": phase1_speed,
                                "is_pause": False,
                                "speaker": speaker,
                                "phase": 1
                            })
                        
                        if d2 > 0.1:  # Phase 2 segment is meaningful
                            speed_segments.append({
                                "start": original_start + d1 if d1 > 0.1 else original_start,
                                "end": original_end,
                                "speed_factor": phase2_speed,
                                "is_pause": False,
                                "speaker": speaker,
                                "phase": 2
                            })
                        elif d1 <= 0.1:
                            # Both segments too small, use average speed
                            speed_segments.append({
                                "start": original_start,
                                "end": original_end,
                                "speed_factor": required_speed,
                                "is_pause": False,
                                "speaker": speaker
                            })
                    else:
                        # Speeds are too similar, use required speed directly
                        speed_segments.append({
                            "start": original_start,
                            "end": original_end,
                            "speed_factor": min(required_speed, video_speed_max),
                            "is_pause": False,
                            "speaker": speaker
                        })
                else:
                    # Exceeds max - cap at video_speed_max
                    speed_segments.append({
                        "start": original_start,
                        "end": original_end,
                        "speed_factor": video_speed_max,
                        "is_pause": False,
                        "speaker": speaker
                    })
            
            prev_end = original_end
        
        # Handle trailing segment after last speech
        if prev_end < video_duration:
            speed_segments.append({
                "start": prev_end,
                "end": video_duration,
                "speed_factor": 1.0,
                "is_pause": True
            })
        
        return speed_segments
    
    def _apply_video_speedup(
        self,
        command: List[str],
        video_path: str,
        segment_positions: List[Dict],
        min_pause_duration: float,
        preserve_pause_duration: float,
        video_speed_min: float,
        video_speed_comfortable: float,
        video_speed_max: float,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[str], List[str], List[Dict[str, float]]]:
        """Apply variable speed to video segments based on dubbed audio timing.
        
        Uses two-phase speedup: comfortable speed first, extended for remainder.
        
        Args:
            command: Original FFmpeg command
            video_path: Path to video file
            segment_positions: List of segment positions with timing info
            min_pause_duration: Minimum pause duration to process
            preserve_pause_duration: Duration of pause to preserve
            video_speed_min: Minimum video speed
            video_speed_comfortable: Comfortable speedup limit for phase 1
            video_speed_max: Extended speedup limit for phase 2 remainder
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (Modified FFmpeg command, temp files to cleanup, pause adjustments)
        """
        video_duration = self._get_video_duration(video_path)
        
        # Calculate speed segments with two-phase approach
        speed_segments = self._calculate_speed_segments(
            segment_positions, video_duration,
            min_pause_duration, preserve_pause_duration,
            video_speed_min, video_speed_comfortable, video_speed_max
        )
        
        if not speed_segments:
            logger.warning("No speed segments calculated, skipping speedup")
            return command, [], []
        
        # Log speed segment info with phase breakdown
        total_speedup_time = 0.0
        segments_to_speedup = []
        phase1_count = 0
        phase2_count = 0
        comfortable_count = 0
        
        for seg in speed_segments:
            if seg["speed_factor"] > 1.0:
                original_dur = seg["end"] - seg["start"]
                new_dur = original_dur / seg["speed_factor"]
                time_saved = original_dur - new_dur
                total_speedup_time += time_saved
                segments_to_speedup.append(seg)
                
                # Count phases
                phase = seg.get("phase")
                if phase == 1:
                    phase1_count += 1
                elif phase == 2:
                    phase2_count += 1
                elif seg["speed_factor"] <= video_speed_comfortable:
                    comfortable_count += 1
        
        if not segments_to_speedup:
            logger.info("No segments need speedup, all within normal speed range")
            return command, [], []
        
        logger.info(f"Speedup plan: {len(segments_to_speedup)} segments will be sped up")
        if phase1_count > 0 or phase2_count > 0:
            logger.info(f"Two-phase breakdown: {phase1_count} comfortable ({video_speed_comfortable}x), "
                       f"{phase2_count} extended ({video_speed_max}x)")
        if comfortable_count > 0:
            logger.info(f"Single-phase segments: {comfortable_count} within comfortable limit")
        logger.info(f"Estimated time reduction: {total_speedup_time:.2f}s")
        
        # Create temporary sped-up video
        temp_video_path = "artifacts/temp_speedup_video.mp4"
        temp_audio_path = "artifacts/temp_speedup_audio.wav"
        temp_files_to_cleanup = [temp_video_path, temp_audio_path]
        
        # Build FFmpeg command with variable speed using trim + setpts + concat
        try:
            self._create_variable_speed_video(
                video_path, speed_segments, temp_video_path,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Failed to create sped-up video: {e}")
            return command, [], []
        
        if not os.path.exists(temp_video_path):
            logger.error("Sped-up video was not created")
            return command, [], []
        
        # Calculate pause adjustments for subtitle timing
        pause_adjustments = self._calculate_speedup_adjustments(speed_segments)
        
        # Modify original command to use the sped-up video
        modified_command = []
        for i, arg in enumerate(command):
            if arg == video_path and i > 0 and command[i-1] == "-i":
                modified_command.append(temp_video_path)
            else:
                modified_command.append(arg)
        
        # Also create sped-up versions of audio files
        for i, arg in enumerate(command):
            if arg == "-i" and i + 1 < len(command):
                input_file = command[i + 1]
                if input_file != video_path and input_file != temp_video_path and os.path.exists(input_file):
                    # Check if it's an audio file
                    try:
                        probe_cmd = [
                            'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_file
                        ]
                        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                        if 'audio' in result.stdout:
                            sped_audio = self._create_sped_audio_file(input_file, speed_segments)
                            if sped_audio:
                                temp_files_to_cleanup.append(sped_audio)
                                # Replace in command
                                for j, cmd_arg in enumerate(modified_command):
                                    if cmd_arg == input_file:
                                        modified_command[j] = sped_audio
                    except subprocess.CalledProcessError:
                        pass
        
        logger.info(f"Video speedup applied, total time saved: {total_speedup_time:.2f}s")
        
        return modified_command, temp_files_to_cleanup, pause_adjustments
    
    def _create_variable_speed_video(
        self,
        video_path: str,
        speed_segments: List[Dict],
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Create a video with variable speed applied to different segments.
        
        Uses FFmpeg's trim + setpts + concat filters for precise speed control.
        """
        if not speed_segments:
            return
        
        video_info = self._get_video_info(video_path)
        
        # Build FFmpeg filter for variable speed
        # We'll create separate trimmed+sped segments and concat them
        filter_parts = []
        video_labels = []
        audio_labels = []
        
        for i, seg in enumerate(speed_segments):
            start = seg["start"]
            end = seg["end"]
            speed = seg["speed_factor"]
            
            # Trim the segment
            # setpts=PTS/speed speeds up video (lower PTS = faster)
            # atempo=speed speeds up audio (must be between 0.5 and 2.0, chain for more)
            pts_factor = 1.0 / speed
            
            # Video: trim and speed up
            filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)*{pts_factor}[v{i}]"
            )
            video_labels.append(f"[v{i}]")
            
            # Audio: trim and speed up using atempo
            # atempo only supports 0.5-2.0, so we chain filters for extreme speeds
            atempo_filters = self._build_atempo_chain(speed)
            filter_parts.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,{atempo_filters}[a{i}]"
            )
            audio_labels.append(f"[a{i}]")
        
        # Concat all segments
        n_segments = len(speed_segments)
        concat_video = "".join(video_labels)
        concat_audio = "".join(audio_labels)
        filter_parts.append(
            f"{concat_video}concat=n={n_segments}:v=1:a=0[outv]"
        )
        filter_parts.append(
            f"{concat_audio}concat=n={n_segments}:v=0:a=1[outa]"
        )
        
        filter_complex = ";".join(filter_parts)
        
        # Build command
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ]
        
        logger.debug(f"Creating variable speed video: {' '.join(cmd)}")
        
        # Calculate expected output duration for progress tracking
        total_output_duration = sum(
            (seg["end"] - seg["start"]) / seg["speed_factor"]
            for seg in speed_segments
        )
        
        self._run_ffmpeg_with_progress(
            cmd, total_output_duration, progress_callback, "Pauses speedup"
        )
    
    def _build_atempo_chain(self, speed: float) -> str:
        """Build atempo filter chain for a given speed factor.
        
        atempo only supports 0.5-2.0, so we chain multiple filters for extreme speeds.
        """
        if speed <= 0:
            return "atempo=1.0"
        
        filters = []
        remaining_speed = speed
        
        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0
        
        while remaining_speed < 0.5:
            filters.append("atempo=0.5")
            remaining_speed /= 0.5
        
        if 0.5 <= remaining_speed <= 2.0:
            filters.append(f"atempo={remaining_speed}")
        
        return ",".join(filters) if filters else "atempo=1.0"
    
    def _create_sped_audio_file(
        self,
        audio_path: str,
        speed_segments: List[Dict]
    ) -> Optional[str]:
        """Create a sped-up version of an audio file matching video speed changes."""
        if not speed_segments or not os.path.exists(audio_path):
            return None
        
        audio_dir = os.path.dirname(audio_path)
        audio_name = os.path.basename(audio_path)
        name, ext = os.path.splitext(audio_name)
        output_path = os.path.join(audio_dir, f"{name}_sped.wav")
        
        try:
            # Build filter similar to video
            filter_parts = []
            audio_labels = []
            
            for i, seg in enumerate(speed_segments):
                start = seg["start"]
                end = seg["end"]
                speed = seg["speed_factor"]
                
                atempo_filters = self._build_atempo_chain(speed)
                filter_parts.append(
                    f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,{atempo_filters}[a{i}]"
                )
                audio_labels.append(f"[a{i}]")
            
            n_segments = len(speed_segments)
            concat_audio = "".join(audio_labels)
            filter_parts.append(
                f"{concat_audio}concat=n={n_segments}:v=0:a=1[outa]"
            )
            
            filter_complex = ";".join(filter_parts)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-filter_complex", filter_complex,
                "-map", "[outa]",
                "-c:a", "pcm_s16le",
                output_path
            ]
            
            logger.debug(f"Creating sped audio: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create sped audio: {e.stderr}")
            return None
    
    def _calculate_speedup_adjustments(
        self,
        speed_segments: List[Dict]
    ) -> List[Dict[str, float]]:
        """Calculate timing adjustments for subtitles based on speed changes."""
        adjustments = []
        cumulative_offset = 0.0
        
        for seg in speed_segments:
            original_duration = seg["end"] - seg["start"]
            new_duration = original_duration / seg["speed_factor"]
            time_change = original_duration - new_duration
            
            if abs(time_change) > 0.01:  # Only track significant changes
                cumulative_offset += time_change
                adjustments.append({
                    "original_start": seg["start"],
                    "original_end": seg["end"],
                    "time_removed": time_change,
                    "cumulative_offset": cumulative_offset
                })
        
        return adjustments
    
    def _build_minterpolate_filter(self, speed_factor: float, target_fps: int = 30) -> str:
        """Build minterpolate filter for smooth video slowdown.
        
        minterpolate creates interpolated frames for smoother slow motion effect.
        This is used when slowing down video beyond the comfort zone.
        
        Args:
            speed_factor: Speed factor (< 1.0 for slowdown)
            target_fps: Target frame rate for output
            
        Returns:
            FFmpeg filter string for minterpolate + setpts
        """
        # minterpolate interpolates frames for smoother slow motion
        # mi_mode=mci: Motion Compensated Interpolation
        # mc_mode=aobmc: Adaptive Overlapped Block Motion Compensation
        # me_mode=bidir: Bidirectional motion estimation
        pts_factor = 1.0 / speed_factor  # PTS multiplier for slowdown
        
        return (
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
            f"setpts={pts_factor}*PTS"
        )
    
    def _build_setpts_filter(self, speed_factor: float) -> str:
        """Build setpts filter for video speed change.
        
        Args:
            speed_factor: Speed factor (> 1.0 for speedup, < 1.0 for slowdown)
            
        Returns:
            FFmpeg setpts filter string
        """
        pts_factor = 1.0 / speed_factor
        return f"setpts={pts_factor}*PTS"
    
    def _apply_per_segment_video_speed(
        self,
        video_path: str,
        segments_with_speed: List[Dict],
        output_path: str,
        video_min_speed: float = 0.75,
        use_minterpolate_for_slowdown: bool = True,
        target_duration: Optional[float] = None,
        progress_callback: Optional[callable] = None,
        log_callback: Optional[callable] = None
    ) -> Tuple[str, List[Dict[str, float]]]:
        """Apply per-segment video speed adjustments based on dubbed audio timing.
        
        This method processes segments that have 'video_speed_required' attribute
        and applies variable speed changes to sync video with dubbed audio.
        
        Args:
            video_path: Path to the original video file
            segments_with_speed: List of segment dicts with 'video_speed_required' field
            output_path: Path for the output video
            video_min_speed: Minimum video speed (slowdown threshold for minterpolate)
            use_minterpolate_for_slowdown: Use minterpolate for smooth slowdown
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (output video path, timing adjustments for subtitles)
        """
        def log(message: str):
            logger.info(message)
            if log_callback:
                log_callback(message)
        
        # Filter segments that have video speed requirements
        speed_segments = [
            seg for seg in segments_with_speed 
            if seg.get('video_speed_required') is not None
        ]
        
        if not speed_segments:
            log("No segments require video speed adjustment")
            return video_path, []

        log(f"Applying video speed adjustment to {len(speed_segments)} segments...")
        log(f"Minterpolate threshold: {video_min_speed}x (speeds below this use smooth frame interpolation)")

        # Get video info
        video_info = self._get_video_info(video_path)
        video_duration = self._get_video_duration(video_path)
        
        # Get target FPS from video info
        video_stream = video_info.get('video_stream', {})
        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                target_fps = int(float(num) / float(den))
            else:
                target_fps = int(float(fps_str))
        except (ValueError, ZeroDivisionError):
            target_fps = 30
        
        # Sort segments by start time
        sorted_segments = sorted(speed_segments, key=lambda x: x.get('video_sync_start', x.get('start', 0)))
        
        # Build speed segment list with gaps filled
        full_speed_segments = []
        prev_end = 0.0
        
        for seg in sorted_segments:
            seg_start = seg.get('video_sync_start', seg.get('start', 0))
            seg_end = seg.get('video_sync_end', seg.get('end', 0))
            speed_factor = seg.get('video_speed_required', 1.0)
            
            # Add gap before this segment at normal speed
            if seg_start > prev_end + 0.01:
                full_speed_segments.append({
                    "start": prev_end,
                    "end": seg_start,
                    "speed_factor": 1.0,
                    "is_gap": True
                })
            
            # Add the speed-adjusted segment
            full_speed_segments.append({
                "start": seg_start,
                "end": seg_end,
                "speed_factor": speed_factor,
                "is_gap": False,
                "use_minterpolate": use_minterpolate_for_slowdown and speed_factor < video_min_speed
            })
            
            prev_end = seg_end
        
        # Add final gap if video continues after last segment
        if prev_end < video_duration - 0.01:
            full_speed_segments.append({
                "start": prev_end,
                "end": video_duration,
                "speed_factor": 1.0,
                "is_gap": True
            })
        
        # Log the speed segments
        speedup_count = sum(1 for s in full_speed_segments if s["speed_factor"] > 1.0)
        slowdown_count = sum(1 for s in full_speed_segments if s["speed_factor"] < 1.0 and not s.get("is_gap"))
        minterpolate_count = sum(1 for s in full_speed_segments if s.get("use_minterpolate", False))
        
        log(f"Video speed adjustments: {speedup_count} speedup, {slowdown_count} slowdown "
            f"({minterpolate_count} using minterpolate for smooth slowdown)")
        
        # Build FFmpeg filter for variable speed
        filter_parts = []
        video_labels = []
        audio_labels = []
        
        for i, seg in enumerate(full_speed_segments):
            start = seg["start"]
            end = seg["end"]
            speed = seg["speed_factor"]
            
            # Video processing
            if seg.get("use_minterpolate", False):
                # Use minterpolate for smooth slowdown
                video_filter = self._build_minterpolate_filter(speed, target_fps)
                filter_parts.append(
                    f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,{video_filter}[v{i}]"
                )
            else:
                # Use simple setpts for speedup or regular slowdown
                pts_factor = 1.0 / speed
                filter_parts.append(
                    f"[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)*{pts_factor}[v{i}]"
                )
            video_labels.append(f"[v{i}]")
            
            # Audio processing
            atempo_filters = self._build_atempo_chain(speed)
            filter_parts.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,{atempo_filters}[a{i}]"
            )
            audio_labels.append(f"[a{i}]")
        
        # Concat all segments
        n_segments = len(full_speed_segments)
        concat_video = "".join(video_labels)
        concat_audio = "".join(audio_labels)
        filter_parts.append(
            f"{concat_video}concat=n={n_segments}:v=1:a=0[outv]"
        )
        filter_parts.append(
            f"{concat_audio}concat=n={n_segments}:v=0:a=1[outa]"
        )

        # Calculate expected output duration for progress tracking
        total_output_duration = sum(
            (seg["end"] - seg["start"]) / seg["speed_factor"]
            for seg in full_speed_segments
        )

        video_out_label = "[outv]"
        audio_out_label = "[outa]"
        if target_duration and target_duration > total_output_duration + 0.05:
            pad_duration = target_duration - total_output_duration
            filter_parts.append(
                f"{video_out_label}tpad=stop_mode=clone:stop_duration={pad_duration}[outv_pad]"
            )
            filter_parts.append(
                f"{audio_out_label}apad=pad_dur={pad_duration}[outa_pad]"
            )
            video_out_label = "[outv_pad]"
            audio_out_label = "[outa_pad]"
            total_output_duration = target_duration
            log(f"Padding speed-adjusted video by {pad_duration:.2f}s to match audio duration")

        filter_complex = ";".join(filter_parts)
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", video_out_label, "-map", audio_out_label,
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ]
        
        logger.debug(f"Creating per-segment speed video: {' '.join(cmd)}")
        
        try:
            self._run_ffmpeg_with_progress(
                cmd, total_output_duration, progress_callback, "Per-segment video speed"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply per-segment video speed: {e.stderr}")
            raise
        
        # Calculate timing adjustments for subtitles
        timing_adjustments = self._calculate_speedup_adjustments(full_speed_segments)
        
        logger.info(f"Per-segment video speed adjustment completed. Output: {output_path}")
        
        return output_path, timing_adjustments
