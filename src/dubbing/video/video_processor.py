"""Video processing for the Smart Dubbing system."""

import os
import subprocess
import json
import tempfile
import shutil
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        # Coarse pre-seek for per-segment processing.
        # We still do exact trim in filters, so timing precision is preserved.
        self.video_segment_seek_padding = max(
            0.0,
            float(self.config.get('video_segment_seek_padding', 1.0))
        )
    
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

    def _parse_ffprobe_fps(self, raw_fps: Any) -> Optional[float]:
        """Parse ffprobe frame-rate strings like '30000/1001' or '30'."""
        if raw_fps is None:
            return None
        text = str(raw_fps).strip()
        if not text or text in {"0/0", "N/A"}:
            return None
        try:
            if "/" in text:
                num_str, den_str = text.split("/", 1)
                num = float(num_str)
                den = float(den_str)
                if den <= 0:
                    return None
                fps = num / den
            else:
                fps = float(text)
            if fps <= 0:
                return None
            return fps
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _resolve_speed_output_fps(self, video_info: Dict[str, Any], needs_high_fps: bool) -> float:
        """Resolve a stable CFR target FPS for per-segment speed rendering."""
        default_fps = 30.0
        video_stream = video_info.get("video_stream") or {}
        parsed_fps = (
            self._parse_ffprobe_fps(video_stream.get("avg_frame_rate"))
            or self._parse_ffprobe_fps(video_stream.get("r_frame_rate"))
            or default_fps
        )
        # Clamp to a sane playback range.
        parsed_fps = max(15.0, min(parsed_fps, 120.0))
        if needs_high_fps:
            return max(parsed_fps, 60.0)
        return parsed_fps

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
    
    def _build_atempo_chain(self, speed: float) -> str:
        """Build chained atempo filters for speed outside 0.5-2.0 range.
        
        FFmpeg atempo filter is limited to 0.5-2.0 range, so we chain
        multiple filters for larger speed changes.
        """
        if 0.5 <= speed <= 2.0:
            return f"atempo={speed:.6f}"
        
        filters = []
        remaining = speed
        
        while abs(remaining - 1.0) > 0.001:
            if remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2.0
            elif remaining < 0.5:
                filters.append("atempo=0.5")
                remaining /= 0.5
            else:
                filters.append(f"atempo={remaining:.6f}")
                break
            
            # Safety limit
            if len(filters) > 10:
                logger.warning(f"Too many atempo stages ({len(filters)}), using final value")
                break
        
        return ",".join(filters) if filters else "atempo=1.0"

    def _align_audio_duration_to_video(
        self,
        audio_path: str,
        target_duration: float,
        tolerance_seconds: float = 0.25,
        log_callback: Optional[callable] = None,
    ) -> str:
        """Apply a tiny global tempo correction so dubbed audio matches rendered video duration.

        Per-segment video rendering can accumulate fractional duration drift on long timelines.
        This correction keeps final A/V sync stable without altering segment-level timing logic.
        """
        def log(message: str):
            logger.info(message)
            if log_callback:
                log_callback(message)

        if target_duration <= 0 or not os.path.exists(audio_path):
            return audio_path

        source_duration = self._get_video_duration(audio_path)
        if source_duration <= 0:
            logger.warning(
                "Could not measure translated audio duration for '%s', skipping A/V duration alignment.",
                audio_path,
            )
            return audio_path

        duration_delta = source_duration - target_duration
        if abs(duration_delta) <= tolerance_seconds:
            return audio_path

        tempo_speed = source_duration / target_duration
        atempo_filter = self._build_atempo_chain(tempo_speed)
        base, ext = os.path.splitext(audio_path)
        if not ext:
            ext = ".wav"
        synced_audio_path = f"{base}_synced{ext}"

        log(
            "Translated audio/video duration mismatch detected: "
            f"audio={source_duration:.3f}s, video={target_duration:.3f}s "
            f"(delta={duration_delta:+.3f}s). "
            f"Applying global tempo correction (speed={tempo_speed:.6f})."
        )

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-filter:a",
                atempo_filter,
                "-vn",
                synced_audio_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            synced_duration = self._get_video_duration(synced_audio_path)
            log(
                "Duration alignment complete: "
                f"new_audio={synced_duration:.3f}s (target_video={target_duration:.3f}s)."
            )
            return synced_audio_path
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Failed to align translated audio duration to video: %s. "
                "Using original translated audio.",
                exc.stderr if hasattr(exc, "stderr") else exc,
            )
            return audio_path
    
    def _apply_per_segment_video_speed(
        self,
        video_path: str,
        speed_segments: List[Dict],
        output_path: str,
        progress_callback: Optional[callable] = None,
        log_callback: Optional[callable] = None
    ) -> Tuple[str, List[Dict[str, float]]]:
        """
        Apply different speed to different video segments for video/audio_and_video modes.
        
        Creates a video where each segment plays at its specified speed.
        Final video duration will differ from original.
        
        Args:
            video_path: Path to original video
            speed_segments: List of dicts with keys: original_start, original_end, video_speed, use_minterpolate
            output_path: Path for output video
            progress_callback: Optional progress callback
            log_callback: Optional log callback
            
        Returns:
            Tuple of (output video path, list of timing adjustments for subtitles)
        """
        def log(message: str):
            logger.info(message)
            if log_callback:
                log_callback(message)
        
        if not speed_segments:
            log("No video speed segments provided, returning original video")
            return video_path, []

        log(f"Applying per-segment video speed adjustment to {len(speed_segments)} segments...")

        # Get video info for quality settings
        video_info = self._get_video_info(video_path)
        total_original_duration = video_info.get('duration', 0)

        # Build valid segments.
        # We compute subtitle timing adjustments from actual rendered segment durations
        # later to avoid drift from per-segment encoder/mux rounding.
        valid_segments = []
        requested_total_new_time = 0.0
        for i, seg in enumerate(speed_segments):
            original_start = float(seg.get('original_start', 0))
            original_end = float(seg.get('original_end', 0))
            video_speed = float(seg.get('video_speed', 1.0))
            use_minterpolate = bool(seg.get('use_minterpolate', False))

            original_duration = original_end - original_start
            if original_duration <= 0:
                continue

            if video_speed <= 0:
                logger.warning(f"Invalid video_speed={video_speed} for segment {i}, using 1.0")
                video_speed = 1.0

            requested_new_duration = original_duration / video_speed
            valid_segments.append({
                "index": i,
                "original_start": original_start,
                "original_end": original_end,
                "video_speed": video_speed,
                "use_minterpolate": use_minterpolate,
                "requested_new_duration": requested_new_duration
            })
            requested_total_new_time += requested_new_duration

        if not valid_segments:
            log("No valid segments to process")
            return video_path, []
        segment_meta_by_index = {seg["index"]: seg for seg in valid_segments}
        uses_minterpolate = any(
            seg["use_minterpolate"] and seg["video_speed"] < self.video_minterpolate_threshold
            for seg in valid_segments
        )
        target_output_fps = self._resolve_speed_output_fps(video_info, uses_minterpolate)
        target_output_fps_arg = f"{target_output_fps:.6f}"
        log(f"Using stable CFR for speed segments: {target_output_fps:.2f} fps")

        cpu_total = os.cpu_count() or 1
        segment_workers = max(1, cpu_total - 1)
        log(f"Parallel video speed stretching: workers={segment_workers} (host_cpu={cpu_total})")

        temp_dir = tempfile.mkdtemp(prefix="video_speed_segments_")
        segment_outputs: List[Tuple[int, str]] = []
        completed = 0

        def process_segment(seg_meta: Dict[str, Any]) -> Tuple[int, str]:
            seg_idx = seg_meta["index"]
            original_start = seg_meta["original_start"]
            original_end = seg_meta["original_end"]
            video_speed = seg_meta["video_speed"]
            use_minterpolate = seg_meta["use_minterpolate"]

            # Use coarse input seek to avoid decoding from the beginning for each segment.
            # Precision is preserved by applying exact trim on top of the seeked input.
            input_seek = max(0.0, original_start - self.video_segment_seek_padding)
            trim_start = max(0.0, original_start - input_seek)
            trim_end = max(trim_start, original_end - input_seek)

            start_s = f"{trim_start:.6f}"
            end_s = f"{trim_end:.6f}"
            setpts_factor = 1.0 / video_speed
            speed_is_unity = abs(video_speed - 1.0) < 0.001
            logger.debug(
                "Segment %s seek plan: input_seek=%.6f, trim_start=%.6f, trim_end=%.6f, "
                "original=[%.6f, %.6f], speed=%.4f",
                seg_idx,
                input_seek,
                trim_start,
                trim_end,
                original_start,
                original_end,
                video_speed,
            )

            segment_output = os.path.join(temp_dir, f"segment_{seg_idx:06d}.mp4")
            segment_cmd = ["ffmpeg", "-y"]
            if input_seek > 0:
                segment_cmd.extend(["-ss", f"{input_seek:.6f}"])
            segment_cmd.extend(["-i", video_path])

            if use_minterpolate and video_speed < self.video_minterpolate_threshold:
                vf_parts = [
                    f"trim=start={start_s}:end={end_s}",
                    f"setpts={setpts_factor:.6f}*(PTS-STARTPTS)",
                    "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                    "setpts=PTS-STARTPTS",
                ]
                log(
                    f"Segment {seg_idx}: {original_start:.2f}-{original_end:.2f}s, "
                    f"speed={video_speed:.2f}x with minterpolate"
                )
            elif speed_is_unity:
                # Fast path for 1.0x segments: exact trim without speed transform.
                vf_parts = [
                    f"trim=start={start_s}:end={end_s}",
                    "setpts=PTS-STARTPTS",
                ]
                log(f"Segment {seg_idx}: {original_start:.2f}-{original_end:.2f}s, speed=1.00x")
            else:
                vf_parts = [
                    f"trim=start={start_s}:end={end_s}",
                    f"setpts={setpts_factor:.6f}*(PTS-STARTPTS)",
                ]
                log(f"Segment {seg_idx}: {original_start:.2f}-{original_end:.2f}s, speed={video_speed:.2f}x")
            # Keep all rendered segments on one constant frame rate/timebase.
            vf_parts.append(f"fps=fps={target_output_fps_arg}")
            vf = ",".join(vf_parts)

            segment_cmd.extend([
                "-vf", vf,
                "-an",
                "-threads", "1",
                "-vsync", "cfr",
                "-r", target_output_fps_arg,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-tag:v", "avc1",
                segment_output
            ])
            logger.debug(f"Segment {seg_idx} FFmpeg command: {' '.join(segment_cmd)}")
            subprocess.run(segment_cmd, capture_output=True, text=True, check=True)
            return seg_idx, segment_output

        try:
            with ThreadPoolExecutor(max_workers=segment_workers) as executor:
                futures = [executor.submit(process_segment, seg_meta) for seg_meta in valid_segments]
                for future in as_completed(futures):
                    seg_idx, seg_path = future.result()
                    segment_outputs.append((seg_idx, seg_path))
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(valid_segments), "Video speed segment processing")

            segment_outputs.sort(key=lambda item: item[0])

            # Build timing adjustments from the real encoded segment durations.
            timing_adjustments = []
            cumulative_actual_time = 0.0
            for seg_idx, seg_path in segment_outputs:
                seg_meta = segment_meta_by_index.get(seg_idx)
                if not seg_meta:
                    continue
                original_start = seg_meta["original_start"]
                original_end = seg_meta["original_end"]
                requested_speed = seg_meta["video_speed"]
                original_duration = max(0.0, original_end - original_start)
                segment_actual_duration = self._get_video_duration(seg_path)
                if segment_actual_duration <= 0:
                    segment_actual_duration = seg_meta.get("requested_new_duration", 0.0)
                    logger.warning(
                        "Could not measure rendered duration for segment %s. "
                        "Falling back to requested duration %.6fs.",
                        seg_idx,
                        segment_actual_duration,
                    )

                effective_video_speed = requested_speed
                if segment_actual_duration > 0 and original_duration > 0:
                    effective_video_speed = original_duration / segment_actual_duration

                timing_adjustments.append({
                    "original_start": original_start,
                    "original_end": original_end,
                    "new_start": cumulative_actual_time,
                    "new_end": cumulative_actual_time + segment_actual_duration,
                    "video_speed": effective_video_speed,
                    "requested_video_speed": requested_speed,
                    "offset": cumulative_actual_time - original_start
                })
                cumulative_actual_time += segment_actual_duration
            concat_list_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for _, seg_path in segment_outputs:
                    safe_path = seg_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-fflags", "+genpts",
                "-i", concat_list_file,
                "-an",
                "-vsync", "cfr",
                "-r", target_output_fps_arg,
                "-c:v", "libx264",
                "-crf", "16",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-tag:v", "avc1",
                "-movflags", "+faststart",
                output_path
            ]
            logger.debug(f"Concat FFmpeg command: {' '.join(concat_cmd)}")
            subprocess.run(concat_cmd, capture_output=True, text=True, check=True)

            muxed_duration = self._get_video_duration(output_path)
            if muxed_duration > 0 and cumulative_actual_time > 0:
                # Concat muxing/re-encode can slightly shift cumulative duration.
                # Scale the timing map so subtitles follow the real muxed output.
                duration_scale = muxed_duration / cumulative_actual_time
                if abs(duration_scale - 1.0) > 1e-4:
                    for adj in timing_adjustments:
                        adj["new_start"] *= duration_scale
                        adj["new_end"] *= duration_scale
                        adj["offset"] = adj["new_start"] - adj["original_start"]
                    logger.warning(
                        "Applied timing scale %.8f to subtitle mapping "
                        "(segment_sum=%.6fs, muxed=%.6fs).",
                        duration_scale,
                        cumulative_actual_time,
                        muxed_duration,
                    )
                    cumulative_actual_time = muxed_duration

            log(
                "Video speed adjustment complete. "
                f"Requested duration: {requested_total_new_time:.2f}s, "
                f"Rendered segments: {cumulative_actual_time:.2f}s "
                f"(original: {total_original_duration:.2f}s)"
            )
            return output_path, timing_adjustments

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply video speed adjustment: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
            return video_path, []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
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
                                keyframe_buffer: float = 0.2,
                                ffmpeg_batch_size: int = 50,
                                dubbed_volume: float = 1.0,
                                background_volume: float = 0.562341,
                                max_output_height: Optional[int] = None,
                                upscale_factor: float = 1.0,
                                upscale_sharpen: bool = True,
                                progress_callback: Optional[callable] = None,
                                log_callback: Optional[callable] = None,
                                video_speed_segments: Optional[List[Dict]] = None) -> Tuple[str, List[Dict[str, float]]]:
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
            pause_removal: Pause removal mode: 'cut' (remove pauses), 'disabled'
            min_pause_duration: Minimum silence duration to consider for processing (seconds)
            preserve_pause_duration: The duration of pause to keep after processing (seconds)
            keyframe_buffer: Buffer around keyframes to preserve (seconds)
            ffmpeg_batch_size: Number of cuts to process in a single ffmpeg command
            dubbed_volume: Gain multiplier for the translated track (e.g., 1.2 for +1.6 dB)
            upscale_factor: Video scale factor (>1.0 to upscale; default 1.0 disables)
            upscale_sharpen: Apply mild sharpening after scaling
            max_output_height: Maximum output video height (e.g., 720/1080). None preserves original.
            progress_callback: Optional callback(current, total, message) for progress updates
            log_callback: Optional callback(message) for logging to external systems
            video_speed_segments: Optional list of VideoSpeedSegment dicts for per-segment video speed adjustment
                Each dict contains: original_start, original_end, video_speed, use_minterpolate

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

        # Track video speed timing adjustments for subtitle synchronization
        video_speed_timing_adjustments = []
        
        # Apply per-segment video speed if provided (video/audio_and_video modes)
        effective_video_path = video_path
        if video_speed_segments and len(video_speed_segments) > 0:
            log(f"Applying per-segment video speed adjustment ({len(video_speed_segments)} segments)...")
            temp_speed_adjusted_video = "artifacts/temp_speed_adjusted_video.mp4"
            
            # Convert VideoSpeedSegment dataclass instances to dicts if needed
            speed_segments_dicts = []
            for seg in video_speed_segments:
                if hasattr(seg, '__dict__'):
                    # It's a dataclass or object with attributes
                    speed_segments_dicts.append({
                        'original_start': getattr(seg, 'original_start', seg.get('original_start', 0) if isinstance(seg, dict) else 0),
                        'original_end': getattr(seg, 'original_end', seg.get('original_end', 0) if isinstance(seg, dict) else 0),
                        'video_speed': getattr(seg, 'video_speed', seg.get('video_speed', 1.0) if isinstance(seg, dict) else 1.0),
                        'use_minterpolate': getattr(seg, 'use_minterpolate', seg.get('use_minterpolate', False) if isinstance(seg, dict) else False)
                    })
                else:
                    speed_segments_dicts.append(seg)
            
            effective_video_path, video_speed_timing_adjustments = self._apply_per_segment_video_speed(
                video_path,
                speed_segments_dicts,
                temp_speed_adjusted_video,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if effective_video_path != video_path:
                log(f"Video speed adjustment complete, using adjusted video")
                # In video/audio_and_video modes, disable pause_removal as timing is already adjusted
                if pause_removal != 'disabled':
                    log("Disabling pause_removal for video speed modes (timing already adjusted)")
                    pause_removal = 'disabled'

        # Audio normalization is now done per-segment before combination
        # Skip whole-track normalization since segments are already normalized
        normalized_translated_audio_path = translated_audio_path
        duration_synced_audio_temp = None
        logger.debug("Using pre-normalized per-segment audio (normalization disabled at track level)")

        # Initialize logo dimensions to default values
        logo_width = 0
        logo_height = 0

        # Check if the files exist before trying to use them
        if not os.path.exists(effective_video_path):
            raise FileNotFoundError(f"Input video file not found: {effective_video_path}")
        if not os.path.exists(normalized_translated_audio_path):
            raise FileNotFoundError(f"Normalized translated audio file not found: {normalized_translated_audio_path}")
        if background_audio_path and not os.path.exists(background_audio_path):
            raise FileNotFoundError(f"Background audio file not found: {background_audio_path}")

        # In video/audio_and_video modes, align dubbed audio duration to the
        # real rendered speed-adjusted video duration to prevent cumulative drift.
        if effective_video_path != video_path:
            effective_video_duration = self._get_video_duration(effective_video_path)
            if effective_video_duration > 0:
                maybe_synced_audio = self._align_audio_duration_to_video(
                    normalized_translated_audio_path,
                    effective_video_duration,
                    tolerance_seconds=0.25,
                    log_callback=log_callback,
                )
                if maybe_synced_audio != normalized_translated_audio_path:
                    duration_synced_audio_temp = maybe_synced_audio
                    normalized_translated_audio_path = maybe_synced_audio

        # Get source video metadata once for filter and encoding decisions.
        video_info = self._get_video_info(video_path)

        # Use list-based approach to build command to avoid quoting issues
        command = ["ffmpeg", "-y"]

        # Validate dubbed volume (must be positive)
        if dubbed_volume <= 0:
            logger.warning(f"Invalid dubbed_volume {dubbed_volume}, defaulting to 1.0")
            dubbed_volume = 1.0

        # Add start time if specified (only for original video, not speed-adjusted)
        if start_time is not None and effective_video_path == video_path:
            command.extend(["-ss", str(start_time)])

        # Input 0: Video (original or speed-adjusted)
        command.extend(["-i", effective_video_path])
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

        # Optional output resolution cap (downscale only, never upscale).
        try:
            requested_max_height = int(max_output_height) if max_output_height is not None else None
        except (TypeError, ValueError):
            requested_max_height = None

        source_height = 0
        if video_info.get("video_stream"):
            source_height = int(video_info["video_stream"].get("height") or 0)

        if requested_max_height and requested_max_height > 0 and source_height > requested_max_height:
            all_filter_complex_parts.append(
                f"[{current_video_label}]scale=-2:{requested_max_height}:flags=lanczos[vid_capped]"
            )
            current_video_label = "vid_capped"
            video_map_option = "[vid_capped]"
            log(
                f"Applying video quality cap: {requested_max_height}p "
                f"(source height: {source_height}p)"
            )

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
        if duration_synced_audio_temp and duration_synced_audio_temp != translated_audio_path:
            temp_files_to_cleanup.append(duration_synced_audio_temp)
        
        logger.info(f"Pause removal mode: {pause_removal}")
        
        if pause_removal == "cut":
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

        # Return video speed timing adjustments if used, otherwise pause adjustments
        final_adjustments = video_speed_timing_adjustments if video_speed_timing_adjustments else pause_adjustments
        return output_video_path, final_adjustments 

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
            
