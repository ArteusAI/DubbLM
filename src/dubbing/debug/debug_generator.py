"""Debug video generation for the Smart Dubbing system."""

import os
import subprocess
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.utils.time_utils import format_seconds_to_srt
from ..utils.subtitle_utils import SubtitleManager
from ..core.log_config import get_logger

logger = get_logger(__name__)


class DebugGenerator:
    """Generates debug videos with comprehensive annotations for the Smart Dubbing system."""
    
    def __init__(self):
        """Initialize the debug generator."""
        self.subtitle_manager = SubtitleManager()
    
    def generate_debug_video(self, 
                           video_path: str,
                           debug_data: Dict[str, Any],
                           start_time: Optional[float] = None,
                           duration: Optional[float] = None,
                           total_duration: Optional[float] = None) -> None:
        """Generate a debug video with comprehensive annotations."""
        logger.debug("Generating comprehensive debug video...")
        debug_dir = "artifacts/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Choose debug video method based on video duration
        if total_duration and total_duration < 1800:  # 30 minutes = 1800 seconds
            self._generate_detailed_debug_video(video_path, debug_data, debug_dir, start_time, duration)
        else:
            self._generate_simple_debug_video(video_path, debug_data, debug_dir, start_time, duration)
    
    def _generate_simple_debug_video(self, 
                                   video_path: str,
                                   debug_data: Dict[str, Any],
                                   debug_dir: str,
                                   start_time: Optional[float],
                                   duration: Optional[float]) -> None:
        """Generate simpler debug video focusing only on showing transcription segments."""
        logger.debug("Generating simple debug video with transcription segments...")
        
        # Start time for trim if needed
        trim_option = f"-ss {start_time} " if start_time is not None else ""
        duration_option = f"-t {duration} " if duration is not None else ""
        
        base_video = os.path.join(debug_dir, "base_video.mp4")
        final_debug_video = os.path.join(debug_dir, "dubbing_debug.mp4")
        
        try:
            # Create base video with minimal processing
            self._run_ffmpeg_command(
                f"ffmpeg -y -loglevel error {trim_option}-i \"{video_path}\" {duration_option} "
                f"-c:v copy -c:a copy \"{base_video}\"",
                "Creating base video..."
            )
            
            # Create subtitle file with transcription segments
            srt_file = os.path.join(debug_dir, "transcription.srt")
            if debug_data.get("transcription"):
                with open(srt_file, 'w', encoding='utf-8') as f:
                    for i, segment in enumerate(debug_data["transcription"]):
                        speaker = segment.get("speaker", "UNKNOWN")
                        start_time = format_seconds_to_srt(segment["start"])
                        end_time = format_seconds_to_srt(segment["end"])
                        text = segment["text"]
                        
                        f.write(f"{i+1}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"[{speaker}]: {text}\n\n")
            
            # Use subtitles
            self._run_ffmpeg_command(
                f"ffmpeg -y -loglevel error -i \"{base_video}\" "
                f"-vf \"subtitles='{srt_file}'\" -c:a copy \"{final_debug_video}\"",
                "Adding transcription subtitles..."
            )
            
            logger.info(f"Simple debug video created at {final_debug_video}")
            
            # Clean up
            if os.path.exists(base_video):
                os.remove(base_video)
                
        except Exception as e:
            logger.error(f"Error generating simple debug video: {e}", exc_info=True)
    
    def _generate_detailed_debug_video(self, 
                                     video_path: str,
                                     debug_data: Dict[str, Any],
                                     debug_dir: str,
                                     start_time: Optional[float],
                                     duration: Optional[float]) -> None:
        """Generate detailed debug video - fallback to simple for now."""
        self._generate_simple_debug_video(video_path, debug_data, debug_dir, start_time, duration)
    
    def _run_ffmpeg_command(self, command: str, message: str = None) -> None:
        """Run an FFmpeg command with proper error handling."""
        if message:
            logger.debug(message)
            
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else "No error details available"
            logger.error(f"Command failed: {error_output}")
            raise 