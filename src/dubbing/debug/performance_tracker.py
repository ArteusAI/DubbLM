"""Performance tracking for the Smart Dubbing system."""

import time
from typing import Dict, Optional

from ..core.log_config import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks performance metrics for the Smart Dubbing pipeline."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics = {
            "extract_audio": 0.0,
            "diarization": 0.0,
            "speaker_audio": 0.0,
            "transcription": 0.0,
            "context_analysis": 0.0,
            "translation": 0.0,
            "emotion_analysis": 0.0,
            "speech_synthesis": 0.0,
            "background_audio": 0.0,
            "audio_normalization": 0.0,
            "video_creation": 0.0,
            "total": 0.0
        }
        self._start_times = {}
    
    def start_timing(self, step_name: str) -> None:
        """Start timing a step.
        
        Args:
            step_name: Name of the step to time
        """
        self._start_times[step_name] = time.perf_counter()
    
    def end_timing(self, step_name: str) -> float:
        """End timing a step and record the duration.
        
        Args:
            step_name: Name of the step to end timing for
            
        Returns:
            Duration in seconds
        """
        if step_name in self._start_times:
            duration = time.perf_counter() - self._start_times[step_name]
            self.metrics[step_name] = duration
            del self._start_times[step_name]
            return duration
        return 0.0
    
    def record_metric(self, step_name: str, duration: float) -> None:
        """Record a metric directly.
        
        Args:
            step_name: Name of the step
            duration: Duration in seconds
        """
        self.metrics[step_name] = duration
    
    def get_metric(self, step_name: str) -> float:
        """Get a metric value.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Duration in seconds
        """
        return self.metrics.get(step_name, 0.0)
    
    def write_performance_summary(self, total_duration: Optional[float] = None) -> None:
        """Write performance summary metrics including processing time and speed relative to video duration.
        
        Args:
            total_duration: Total video/segment duration for speed calculations
        """
        logger.info("Performance summary:")
        
        # Get total time for percentage calculations
        total_elapsed = self.metrics.get("total", 0)
        
        # Display all recorded metrics
        summary_lines = []
        summary_lines.append(f"{'Step':<25} {'Time (sec)':<15} {'Time (min)':<15} {'Percentage':<10}")
        summary_lines.append("-" * 65)
        
        # Sort metrics: First key processing steps in pipeline order, then total at end
        step_order = [
            "extract_audio", "diarization", "speaker_audio", "transcription", 
            "context_analysis", "translation", "emotion_analysis", "speech_synthesis", 
            "background_audio", "audio_normalization", "video_creation"
        ]
        
        for step in step_order:
            time_sec = self.metrics.get(step, 0)
            time_min = time_sec / 60
            percentage = (time_sec / total_elapsed * 100) if total_elapsed > 0 else 0
            summary_lines.append(f"{step.replace('_', ' ').title():<25} {time_sec:<15.2f} {time_min:<15.2f} {percentage:<10.1f}%")
        
        # Add a separator before total
        summary_lines.append("-" * 65)
        summary_lines.append(f"{'Total Pipeline':<25} {total_elapsed:<15.2f} {total_elapsed/60:<15.2f} {100:<10.1f}%")
        
        # Calculate and display the processing speed relative to video duration
        if total_duration and total_duration > 0:
            speed_ratio = total_elapsed / total_duration
            summary_lines.append(f"\nProcessing speed:")
            summary_lines.append(f" - Video duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            summary_lines.append(f" - Processing-to-video ratio: {speed_ratio:.2f}x (higher means slower)")
            summary_lines.append(f" - Real-time factor: {1/speed_ratio:.2f}x (how many seconds of video processed per second)")

        # Log all lines
        for line in summary_lines:
            logger.info(line)
    
    def write_performance_summary_for_report(self) -> None:
        """Write performance summary metrics for the speaker report generation process."""
        logger.info("\nPerformance Summary for Speaker Report Generation:")
        
        total_report_time = self.metrics.get("total_report_generation", 0)
        
        summary_lines = []
        summary_lines.append(f"{'Step':<25} {'Time (sec)':<15} {'Time (min)':<15} {'Percentage':<10}")
        summary_lines.append("-" * 65)
        
        report_steps = [
            "extract_audio", "diarization", "transcription", "speaker_audio"
        ]
        
        accumulated_step_time = 0
        for step in report_steps:
            time_sec = self.metrics.get(step, 0)
            # Some steps might be combined, like diarization and transcription
            # Avoid double-counting if one is a sub-part of another and both are logged.
            # For this report, we assume they are distinct timings as per current metrics structure.
            accumulated_step_time += time_sec
            time_min = time_sec / 60
            percentage = (time_sec / total_report_time * 100) if total_report_time > 0 else 0
            # Ensure step name is title cased and underscores replaced
            step_display_name = step.replace('_', ' ').title()
            summary_lines.append(f"{step_display_name:<25} {time_sec:<15.2f} {time_min:<15.2f} {percentage:<10.1f}%")
        
        # Calculate "Report Creation" time as the remainder
        report_creation_time_sec = total_report_time - accumulated_step_time
        report_creation_time_min = report_creation_time_sec / 60
        report_creation_percentage = (report_creation_time_sec / total_report_time * 100) if total_report_time > 0 else 0
        summary_lines.append(f"{'Report File Creation':<25} {report_creation_time_sec:<15.2f} {report_creation_time_min:<15.2f} {report_creation_percentage:<10.1f}%")

        summary_lines.append("-" * 65)
        summary_lines.append(f"{'Total Report Generation':<25} {total_report_time:<15.2f} {total_report_time/60:<15.2f} {100:<10.1f}%")

        total_duration = self.metrics.get("video_duration", 0)
        if total_duration and total_duration > 0:
            speed_ratio = total_report_time / total_duration
            summary_lines.append(f"\nProcessing speed relative to video/segment duration:")
            summary_lines.append(f" - Video/Segment duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            summary_lines.append(f" - Processing-to-duration ratio: {speed_ratio:.2f}x (higher means slower)")
    
        for line in summary_lines:
            logger.info(line)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary.
        
        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0.0
        self._start_times.clear() 