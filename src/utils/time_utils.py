import math

def format_seconds_to_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted SRT timestamp
    """
    if not math.isfinite(seconds):
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    milliseconds = int((sec - int(sec)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(sec):02d},{milliseconds:03d}"

def format_seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH.MM.SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted HH.MM.SS timestamp
    """
    if not math.isfinite(seconds):
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)

    return f"{hours:02d}.{minutes:02d}.{sec:02d}" 