"""Logging configuration for the Smart Dubbing system."""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

# Global variable to store the current log file path
_current_log_file: Optional[str] = None


def get_current_log_file() -> Optional[str]:
    """Get the path to the current log file."""
    return _current_log_file


def setup_logging(level=logging.INFO, output_dir: Optional[str] = None, include_console: bool = True):
    """
    Set up logging for the application.
    
    Args:
        level: The logging level to use for console output (e.g., logging.INFO, logging.DEBUG)
        output_dir: Optional directory to save log file. If None, uses 'logs/' directory.
                   If provided, saves to that directory (e.g., project's artifacts/debug/).
        include_console: Whether to include a console handler. Set to False when running 
                        in environments like Celery that already handle stdout/stderr.
    """
    global _current_log_file
    
    # Determine log directory
    if output_dir:
        log_dir = output_dir
    else:
        log_dir = "logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a filename with the current date and time
    log_filename = datetime.now().strftime(os.path.join(log_dir, 'dubbing_%Y-%m-%d_%H-%M-%S.log'))
    _current_log_file = log_filename

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set root logger to capture all levels

    # Clear existing handlers to avoid duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set logging level for noisy libraries to reduce verbosity
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("http").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("speechbrain.utils.fetching").setLevel(logging.WARNING)
    logging.getLogger("speechbrain.utils.parameter_transfer").setLevel(logging.WARNING)
    logging.getLogger("speechbrain.utils.checkpoints").setLevel(logging.WARNING)

    # Console Handler (prints INFO and above to stdout)
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # Use a simpler format for the console
        console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File Handler (prints DEBUG and above to a file)
    file_handler = logging.FileHandler(log_filename, 'a', 'utf-8')
    file_handler.setLevel(logging.DEBUG)
    # Use a more detailed format for the file
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name) 