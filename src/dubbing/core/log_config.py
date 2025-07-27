"""Logging configuration for the Smart Dubbing system."""

import logging
import sys
import os
from datetime import datetime

def setup_logging(level=logging.INFO):
    """
    Set up logging for the application.
    
    Args:
        level: The logging level to use for console output (e.g., logging.INFO, logging.DEBUG)
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a filename with the current date and time
    log_filename = datetime.now().strftime(os.path.join(log_dir, 'dubbing_%Y-%m-%d_%H-%M-%S.log'))

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