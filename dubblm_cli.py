#!/usr/bin/env python3
"""
Video Dubbing Tool

Usage:
    python dubblm_cli.py --input video.mp4 --source_language en --target_language es

For configuration file usage:
    python dubblm_cli.py --config my_config.yml

For speaker report generation:
    python dubblm_cli.py --input video.mp4 --source_language en --generate_speaker_report

"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dubbing.cli.main import main

if __name__ == "__main__":
    main() 