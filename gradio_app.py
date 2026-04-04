#!/usr/bin/env python3
"""Top-level launcher for the DubbLM Gradio interface."""

import os
import sys


SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dubbing.ui.gradio_app import main


if __name__ == "__main__":
    main()
