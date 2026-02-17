#!/usr/bin/env python
"""
Entry point module for Find 3D CLI.
Allows running as: python -m cli.gradio_app or uv run find3d
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from cli.gradio_app import main

if __name__ == "__main__":
    main()
