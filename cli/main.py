"""
Find3D CLI entry point.
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from cli.gradio_app import main

def run():
    """Entry point for 'uv run find3d' command."""
    main()

if __name__ == "__main__":
    run()
