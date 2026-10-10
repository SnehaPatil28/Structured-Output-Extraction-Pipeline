"""Pytest configuration ensuring source directory is in sys.path."""

import sys
from pathlib import Path

# Add project root (parent of tests) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
