"""
validate_phase4.py
Top-level runner for tests.validate_phase4
"""

import os
import sys
from pathlib import Path

# Add fashion_project to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from tests.validate_phase4 import run_all_checks

if __name__ == "__main__":
    proj_dir = "fashion_project" if os.path.exists("fashion_project") else "."
    passed, total, _ = run_all_checks(proj_dir)
    sys.exit(0 if passed == total else 1)
