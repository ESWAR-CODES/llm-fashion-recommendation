"""
app.py (Root Launcher)
Delegates execution to fashion_project/app.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FASHION_DIR = ROOT_DIR / "fashion_project"
if str(FASHION_DIR) not in sys.path:
    sys.path.insert(0, str(FASHION_DIR))

from app import main

if __name__ == "__main__":
    main()
