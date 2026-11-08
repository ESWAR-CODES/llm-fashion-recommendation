"""
scripts/export_package.py
Phase 5: Step 11 Project Archive Exporter
Packages the fashion_project directory into a standalone zip file excluding heavy raw archives.
"""

import os
import sys
import zipfile
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FASHION_PROJECT = PROJECT_ROOT

EXPORT_ZIP = FASHION_PROJECT / "fashion_project_export.zip"

EXCLUDE_PATTERNS = [
    "data/raw",
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "venv",
    ".venv",
    ".cache",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".env",
    ".pem",
    ".key",
    "fashion_project_export.zip"
]


def should_include(rel_path_str: str) -> bool:
    normalized = rel_path_str.replace("\\", "/")
    for exc in EXCLUDE_PATTERNS:
        if exc in normalized:
            return False
    return True


def package_project():
    print(f"[EXPORT] Packaging fashion_project into {EXPORT_ZIP}...")
    file_count = 0

    with zipfile.ZipFile(EXPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(FASHION_PROJECT):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(FASHION_PROJECT)
                rel_path_str = str(rel_path)

                if should_include(rel_path_str):
                    zipf.write(full_path, arcname=rel_path_str)
                    file_count += 1

    size_mb = EXPORT_ZIP.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Packaged {file_count} files into {EXPORT_ZIP} ({size_mb:.2f} MB)")
    return EXPORT_ZIP


if __name__ == "__main__":
    package_project()
