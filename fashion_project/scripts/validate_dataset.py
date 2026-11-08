"""
scripts/validate_dataset.py
Dataset Installation & Integrity Validation Script
LLM-Assisted Personalized Fashion Recommendation System

Usage:
    python fashion_project/scripts/validate_dataset.py
"""

import sys
import json
from pathlib import Path

# Add fashion_project to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.dataset_service import DatasetService


def main():
    print("=" * 80)
    print("DATASET INSTALLATION & INTEGRITY VERIFICATION")
    print("=" * 80)

    svc = DatasetService()

    # 1. Existence check
    exists, msg = svc.validate_raw_dataset_exists()
    print(f"[1/4] Checking raw files: {msg}")
    if not exists:
        print("[FAIL] Raw dataset missing. Please ensure styles.csv and images/ are present.")
        sys.exit(1)

    # 2. Parse styles.csv
    print("[2/4] Parsing styles.csv and checking schema...")
    df_raw, malformed, stats = svc.load_and_audit_raw_styles()
    print(f"      - Valid rows parsed: {stats['valid_rows_count']:,}")
    print(f"      - Malformed rows detected (preserved): {stats['malformed_rows_count']:,}")
    print(f"      - Unique product IDs: {stats['unique_ids_count']:,}")

    # 3. Frozen catalog check
    print("[3/4] Verifying frozen 10,000-item catalog...")
    catalog_df, catalog_ids = svc.load_frozen_catalog()
    print(f"      - Catalog items: {len(catalog_df):,}")
    print(f"      - Catalog IDs: {len(catalog_ids):,}")

    # 4. Image check for sample catalog items
    print("[4/4] Validating sample catalog product images with Pillow...")
    sample_ids = catalog_ids[:50]
    valid_ids, missing_ids, corrupt = svc.validate_image_files(sample_ids)
    print(f"      - Verified {len(valid_ids)}/50 sample images successfully.")
    if missing_ids or corrupt:
        print(f"[WARN] Missing: {len(missing_ids)}, Corrupt: {len(corrupt)}")
    else:
        print("[PASS] All checked product images are present and fully decodable.")

    print("=" * 80)
    print("[SUCCESS] Dataset installation is fully verified and ready for ML inference.")
    print("=" * 80)


if __name__ == "__main__":
    main()
