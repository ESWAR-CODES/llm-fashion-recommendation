"""
services/dataset_service.py
Robust Dataset Loader, Validator, and Audit Generator
LLM-Assisted Personalized Fashion Recommendation System

Authoritative Specification: End_to_End PipeLine.pdf (Steps 1 & 2)
"""

import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
from PIL import Image

from services.config_service import resolve_path, ConfigService

EXPECTED_METADATA_FIELDS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName"
]

NA_STRINGS = {"", "NA", "NaN", "null", "none", "nan", "None"}


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    p = Path(filepath)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


class DatasetService:
    """
    Centralized dataset service responsible for dataset loading,
    schema validation, Pillow image verification, and audit generation.
    """

    def __init__(
        self,
        styles_csv_path: Optional[str | Path] = None,
        images_dir_path: Optional[str | Path] = None,
        catalog_csv_path: Optional[str | Path] = None,
        catalog_ids_path: Optional[str | Path] = None,
    ):
        cfg = ConfigService.get_instance()
        self.styles_csv = resolve_path(styles_csv_path) if styles_csv_path else cfg.get_path("styles_csv")
        self.images_dir = resolve_path(images_dir_path) if images_dir_path else cfg.get_path("images_dir")
        self.catalog_csv = resolve_path(catalog_csv_path) if catalog_csv_path else cfg.get_path("catalog_csv")
        self.catalog_ids_path = resolve_path(catalog_ids_path) if catalog_ids_path else cfg.get_path("catalog_ids_json")

    def validate_raw_dataset_exists(self) -> Tuple[bool, str]:
        """Verify presence of styles.csv and images directory."""
        if not self.styles_csv.exists():
            return False, f"styles.csv not found at: {self.styles_csv}"
        if not self.images_dir.exists():
            return False, f"images directory not found at: {self.images_dir}"
        return True, "Raw dataset files located successfully."

    def load_and_audit_raw_styles(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Carefully parse styles.csv without silently dropping malformed rows.
        Returns:
            - valid_df: DataFrame of syntactically valid rows with IDs as strings.
            - malformed_records: List of malformed row details (line number, tokens).
            - stats: Summary counts of raw rows and fields.
        """
        if not self.styles_csv.exists():
            raise FileNotFoundError(f"styles.csv does not exist at {self.styles_csv}")

        malformed_records: List[Dict[str, Any]] = []
        valid_rows: List[List[Optional[str]]] = []
        total_lines = 0

        with open(self.styles_csv, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("styles.csv is completely empty.")
            if header != EXPECTED_METADATA_FIELDS:
                raise ValueError(
                    f"Column header mismatch! Expected: {EXPECTED_METADATA_FIELDS}, Found: {header}"
                )

            for line_idx, row in enumerate(reader, start=2):
                total_lines += 1
                if len(row) != 10:
                    malformed_records.append({
                        "line_number": line_idx,
                        "token_count": len(row),
                        "tokens": row,
                        "id": row[0].strip() if len(row) > 0 else f"unknown_line_{line_idx}",
                        "reason": "field_count_mismatch"
                    })
                else:
                    cleaned_row = [
                        val.strip() if val.strip() not in NA_STRINGS else None
                        for val in row
                    ]
                    valid_rows.append(cleaned_row)

        df = pd.DataFrame(valid_rows, columns=EXPECTED_METADATA_FIELDS)
        df["id"] = df["id"].astype(str).str.strip()

        # Compute missing attributes per column
        missing_by_col = {col: int(df[col].isnull().sum()) for col in EXPECTED_METADATA_FIELDS}

        stats = {
            "total_lines_read": total_lines,
            "valid_rows_count": len(df),
            "malformed_rows_count": len(malformed_records),
            "unique_ids_count": int(df["id"].nunique()),
            "duplicate_ids_count": int(len(df) - df["id"].nunique()),
            "missing_attributes_by_column": missing_by_col
        }

        return df, malformed_records, stats

    def validate_image_files(
        self,
        product_ids: List[str],
        max_workers: int = 16
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        """
        Validate product images on disk using Pillow.
        Returns:
            - valid_ids: Product IDs whose images exist and decode correctly.
            - missing_ids: Product IDs whose images do not exist on disk.
            - corrupt_details: Details of unreadable / corrupt image files.
        """
        if not self.images_dir.exists():
            return [], product_ids, [{"error": "Images directory does not exist"}]

        # Fast set lookup of disk files
        existing_filenames = set(os.listdir(self.images_dir))

        missing_ids = []
        present_ids = []
        for pid in product_ids:
            fn = f"{pid}.jpg"
            if fn in existing_filenames:
                present_ids.append(pid)
            else:
                missing_ids.append(pid)

        corrupt_details: List[Dict[str, str]] = []
        valid_ids: List[str] = []

        def check_img(pid: str) -> Tuple[str, bool, Optional[str]]:
            img_path = self.images_dir / f"{pid}.jpg"
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.load()
                return pid, True, None
            except Exception as e:
                return pid, False, str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for pid, is_valid, err_msg in executor.map(check_img, present_ids):
                if is_valid:
                    valid_ids.append(pid)
                else:
                    corrupt_details.append({"id": pid, "error": str(err_msg)})

        return valid_ids, missing_ids, corrupt_details

    def generate_comprehensive_audit(self) -> Dict[str, Any]:
        """
        Perform a full dataset audit and return a structured dictionary.
        """
        df_raw, malformed_records, stats = self.load_and_audit_raw_styles()
        all_ids = df_raw["id"].tolist()

        valid_ids, missing_ids, corrupt_details = self.validate_image_files(all_ids)

        # Apparel & Footwear filtering breakdown
        apparel_mask = df_raw["masterCategory"] == "Apparel"
        footwear_mask = df_raw["masterCategory"] == "Footwear"
        apparel_footwear_mask = apparel_mask | footwear_mask

        # Cleanable records (in Apparel/Footwear, valid image, no missing critical metadata)
        valid_id_set = set(valid_ids)
        has_valid_image = df_raw["id"].isin(valid_id_set)
        no_missing_critical = (
            df_raw["articleType"].notnull() &
            df_raw["baseColour"].notnull() &
            df_raw["usage"].notnull() &
            df_raw["productDisplayName"].notnull()
        )

        eligible_mask = apparel_footwear_mask & has_valid_image & no_missing_critical
        eligible_count = int(eligible_mask.sum())

        audit = {
            "dataset_name": "Fashion Product Images Small",
            "source_url": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
            "styles_csv_path": str(self.styles_csv),
            "images_dir_path": str(self.images_dir),
            "original_csv_row_count": stats["total_lines_read"],
            "valid_field_count_rows": stats["valid_rows_count"],
            "malformed_rows_count": stats["malformed_rows_count"],
            "unique_product_ids_count": stats["unique_ids_count"],
            "duplicate_ids_count": stats["duplicate_ids_count"],
            "missing_metadata_by_column": stats["missing_attributes_by_column"],
            "missing_image_count": len(missing_ids),
            "missing_image_ids_sample": missing_ids[:20],
            "corrupt_image_count": len(corrupt_details),
            "valid_image_count": len(valid_ids),
            "apparel_count": int(apparel_mask.sum()),
            "footwear_count": int(footwear_mask.sum()),
            "apparel_and_footwear_total": int(apparel_footwear_mask.sum()),
            "eligible_candidates_after_cleaning": eligible_count,
            "exclusion_reasons": {
                "malformed_csv_token_count": stats["malformed_rows_count"],
                "non_clothing_footwear_category": int((~apparel_footwear_mask).sum()),
                "missing_or_unreadable_image": len(missing_ids) + len(corrupt_details),
                "missing_critical_metadata": int((apparel_footwear_mask & ~no_missing_critical).sum())
            }
        }
        return audit

    def load_frozen_catalog(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load the authoritative frozen 10,000-product catalog.
        Validates ID types, ordering, and consistency.
        """
        if not self.catalog_csv.exists():
            raise FileNotFoundError(f"Frozen catalog not found at {self.catalog_csv}")
        if not self.catalog_ids_path.exists():
            raise FileNotFoundError(f"catalog_ids.json not found at {self.catalog_ids_path}")

        catalog_df = pd.read_csv(self.catalog_csv, dtype={"id": str})
        catalog_df["id"] = catalog_df["id"].astype(str).str.strip()

        with open(self.catalog_ids_path, "r", encoding="utf-8") as f:
            catalog_ids = json.load(f)

        if len(catalog_df) != 10000:
            raise ValueError(f"Expected 10,000 catalog rows, got {len(catalog_df)}")
        if catalog_df["id"].tolist() != catalog_ids:
            raise ValueError("catalog_ids.json does not strictly match catalog.csv row ordering!")

        return catalog_df, catalog_ids


if __name__ == "__main__":
    print("[DATASET SERVICE] Running standalone dataset audit...")
    svc = DatasetService()
    exists, msg = svc.validate_raw_dataset_exists()
    print(f"Dataset status: {msg}")
    if exists:
        df, malformed, stats = svc.load_and_audit_raw_styles()
        print(f"Parsed {stats['valid_rows_count']} valid rows, {stats['malformed_rows_count']} malformed rows.")
        cat, cids = svc.load_frozen_catalog()
        print(f"Loaded frozen catalog successfully: {len(cat)} items.")
