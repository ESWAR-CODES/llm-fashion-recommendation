"""
test_catalog.py
Test Suite for Phase 1: Catalog Preparation, Ingestion, & Verification
Authoritative Specification: End_to_End PipeLine.pdf (Steps 1, 2, 3)
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_manager import resolve_path


class TestCatalog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = resolve_path("fashion_project/config.json")
        cls.catalog_path = resolve_path("fashion_project/artifacts/catalog.csv")
        cls.ids_path = resolve_path("fashion_project/artifacts/catalog_ids.json")
        cls.manifest_path = resolve_path("fashion_project/artifacts/catalog_manifest.json")
        cls.images_dir = resolve_path("fashion_project/data/raw/images")

        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

        cls.catalog_df = pd.read_csv(cls.catalog_path, dtype={"id": str})
        with open(cls.ids_path, "r", encoding="utf-8") as f:
            cls.catalog_ids = json.load(f)
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_01_config_exists_and_valid(self):
        self.assertTrue(self.config_path.exists(), "config.json does not exist")
        self.assertIn("project_title", self.config)
        self.assertIn("random_seed", self.config)
        self.assertEqual(self.config["random_seed"], 42)

    def test_02_catalog_size_exactly_10000(self):
        self.assertEqual(len(self.catalog_df), 10000, "Catalog must contain exactly 10,000 items")
        self.assertEqual(len(self.catalog_ids), 10000, "catalog_ids list must have 10,000 items")

    def test_03_expected_columns(self):
        expected_cols = [
            "id", "gender", "masterCategory", "subCategory", "articleType",
            "baseColour", "season", "year", "usage", "productDisplayName"
        ]
        for col in expected_cols:
            self.assertIn(col, self.catalog_df.columns, f"Missing required column: {col}")

    def test_04_product_ids_are_strings_and_unique(self):
        ids = self.catalog_df["id"].tolist()
        self.assertTrue(all(isinstance(x, str) for x in ids), "All product IDs must be strings")
        self.assertEqual(len(ids), len(set(ids)), "Product IDs must contain zero duplicates")

    def test_05_id_list_matches_csv_ordering(self):
        self.assertEqual(
            self.catalog_df["id"].tolist(),
            self.catalog_ids,
            "catalog_ids.json must exactly match catalog.csv row ordering"
        )

    def test_06_categories_restricted_to_apparel_and_footwear(self):
        categories = set(self.catalog_df["masterCategory"].unique())
        self.assertTrue(categories.issubset({"Apparel", "Footwear"}), f"Invalid categories: {categories}")

    def test_07_no_missing_metadata_in_critical_fields(self):
        critical = ["id", "masterCategory", "articleType", "baseColour", "usage", "productDisplayName"]
        for col in critical:
            null_count = self.catalog_df[col].isnull().sum()
            self.assertEqual(null_count, 0, f"Found {null_count} nulls in column {col}")

    def test_08_sample_images_valid_and_decodable(self):
        # Sample 50 items across the catalog and verify image files exist and open with Pillow
        sample_ids = self.catalog_ids[::200]
        for pid in sample_ids:
            img_p = self.images_dir / f"{pid}.jpg"
            self.assertTrue(img_p.exists(), f"Image not found: {img_p}")
            with Image.open(img_p) as img:
                img.verify()

    def test_09_manifest_file_hashes_match(self):
        def compute_sha256(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        for fname, meta in self.manifest.get("files", {}).items():
            fp = resolve_path(f"fashion_project/artifacts/{fname}")
            if fp.exists():
                actual_hash = compute_sha256(fp)
                self.assertEqual(actual_hash, meta["sha256"], f"Hash mismatch for {fname}")


if __name__ == "__main__":
    unittest.main()
