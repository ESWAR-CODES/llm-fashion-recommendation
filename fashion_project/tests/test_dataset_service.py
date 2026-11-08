"""
test_dataset_service.py
Unit tests for DatasetService
LLM-Assisted Personalized Fashion Recommendation System
"""

import sys
import unittest
from pathlib import Path

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.dataset_service import DatasetService, EXPECTED_METADATA_FIELDS


class TestDatasetService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = DatasetService()

    def test_01_raw_dataset_exists(self):
        exists, msg = self.service.validate_raw_dataset_exists()
        self.assertTrue(exists, f"Raw dataset existence failed: {msg}")

    def test_02_load_and_audit_raw_styles(self):
        df_raw, malformed, stats = self.service.load_and_audit_raw_styles()
        self.assertGreaterEqual(stats["valid_rows_count"], 44000)
        self.assertEqual(stats["malformed_rows_count"], 22)
        self.assertEqual(len(EXPECTED_METADATA_FIELDS), 10)
        for col in EXPECTED_METADATA_FIELDS:
            self.assertIn(col, df_raw.columns)
        self.assertTrue(all(isinstance(x, str) for x in df_raw["id"].iloc[:50]))

    def test_03_frozen_catalog_integrity(self):
        catalog_df, catalog_ids = self.service.load_frozen_catalog()
        self.assertEqual(len(catalog_df), 10000)
        self.assertEqual(len(catalog_ids), 10000)
        self.assertEqual(catalog_df["id"].tolist(), catalog_ids)

    def test_04_validate_at_least_20_actual_images(self):
        catalog_df, catalog_ids = self.service.load_frozen_catalog()
        test_ids = catalog_ids[:25]
        valid_ids, missing_ids, corrupt = self.service.validate_image_files(test_ids)
        self.assertEqual(len(valid_ids), len(test_ids), f"Missing images: {missing_ids}")
        self.assertEqual(len(corrupt), 0, f"Corrupt images: {corrupt}")

    def test_05_missing_images_detection(self):
        fake_ids = ["missing_id_999901", "missing_id_999902"]
        valid_ids, missing_ids, corrupt = self.service.validate_image_files(fake_ids)
        self.assertEqual(len(valid_ids), 0)
        self.assertEqual(len(missing_ids), 2)


if __name__ == "__main__":
    unittest.main()
