"""
test_image_service.py
Unit tests for ProductImageService
LLM-Assisted Personalized Fashion Recommendation System
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.image_service import ProductImageService
from services.catalog_service import CatalogService


class TestImageService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_svc = ProductImageService.get_instance()
        cls.cat_svc = CatalogService.get_instance()

    def test_01_load_at_least_20_actual_product_images(self):
        """Verify at least 20 actual product images load and resize correctly."""
        sample_ids = self.cat_svc.catalog_ids[:25]
        loaded_count = 0
        for pid in sample_ids:
            self.assertTrue(self.img_svc.has_image(pid), f"Missing image for catalog ID: {pid}")
            img = self.img_svc.load_image(pid, target_size=(120, 160))
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (120, 160))
            loaded_count += 1
        self.assertGreaterEqual(loaded_count, 20)

    def test_02_base64_data_uri_generation(self):
        pid = self.cat_svc.catalog_ids[0]
        data_uri = self.img_svc.get_image_base64(pid, target_size=(100, 130))
        self.assertIsInstance(data_uri, str)
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(data_uri), 100)

    def test_03_in_memory_lru_cache(self):
        pid = self.cat_svc.catalog_ids[1]
        img1 = self.img_svc.load_image(pid, target_size=(100, 130))
        img2 = self.img_svc.load_image(pid, target_size=(100, 130))
        self.assertIs(img1, img2, "Cached image instance must be returned without re-decoding disk")

    def test_04_graceful_deterministic_placeholder_for_missing_ids(self):
        fake_id = "non_existent_999999"
        self.assertFalse(self.img_svc.has_image(fake_id))
        placeholder = self.img_svc.load_image(fake_id, target_size=(150, 200))
        self.assertIsInstance(placeholder, Image.Image)
        self.assertEqual(placeholder.size, (150, 200))

    def test_05_placeholder_base64(self):
        fake_id = "non_existent_888888"
        b64 = self.img_svc.get_image_base64(fake_id, target_size=(150, 200))
        self.assertTrue(b64.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
