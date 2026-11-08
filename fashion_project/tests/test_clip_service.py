"""
test_clip_service.py
Unit tests for CLIPService and Visual Retrieval Foundation
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

from services.clip_service import CLIPService


class TestCLIPService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_svc = CLIPService.get_instance()

    def test_01_clip_status_reporting(self):
        status = self.clip_svc.get_status()
        self.assertIn("ready", status)
        self.assertIn("model_name", status)
        self.assertEqual(status["dimension"], 512)
        self.assertIn("message", status)

    def test_02_text_only_fallback_behavior(self):
        """When precomputed CLIP embeddings are absent, visual search reports not ready cleanly."""
        if not self.clip_svc.is_visual_retrieval_ready():
            self.assertFalse(self.clip_svc.is_visual_retrieval_ready())
            # Must not crash; reports offline indexing required
            status = self.clip_svc.get_status()
            self.assertIn("offline generation required", status["message"].lower())


if __name__ == "__main__":
    unittest.main()
