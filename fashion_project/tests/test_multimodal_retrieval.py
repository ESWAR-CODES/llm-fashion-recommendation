"""
tests/test_multimodal_retrieval.py
Phase 2 Test Suite: Multimodal Retrieval & Hybrid Fashion Recommendation
LLM-Assisted Personalized Fashion Recommendation System

Covers:
- Upload validation (valid formats, oversize, corrupt bytes, empty files).
- Weight normalization and dynamic reweighting.
- Hybrid score bounds and calibration ([0.0, 1.0]).
- Graceful missing modality / text-only fallback.
- Multi-field hard constraint enforcement.
- Dismissal exclusion and deduplication.
- Personalization influence without overriding constraints.
- Grounded multimodal explanation validation.
- Discovery and Assistant UI component integration.
"""

import io
import sys
import json
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config_service import ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
from services.retrieval_service import RetrievalService
from services.ranking_service import RankingService, filter_candidates
from services.persistence_service import PersistenceService
from services.explanation_service import ExplanationService


class TestMultimodalRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = ConfigService.get_instance()
        cls.cat_svc = CatalogService.get_instance()
        cls.emb_svc = EmbeddingService.get_instance()
        cls.clip_svc = CLIPService.get_instance()
        cls.ret_svc = RetrievalService.get_instance()
        cls.rank_svc = RankingService.get_instance()
        cls.pers_svc = PersistenceService.get_instance()
        cls.expl_svc = ExplanationService.get_instance()

    def test_01_clip_upload_validation_valid_formats(self):
        """Verify JPEG, PNG, and RGB conversion for valid uploaded images."""
        # 1. Valid JPEG
        img_rgb = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG")
        buf.seek(0)
        ok, dec_img, err = CLIPService.validate_uploaded_image(buf)
        self.assertTrue(ok)
        self.assertIsNotNone(dec_img)
        self.assertEqual(dec_img.mode, "RGB")
        self.assertIsNone(err)

        # 2. Valid PNG with RGBA (transparent)
        img_rgba = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
        buf_png = io.BytesIO()
        img_rgba.save(buf_png, format="PNG")
        buf_png.seek(0)
        ok2, dec_img2, err2 = CLIPService.validate_uploaded_image(buf_png)
        self.assertTrue(ok2)
        self.assertEqual(dec_img2.mode, "RGB")

    def test_02_clip_upload_validation_invalid_files(self):
        """Verify rejection of empty files, corrupt data, and oversized inputs."""
        # 1. Empty bytes
        ok, img, err = CLIPService.validate_uploaded_image(b"")
        self.assertFalse(ok)
        self.assertIn("empty", err.lower())

        # 2. Corrupt bytes
        ok, img, err = CLIPService.validate_uploaded_image(b"NOT_A_REAL_IMAGE_DATA_123456")
        self.assertFalse(ok)
        self.assertIn("failed to decode", err.lower())

        # 3. Oversized check (simulated with 10 bytes limit)
        ok, img, err = CLIPService.validate_uploaded_image(b"0" * 20, max_bytes=10)
        self.assertFalse(ok)
        self.assertIn("exceeds maximum allowed size", err.lower())

    def test_03_weight_normalization_and_dynamic_reweighting(self):
        """Verify that ranking weights always sum to exactly 1.0 under all modality combinations."""
        # A. All 3 modalities active
        w1 = self.rank_svc._resolve_weights(has_text=True, has_image=True, has_profile=True)
        self.assertAlmostEqual(sum(w1.values()), 1.0, places=4)
        self.assertGreater(w1["text_weight"], 0)
        self.assertGreater(w1["visual_weight"], 0)
        self.assertGreater(w1["preference_weight"], 0)

        # B. Text + Image, no profile (renormalized)
        w2 = self.rank_svc._resolve_weights(has_text=True, has_image=True, has_profile=False)
        self.assertAlmostEqual(sum(w2.values()), 1.0, places=4)
        self.assertEqual(w2["preference_weight"], 0.0)

        # C. Text + Profile, no image (baseline compatibility: 0.8 / 0.2)
        w3 = self.rank_svc._resolve_weights(has_text=True, has_image=False, has_profile=True)
        self.assertAlmostEqual(sum(w3.values()), 1.0, places=4)
        self.assertAlmostEqual(w3["text_weight"], 0.8, places=4)
        self.assertAlmostEqual(w3["preference_weight"], 0.2, places=4)

        # D. Image only (no text, no profile)
        w4 = self.rank_svc._resolve_weights(has_text=False, has_image=True, has_profile=False)
        self.assertAlmostEqual(w4["visual_weight"], 1.0, places=4)

    def test_04_hybrid_score_calculation_bounds(self):
        """Verify that hybrid scores are strictly bounded in [0.0, 1.0]."""
        res = self.rank_svc.recommend_multimodal(
            query_text="casual blue cotton t-shirt",
            constraints={"masterCategory": ["Apparel"]},
            top_k=5
        )
        self.assertEqual(res["status"], "success")
        for p in res["results"]:
            score = p["score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertIn("text_similarity", p)
            self.assertIn("visual_similarity", p)
            self.assertIn("preference_similarity", p)

    def test_05_multimodal_retrieval_text_only_fallback(self):
        """Verify graceful fallback to text search when visual embeddings or image is missing."""
        # Query with image=None
        res = self.rank_svc.recommend_multimodal(
            query_text="black formal shoes",
            query_image=None,
            top_k=5
        )
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["results"]), 0)
        self.assertEqual(res["weights_applied"]["text_weight"], 1.0)
        self.assertEqual(res["weights_applied"]["visual_weight"], 0.0)

    def test_06_multimodal_ranking_hard_constraints_and_filters(self):
        """Verify strict constraint filtering in multimodal recommendation."""
        # Test multi-field AND/OR filtering
        cons = {
            "masterCategory": ["Footwear"],
            "baseColour": ["Black", "White"],
            "usage": ["Sports"]
        }
        res = self.rank_svc.recommend_multimodal(
            query_text="running sneakers",
            constraints=cons,
            top_k=5
        )
        self.assertEqual(res["status"], "success")
        for p in res["results"]:
            self.assertEqual(p["masterCategory"], "Footwear")
            self.assertIn(p["baseColour"], ["Black", "White"])
            self.assertEqual(p["usage"], "Sports")

    def test_07_multimodal_ranking_dismissal_and_deduplication(self):
        """Verify that dismissed items are strictly excluded and no duplicate IDs exist."""
        profile_id = self.pers_svc.create_profile()
        top_candidates = self.rank_svc.recommend(query="casual shirts", top_k=5)["results"]
        target_pid = str(top_candidates[0]["id"])

        # Dismiss target product
        self.pers_svc.save_feedback(profile_id, target_pid, "dismiss")

        # Rerun recommendation for this profile
        res = self.rank_svc.recommend_multimodal(
            query_text="casual shirts",
            profile_id=profile_id,
            top_k=5
        )
        result_pids = [str(r["id"]) for r in res["results"]]
        self.assertNotIn(target_pid, result_pids)
        self.assertEqual(len(result_pids), len(set(result_pids)))  # Deduplication check

    def test_08_multimodal_personalization_influence(self):
        """Verify that user likes synthesize preference vector that influences candidate ranking."""
        profile_id = self.pers_svc.create_profile()
        # Like a navy blue product
        sample_prod = self.cat_svc.get_product("10031")  # Navy Blue Tshirts
        self.pers_svc.save_feedback(profile_id, "10031", "like")

        user_vec = self.pers_svc.build_user_vector(profile_id)
        self.assertIsNotNone(user_vec)
        self.assertAlmostEqual(float(np.linalg.norm(user_vec)), 1.0, places=4)

        # Recommendation should incorporate preference weight
        res = self.rank_svc.recommend_multimodal(
            query_text="tshirts",
            profile_id=profile_id,
            top_k=5
        )
        self.assertTrue(res["has_personalized_vector"])
        self.assertGreater(res["weights_applied"]["preference_weight"], 0.0)

    def test_09_multimodal_explanation_grounding(self):
        """Verify that recommendation explanations are strictly grounded in catalog metadata."""
        sample_prod = self.cat_svc.get_product(self.cat_svc.catalog_ids[0])
        explanation = self.expl_svc.explain(
            sample_prod,
            active_constraints={"masterCategory": [sample_prod["masterCategory"]]}
        )
        self.assertIsInstance(explanation, str)
        self.assertIn(sample_prod["baseColour"].lower(), explanation.lower())
        self.assertIn(sample_prod["articleType"].lower(), explanation.lower())

    def test_10_undo_feedback_lifecycle(self):
        """Verify removing feedback (unlike / undo dismissal)."""
        pid = self.pers_svc.create_profile()
        item = "10015"
        self.pers_svc.save_feedback(pid, item, "like")
        fb = self.pers_svc.get_feedback(pid)
        self.assertIn(item, fb["likes"])

        # Undo
        ok = self.pers_svc.remove_feedback(pid, item)
        self.assertTrue(ok)
        fb_after = self.pers_svc.get_feedback(pid)
        self.assertNotIn(item, fb_after["likes"])


if __name__ == "__main__":
    unittest.main()
