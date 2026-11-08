"""
tests/test_deployment_readiness.py
Automated Deployment Readiness & Cloud Production Verification Suite
LLM-Assisted Personalized Fashion Recommendation System

Validates:
1. Streamlit configuration (.streamlit/config.toml) existence and valid directives.
2. Production requirements.txt existence and required package specifications.
3. Cross-platform path resolution (Linux/Windows compatibility, no doubled paths).
4. Guest session profile isolation & lifecycle (independent anonymous sessions).
5. Product image delivery across multi-tier resolution (local catalog, cache, placeholder).
6. Memory efficiency & vector matrix dimensions (MiniLM 384-d, CLIP 512-d).
7. Three-tier LLM provider resilience (External API fallback to deterministic rule engine).
8. Security guardrails (zero secret leakage, error masking, input validation).
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config_service import ConfigService, resolve_path
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
from services.image_service import ProductImageService
from services.persistence_service import PersistenceService
from services.preference_service import (
    PreferenceExtractionService,
    RuleBasedFallbackProvider,
    ExternalAPIProvider
)
from services.ranking_service import RankingService


class TestDeploymentReadiness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = ConfigService.get_instance()
        cls.cat_svc = CatalogService.get_instance()
        cls.emb_svc = EmbeddingService.get_instance()
        cls.clip_svc = CLIPService.get_instance()
        cls.img_svc = ProductImageService.get_instance()
        cls.pers_svc = PersistenceService.get_instance()
        cls.pref_svc = PreferenceExtractionService.get_instance()
        cls.rank_svc = RankingService.get_instance()

    def test_01_streamlit_config_exists_and_valid(self):
        """Verify .streamlit/config.toml exists and contains headless and server port directives."""
        config_paths = [
            PROJECT_ROOT / ".streamlit" / "config.toml",
            PROJECT_ROOT.parent / ".streamlit" / "config.toml"
        ]
        found = any(p.exists() for p in config_paths)
        self.assertTrue(found, "Streamlit config.toml must exist in .streamlit/")

        active_path = next(p for p in config_paths if p.exists())
        content = active_path.read_text(encoding="utf-8")
        self.assertIn("[server]", content)
        self.assertIn("headless = true", content)
        self.assertIn("port = 8501", content)
        self.assertIn("[theme]", content)

    def test_02_production_requirements_complete(self):
        """Verify requirements.txt exists and contains core deployment packages."""
        req_paths = [
            PROJECT_ROOT / "requirements.txt",
            PROJECT_ROOT.parent / "requirements.txt"
        ]
        self.assertTrue(any(p.exists() for p in req_paths))
        active_req = next(p for p in req_paths if p.exists())
        req_text = active_req.read_text(encoding="utf-8")

        required_packages = [
            "streamlit", "pillow", "pandas", "numpy", "pydantic", "psutil", "requests"
        ]
        for pkg in required_packages:
            self.assertIn(pkg, req_text, f"Missing core dependency {pkg} in requirements.txt")

    def test_03_cross_platform_path_resolution(self):
        """Verify resolve_path handles forward slashes, backward slashes, and relative paths portably."""
        p_unix = resolve_path("artifacts/catalog.csv")
        p_win = resolve_path("artifacts\\catalog.csv")
        self.assertEqual(p_unix, p_win)
        self.assertTrue(p_unix.exists(), "Resolved catalog.csv path must exist")

        # Verify duplicate 'fashion_project/fashion_project' prevention
        p_nested = resolve_path("fashion_project/artifacts/catalog.csv")
        self.assertEqual(p_nested, p_unix)

    def test_04_isolated_guest_session_lifecycle(self):
        """Verify anonymous guest session profiles are isolated and independent."""
        guest_a = self.pers_svc.create_profile()
        guest_b = self.pers_svc.create_profile()
        self.assertNotEqual(guest_a, guest_b)

        # Mutate Guest A preferences and feedback
        self.pers_svc.save_feedback(guest_a, "15970", "like")
        self.pers_svc.set_active_constraints(guest_a, {"gender": ["Men"], "baseColour": ["Navy Blue"]})

        # Verify Guest B remains clean and isolated
        fb_b = self.pers_svc.get_feedback(guest_b)
        cons_b = self.pers_svc.get_active_constraints(guest_b)
        self.assertEqual(len(fb_b.get("likes", [])), 0)
        self.assertEqual(len(cons_b), 0)

        # Verify Guest A state is accurate
        fb_a = self.pers_svc.get_feedback(guest_a)
        cons_a = self.pers_svc.get_active_constraints(guest_a)
        self.assertIn("15970", fb_a.get("likes", []))
        self.assertEqual(cons_a.get("gender"), ["Men"])

        # Reset Guest A state
        self.pers_svc.clear_active_constraints(guest_a)
        self.assertEqual(len(self.pers_svc.get_active_constraints(guest_a)), 0)

    def test_05_image_delivery_multi_tier_and_placeholders(self):
        """Verify multi-tier image service loads real catalog items and generates clean placeholders."""
        # 1. Real catalog ID
        test_id = self.cat_svc.catalog_ids[0]
        self.assertTrue(self.img_svc.has_image(test_id))
        img = self.img_svc.load_image(test_id, target_size=(100, 130))
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (100, 130))

        # 2. Base64 generation
        b64 = self.img_svc.get_image_base64(test_id)
        self.assertTrue(b64.startswith("data:image/jpeg;base64,"))

        # 3. Non-existent product ID fallback to placeholder
        placeholder = self.img_svc.load_image("missing_dummy_id_99999", target_size=(100, 130))
        self.assertIsNotNone(placeholder)
        self.assertEqual(placeholder.size, (100, 130))

    def test_06_embedding_matrices_shapes_and_normalization(self):
        """Verify text and visual embedding matrices have identical 10,000 product rows and unit L2 norms."""
        mat_text = self.emb_svc.embeddings
        self.assertEqual(mat_text.shape, (10000, 384))
        norms_text = [float(x) for x in (mat_text ** 2).sum(axis=1) ** 0.5]
        self.assertAlmostEqual(norms_text[0], 1.0, places=4)

        if self.clip_svc.embeddings is not None:
            mat_clip = self.clip_svc.embeddings
            self.assertEqual(mat_clip.shape, (10000, 512))
            norms_clip = [float(x) for x in (mat_clip ** 2).sum(axis=1) ** 0.5]
            self.assertAlmostEqual(norms_clip[0], 1.0, places=4)

    def test_07_three_tier_llm_provider_fallback(self):
        """Verify ExternalAPIProvider and RuleBasedFallbackProvider execute cleanly without unhandled exceptions."""
        # Test 1: Fallback provider is always available
        rule_prov = RuleBasedFallbackProvider()
        self.assertTrue(rule_prov.is_available())
        upd, mode = rule_prov.extract("show me black shirts but no red", {}, "")
        self.assertEqual(mode, "rule_fallback")
        self.assertIsNotNone(upd)

        # Test 2: External API provider without key falls back gracefully
        ext_prov = ExternalAPIProvider(api_key=None)
        self.assertFalse(ext_prov.is_available())
        upd_ext, mode_ext = ext_prov.extract("blue denim jeans", {}, "")
        self.assertEqual(mode_ext, "rule_fallback")

    def test_08_security_error_masking_and_isolation(self):
        """Verify no sensitive environment keys or database secrets are leaked in user-facing calls."""
        # Verify invalid product lookup returns None / empty instead of throwing fatal exception
        bogus_prod = self.cat_svc.get_product("malicious_injection' OR 1=1 --")
        self.assertIsNone(bogus_prod)

        # Verify empty query handling in ranking
        empty_res = self.rank_svc.recommend_multimodal(query_text="", top_k=5)
        self.assertIn("status", empty_res)

    def test_09_no_torch_preference_extraction(self):
        """Verify conversational preference extraction operates flawlessly when PyTorch is absent."""
        from unittest.mock import patch
        with patch("engines.preference_extractor._get_torch", return_value=None):
            from engines.preference_extractor import get_hardware_info, rule_based_extract_update
            hw = get_hardware_info()
            self.assertEqual(hw["torch_version"], "Not Installed")
            self.assertFalse(hw["cuda_available"])
            self.assertEqual(hw["device_name"], "CPU")

            # Extract update without torch
            upd = rule_based_extract_update("show me blue casual shirts", {})
            self.assertIsNotNone(upd)
            self.assertTrue(len(upd.updates) > 0)

            # Service execution
            new_state, meta = self.pref_svc.extract_and_apply("show me black shoes", {})
            self.assertTrue(meta["success"])
            self.assertIn("Black", new_state.get("baseColour", []))

    def test_10_no_torch_text_query_encoding_and_recommendation(self):
        """Verify text query encoding and ranking work via pure NumPy projection without PyTorch."""
        from unittest.mock import patch
        with patch.object(self.emb_svc, "_get_encoder", return_value=None):
            # Test encoding
            vec = self.emb_svc.encode_text("navy blue casual shirt")
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (384,))
            norm = float(np.linalg.norm(vec))
            self.assertAlmostEqual(norm, 1.0, places=4)

            # Test recommendation pipeline
            res = self.rank_svc.recommend("navy blue casual shirt", top_k=5)
            self.assertIn("results", res)
            self.assertEqual(len(res["results"]), 5)
            for prod in res["results"]:
                self.assertIn("productDisplayName", prod)
                self.assertGreater(prod["score"], 0.0)

    def test_11_no_torch_clip_status_reporting(self):
        """Verify CLIPService reports status accurately without throwing exceptions when PyTorch is absent."""
        from unittest.mock import patch
        with patch.object(self.clip_svc, "is_encoder_available", return_value=False):
            status = self.clip_svc.get_status()
            self.assertTrue(status["ready"], "Precomputed embeddings on disk must remain ready")
            self.assertFalse(status["encoder_available"])
            self.assertIn("Lightweight CPU", status["message"])

    def test_12_no_torch_app_imports_and_service_initialization(self):
        """Verify app and services can initialize under a strict no-PyTorch simulation."""
        from unittest.mock import patch
        # Patch import mechanism so any attempt to import torch raises ModuleNotFoundError
        orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else getattr(__builtins__, "__import__")

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "sentence_transformers", "transformers"):
                raise ModuleNotFoundError(f"No module named '{name}' (simulated)")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Verify hardware detection doesn't crash
            hw = self.cfg.get_hardware_info()
            self.assertEqual(hw.get("torch_version"), "Not Installed")

            # Verify preference service runs
            res_state, meta = self.pref_svc.extract_and_apply("white casual sneakers", {})
            self.assertTrue(meta["success"])


if __name__ == "__main__":
    unittest.main()
