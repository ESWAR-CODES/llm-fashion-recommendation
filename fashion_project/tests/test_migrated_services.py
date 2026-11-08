"""
test_migrated_services.py
Unit tests for migrated ML services: Catalog, Embedding, Retrieval, Ranking, Preference, Explanation
LLM-Assisted Personalized Fashion Recommendation System
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.ranking_service import RankingService, filter_candidates
from services.preference_service import PreferenceExtractionService, RuleBasedFallbackProvider
from services.explanation_service import ExplanationService
from services.persistence_service import PersistenceService


class TestMigratedServices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cat_svc = CatalogService.get_instance()
        cls.emb_svc = EmbeddingService.get_instance()
        cls.ret_svc = RetrievalService.get_instance()
        cls.rank_svc = RankingService.get_instance()
        cls.pref_svc = PreferenceExtractionService.get_instance()
        cls.expl_svc = ExplanationService.get_instance()
        cls.pers_svc = PersistenceService.get_instance()

    def test_01_catalog_service_lookup_and_domains(self):
        first_id = self.cat_svc.catalog_ids[0]
        prod = self.cat_svc.get_product(first_id)
        self.assertIsNotNone(prod)
        self.assertEqual(str(prod["id"]), first_id)
        self.assertIn(prod["masterCategory"], ["Apparel", "Footwear"])
        
        cats = self.cat_svc.get_categories()
        self.assertIn("Apparel", cats)
        self.assertIn("Footwear", cats)

    def test_02_embedding_service_zero_recomputation(self):
        matrix = self.emb_svc.embeddings
        self.assertEqual(matrix.shape, (10000, 384))
        self.assertEqual(matrix.dtype, np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    def test_03_retrieval_service_semantic_search(self):
        results = self.ret_svc.retrieve("casual navy blue cotton shirt", top_k=5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("cosine_similarity", r)
            self.assertIn("id", r)
            self.assertTrue(self.cat_svc.contains(r["id"]))

    def test_04_ranking_service_hard_constraints_and_no_duplicates(self):
        constraints = {"articleType": ["Shirts"], "exclude_colors": ["Red"]}
        res = self.rank_svc.recommend(query="casual shirts", constraints=constraints, top_k=5)
        self.assertEqual(res["status"], "success")
        items = res["results"]
        self.assertLessEqual(len(items), 5)
        
        seen_ids = set()
        for item in items:
            self.assertNotIn(item["id"], seen_ids, "Duplicate product returned!")
            seen_ids.add(item["id"])
            self.assertEqual(item["articleType"].lower(), "shirts")
            self.assertNotEqual(item["baseColour"].lower(), "red")

    def test_05_ranking_service_no_match_handling(self):
        impossible = {"articleType": ["Kurtas"], "usage": ["Formal"], "baseColour": ["Neon Pink"]}
        res = self.rank_svc.recommend(query="formal kurtas", constraints=impossible, top_k=5)
        self.assertEqual(res["status"], "no_match")
        self.assertEqual(len(res["results"]), 0)
        self.assertIn("revised_constraints_suggestion", res)

    def test_06_preference_extraction_service_operations(self):
        current_state = {"articleType": ["Kurtas"]}
        new_state, meta = self.pref_svc.extract_and_apply("Show me shirts instead", current_state)
        self.assertIn("Shirts", new_state.get("articleType", []))
        self.assertNotIn("Kurtas", new_state.get("articleType", []))
        self.assertEqual(meta["provider_name"], "rule_fallback")

    def test_07_preference_service_unsupported_detection(self):
        new_state, meta = self.pref_svc.extract_and_apply("size 10 shoes under $50 in stock", {})
        self.assertGreater(len(meta["unsupported"]), 0)

    def test_08_explanation_service_grounding(self):
        prod = self.cat_svc.get_product(self.cat_svc.catalog_ids[0])
        explanation = self.expl_svc.explain(prod, active_constraints={"articleType": [prod["articleType"]]})
        self.assertIsInstance(explanation, str)
        self.assertIn(prod["articleType"].lower(), explanation.lower())
        self.assertIn(prod["baseColour"].lower(), explanation.lower())


if __name__ == "__main__":
    unittest.main()
