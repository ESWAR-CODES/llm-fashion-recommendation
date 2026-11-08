"""
test_ranking.py
Test Suite for Phase 3: Candidate Filtering, User Vectors & Personalized Ranking
Authoritative Specification: End_to_End PipeLine.pdf (Step 7)
"""

import os
import sys
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_manager import (
    resolve_path,
    init_db,
    create_profile,
    update_profile,
    save_feedback,
    build_user_vector
)
from personalized_ranker import filter_candidates, recommend_personalized
from embedding_engine import load_embeddings, load_embedding_model


class TestRanking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalog_path = resolve_path("fashion_project/artifacts/catalog.csv")
        cls.ids_path = resolve_path("fashion_project/artifacts/catalog_ids.json")
        cls.catalog_df = pd.read_csv(cls.catalog_path, dtype={"id": str})
        with open(cls.ids_path, "r", encoding="utf-8") as f:
            cls.catalog_ids = json.load(f)
        cls.embeddings_matrix, _ = load_embeddings(str(cls.catalog_path.parent))
        cls.model = load_embedding_model()

        cls.test_db = str(PROJECT_ROOT / "backups" / "test_ranking_profiles.sqlite")
        init_db(cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if Path(cls.test_db).exists():
            try:
                Path(cls.test_db).unlink()
            except Exception:
                pass

    def test_01_filter_candidates_hard_constraints(self):
        # Strict category filter
        filtered_shirts = filter_candidates(self.catalog_df, {"articleType": ["Shirts"]})
        self.assertTrue(all(filtered_shirts["articleType"] == "Shirts"))
        self.assertGreater(len(filtered_shirts), 0)

        # Excluded colors
        filtered_no_red = filter_candidates(self.catalog_df, {"articleType": ["Shirts"], "exclude_colors": ["Red"]})
        self.assertTrue(all(filtered_no_red["baseColour"] != "Red"))

        # Dismissed items excluded
        dismissed = {"10051"}
        filtered_no_dismiss = filter_candidates(self.catalog_df, {"articleType": ["Shirts"]}, dismissed_ids=dismissed)
        self.assertNotIn("10051", filtered_no_dismiss["id"].tolist())

    def test_02_user_vector_synthesis_and_normalization(self):
        pid = create_profile(db_path=self.test_db)
        save_feedback(pid, "10051", "like", db_path=self.test_db)
        save_feedback(pid, "10440", "like", db_path=self.test_db)

        u_vec = build_user_vector(
            profile_id=pid,
            embeddings=self.embeddings_matrix,
            catalog_ids=self.catalog_ids,
            catalog=self.catalog_df,
            model=self.model,
            db_path=self.test_db
        )
        self.assertIsNotNone(u_vec)
        self.assertEqual(u_vec.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(u_vec)), 1.0, places=4)

    def test_03_personalized_ranking_formula_weights(self):
        # Verify 0.8 * q + 0.2 * u ranking
        pid = create_profile(db_path=self.test_db)
        save_feedback(pid, "10051", "like", db_path=self.test_db)

        rec = recommend_personalized(
            profile_id=pid,
            query="casual shirts",
            constraints={"articleType": ["Shirts"]},
            embeddings=self.embeddings_matrix,
            catalog_ids=self.catalog_ids,
            catalog=self.catalog_df,
            model=self.model,
            db_path=self.test_db,
            top_k=5
        )
        self.assertEqual(rec["status"], "success")
        self.assertEqual(len(rec["results"]), 5)
        # Verify hard constraint is maintained for all returned results
        for item in rec["results"]:
            self.assertEqual(item["articleType"], "Shirts")

    def test_04_no_matches_handling(self):
        # Valid catalog attributes with zero co-occurrence
        rec = recommend_personalized(
            query="shoes",
            constraints={"articleType": ["Flip Flops"], "usage": ["Formal"]},
            embeddings=self.embeddings_matrix,
            catalog_ids=self.catalog_ids,
            catalog=self.catalog_df,
            model=self.model,
            db_path=self.test_db,
            top_k=5
        )
        self.assertEqual(rec["status"], "no_match")
        self.assertEqual(len(rec["results"]), 0)
        self.assertIn("revised_constraints_suggestion", rec)


if __name__ == "__main__":
    unittest.main()
