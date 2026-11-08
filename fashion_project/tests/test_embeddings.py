"""
test_embeddings.py
Test Suite for Phase 2: Product Text Representations, MiniLM Embeddings & Semantic Retrieval
Authoritative Specification: End_to_End PipeLine.pdf (Step 4)
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

from profile_manager import resolve_path
from embedding_engine import (
    load_embeddings,
    load_embedding_model,
    retrieve_products,
    build_product_description
)


class TestEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalog_path = resolve_path("fashion_project/artifacts/catalog.csv")
        cls.ids_path = resolve_path("fashion_project/artifacts/catalog_ids.json")
        cls.embeddings_dir = resolve_path("fashion_project/artifacts")
        cls.desc_path = resolve_path("fashion_project/artifacts/product_descriptions.csv")

        cls.catalog_df = pd.read_csv(cls.catalog_path, dtype={"id": str})
        with open(cls.ids_path, "r", encoding="utf-8") as f:
            cls.catalog_ids = json.load(f)

        cls.embeddings_matrix, cls.manifest = load_embeddings(str(cls.embeddings_dir))
        cls.model = load_embedding_model()

    def test_01_product_descriptions_file(self):
        self.assertTrue(self.desc_path.exists(), "product_descriptions.csv not found")
        desc_df = pd.read_csv(self.desc_path, dtype={"id": str})
        self.assertEqual(len(desc_df), 10000, "Description count must equal 10,000")
        self.assertEqual(desc_df["id"].tolist(), self.catalog_ids, "Description IDs must match catalog IDs exactly")

    def test_02_embedding_matrix_shape_and_dtype(self):
        self.assertEqual(self.embeddings_matrix.shape, (10000, 384), "Embeddings must be (10000, 384)")
        self.assertEqual(self.embeddings_matrix.dtype, np.float32, "Embeddings dtype must be float32")

    def test_03_no_nan_or_inf_in_embeddings(self):
        self.assertTrue(np.all(np.isfinite(self.embeddings_matrix)), "Embeddings contain non-finite values")

    def test_04_l2_unit_normalization(self):
        norms = np.linalg.norm(self.embeddings_matrix, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4), "All embedding vectors must have unit L2 norm")

    def test_05_semantic_retrieval_returns_top_5(self):
        results = retrieve_products(
            query="navy blue casual shirts",
            top_k=5,
            model=self.model,
            embeddings=self.embeddings_matrix,
            catalog=self.catalog_df
        )
        self.assertEqual(len(results), 5, "Semantic retrieval must return exactly 5 items")
        retrieved_ids = [r["id"] for r in results]
        self.assertEqual(len(set(retrieved_ids)), 5, "Retrieved IDs must be unique")
        # Ensure scores are between -1 and 1
        for r in results:
            self.assertGreaterEqual(r["retrieval_score"], -1.0)
            self.assertLessEqual(r["retrieval_score"], 1.0)

    def test_06_empty_query_handling(self):
        results_empty = retrieve_products(
            query="",
            top_k=5,
            model=self.model,
            embeddings=self.embeddings_matrix,
            catalog=self.catalog_df
        )
        self.assertEqual(results_empty, [], "Empty query must return empty results")

        results_space = retrieve_products(
            query="   ",
            top_k=5,
            model=self.model,
            embeddings=self.embeddings_matrix,
            catalog=self.catalog_df
        )
        self.assertEqual(results_space, [], "Whitespace query must return empty results")


if __name__ == "__main__":
    unittest.main()
