"""
services/retrieval_service.py
Pure NumPy Semantic Retrieval Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Vectorized dot product cosine similarity calculation over unit-normalized embeddings.
- Sub-millisecond similarity scoring on eligible candidate subsets.
- Return structured retrieval records with stable IDs and similarity scores.
- Terminology guardrail: Cosine similarity represents semantic proximity in embedding space,
  never to be labeled or described as "recommendation accuracy".
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService


class RetrievalService:
    """
    High-performance semantic retrieval service executing matrix dot products.
    """

    _instance: Optional["RetrievalService"] = None

    def __init__(
        self,
        catalog_svc: Optional[CatalogService] = None,
        embedding_svc: Optional[EmbeddingService] = None
    ):
        self.catalog_svc = catalog_svc or CatalogService.get_instance()
        self.embedding_svc = embedding_svc or EmbeddingService.get_instance()

        # Build fast O(1) mapping: product_id -> embedding row index
        self._id_to_idx = {pid: i for i, pid in enumerate(self.catalog_svc.catalog_ids)}

    @classmethod
    def get_instance(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_index_for_id(self, product_id: str | int) -> Optional[int]:
        return self._id_to_idx.get(str(product_id).strip())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        eligible_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic retrieval for a natural-language query.
        Args:
            query: User's search text.
            top_k: Number of products to return (default: 5).
            eligible_ids: Optional subset of candidate product IDs to filter before retrieval.
        Returns:
            List of dicts containing 'id', 'cosine_similarity', and catalog product attributes.
        """
        if not query or not query.strip():
            return []

        # 1. Encode query
        q_vec = self.embedding_svc.encode_text(query)
        if np.allclose(q_vec, 0.0):
            return []

        # 2. Select embedding subset
        embeddings_matrix = self.embedding_svc.embeddings
        if eligible_ids is not None:
            clean_ids = [str(x).strip() for x in eligible_ids]
            valid_indices = [self._id_to_idx[pid] for pid in clean_ids if pid in self._id_to_idx]
            if not valid_indices:
                return []
            sub_matrix = embeddings_matrix[valid_indices]
            sub_ids = [self.catalog_svc.catalog_ids[idx] for idx in valid_indices]
        else:
            sub_matrix = embeddings_matrix
            sub_ids = self.catalog_svc.catalog_ids

        # 3. Pure NumPy dot product cosine similarity: (M, 384) . (384,) -> (M,)
        sims = np.dot(sub_matrix, q_vec)

        # 4. Deterministic tie-breaking sort (score descending, ID ascending)
        items = [
            (float(sims[i]), sub_ids[i])
            for i in range(len(sub_ids))
        ]
        items.sort(key=lambda x: (-round(x[0], 7), str(x[1])))

        # 5. Take top-k
        top_items = items[:top_k]

        results = []
        for sim_score, pid in top_items:
            prod = self.catalog_svc.get_product(pid)
            if prod:
                rec = dict(prod)
                rec["cosine_similarity"] = float(sim_score)
                results.append(rec)

        return results
