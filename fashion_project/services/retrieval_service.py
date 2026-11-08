"""
services/retrieval_service.py
Pure NumPy Multimodal Retrieval Service (Text, Visual & Combined)
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Mode A (Text Search): Vectorized dot product cosine similarity with MiniLM.
- Mode B (Image Search): Vectorized dot product visual similarity with CLIP ViT-B/32.
- Mode C (Combined Multimodal): Dual-modality score extraction for candidate fusion.
- Sub-millisecond similarity scoring on eligible candidate subsets.
- Terminology guardrail: Cosine similarity represents semantic/visual proximity in embedding space,
  never to be labeled or described as "recommendation accuracy".
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from PIL import Image

from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService


class RetrievalService:
    """
    High-performance multimodal retrieval service executing matrix dot products.
    Supports Mode A (Text), Mode B (Image), and Mode C (Combined).
    """

    _instance: Optional["RetrievalService"] = None

    def __init__(
        self,
        catalog_svc: Optional[CatalogService] = None,
        embedding_svc: Optional[EmbeddingService] = None,
        clip_svc: Optional[CLIPService] = None
    ):
        self.catalog_svc = catalog_svc or CatalogService.get_instance()
        self.embedding_svc = embedding_svc or EmbeddingService.get_instance()
        self.clip_svc = clip_svc or CLIPService.get_instance()

        # Build fast O(1) mapping: product_id -> embedding row index
        self._id_to_idx = {pid: i for i, pid in enumerate(self.catalog_svc.catalog_ids)}

    @classmethod
    def get_instance(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_index_for_id(self, product_id: str | int) -> Optional[int]:
        return self._id_to_idx.get(str(product_id).strip())

    # --------------------------------------------------------------------------
    # MODE A: TEXT SEARCH (MiniLM)
    # --------------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        eligible_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic retrieval for a natural-language query using MiniLM.
        Preserves original Phase 1/Phase 2 contract.
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
                rec["text_similarity"] = float(sim_score)
                results.append(rec)

        return results

    # --------------------------------------------------------------------------
    # MODE B: IMAGE SEARCH (CLIP)
    # --------------------------------------------------------------------------
    def retrieve_by_image(
        self,
        query_image: Image.Image,
        top_k: int = 5,
        eligible_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute visual retrieval for an uploaded image using precomputed CLIP embeddings.
        """
        return self.clip_svc.search_by_image(
            query_image=query_image,
            top_k=top_k,
            eligible_ids=eligible_ids
        )

    # --------------------------------------------------------------------------
    # MODE C: CANDIDATE SIMILARITY EXTRACTION (MULTIMODAL FUSION)
    # --------------------------------------------------------------------------
    def get_candidate_scores(
        self,
        candidate_ids: List[str],
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute text and/or visual similarity for a set of candidate product IDs.
        Returns:
            {pid: {"text_similarity": float, "visual_similarity": float}}
        """
        scores: Dict[str, Dict[str, float]] = {
            str(pid): {"text_similarity": 0.0, "visual_similarity": 0.0}
            for pid in candidate_ids
        }

        # 1. Text similarities via MiniLM
        if query_text and query_text.strip():
            q_vec = self.embedding_svc.encode_text(query_text)
            if not np.allclose(q_vec, 0.0):
                clean_ids = [str(x).strip() for x in candidate_ids]
                valid_indices = [self._id_to_idx[pid] for pid in clean_ids if pid in self._id_to_idx]
                if valid_indices:
                    sub_matrix = self.embedding_svc.embeddings[valid_indices]
                    sims = np.dot(sub_matrix, q_vec)
                    for idx, s in zip(valid_indices, sims):
                        pid_str = self.catalog_svc.catalog_ids[idx]
                        scores[pid_str]["text_similarity"] = float(max(0.0, min(1.0, float(s))))

        # 2. Visual similarities via CLIP
        if query_image is not None and self.clip_svc.is_visual_retrieval_ready():
            try:
                vis_scores = self.clip_svc.get_visual_similarities(query_image, candidate_ids)
                for pid_str, vs in vis_scores.items():
                    if pid_str in scores:
                        scores[pid_str]["visual_similarity"] = vs
            except Exception:
                pass

        return scores
