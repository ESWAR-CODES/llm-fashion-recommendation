"""
services/ranking_service.py
Candidate Filtering & Personalized Ranking Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Strict hard constraint filtering (AND across fields, OR within field, exclusion rules).
- Authoritative personalized scoring formula:
    score(i) = 0.8 * cos(q, v_i) + 0.2 * cos(u, v_i)
- Deterministic tie-breaking (score descending, canonical string ID ascending).
- Zero silent relaxation: when zero products match, return empty result with actionable suggestions.
- Strict candidate exclusion for dismissed products and onboarding likes.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set, Union

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from services.config_service import ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.persistence_service import PersistenceService


def filter_candidates(
    catalog_df: pd.DataFrame,
    constraints: Optional[Dict[str, Any]] = None,
    dismissed_ids: Optional[Union[Set[str], List[str]]] = None
) -> pd.DataFrame:
    """
    Apply strict hard constraints and remove dismissed products before scoring.
    
    Rules:
    - OR within values of the same field.
    - AND across different fields.
    - Excluded colors always fail.
    - Missing attributes fail any strict constraint depending on them.
    - Dismissed products are strictly excluded.
    - Zero silent relaxation.
    """
    if catalog_df is None or len(catalog_df) == 0:
        return pd.DataFrame(columns=catalog_df.columns if catalog_df is not None else [])

    df = catalog_df.copy()

    # Rule 1: Exclude dismissed items
    if dismissed_ids:
        dismissed_set = set(str(x).strip() for x in dismissed_ids if str(x).strip())
        if dismissed_set:
            df = df[~df["id"].astype(str).isin(dismissed_set)]

    if not constraints:
        return df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Rule 2: Excluded colors strictly fail
    exclude_colors = constraints.get("exclude_colors", [])
    if isinstance(exclude_colors, str):
        exclude_colors = [exclude_colors]
    if exclude_colors:
        clean_exclude = set(str(c).strip().lower() for c in exclude_colors if c)
        if clean_exclude:
            color_matches_exclude = df["baseColour"].apply(
                lambda val: False if pd.isna(val) or str(val).strip() in ("", "NA", "null", "none")
                else str(val).strip().lower() in clean_exclude
            )
            df = df[~color_matches_exclude]

    # Rule 3: Field constraints (AND across fields, OR within same field)
    field_keys = ["articleType", "baseColour", "usage", "masterCategory", "gender"]
    for field in field_keys:
        allowed = constraints.get(field)
        if not allowed or field not in df.columns:
            continue
        if isinstance(allowed, str):
            allowed = [allowed]
        clean_allowed = set(str(a).strip().lower() for a in allowed if str(a).strip())
        if not clean_allowed:
            continue

        def check_field_match(val):
            if pd.isna(val) or str(val).strip() in ("", "NA", "null", "none"):
                return False
            return str(val).strip().lower() in clean_allowed

        df = df[df[field].apply(check_field_match)]

    return df.drop_duplicates(subset=["id"]).reset_index(drop=True)


class RankingService:
    """
    Service executing personalized multi-stage recommendation ranking.
    """

    _instance: Optional["RankingService"] = None

    def __init__(
        self,
        catalog_svc: Optional[CatalogService] = None,
        embedding_svc: Optional[EmbeddingService] = None,
        persistence_svc: Optional[PersistenceService] = None
    ):
        self.catalog_svc = catalog_svc or CatalogService.get_instance()
        self.embedding_svc = embedding_svc or EmbeddingService.get_instance()
        self.persistence_svc = persistence_svc or PersistenceService.get_instance()
        self._id_to_idx = {pid: i for i, pid in enumerate(self.catalog_svc.catalog_ids)}

    @classmethod
    def get_instance(cls) -> "RankingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def recommend(
        self,
        query: str,
        profile_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        exclude_ids: Optional[Union[Set[str], List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end personalized ranking.
        """
        # 1. Gather all exclusions: profile dismissals + caller exclude_ids
        all_excluded: Set[str] = set()
        if exclude_ids:
            all_excluded.update(str(x).strip() for x in exclude_ids)

        if profile_id:
            fb = self.persistence_svc.get_feedback(profile_id)
            all_excluded.update(fb.get("dismissals", []))

        # 2. Merge active constraints if profile provided and none passed explicitly
        active_constraints = dict(constraints or {})
        if profile_id and not constraints:
            saved_cons = self.persistence_svc.get_active_constraints(profile_id)
            if saved_cons:
                active_constraints.update(saved_cons)

        # 3. Apply strict candidate filtering
        catalog_df = self.catalog_svc.catalog_df
        eligible_df = filter_candidates(catalog_df, active_constraints, all_excluded)

        if len(eligible_df) == 0:
            return {
                "status": "no_match",
                "query": query,
                "profile_id": profile_id,
                "has_personalized_vector": False,
                "constraints_applied": active_constraints,
                "dismissed_count": len(all_excluded),
                "total_eligible": 0,
                "results_count": 0,
                "results": [],
                "message": "No products matched the given constraints.",
                "revised_constraints_suggestion": "Try removing excluded colors or broadening category / usage filters."
            }

        # 4. Encode query vector q
        q_vec = self.embedding_svc.encode_text(query)

        # 5. Synthesize user preference vector u if profile exists
        user_vec = None
        has_personalized_vector = False
        if profile_id:
            user_vec = self.persistence_svc.build_user_vector(profile_id)
            has_personalized_vector = user_vec is not None

        # 6. Map eligible candidate IDs to embedding matrix rows
        eligible_ids = eligible_df["id"].astype(str).tolist()
        valid_indices = [self._id_to_idx[pid] for pid in eligible_ids if pid in self._id_to_idx]

        if not valid_indices:
            return {
                "status": "no_match",
                "query": query,
                "profile_id": profile_id,
                "has_personalized_vector": False,
                "constraints_applied": active_constraints,
                "dismissed_count": len(all_excluded),
                "total_eligible": 0,
                "results_count": 0,
                "results": [],
                "message": "No matching embeddings found for eligible items.",
                "revised_constraints_suggestion": "Check catalog embedding alignment."
            }

        eligible_matrix = self.embedding_svc.embeddings[valid_indices]
        q_sims = np.dot(eligible_matrix, q_vec)

        # Authoritative weighting: 0.8 * query + 0.2 * user_vector
        if user_vec is not None and not np.allclose(q_vec, 0.0):
            u_sims = np.dot(eligible_matrix, user_vec)
            scores = 0.8 * q_sims + 0.2 * u_sims
        elif user_vec is not None and np.allclose(q_vec, 0.0):
            # Query empty: purely profile-based
            scores = np.dot(eligible_matrix, user_vec)
        else:
            # No user vector: purely query-based
            scores = q_sims

        # 7. Deterministic tie-breaking (score descending, string ID ascending)
        candidate_items = [
            (float(scores[i]), eligible_ids[i], float(q_sims[i]))
            for i in range(len(eligible_ids))
        ]
        candidate_items.sort(key=lambda x: (-round(x[0], 7), str(x[1])))

        # 8. Return top-k (never pad; return fewer if fewer exist)
        top_items = candidate_items[:top_k]

        results = []
        for final_score, pid, q_sim in top_items:
            prod = self.catalog_svc.get_product(pid)
            if prod:
                rec = dict(prod)
                rec["score"] = float(final_score)
                rec["query_similarity"] = float(q_sim)
                results.append(rec)

        return {
            "status": "success",
            "query": query,
            "profile_id": profile_id,
            "has_personalized_vector": has_personalized_vector,
            "constraints_applied": active_constraints,
            "dismissed_count": len(all_excluded),
            "total_eligible": len(eligible_df),
            "results_count": len(results),
            "results": results
        }
