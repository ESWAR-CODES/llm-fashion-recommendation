"""
services/ranking_service.py
Candidate Filtering & Hybrid Personalized Multimodal Ranking Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Strict hard constraint filtering (AND across fields, OR within field, exclusion rules).
  Supports: articleType, baseColour, usage, masterCategory, subCategory, gender, season, exclude_colors.
- Multimodal hybrid scoring formula:
    score(i) = Wt * TextSimilarity(i) + Wv * VisualSimilarity(i) + Wp * PreferenceSimilarity(i)
    Where Wt + Wv + Wp = 1.0 (with dynamic renormalization when modalities are absent).
- Preserves baseline 0.8 * text + 0.2 * profile when only text and profile are provided.
- Deterministic tie-breaking (score descending, canonical string ID ascending).
- Zero silent relaxation: when zero products match, return empty result with actionable suggestions.
- Strict candidate exclusion for dismissed products and caller exclusions.
- Deduplication and exact constraint enforcement.
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
from PIL import Image

from services.config_service import ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
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
    field_keys = ["articleType", "baseColour", "usage", "masterCategory", "subCategory", "gender", "season"]
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
    Service executing personalized multi-stage recommendation ranking
    with multimodal hybrid score fusion.
    """

    _instance: Optional["RankingService"] = None

    def __init__(
        self,
        catalog_svc: Optional[CatalogService] = None,
        embedding_svc: Optional[EmbeddingService] = None,
        clip_svc: Optional[CLIPService] = None,
        persistence_svc: Optional[PersistenceService] = None
    ):
        self.catalog_svc = catalog_svc or CatalogService.get_instance()
        self.embedding_svc = embedding_svc or EmbeddingService.get_instance()
        self.clip_svc = clip_svc or CLIPService.get_instance()
        self.persistence_svc = persistence_svc or PersistenceService.get_instance()
        self._id_to_idx = {pid: i for i, pid in enumerate(self.catalog_svc.catalog_ids)}

    @classmethod
    def get_instance(cls) -> "RankingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_weights(
        self,
        has_text: bool,
        has_image: bool,
        has_profile: bool,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Dynamically determine and normalize ranking weights based on active modalities.
        Guarantees that active weights sum exactly to 1.0.
        """
        if custom_weights:
            w_t = max(0.0, float(custom_weights.get("text_weight", 0.0))) if has_text else 0.0
            w_v = max(0.0, float(custom_weights.get("visual_weight", 0.0))) if has_image else 0.0
            w_p = max(0.0, float(custom_weights.get("preference_weight", 0.0))) if has_profile else 0.0
            total = w_t + w_v + w_p
            if total > 1e-9:
                return {
                    "text_weight": round(w_t / total, 4),
                    "visual_weight": round(w_v / total, 4),
                    "preference_weight": round(w_p / total, 4)
                }

        cfg = ConfigService.get_instance()
        pers_cfg = cfg.config.get("personalization", {})
        hybrid_cfg = pers_cfg.get("hybrid_weights", {"text_weight": 0.5, "visual_weight": 0.3, "preference_weight": 0.2})

        raw_t = hybrid_cfg.get("text_weight", 0.5) if has_text else 0.0
        raw_v = hybrid_cfg.get("visual_weight", 0.3) if has_image else 0.0
        raw_p = hybrid_cfg.get("preference_weight", 0.2) if has_profile else 0.0

        # Baseline text + profile compatibility (0.8 / 0.2)
        if has_text and has_profile and not has_image:
            raw_t = pers_cfg.get("query_weight", 0.8)
            raw_p = pers_cfg.get("profile_weight", 0.2)
            raw_v = 0.0
        elif has_image and has_profile and not has_text:
            raw_v = 0.8
            raw_p = 0.2
            raw_t = 0.0

        total = raw_t + raw_v + raw_p
        if total <= 1e-9:
            # Fallback when no active query
            return {"text_weight": 1.0, "visual_weight": 0.0, "preference_weight": 0.0}

        return {
            "text_weight": round(raw_t / total, 4),
            "visual_weight": round(raw_v / total, 4),
            "preference_weight": round(raw_p / total, 4)
        }

    def recommend(
        self,
        query: str,
        profile_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        exclude_ids: Optional[Union[Set[str], List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Original Phase 1 baseline interface (text search + personalization).
        Preserved for 100% backward compatibility.
        """
        return self.recommend_multimodal(
            query_text=query,
            query_image=None,
            profile_id=profile_id,
            constraints=constraints,
            top_k=top_k,
            exclude_ids=exclude_ids
        )

    def recommend_multimodal(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        profile_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        exclude_ids: Optional[Union[Set[str], List[str]]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end multimodal personalized ranking:
        Supports:
        - Mode A: Text Search
        - Mode B: Image Search
        - Mode C: Combined Text and Image Search
        - Mode D: Personalized Search
        """
        # 1. Exclusions: profile dismissals + caller exclusions
        all_excluded: Set[str] = set()
        if exclude_ids:
            all_excluded.update(str(x).strip() for x in exclude_ids)

        if profile_id:
            fb = self.persistence_svc.get_feedback(profile_id)
            all_excluded.update(fb.get("dismissals", []))

        # 2. Merge active constraints from profile if not passed explicitly
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
                "query": query_text,
                "profile_id": profile_id,
                "retrieval_mode": "no_match",
                "has_personalized_vector": False,
                "constraints_applied": active_constraints,
                "dismissed_count": len(all_excluded),
                "total_eligible": 0,
                "results_count": 0,
                "results": [],
                "message": "No products matched the given constraints.",
                "revised_constraints_suggestion": "Try removing excluded colors or broadening category / usage filters."
            }

        # 4. Modality detection & visual readiness verification
        has_text = bool(query_text and query_text.strip())
        visual_ready = self.clip_svc.is_visual_retrieval_ready()
        has_image = bool(query_image is not None and visual_ready)
        visual_fallback_triggered = bool(query_image is not None and not visual_ready)

        user_vec = None
        has_personalized_vector = False
        if profile_id:
            user_vec = self.persistence_svc.build_user_vector(profile_id)
            has_personalized_vector = user_vec is not None

        # 5. Resolve and renormalize weights
        active_weights = self._resolve_weights(
            has_text=has_text,
            has_image=has_image,
            has_profile=has_personalized_vector,
            custom_weights=weights
        )

        w_t = active_weights["text_weight"]
        w_v = active_weights["visual_weight"]
        w_p = active_weights["preference_weight"]

        # Determine retrieval mode label
        if has_text and has_image:
            retrieval_mode = "personalized_hybrid" if has_personalized_vector else "hybrid"
        elif has_image:
            retrieval_mode = "personalized_image" if has_personalized_vector else "image"
        elif has_text:
            retrieval_mode = "personalized_text" if has_personalized_vector else "text"
        else:
            retrieval_mode = "profile_recommendation" if has_personalized_vector else "catalog_browse"

        # 6. Candidate vector slicing
        eligible_ids = eligible_df["id"].astype(str).tolist()
        valid_indices = [self._id_to_idx[pid] for pid in eligible_ids if pid in self._id_to_idx]

        if not valid_indices:
            return {
                "status": "no_match",
                "query": query_text,
                "profile_id": profile_id,
                "retrieval_mode": "no_match",
                "has_personalized_vector": False,
                "constraints_applied": active_constraints,
                "dismissed_count": len(all_excluded),
                "total_eligible": 0,
                "results_count": 0,
                "results": [],
                "message": "No matching embeddings found for eligible items.",
                "revised_constraints_suggestion": "Check catalog embedding alignment."
            }

        sub_text_matrix = self.embedding_svc.embeddings[valid_indices]

        # 7. Compute individual similarities
        # A. Text similarity via MiniLM
        if has_text:
            q_vec = self.embedding_svc.encode_text(query_text)
            raw_t = np.dot(sub_text_matrix, q_vec)
            sim_t = np.clip(raw_t, 0.0, 1.0)
        else:
            sim_t = np.zeros(len(valid_indices), dtype=np.float32)

        # B. Visual similarity via CLIP
        if has_image:
            try:
                vis_scores_map = self.clip_svc.get_visual_similarities(query_image, eligible_ids)
                sim_v = np.array([vis_scores_map.get(pid, 0.0) for pid in eligible_ids], dtype=np.float32)
            except Exception:
                sim_v = np.zeros(len(valid_indices), dtype=np.float32)
        else:
            sim_v = np.zeros(len(valid_indices), dtype=np.float32)

        # C. User preference similarity via MiniLM
        if has_personalized_vector:
            raw_p = np.dot(sub_text_matrix, user_vec)
            sim_p = np.clip(raw_p, 0.0, 1.0)
        else:
            sim_p = np.zeros(len(valid_indices), dtype=np.float32)

        # 8. Compute hybrid score
        final_scores = (w_t * sim_t) + (w_v * sim_v) + (w_p * sim_p)

        # 9. Deterministic tie-breaking sort (score descending, string ID ascending)
        candidate_items = [
            (
                float(final_scores[i]),
                eligible_ids[i],
                float(sim_t[i]),
                float(sim_v[i]),
                float(sim_p[i])
            )
            for i in range(len(eligible_ids))
        ]
        candidate_items.sort(key=lambda x: (-round(x[0], 7), str(x[1])))

        # 10. Return top-k (never pad; return fewer if fewer exist)
        top_items = candidate_items[:top_k]

        results = []
        for score_val, pid, st_val, sv_val, sp_val in top_items:
            prod = self.catalog_svc.get_product(pid)
            if prod:
                rec = dict(prod)
                rec["score"] = float(score_val)
                rec["query_similarity"] = float(st_val)
                rec["text_similarity"] = float(st_val)
                rec["visual_similarity"] = float(sv_val)
                rec["preference_similarity"] = float(sp_val)
                results.append(rec)

        return {
            "status": "success",
            "query": query_text,
            "profile_id": profile_id,
            "retrieval_mode": retrieval_mode,
            "has_personalized_vector": has_personalized_vector,
            "constraints_applied": active_constraints,
            "weights_applied": active_weights,
            "visual_fallback_triggered": visual_fallback_triggered,
            "dismissed_count": len(all_excluded),
            "total_eligible": len(eligible_df),
            "results_count": len(results),
            "results": results
        }
