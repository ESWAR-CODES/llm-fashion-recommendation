"""
personalized_ranker.py
Phase 3: Candidate Filtering & Personalized Recommendation Ranking
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Steps 5 & 7)
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set, Union

import numpy as np
import pandas as pd

from profile_manager import (
    resolve_path,
    get_profile,
    get_feedback,
    get_active_constraints,
    validate_constraints,
    build_user_vector,
    DEFAULT_DB_PATH,
    DEFAULT_ARTIFACTS_DIR
)


def filter_candidates(
    catalog: pd.DataFrame,
    constraints: Optional[Dict[str, Any]] = None,
    dismissed_ids: Optional[Union[Set[str], List[str]]] = None
) -> pd.DataFrame:
    """
    Apply strict hard constraints and remove dismissed products before calculating ranking scores.
    
    Rules:
    - OR within values of the same field.
    - AND across different fields.
    - Excluded colors always fail.
    - Missing attributes fail any strict constraint depending on them.
    - Dismissed products must not be returned.
    - Duplicate product IDs must not appear.
    - Do not silently relax constraints.
    """
    if catalog is None or len(catalog) == 0:
        return pd.DataFrame(columns=catalog.columns if catalog is not None else [])

    df = catalog.copy()

    # Rule 1: Remove dismissed products
    if dismissed_ids:
        dismissed_set = set(str(x).strip() for x in dismissed_ids)
        df = df[~df["id"].astype(str).isin(dismissed_set)]

    if not constraints:
        return df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Rule 2: Excluded colors always fail
    exclude_colors = constraints.get("exclude_colors", [])
    if isinstance(exclude_colors, str):
        exclude_colors = [exclude_colors]
    if exclude_colors:
        ex_colors_clean = set(str(c).strip().lower() for c in exclude_colors if c)
        if ex_colors_clean:
            color_matches_exclude = df["baseColour"].apply(
                lambda val: False if pd.isna(val) or str(val).strip() in ("", "NA", "null", "none")
                else str(val).strip().lower() in ex_colors_clean
            )
            df = df[~color_matches_exclude]

    # Rule 3: Field constraints (AND across fields, OR within same field)
    # Supported: articleType, baseColour, usage, masterCategory
    field_keys = ["articleType", "baseColour", "usage", "masterCategory"]
    for field in field_keys:
        allowed = constraints.get(field)
        if not allowed:
            continue
        if isinstance(allowed, str):
            allowed = [allowed]
        allowed_clean = set(str(a).strip().lower() for a in allowed if a)
        if not allowed_clean:
            continue

        def check_field_match(val):
            # Missing attributes fail any strict constraint depending on them
            if pd.isna(val) or str(val).strip() in ("", "NA", "null", "none"):
                return False
            return str(val).strip().lower() in allowed_clean

        mask = df[field].apply(check_field_match)
        df = df[mask]

    # Rule 4: Eliminate duplicate product IDs
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def recommend_personalized(
    profile_id: Optional[str] = None,
    query: str = "",
    constraints: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    model: Optional[Any] = None,
    catalog: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    catalog_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    images_dir: Optional[str] = "fashion_project/data/raw/images"
) -> Dict[str, Any]:
    """
    Personalized Recommendation Ranking Workflow:
    1. Validate profile (handle unknown profile IDs gracefully).
    2. Validate active constraints (or load active constraints from profile).
    3. Construct request representation and encode query with MiniLM.
    4. Apply hard catalog constraints & remove dismissed products.
    5. Construct user preference vector when available (mean of liked embeddings or saved preferences).
    6. Calculate personalized similarity scores using vectorized matrix operations:
       score(item) = 0.8 * cosine(q, item_vector) + 0.2 * cosine(u, item_vector)
       Fallback when user vector is unavailable:
       score(item) = cosine(q, item_vector)
    7. Stable tie-breaking: sort descending by score, tie-breaking ascending by string id.
    8. Return up to top_k items with metadata and verified image paths.
    """
    art_dir = Path(artifacts_dir or DEFAULT_ARTIFACTS_DIR)

    # 1. Load catalog, embeddings, and catalog_ids if not provided
    if catalog is None:
        cat_p = resolve_path(str(art_dir / "catalog.csv"))
        catalog = pd.read_csv(cat_p, dtype={"id": str})

    if embeddings is None or catalog_ids is None:
        emb_p = resolve_path(str(art_dir / "embeddings.npy"))
        ids_p = resolve_path(str(art_dir / "catalog_ids.json"))
        embeddings = np.load(str(emb_p))
        with open(ids_p, "r", encoding="utf-8") as f:
            catalog_ids = json.load(f)

    # 2. Profile Validation & Constraint Resolution
    profile = None
    dismissed_ids: Set[str] = set()
    if profile_id is not None:
        profile = get_profile(profile_id, db_path=db_path)
        if profile is not None:
            fb = get_feedback(profile_id, db_path=db_path)
            dismissed_ids = set(fb.get("dismissals", []))

    # Resolve constraints: passed explicitly OR loaded from profile active_states
    if constraints is None and profile is not None:
        constraints = get_active_constraints(profile_id, db_path=db_path)

    clean_constraints = {}
    if constraints:
        clean_constraints = validate_constraints(constraints)

    # 3. Filter candidates
    eligible_df = filter_candidates(catalog, clean_constraints, dismissed_ids)

    # Handle No-Match scenario
    if len(eligible_df) == 0:
        return {
            "status": "no_match",
            "query": query,
            "profile_id": profile_id,
            "has_personalized_vector": False,
            "constraints_applied": clean_constraints,
            "dismissed_count": len(dismissed_ids),
            "total_eligible": 0,
            "results_count": 0,
            "results": [],
            "message": "No products matched the given constraints.",
            "revised_constraints_suggestion": (
                "Try relaxing specific constraints (e.g. allowing additional colors or article types) "
                "or removing excluded colors to increase matching candidates."
            )
        }

    # 4. Initialize embedding model if needed
    if model is None:
        from embedding_engine import load_embedding_model
        model = load_embedding_model()

    # 5. Construct Query Representation & Vector
    q_str = query.strip() if query and query.strip() else ""
    if not q_str:
        # Construct fallback textual query from constraints
        terms = []
        if clean_constraints.get("usage"):
            terms.extend(clean_constraints["usage"])
        if clean_constraints.get("articleType"):
            terms.extend(clean_constraints["articleType"])
        if clean_constraints.get("baseColour"):
            terms.extend(clean_constraints["baseColour"])
        if clean_constraints.get("masterCategory"):
            terms.extend(clean_constraints["masterCategory"])
        q_str = " ".join(terms) if terms else "fashion apparel clothing footwear"

    q_vec = model.encode(q_str, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    if q_vec.ndim > 1:
        q_vec = q_vec.squeeze()
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 1e-9:
        q_vec = q_vec / q_norm

    # 6. Build User Preference Vector
    user_vec = None
    if profile_id is not None and profile is not None:
        user_vec = build_user_vector(
            profile_id=profile_id,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            catalog=catalog,
            model=model,
            db_path=db_path,
            artifacts_dir=artifacts_dir
        )

    # 7. Fast Vectorized Similarity Scoring
    id_to_idx = {str(pid): i for i, pid in enumerate(catalog_ids)}
    eligible_ids = eligible_df["id"].astype(str).tolist()
    
    # Map eligible rows to embedding indices
    eligible_embedding_indices = [id_to_idx[pid] for pid in eligible_ids if pid in id_to_idx]
    if not eligible_embedding_indices:
        return {
            "status": "no_match",
            "query": query,
            "profile_id": profile_id,
            "has_personalized_vector": False,
            "constraints_applied": clean_constraints,
            "dismissed_count": len(dismissed_ids),
            "total_eligible": 0,
            "results_count": 0,
            "results": [],
            "message": "No valid catalog embeddings found for eligible products.",
            "revised_constraints_suggestion": "Verify catalog ID mapping."
        }

    # Vectorized cosine similarity dot product: (M, 384) x (384,) -> (M,)
    eligible_matrix = embeddings[eligible_embedding_indices]
    q_sims = np.dot(eligible_matrix, q_vec)

    if user_vec is not None:
        u_sims = np.dot(eligible_matrix, user_vec)
        # Authoritative formula: 0.8 * query + 0.2 * profile
        scores = 0.8 * q_sims + 0.2 * u_sims
    else:
        u_sims = None
        scores = q_sims

    # 8. Deterministic Tie-Breaking & Sorting
    # Build candidate tuple: (score, pid, list_pos)
    candidate_items = [
        (float(scores[i]), eligible_ids[i], i)
        for i in range(len(eligible_embedding_indices))
    ]
    # Sort descending by score, tie-breaking ascending by string id
    candidate_items.sort(key=lambda x: (-round(x[0], 7), str(x[1])))

    # 9. Top-K Selection & Metadata Enrichment (Only on top-k items)
    actual_k = min(max(1, top_k), len(candidate_items))
    selected_candidates = candidate_items[:actual_k]

    img_dir_path = resolve_path(images_dir) if images_dir else None
    top_results = []

    for rank, (score_val, pid, orig_pos) in enumerate(selected_candidates, start=1):
        row = eligible_df.iloc[orig_pos]
        img_p = img_dir_path / f"{pid}.jpg" if img_dir_path else None
        
        sim_q_val = float(q_sims[orig_pos])
        sim_u_val = float(u_sims[orig_pos]) if u_sims is not None else None

        top_results.append({
            "rank": rank,
            "id": pid,
            "productDisplayName": str(row["productDisplayName"]),
            "articleType": str(row["articleType"]),
            "baseColour": str(row["baseColour"]),
            "usage": str(row["usage"]),
            "masterCategory": str(row["masterCategory"]),
            "subCategory": str(row["subCategory"]),
            "season": str(row["season"]),
            "year": str(row["year"]),
            "score": score_val,
            "query_similarity": sim_q_val,
            "profile_similarity": sim_u_val,
            "image_path": str(img_p) if img_p else None,
            "image_exists": img_p.exists() if img_p else False
        })

    return {
        "status": "success",
        "query": query,
        "profile_id": profile_id,
        "has_personalized_vector": user_vec is not None,
        "constraints_applied": clean_constraints,
        "dismissed_count": len(dismissed_ids),
        "total_eligible": len(eligible_df),
        "results_count": len(top_results),
        "results": top_results
    }
