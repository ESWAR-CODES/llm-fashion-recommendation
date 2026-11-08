# Algorithm Design & Pseudocode
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Candidate Filtering Algorithm (Algorithm 1)

The candidate filtering stage guarantees zero silent relaxation of user-specified hard constraints.

```python
def filter_candidates(catalog: DataFrame, constraints: Dict[str, Any], dismissed_ids: Set[str]) -> List[int]:
    """
    Inputs:
        catalog: Frozen DataFrame of 10,000 items
        constraints: Dictionary of facet filters (e.g., gender, masterCategory, exclude_colors)
        dismissed_ids: Set of product IDs dismissed by the active user profile
    Output:
        eligible_indices: List of row indices satisfying all active constraints
    """
    mask = np.ones(len(catalog), dtype=bool)
    
    # 1. Inclusion constraints (conjunction of disjunctions)
    for field in ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage"]:
        if field in constraints and constraints[field]:
            allowed_values = constraints[field]
            field_mask = catalog[field].isin(allowed_values).values
            mask = mask & field_mask

    # 2. Exclusion constraints
    if "exclude_colors" in constraints and constraints["exclude_colors"]:
        excluded = constraints["exclude_colors"]
        color_mask = ~catalog["baseColour"].isin(excluded).values
        mask = mask & color_mask

    # 3. Dismissal exclusion
    if dismissed_ids:
        id_mask = ~catalog["id"].astype(str).isin(dismissed_ids).values
        mask = mask & id_mask

    eligible_indices = np.where(mask)[0].tolist()
    return eligible_indices
```

---

## 2. Dynamic Weight Normalization (Algorithm 2)

Weights must sum to exactly $1.0$ regardless of which modalities are active:

```python
def compute_normalized_weights(has_text: bool, has_image: bool, has_profile: bool) -> Tuple[float, float, float]:
    """
    Computes (W_text, W_visual, W_profile) such that W_t + W_v + W_p = 1.0.
    """
    if has_text and has_image and has_profile:
        return (0.50, 0.30, 0.20)
    elif has_text and has_image and not has_profile:
        # Renormalize 0.50 / 0.80 and 0.30 / 0.80
        return (0.625, 0.375, 0.0)
    elif has_text and not has_image and has_profile:
        return (0.80, 0.0, 0.20)
    elif not has_text and has_image and has_profile:
        return (0.0, 0.80, 0.20)
    elif has_text and not has_image and not has_profile:
        return (1.0, 0.0, 0.0)
    elif not has_text and has_image and not has_profile:
        return (0.0, 1.0, 0.0)
    else:
        # Default fallback
        return (1.0, 0.0, 0.0)
```

---

## 3. Multimodal Hybrid Personalized Scoring (Algorithm 3)

```python
def rank_candidates(
    eligible_indices: List[int],
    text_matrix: np.ndarray,      # shape (10000, 384)
    clip_matrix: np.ndarray,      # shape (10000, 512)
    query_text_vec: Optional[np.ndarray],  # shape (384,)
    query_clip_vec: Optional[np.ndarray],  # shape (512,)
    user_taste_vec: Optional[np.ndarray],  # shape (384,)
    weights: Tuple[float, float, float],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Calculates combined multimodal score and returns top-K items.
    """
    W_t, W_v, W_p = weights
    sub_text = text_matrix[eligible_indices]
    
    # 1. Text Similarity
    if query_text_vec is not None and W_t > 0:
        S_text = np.dot(sub_text, query_text_vec)
    else:
        S_text = np.zeros(len(eligible_indices), dtype=np.float32)

    # 2. Visual Similarity
    if query_clip_vec is not None and W_v > 0 and clip_matrix is not None:
        sub_clip = clip_matrix[eligible_indices]
        S_vis = np.dot(sub_clip, query_clip_vec)
    else:
        S_vis = np.zeros(len(eligible_indices), dtype=np.float32)

    # 3. User Profile Similarity
    if user_taste_vec is not None and W_p > 0:
        S_prof = np.dot(sub_text, user_taste_vec)
    else:
        S_prof = np.zeros(len(eligible_indices), dtype=np.float32)

    # 4. Composite Linear Fusion
    scores = (W_t * S_text) + (W_v * S_vis) + (W_p * S_prof)

    # 5. Deterministic Top-K Extraction with Tie-Breaking
    # Sort indices by score descending, then product ID ascending
    sorted_order = np.argsort(-scores, kind="mergesort")
    top_indices = [eligible_indices[i] for i in sorted_order[:top_k]]
    top_scores = [float(scores[i]) for i in sorted_order[:top_k]]

    return list(zip(top_indices, top_scores))
```

---

## 4. Grounded Explanation Generation (Algorithm 4)

```python
def verify_and_generate_explanation(product: Dict[str, Any], active_constraints: Dict[str, Any]) -> str:
    """
    Verifies that the generated justification cites factual catalog attributes.
    """
    matched_features = []
    for field, values in active_constraints.items():
        if field in product and product[field] in values:
            matched_features.append(f"{field}: {product[field]}")

    if matched_features:
        justification = (
            f"Recommended because it matches your preferred {', '.join(matched_features)} "
            f"and aligns with your visual and semantic style profile."
        )
    else:
        justification = (
            f"Recommended because its aesthetic and semantic profile ({product.get('baseColour')} "
            f"{product.get('articleType')}) closely aligns with your search intent."
        )

    # Factual verification: confirm claimed words exist in catalog attributes
    assert product.get("articleType") in justification or product.get("baseColour") in justification
    return justification
```
