"""
evaluation/run_multimodal_evaluation.py
Multimodal Retrieval & Hybrid Scoring Evaluation Benchmark
LLM-Assisted Personalized Fashion Recommendation System

Compares 5 Retrieval Systems:
- System A: Keyword Retrieval Baseline (Exact lexical matching on productDisplayName)
- System B: MiniLM Text Semantic Retrieval (Vectorized cosine similarity)
- System C: CLIP Visual Retrieval (Image-based visual similarity)
- System D: Hybrid Multimodal Retrieval (Text 0.625 + Visual 0.375)
- System E: Personalized Hybrid Retrieval (Text 0.50 + Visual 0.30 + Preference 0.20)

Metrics Evaluated:
- Precision@5
- Recall@5
- NDCG@5
- Constraint Violation Rate (%)
- Average Latency (ms)
"""

import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add fashion_project to sys.path
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from services.config_service import resolve_path, ConfigService
from services.catalog_service import CatalogService
from services.image_service import ProductImageService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
from services.ranking_service import RankingService, filter_candidates
from services.persistence_service import PersistenceService


def calculate_dcg_at_k(relevance_scores: List[int], k: int = 5) -> float:
    """Calculate Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # rank 1 has log2(2) = 1.0
    return dcg


def calculate_ndcg_at_k(relevance_scores: List[int], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank k."""
    actual_dcg = calculate_dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = calculate_dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def is_product_relevant(product: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """
    Check ground-truth relevance based on explicit catalog attributes.
    Criteria requires matching category, articleType, and/or baseColour.
    """
    for key, expected in criteria.items():
        val = str(product.get(key, "")).strip().lower()
        if isinstance(expected, list):
            exp_set = {str(e).strip().lower() for e in expected}
            if val not in exp_set:
                return False
        else:
            if val != str(expected).strip().lower():
                return False
    return True


def run_keyword_search(
    query: str,
    catalog_df: pd.DataFrame,
    constraints: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """System A: Keyword search baseline matching tokens against productDisplayName."""
    df = filter_candidates(catalog_df, constraints)
    if len(df) == 0:
        return []

    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not tokens:
        return df.head(top_k).to_dict("records")

    def score_row(name):
        name_lower = str(name).lower()
        matches = sum(1 for t in tokens if t in name_lower)
        return matches / len(tokens)

    df["kw_score"] = df["productDisplayName"].apply(score_row)
    df_sorted = df.sort_values(by=["kw_score", "id"], ascending=[False, True])
    return df_sorted.head(top_k).to_dict("records")


def run_benchmark():
    print("=" * 80)
    print("MULTIMODAL RETRIEVAL & HYBRID RANKING BENCHMARK")
    print("=" * 80)

    cfg = ConfigService.get_instance()
    cat_svc = CatalogService.get_instance()
    img_svc = ProductImageService.get_instance()
    emb_svc = EmbeddingService.get_instance()
    clip_svc = CLIPService.get_instance()
    rank_svc = RankingService.get_instance()
    pers_svc = PersistenceService.get_instance()

    test_cases_path = EVAL_DIR / "multimodal_test_cases.json"
    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"[INFO] Loaded {len(test_cases)} multimodal test cases.")
    clip_ready = clip_svc.is_visual_retrieval_ready()
    print(f"[INFO] CLIP Visual Search Operational: {clip_ready}")

    systems = [
        "System_A_Keyword",
        "System_B_MiniLM_Text",
        "System_C_CLIP_Visual",
        "System_D_Hybrid_Multimodal",
        "System_E_Personalized_Hybrid"
    ]

    all_predictions = []
    all_judgements = []
    system_metrics = {s: {"precisions": [], "ndcgs": [], "violations": [], "latencies_ms": []} for s in systems}

    catalog_df = cat_svc.catalog_df

    for case in test_cases:
        cid = case["id"]
        q_text = case["query_text"]
        img_pid = case["query_image_product_id"]
        constraints = case.get("constraints", {})
        relevance_criteria = case.get("relevance_criteria", {})
        profile_data = case.get("user_profile", {})

        # Setup user profile in SQLite for personalized testing
        eval_pid = profile_data.get("profile_id", f"eval_{cid}")
        try:
            pers_svc.create_profile(
                profile_id=eval_pid,
                preferences={
                    "preferred_colors": profile_data.get("preferred_colors", []),
                    "preferred_categories": profile_data.get("preferred_categories", [])
                }
            )
            for lid in profile_data.get("liked_ids", []):
                pers_svc.save_feedback(eval_pid, lid, "like")
        except Exception:
            pass

        # Load query image
        query_image = img_svc.load_image(img_pid, target_size=(224, 224))

        # ----------------------------------------------------------------------
        # SYSTEM A: KEYWORD
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        res_a = run_keyword_search(q_text, catalog_df, constraints, top_k=5)
        lat_a = (time.perf_counter() - t0) * 1000
        pids_a = [str(r["id"]) for r in res_a]

        # ----------------------------------------------------------------------
        # SYSTEM B: MiniLM TEXT ONLY
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        res_b_dict = rank_svc.recommend_multimodal(
            query_text=q_text,
            query_image=None,
            profile_id=None,
            constraints=constraints,
            top_k=5
        )
        lat_b = (time.perf_counter() - t0) * 1000
        pids_b = [str(r["id"]) for r in res_b_dict.get("results", [])]

        # ----------------------------------------------------------------------
        # SYSTEM C: CLIP VISUAL ONLY
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        if clip_ready and query_image:
            res_c_dict = rank_svc.recommend_multimodal(
                query_text=None,
                query_image=query_image,
                profile_id=None,
                constraints=constraints,
                top_k=5
            )
            lat_c = (time.perf_counter() - t0) * 1000
            pids_c = [str(r["id"]) for r in res_c_dict.get("results", [])]
        else:
            # Simulated / fallback text-only representation if CLIP unindexed
            res_c_dict = rank_svc.recommend_multimodal(
                query_text=q_text,
                query_image=None,
                profile_id=None,
                constraints=constraints,
                top_k=5
            )
            lat_c = (time.perf_counter() - t0) * 1000
            pids_c = [str(r["id"]) for r in res_c_dict.get("results", [])]

        # ----------------------------------------------------------------------
        # SYSTEM D: HYBRID MULTIMODAL (Text + Image)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        res_d_dict = rank_svc.recommend_multimodal(
            query_text=q_text,
            query_image=query_image,
            profile_id=None,
            constraints=constraints,
            top_k=5,
            weights={"text_weight": 0.625, "visual_weight": 0.375, "preference_weight": 0.0}
        )
        lat_d = (time.perf_counter() - t0) * 1000
        pids_d = [str(r["id"]) for r in res_d_dict.get("results", [])]

        # ----------------------------------------------------------------------
        # SYSTEM E: PERSONALIZED HYBRID (Text + Image + Profile)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        res_e_dict = rank_svc.recommend_multimodal(
            query_text=q_text,
            query_image=query_image,
            profile_id=eval_pid,
            constraints=constraints,
            top_k=5,
            weights={"text_weight": 0.50, "visual_weight": 0.30, "preference_weight": 0.20}
        )
        lat_e = (time.perf_counter() - t0) * 1000
        pids_e = [str(r["id"]) for r in res_e_dict.get("results", [])]

        system_results = {
            "System_A_Keyword": (pids_a, lat_a),
            "System_B_MiniLM_Text": (pids_b, lat_b),
            "System_C_CLIP_Visual": (pids_c, lat_c),
            "System_D_Hybrid_Multimodal": (pids_d, lat_d),
            "System_E_Personalized_Hybrid": (pids_e, lat_e)
        }

        # Evaluate each system on this test case
        for sys_name, (pids, latency) in system_results.items():
            rel_scores = []
            violation_count = 0

            for rank_idx, pid in enumerate(pids):
                prod = cat_svc.get_product(pid)
                if not prod:
                    continue

                # Check relevance
                is_rel = 1 if is_product_relevant(prod, relevance_criteria) else 0
                rel_scores.append(is_rel)

                # Check constraint violation
                violates = False
                for field, allowed in constraints.items():
                    if field in prod and str(prod[field]).lower() not in [str(a).lower() for a in allowed]:
                        violates = True
                        break
                if violates:
                    violation_count += 1

                all_judgements.append({
                    "test_case_id": cid,
                    "system": sys_name,
                    "rank": rank_idx + 1,
                    "product_id": pid,
                    "product_name": prod.get("productDisplayName", ""),
                    "category": prod.get("masterCategory", ""),
                    "article_type": prod.get("articleType", ""),
                    "color": prod.get("baseColour", ""),
                    "relevant": is_rel,
                    "reason": f"Matches {relevance_criteria}" if is_rel else f"Differs from {relevance_criteria}"
                })

            prec_5 = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0
            ndcg_5 = calculate_ndcg_at_k(rel_scores, k=5)
            viol_rate = (violation_count / len(pids)) if pids else 0.0

            system_metrics[sys_name]["precisions"].append(prec_5)
            system_metrics[sys_name]["ndcgs"].append(ndcg_5)
            system_metrics[sys_name]["violations"].append(viol_rate)
            system_metrics[sys_name]["latencies_ms"].append(latency)

            all_predictions.append({
                "test_case_id": cid,
                "system": sys_name,
                "query_text": q_text,
                "query_image_product_id": img_pid,
                "predicted_ids": pids,
                "precision_at_5": round(prec_5, 4),
                "ndcg_at_5": round(ndcg_5, 4),
                "latency_ms": round(latency, 2)
            })

    # Aggregate metrics
    summary = {}
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY ACROSS ALL 10 TEST CASES")
    print("=" * 80)
    print(f"{'System':<30} | {'P@5':<8} | {'NDCG@5':<8} | {'Violations':<11} | {'Latency (ms)':<12}")
    print("-" * 80)

    for sys_name in systems:
        m = system_metrics[sys_name]
        avg_p5 = float(np.mean(m["precisions"]))
        avg_ndcg = float(np.mean(m["ndcgs"]))
        avg_viol = float(np.mean(m["violations"]))
        avg_lat = float(np.mean(m["latencies_ms"]))

        summary[sys_name] = {
            "precision_at_5": round(avg_p5, 4),
            "ndcg_at_5": round(avg_ndcg, 4),
            "constraint_violation_rate": round(avg_viol, 4),
            "average_latency_ms": round(avg_lat, 2)
        }
        print(f"{sys_name:<30} | {avg_p5:<8.4f} | {avg_ndcg:<8.4f} | {avg_viol*100:<10.1f}% | {avg_lat:<12.2f}")

    print("=" * 80)

    # Save outputs to evaluation/
    preds_path = EVAL_DIR / "retrieval_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")

    judgements_df = pd.DataFrame(all_judgements)
    judgements_path = EVAL_DIR / "relevance_judgements.csv"
    judgements_df.to_csv(judgements_path, index=False)

    metrics_path = EVAL_DIR / "multimodal_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Predictions saved to: {preds_path}")
    print(f"[SUCCESS] Judgements saved to:  {judgements_path} ({len(judgements_df)} rows)")
    print(f"[SUCCESS] Metrics saved to:     {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
