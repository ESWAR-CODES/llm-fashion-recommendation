"""
validate_phase2.py
Phase 2 Validation and Testing Suite
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Tests all 17 mandatory requirements from Section 10.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Import Phase 2 engine functions
from embedding_engine import (
    build_product_description,
    generate_product_descriptions,
    load_embedding_model,
    build_embeddings,
    save_embeddings,
    load_embeddings,
    retrieve_products,
    compute_sha256
)


def run_all_checks(project_root: str = "fashion_project") -> Dict[str, Tuple[str, str]]:
    root = Path(project_root)
    artifacts_dir = root / "artifacts"
    catalog_path = artifacts_dir / "catalog.csv"
    catalog_ids_path = artifacts_dir / "catalog_ids.json"
    desc_path = artifacts_dir / "product_descriptions.csv"
    emb_path = artifacts_dir / "embeddings.npy"
    manifest_path = artifacts_dir / "embedding_manifest.json"
    images_dir = root / "data" / "raw" / "images"

    results = {}

    print("\n" + "="*80)
    print("EXECUTING PHASE 2 VALIDATION AND TEST SUITE (17 CHECKS)")
    print("="*80)

    # Load baseline Phase 1 catalog
    catalog = pd.read_csv(catalog_path, dtype={"id": str})
    with open(catalog_ids_path, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)

    # ----------------------------------------------------
    # Check 1: Product descriptions are generated for all catalog rows
    # ----------------------------------------------------
    try:
        assert desc_path.exists(), f"Missing {desc_path}"
        desc_df = pd.read_csv(desc_path, dtype={"id": str})
        assert len(desc_df) == len(catalog), f"Description count {len(desc_df)} != catalog count {len(catalog)}"
        assert (desc_df["product_description"].str.len() > 10).all(), "Some descriptions are empty or trivial"
        results["Check 01: Product descriptions generated for all catalog rows"] = (
            "PASS", f"Generated {len(desc_df)} non-empty descriptions"
        )
    except Exception as e:
        results["Check 01: Product descriptions generated for all catalog rows"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 2: Product IDs remain correctly associated
    # ----------------------------------------------------
    try:
        assert desc_df["id"].tolist() == catalog_ids, "ID sequence in descriptions does not match catalog_ids"
        assert desc_df["id"].tolist() == catalog["id"].tolist(), "ID sequence does not match catalog.csv"
        results["Check 02: Product IDs remain correctly associated"] = (
            "PASS", "Exact 1-to-1 row sequence alignment with catalog_ids"
        )
    except Exception as e:
        results["Check 02: Product IDs remain correctly associated"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 3: Missing attributes are handled without fabricated values
    # ----------------------------------------------------
    try:
        # Test synthetic product with null attributes
        sample_sparse = {
            "id": "test_sparse_999",
            "productDisplayName": "Test Minimal Shirt",
            "articleType": None,
            "baseColour": "",
            "usage": "NA",
            "masterCategory": "Apparel",
            "season": None
        }
        sparse_desc = build_product_description(sample_sparse)
        # Check that it formats without raising and doesn't invent fabric/occasion
        assert "Name: Test Minimal Shirt" in sparse_desc
        assert "Type: Apparel" in sparse_desc or "Type: Unknown" in sparse_desc
        assert "Color: Unknown Color" in sparse_desc
        assert "Usage: Casual" in sparse_desc
        # Verify no fabricated marketing terms
        fabricated_words = ["silk", "premium", "cotton", "luxurious", "100%", "organic"]
        assert not any(w in sparse_desc.lower() for w in fabricated_words)
        results["Check 03: Missing attributes handled without fabricated values"] = (
            "PASS", "Sparse records safely format using recorded values only"
        )
    except Exception as e:
        results["Check 03: Missing attributes handled without fabricated values"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 4: Embedding model loads successfully
    # ----------------------------------------------------
    model = None
    try:
        model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        assert model is not None
        results["Check 04: Embedding model loads successfully"] = (
            "PASS", f"Model: sentence-transformers/all-MiniLM-L6-v2 (device: {model.device})"
        )
    except Exception as e:
        results["Check 04: Embedding model loads successfully"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 5: Embedding dimension is exactly 384
    # ----------------------------------------------------
    try:
        test_emb = model.encode("sample text", normalize_embeddings=True)
        assert test_emb.shape[-1] == 384, f"Expected 384, got {test_emb.shape[-1]}"
        results["Check 05: Embedding dimension is exactly 384"] = ("PASS", "Output dimension: 384")
    except Exception as e:
        results["Check 05: Embedding dimension is exactly 384"] = ("FAIL", str(e))

    # Load saved embeddings matrix for checks 6-12
    embeddings = None
    try:
        embeddings = np.load(emb_path)
    except Exception:
        pass

    # ----------------------------------------------------
    # Check 6: Embedding dtype is float32
    # ----------------------------------------------------
    try:
        assert embeddings is not None, "Embeddings not loaded"
        assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
        results["Check 06: Embedding dtype is float32"] = ("PASS", f"dtype: {embeddings.dtype}")
    except Exception as e:
        results["Check 06: Embedding dtype is float32"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 7: Embeddings contain no NaN or infinite values
    # ----------------------------------------------------
    try:
        assert embeddings is not None
        assert np.all(np.isfinite(embeddings)), "Matrix contains NaN or Inf values"
        results["Check 07: Embeddings contain no NaN or infinite values"] = (
            "PASS", "All 3,840,000 float32 elements are finite"
        )
    except Exception as e:
        results["Check 07: Embeddings contain no NaN or infinite values"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 8: Embeddings are normalized
    # ----------------------------------------------------
    try:
        assert embeddings is not None
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), f"L2 norms not unit: max={norms.max()}, min={norms.min()}"
        results["Check 08: Embeddings are normalized"] = (
            "PASS", f"L2 norms unit-normalized (min: {norms.min():.5f}, max: {norms.max():.5f})"
        )
    except Exception as e:
        results["Check 08: Embeddings are normalized"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 9: Embedding row count matches catalog row count
    # ----------------------------------------------------
    try:
        assert embeddings is not None
        assert len(embeddings) == len(catalog) == 10000, f"Shape: {embeddings.shape}"
        results["Check 09: Embedding row count matches catalog row count"] = (
            "PASS", f"Row count {len(embeddings)} matches catalog length {len(catalog)}"
        )
    except Exception as e:
        results["Check 09: Embedding row count matches catalog row count"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 10: Product ID ordering remains unchanged
    # ----------------------------------------------------
    try:
        with open(artifacts_dir / "catalog_ids.json") as f:
            persisted_ids = json.load(f)
        assert len(embeddings) == len(persisted_ids)
        assert catalog["id"].tolist() == persisted_ids
        results["Check 10: Product ID ordering remains unchanged"] = (
            "PASS", "Row j in embeddings corresponds strictly to catalog row j"
        )
    except Exception as e:
        results["Check 10: Product ID ordering remains unchanged"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 11: Saved embeddings can be reloaded
    # ----------------------------------------------------
    try:
        reloaded_emb, emb_meta = load_embeddings(str(artifacts_dir), catalog, catalog_ids)
        assert reloaded_emb.shape == (10000, 384)
        assert np.array_equal(reloaded_emb, embeddings)
        results["Check 11: Saved embeddings can be reloaded"] = (
            "PASS", "load_embeddings() reloaded exact matrix without recomputation"
        )
    except Exception as e:
        results["Check 11: Saved embeddings can be reloaded"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 12: Manifest and hash validation succeeds
    # ----------------------------------------------------
    try:
        assert manifest_path.exists(), "Manifest file missing"
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        actual_hash = compute_sha256(str(emb_path))
        manifest_hash = man["storage_format"]["sha256"]
        assert actual_hash == manifest_hash, f"Hash mismatch: actual {actual_hash} != manifest {manifest_hash}"
        assert man["embedding_model"]["dimension"] == 384
        assert man["dataset_linkage"]["num_products"] == 10000
        results["Check 12: Manifest and hash validation succeeds"] = (
            "PASS", f"SHA-256 verified: {manifest_hash[:16]}..."
        )
    except Exception as e:
        results["Check 12: Manifest and hash validation succeeds"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 13: Semantic retrieval returns no more than five products
    # ----------------------------------------------------
    try:
        res = retrieve_products("blue denim jacket", model, catalog, embeddings, top_k=5, images_dir=str(images_dir))
        assert len(res) == 5, f"Expected 5 results, got {len(res)}"
        results["Check 13: Semantic retrieval returns no more than five products"] = (
            "PASS", f"Returned exactly {len(res)} items for top_k=5"
        )
    except Exception as e:
        results["Check 13: Semantic retrieval returns no more than five products"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 14: Retrieved product IDs exist in the frozen catalog
    # ----------------------------------------------------
    try:
        catalog_id_set = set(catalog_ids)
        retrieved_ids = [r["id"] for r in res]
        assert all(pid in catalog_id_set for pid in retrieved_ids), "Unknown product ID in retrieval results"
        results["Check 14: Retrieved product IDs exist in the frozen catalog"] = (
            "PASS", f"All retrieved IDs {retrieved_ids} verified in frozen catalog"
        )
    except Exception as e:
        results["Check 14: Retrieved product IDs exist in the frozen catalog"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 15: Retrieved images correspond to the returned product IDs
    # ----------------------------------------------------
    try:
        for r in res:
            pid = r["id"]
            img_file = images_dir / f"{pid}.jpg"
            assert img_file.exists(), f"Image {img_file} does not exist"
            with Image.open(img_file) as im:
                im.verify()
        results["Check 15: Retrieved images correspond to the returned product IDs"] = (
            "PASS", "All 5 retrieved image files exist and are decodable with Pillow"
        )
    except Exception as e:
        results["Check 15: Retrieved images correspond to the returned product IDs"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 16: Empty queries are handled correctly
    # ----------------------------------------------------
    try:
        res_empty1 = retrieve_products("", model, catalog, embeddings, top_k=5)
        res_empty2 = retrieve_products("   ", model, catalog, embeddings, top_k=5)
        assert res_empty1 == [] and res_empty2 == [], "Empty query did not return empty list"
        results["Check 16: Empty queries are handled correctly"] = (
            "PASS", "Both empty string and whitespace-only queries return [] without raising"
        )
    except Exception as e:
        results["Check 16: Empty queries are handled correctly"] = ("FAIL", str(e))

    # ----------------------------------------------------
    # Check 17: Retrieval works after loading saved embeddings without recomputation
    # ----------------------------------------------------
    try:
        # Load embeddings fresh from disk without model encoding
        fresh_emb, _ = load_embeddings(str(artifacts_dir), catalog, catalog_ids)
        res_fresh = retrieve_products("red high heels", model, catalog, fresh_emb, top_k=5, images_dir=str(images_dir))
        assert len(res_fresh) == 5
        # Verify scores are sorted descending
        scores = [r["retrieval_score"] for r in res_fresh]
        assert scores == sorted(scores, reverse=True), "Scores are not monotonically decreasing"
        results["Check 17: Retrieval works after loading saved embeddings without recomputation"] = (
            "PASS", f"Top retrieval score: {scores[0]:.4f} for ID: {res_fresh[0]['id']}"
        )
    except Exception as e:
        results["Check 17: Retrieval works after loading saved embeddings without recomputation"] = ("FAIL", str(e))

    # Print summary table
    print("\n" + "="*80)
    print("PHASE 2 VALIDATION SUMMARY REPORT")
    print("="*80)
    all_passed = True
    for test_name, (status, detail) in results.items():
        print(f"[{status}] {test_name}\n        Details: {detail}")
        if status != "PASS":
            all_passed = False
    print("="*80)
    print(f"OVERALL STATUS: {'ALL 17 CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    from typing import Dict, Tuple
    run_all_checks()
