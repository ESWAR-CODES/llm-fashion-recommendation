"""
embedding_engine.py
Phase 2: Product Text Embeddings & Semantic Retrieval
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Steps 3 & 4)
"""

import os
import sys
import json
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Global cache for the embedding model to avoid repeated initialization
_CACHED_MODEL = None
_CACHED_MODEL_NAME = None


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


def build_product_description(product: Any) -> str:
    """
    Generate a concise textual representation using only actual recorded product metadata.
    Specification Format:
    Name: {productDisplayName}
    Type: {articleType} | Color: {baseColour} | Usage: {usage}
    Category: {masterCategory} | Season: {season}

    Handles missing/unrecorded fields gracefully without inventing attributes.
    """
    def _get(field_name: str, fallback: str = "Unknown") -> str:
        if isinstance(product, dict):
            val = product.get(field_name)
        elif hasattr(product, "__getitem__"):
            try:
                val = product[field_name]
            except (KeyError, IndexError):
                val = getattr(product, field_name, None)
        else:
            val = getattr(product, field_name, None)

        if val is None or pd.isna(val) or str(val).strip() in ("", "nan", "None", "NA"):
            return fallback
        return str(val).strip()

    name = _get("productDisplayName", "Unknown Product")
    article_type = _get("articleType", "Apparel")
    color = _get("baseColour", "Unknown Color")
    usage = _get("usage", "Casual")
    category = _get("masterCategory", "Apparel")
    season = _get("season", "All-Season")

    description = (
        f"Name: {name}\n"
        f"Type: {article_type} | Color: {color} | Usage: {usage}\n"
        f"Category: {category} | Season: {season}"
    )
    return description


def generate_product_descriptions(
    catalog: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate descriptions for every catalog item in exact catalog row order.
    Saves to artifacts/product_descriptions.csv if output_path is provided.
    """
    if "id" not in catalog.columns:
        raise ValueError("Catalog DataFrame must contain an 'id' column.")

    descriptions = [build_product_description(row) for _, row in catalog.iterrows()]
    
    desc_df = pd.DataFrame({
        "id": catalog["id"].astype(str).tolist(),
        "productDisplayName": catalog["productDisplayName"].tolist(),
        "product_description": descriptions
    })

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        desc_df.to_csv(out_p, index=False)
        print(f"[INFO] Product descriptions saved to: {out_p} ({len(desc_df)} records)")

    return desc_df


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None
):
    """
    Load the sentence-transformers model once with proper hardware detection.
    Supports CPU and CUDA. Centralizes model initialization and inference configuration.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME

    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name:
        return _CACHED_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Please run: pip install sentence-transformers"
        ) from e

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Initializing embedding model '{model_name}' on device '{device}'...")
    try:
        model = SentenceTransformer(model_name, device=device)
        model.eval()  # Inference mode
        _CACHED_MODEL = model
        _CACHED_MODEL_NAME = model_name
        print(f"[SUCCESS] Model loaded successfully. Dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model '{model_name}': {e}")
        raise


def build_embeddings(
    catalog: pd.DataFrame,
    model: Any,
    batch_size: int = 64,
    descriptions: Optional[List[str]] = None
) -> np.ndarray:
    """
    Encode product descriptions into 384-dimensional normalized float32 embeddings.
    Preserves exact catalog row ordering.
    """
    if descriptions is None:
        print("[INFO] Generating product descriptions from catalog...")
        descriptions = [build_product_description(row) for _, row in catalog.iterrows()]

    if len(descriptions) != len(catalog):
        raise ValueError(
            f"Descriptions count ({len(descriptions)}) does not match catalog length ({len(catalog)})"
        )

    print(f"[INFO] Encoding {len(descriptions):,} product descriptions (batch_size={batch_size})...")
    # Generate embeddings with SentenceTransformer, normalized directly
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Ensure float32 dtype
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Rigorous validation assertions
    if len(embeddings) != len(catalog):
        raise AssertionError(f"Embedding count ({len(embeddings)}) != catalog count ({len(catalog)})")
    if embeddings.shape[1] != 384:
        raise AssertionError(f"Expected embedding dimension 384, got {embeddings.shape[1]}")
    if not np.all(np.isfinite(embeddings)):
        raise AssertionError("Embedding matrix contains NaN or Inf values!")

    # Check normalization within tolerance (norm should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise AssertionError(f"Embeddings are not unit-normalized! Max norm: {norms.max()}, Min norm: {norms.min()}")

    print(f"[SUCCESS] Generated embedding matrix with shape {embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    catalog: pd.DataFrame,
    output_dir: str = "fashion_project/artifacts",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
) -> Tuple[str, str]:
    """
    Save the embedding matrix to embeddings.npy and write embedding_manifest.json
    with complete audit metadata, catalog manifest linkage, and SHA-256 hashes.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    manifest_path = out_dir / "embedding_manifest.json"
    catalog_manifest_path = out_dir / "catalog_manifest.json"
    desc_path = out_dir / "product_descriptions.csv"

    # Get catalog manifest hash if available
    catalog_manifest_hash = "UNKNOWN"
    if catalog_manifest_path.exists():
        catalog_manifest_hash = compute_sha256(str(catalog_manifest_path))

    # Save float32 numpy array
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"[INFO] Embeddings array saved to: {emb_path}")

    # Compute SHA-256
    emb_sha256 = compute_sha256(str(emb_path))
    desc_sha256 = compute_sha256(str(desc_path)) if desc_path.exists() else "NOT_SAVED"

    manifest_data = {
        "project_title": "LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval",
        "phase": 2,
        "manifest_version": "1.0.0",
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedding_model": {
            "name": model_name,
            "revision": "default",
            "dimension": int(embeddings.shape[1]),
            "normalize_embeddings": True
        },
        "dataset_linkage": {
            "num_products": len(embeddings),
            "catalog_item_count": len(catalog),
            "catalog_manifest_sha256": catalog_manifest_hash
        },
        "storage_format": {
            "file": "embeddings.npy",
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape),
            "sha256": emb_sha256
        },
        "description_artifact": {
            "file": "product_descriptions.csv",
            "sha256": desc_sha256,
            "template": (
                "Name: {productDisplayName}\\n"
                "Type: {articleType} | Color: {baseColour} | Usage: {usage}\\n"
                "Category: {masterCategory} | Season: {season}"
            )
        },
        "generation_config": {
            "batch_size": batch_size,
            "random_seed": 42
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[SUCCESS] Embedding manifest written to: {manifest_path}")
    return str(emb_path), str(manifest_path)


def load_embeddings(
    artifacts_dir: str = "fashion_project/artifacts",
    catalog: Optional[pd.DataFrame] = None,
    catalog_ids: Optional[List[str]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load persisted embedding matrix and verify alignment with the catalog and manifest.
    Rejects artifacts if row count, ordering, or file hashes are inconsistent.
    """
    art_dir = Path(artifacts_dir)
    emb_path = art_dir / "embeddings.npy"
    manifest_path = art_dir / "embedding_manifest.json"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify SHA-256 hash of embeddings.npy
    current_hash = compute_sha256(str(emb_path))
    expected_hash = manifest.get("storage_format", {}).get("sha256")
    if expected_hash and current_hash != expected_hash:
        raise ValueError(f"Embeddings file hash mismatch! Expected {expected_hash}, got {current_hash}")

    embeddings = np.load(emb_path)

    # Verify shape and properties
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}. Expected (N, 384)")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if catalog is not None and len(embeddings) != len(catalog):
        raise ValueError(f"Embeddings length ({len(embeddings)}) != catalog length ({len(catalog)})")
    if catalog_ids is not None and len(embeddings) != len(catalog_ids):
        raise ValueError(f"Embeddings length ({len(embeddings)}) != catalog_ids length ({len(catalog_ids)})")

    return embeddings, manifest


def retrieve_products(
    query: str,
    model: Any,
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
    images_dir: Optional[str] = "fashion_project/data/raw/images"
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k relevant catalog products for a natural language query using
    cosine similarity (dot product over L2-normalized vectors).

    Returns structured product dictionaries including metadata, retrieval scores,
    and resolved image file paths.
    """
    if not query or not query.strip():
        return []

    # Clamp top_k to available items
    actual_k = min(max(1, top_k), len(catalog))

    # Encode and normalize query vector
    q_vec = model.encode(
        query.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    # Dot product similarity (equivalent to cosine similarity since vectors are unit-normalized)
    # embeddings: (N, 384), q_vec.T: (384, 1) -> scores: (N,)
    scores = np.dot(embeddings, q_vec.T).squeeze()

    # Get top_k indices sorted descending by retrieval score
    top_indices = np.argsort(scores)[::-1][:actual_k]

    results = []
    img_dir_path = Path(images_dir) if images_dir is not None else None

    for rank, idx in enumerate(top_indices, start=1):
        row = catalog.iloc[idx]
        pid = str(row["id"])
        img_path = str(img_dir_path / f"{pid}.jpg") if img_dir_path else None
        
        results.append({
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
            "retrieval_score": float(scores[idx]),
            "image_path": img_path,
            "image_exists": Path(img_path).exists() if img_path else False
        })

    return results


def run_phase2_pipeline(
    project_root: str = "fashion_project",
    batch_size: int = 64,
    force_recompute: bool = False
):
    """
    End-to-end execution of Phase 2:
    - Verifies Phase 1 catalog
    - Generates product_descriptions.csv
    - Loads MiniLM model
    - Computes or reloads embeddings.npy
    - Saves embedding_manifest.json
    - Demonstrates semantic search on sample queries
    """
    root = Path(project_root)
    artifacts_dir = root / "artifacts"
    catalog_path = artifacts_dir / "catalog.csv"
    catalog_ids_path = artifacts_dir / "catalog_ids.json"
    desc_path = artifacts_dir / "product_descriptions.csv"
    emb_path = artifacts_dir / "embeddings.npy"

    if not catalog_path.exists() or not catalog_ids_path.exists():
        raise FileNotFoundError("Phase 1 catalog artifacts not found. Please complete Phase 1 first.")

    print("\n" + "="*70)
    print("PHASE 2: PRODUCT TEXT EMBEDDINGS & SEMANTIC RETRIEVAL PIPELINE")
    print("="*70)

    # 1. Load catalog
    print("[1/5] Loading Phase 1 frozen catalog...")
    catalog = pd.read_csv(catalog_path, dtype={"id": str})
    with open(catalog_ids_path, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)

    assert len(catalog) == 10000, f"Expected 10,000 items, got {len(catalog)}"
    assert catalog["id"].tolist() == catalog_ids, "Catalog ID sequence mismatch!"
    print(f"  Catalog verified: {len(catalog):,} items, 10 columns.")

    # 2. Product Descriptions
    print("\n[2/5] Building product descriptions...")
    if desc_path.exists() and not force_recompute:
        print(f"  Loading existing descriptions from: {desc_path}")
        desc_df = pd.read_csv(desc_path, dtype={"id": str})
        descriptions = desc_df["product_description"].tolist()
    else:
        desc_df = generate_product_descriptions(catalog, str(desc_path))
        descriptions = desc_df["product_description"].tolist()
    print(f"  Sample description for ID {desc_df.iloc[0]['id']}:\n" + "-"*40)
    print(descriptions[0])
    print("-"*40)

    # 3. Model Loading
    print("\n[3/5] Loading embedding model...")
    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

    # 4. Embeddings Generation / Loading
    print("\n[4/5] Processing product embeddings...")
    manifest_path = artifacts_dir / "embedding_manifest.json"
    if emb_path.exists() and manifest_path.exists() and not force_recompute:
        print(f"  Existing embeddings found at {emb_path}. Verifying and loading...")
        try:
            embeddings, manifest = load_embeddings(str(artifacts_dir), catalog, catalog_ids)
            print(f"  Embeddings successfully verified and reloaded! Shape: {embeddings.shape}")
        except Exception as e:
            print(f"  [WARNING] Embedding verification failed ({e}). Recomputing...")
            embeddings = build_embeddings(catalog, model, batch_size=batch_size, descriptions=descriptions)
            save_embeddings(embeddings, catalog, str(artifacts_dir), batch_size=batch_size)
    else:
        embeddings = build_embeddings(catalog, model, batch_size=batch_size, descriptions=descriptions)
        save_embeddings(embeddings, catalog, str(artifacts_dir), batch_size=batch_size)

    # 5. Semantic Retrieval Demonstration
    print("\n[5/5] Demonstrating Semantic Retrieval...")
    test_queries = [
        "casual summer cotton t-shirt for men",
        "black formal leather shoes",
        "traditional ethnic wear for women"
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = retrieve_products(q, model, catalog, embeddings, top_k=3, images_dir=str(root / "data" / "raw" / "images"))
        for r in results:
            print(f"  [{r['rank']}] Score: {r['retrieval_score']:.4f} | ID: {r['id']} | {r['productDisplayName']} ({r['articleType']}, {r['baseColour']})")

    print("\n" + "="*70)
    print("[SUCCESS] Phase 2 Pipeline Finished Successfully!")
    print("="*70 + "\n")
    return catalog, embeddings, model


if __name__ == "__main__":
    run_phase2_pipeline()
