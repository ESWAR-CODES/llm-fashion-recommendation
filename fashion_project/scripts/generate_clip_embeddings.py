"""
scripts/generate_clip_embeddings.py
Offline Batch CLIP Image Embedding Generator & Artifact Validator
LLM-Assisted Personalized Fashion Recommendation System

Usage:
    python fashion_project/scripts/generate_clip_embeddings.py [--device cpu|cuda] [--batch_size 128] [--limit N]
    python fashion_project/scripts/generate_clip_embeddings.py --validate
"""

import os
import sys
import json
import time
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Add fashion_project to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from services.config_service import resolve_path, ConfigService
from services.catalog_service import CatalogService
from services.image_service import ProductImageService


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


def validate_clip_artifacts(
    embeddings_path: Optional[str | Path] = None,
    manifest_path: Optional[str | Path] = None,
    catalog_ids_path: Optional[str | Path] = None,
    expected_dim: int = 512
) -> Tuple[bool, str]:
    """
    Rigorously validate CLIP embedding artifacts according to Section 3 specifications:
    - Verifies file presence
    - Validates dimensions (N, 512)
    - Rejects NaN or Inf
    - Rejects unnormalized vectors (||v|| != 1.0)
    - Rejects catalog and embedding order mismatches
    - Rejects hash discrepancies
    """
    cfg = ConfigService.get_instance()
    emb_p = resolve_path(embeddings_path) if embeddings_path else cfg.get_path("clip_embeddings_npy")
    alt_emb_p = cfg.artifacts_dir / "clip_image_embeddings.npy"
    if not emb_p.exists() and alt_emb_p.exists():
        emb_p = alt_emb_p

    man_p = resolve_path(manifest_path) if manifest_path else cfg.get_path("clip_manifest_json")
    ids_p = resolve_path(catalog_ids_path) if catalog_ids_path else cfg.artifacts_dir / "clip_catalog_ids.json"
    official_ids_p = cfg.get_path("catalog_ids_json")

    if not emb_p.exists():
        return False, f"CLIP embeddings artifact does not exist: {emb_p}"

    if not man_p.exists():
        return False, f"CLIP manifest does not exist: {man_p}"

    # Load matrix
    try:
        matrix = np.load(str(emb_p))
    except Exception as e:
        return False, f"Failed to load numpy matrix: {e}"

    if matrix.ndim != 2:
        return False, f"Expected 2D matrix, got shape {matrix.shape}"

    if matrix.shape[1] != expected_dim:
        return False, f"Incorrect vector dimensions: expected {expected_dim}, got {matrix.shape[1]}"

    if not np.all(np.isfinite(matrix)):
        return False, "Matrix contains NaN or Inf values!"

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        return False, f"Embeddings not unit-normalized: norm range [{norms.min():.5f}, {norms.max():.5f}]"

    # Load manifest
    try:
        with open(man_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Failed to parse manifest JSON: {e}"

    if manifest.get("vector_dimensions") != expected_dim:
        return False, f"Manifest dimension mismatch: {manifest.get('vector_dimensions')} != {expected_dim}"

    if manifest.get("embedding_count") != len(matrix):
        return False, f"Manifest count mismatch: {manifest.get('embedding_count')} != {len(matrix)}"

    # Check ID ordering if clip_catalog_ids.json exists
    if ids_p.exists():
        with open(ids_p, "r", encoding="utf-8") as f:
            clip_ids = json.load(f)
        if len(clip_ids) != len(matrix):
            return False, f"clip_catalog_ids.json count ({len(clip_ids)}) != matrix rows ({len(matrix)})"
        with open(official_ids_p, "r", encoding="utf-8") as f:
            official_ids = json.load(f)
        if clip_ids != official_ids[:len(clip_ids)]:
            return False, "Ordering mismatch between clip_catalog_ids.json and frozen catalog_ids.json!"

    # Verify SHA256 matches manifest if recorded
    actual_hash = compute_sha256(emb_p)
    recorded_hash = manifest.get("sha256")
    if recorded_hash and actual_hash != recorded_hash:
        return False, f"SHA256 checksum mismatch: recorded {recorded_hash}, actual {actual_hash}"

    return True, f"Validation PASSED: {len(matrix)} valid normalized 512-d CLIP vectors verified."


def generate_clip_embeddings(
    device: str = "cpu",
    batch_size: int = 128,
    limit: Optional[int] = None,
    output_npy: Optional[str | Path] = None,
    output_manifest: Optional[str | Path] = None,
    model_name: str = "sentence-transformers/clip-ViT-B-32"
):
    print("=" * 80)
    print("CLIP VISUAL EMBEDDINGS OFFLINE GENERATOR")
    print(f"Model: {model_name} | Device: {device} | Batch Size: {batch_size}")
    print("=" * 80)

    # Maximize CPU efficiency if on CPU
    if device == "cpu":
        try:
            import torch
            avail_threads = os.cpu_count() or 8
            torch.set_num_threads(avail_threads)
            print(f"[INFO] Configured PyTorch CPU threads: {avail_threads}")
        except ImportError:
            pass

    cfg = ConfigService.get_instance()
    out_npy_path = resolve_path(output_npy) if output_npy else cfg.get_path("clip_embeddings_npy")
    alt_out_npy_path = cfg.artifacts_dir / "clip_image_embeddings.npy"
    out_manifest_path = resolve_path(output_manifest) if output_manifest else cfg.get_path("clip_manifest_json")
    out_ids_path = cfg.artifacts_dir / "clip_catalog_ids.json"

    cat_svc = CatalogService.get_instance()
    catalog_ids = cat_svc.catalog_ids
    if limit:
        catalog_ids = catalog_ids[:limit]
        print(f"[INFO] Limiting generation to first {limit} catalog products.")

    total_items = len(catalog_ids)
    print(f"[INFO] Preparing to encode {total_items} catalog images in exact row sequence...")

    # Load SentenceTransformer CLIP model
    from sentence_transformers import SentenceTransformer
    start_time = time.time()
    print(f"[INFO] Loading {model_name} onto {device}...")
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    img_svc = ProductImageService.get_instance()
    all_embeddings = []
    failed_images: List[str] = []

    # Process in batches
    for i in range(0, total_items, batch_size):
        batch_ids = catalog_ids[i:i + batch_size]
        batch_images = []
        for pid in batch_ids:
            # Check original image existence
            raw_path = img_svc.get_image_path(pid)
            if not raw_path or not raw_path.exists():
                failed_images.append(str(pid))
            # Load with deterministic fallback placeholder if missing or corrupt
            img = img_svc.load_image(pid, target_size=(224, 224))
            batch_images.append(img)

        # Encode batch with normalized embeddings
        batch_vecs = model.encode(
            batch_images,
            batch_size=len(batch_images),
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        all_embeddings.append(batch_vecs)
        done = min(i + batch_size, total_items)
        pct = (done / total_items) * 100
        elapsed_so_far = time.time() - start_time
        rate = done / max(elapsed_so_far, 0.001)
        remaining_secs = (total_items - done) / max(rate, 0.001)
        print(f"      - Encoded {done}/{total_items} images ({pct:.1f}%) | {rate:.1f} img/s | ETA: {remaining_secs:.1f}s")

    embeddings_matrix = np.vstack(all_embeddings)

    # Invariant validation
    assert embeddings_matrix.shape == (total_items, 512), f"Unexpected shape {embeddings_matrix.shape}"
    assert np.all(np.isfinite(embeddings_matrix)), "Matrix contains NaN or Inf values!"
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Not unit-normalized: [{norms.min()}, {norms.max()}]"

    # Save artifacts atomically
    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy_path), embeddings_matrix)
    np.save(str(alt_out_npy_path), embeddings_matrix)

    # Save stable clip_catalog_ids.json
    with open(out_ids_path, "w", encoding="utf-8") as f:
        json.dump(catalog_ids, f, indent=2)

    sha256 = compute_sha256(out_npy_path)
    ids_hash = compute_sha256(out_ids_path)
    catalog_hash = compute_sha256(cfg.get_path("catalog_csv"))
    size_mb = out_npy_path.stat().st_size / (1024 * 1024)

    # Write authoritative manifest conforming to Section 3 requirements
    manifest = {
        "model_identifier": model_name,
        "model_revision": "main",
        "vector_dimensions": 512,
        "embedding_count": total_items,
        "image_preprocessing_config": {
            "target_size": [224, 224],
            "resampling": "LANCZOS",
            "channels": 3,
            "color_space": "RGB",
            "normalized": True
        },
        "catalog_hash": catalog_hash,
        "product_id_ordering_hash": ids_hash,
        "generation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_used": device,
        "batch_size": batch_size,
        "failed_image_count": len(failed_images),
        "failed_image_ids": failed_images,
        "sha256": sha256,
        "file_size_mb": round(size_mb, 2),
        "source_catalog": str(cfg.get_path("catalog_csv"))
    }
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[SUCCESS] Saved {total_items} CLIP embeddings to {out_npy_path} ({size_mb:.2f} MB)")
    print(f"[SUCCESS] Also synced to {alt_out_npy_path}")
    print(f"[SUCCESS] Product ID sequence saved to {out_ids_path}")
    print(f"[SUCCESS] Manifest saved to {out_manifest_path} (Time: {elapsed:.2f}s)")
    print("=" * 80)

    # Run verification immediately
    ok, msg = validate_clip_artifacts(out_npy_path, out_manifest_path, out_ids_path)
    if not ok:
        raise RuntimeError(f"Generated artifacts failed validation: {msg}")
    print(f"[VALIDATION] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and validate offline CLIP embeddings for product images.")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for test runs")
    parser.add_argument("--validate", action="store_true", help="Validate existing CLIP artifacts on disk")
    args = parser.parse_args()

    if args.validate:
        ok, msg = validate_clip_artifacts()
        print(f"[RESULT] {msg}")
        sys.exit(0 if ok else 1)

    generate_clip_embeddings(device=args.device, batch_size=args.batch_size, limit=args.limit)
