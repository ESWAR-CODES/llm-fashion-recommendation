"""
scripts/generate_clip_embeddings.py
Offline Batch CLIP Image Embedding Generator
LLM-Assisted Personalized Fashion Recommendation System

Usage:
    python fashion_project/scripts/generate_clip_embeddings.py [--device cpu|cuda] [--batch_size 64] [--limit N]
"""

import os
import sys
import json
import time
import hashlib
import argparse
import datetime
from pathlib import Path

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
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


def generate_clip_embeddings(
    device: str = "cpu",
    batch_size: int = 64,
    limit: Optional[int] = None,
    output_npy: Optional[str | Path] = None,
    output_manifest: Optional[str | Path] = None,
    model_name: str = "sentence-transformers/clip-ViT-B-32"
):
    print("=" * 80)
    print("CLIP VISUAL EMBEDDINGS OFFLINE GENERATOR")
    print(f"Model: {model_name} | Device: {device} | Batch Size: {batch_size}")
    print("=" * 80)

    cfg = ConfigService.get_instance()
    out_npy_path = resolve_path(output_npy) if output_npy else cfg.get_path("clip_embeddings_npy")
    out_manifest_path = resolve_path(output_manifest) if output_manifest else cfg.get_path("clip_manifest_json")

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

    # Process in batches
    for i in range(0, total_items, batch_size):
        batch_ids = catalog_ids[i:i + batch_size]
        batch_images = []
        for pid in batch_ids:
            # Load with fallback placeholder if missing
            img = img_svc.load_image(pid, target_size=(224, 224))
            batch_images.append(img)

        # Encode batch
        batch_vecs = model.encode(
            batch_images,
            batch_size=len(batch_images),
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        all_embeddings.append(batch_vecs)
        done = min(i + batch_size, total_items)
        pct = (done / total_items) * 100
        print(f"      - Processed {done}/{total_items} images ({pct:.1f}%)...")

    embeddings_matrix = np.vstack(all_embeddings)

    # Invariant validation
    assert embeddings_matrix.shape == (total_items, 512), f"Unexpected shape {embeddings_matrix.shape}"
    assert np.all(np.isfinite(embeddings_matrix)), "Matrix contains NaN or Inf values!"
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Not unit-normalized: [{norms.min()}, {norms.max()}]"

    # Save artifact
    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy_path), embeddings_matrix)
    sha256 = compute_sha256(out_npy_path)
    size_mb = out_npy_path.stat().st_size / (1024 * 1024)

    # Write manifest
    manifest = {
        "model_name": model_name,
        "dimension": 512,
        "total_items": total_items,
        "device_used": device,
        "normalized": True,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256": sha256,
        "file_size_mb": round(size_mb, 2),
        "source_catalog": str(cfg.get_path("catalog_csv"))
    }
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[SUCCESS] Saved {total_items} CLIP embeddings to {out_npy_path} ({size_mb:.2f} MB)")
    print(f"[SUCCESS] Manifest saved to {out_manifest_path} (Time: {elapsed:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline CLIP embeddings for product images.")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for test runs")
    args = parser.parse_args()

    generate_clip_embeddings(device=args.device, batch_size=args.batch_size, limit=args.limit)
