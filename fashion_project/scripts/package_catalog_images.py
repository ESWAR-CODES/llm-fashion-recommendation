"""
scripts/package_catalog_images.py
Isolated 10,000-Item Catalog Images Packager
LLM-Assisted Personalized Fashion Recommendation System

Extracts only the 10,000 images needed for the frozen retrieval catalog
from the 44,441 raw images directory into data/catalog_images/ (107 MB).
Allows committing or deploying only the necessary catalog images without
the entire 1.5 GB raw dataset.
"""

import sys
import json
import shutil
from pathlib import Path

# Add fashion_project to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config_service import ConfigService


def package_catalog_images():
    cfg = ConfigService.get_instance()
    ids_path = cfg.get_path("catalog_ids_json")
    raw_images_dir = cfg.get_path("images_dir")
    target_dir = cfg.project_root / "data" / "catalog_images"

    if not ids_path.exists():
        raise FileNotFoundError(f"Catalog IDs not found: {ids_path}")
    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Raw images dir not found: {raw_images_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with open(ids_path, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)

    print("=" * 80)
    print("PACKAGING 10,000 CATALOG IMAGES FOR PRODUCTION DEPLOYMENT")
    print(f"Source: {raw_images_dir}")
    print(f"Target: {target_dir}")
    print("=" * 80)

    copied = 0
    missing = 0
    for pid in catalog_ids:
        src = raw_images_dir / f"{pid}.jpg"
        dst = target_dir / f"{pid}.jpg"
        if src.exists():
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    total_mb = sum(f.stat().st_size for f in target_dir.glob("*.jpg")) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {copied} images to {target_dir} ({total_mb:.2f} MB)")
    if missing > 0:
        print(f"[WARN] {missing} images missing from source.")
    print("=" * 80)


if __name__ == "__main__":
    package_catalog_images()
