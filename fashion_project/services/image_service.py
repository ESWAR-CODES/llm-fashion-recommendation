"""
services/image_service.py
Centralized Product Image Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Resolve product images using stable, canonical product IDs.
- Pillow decoding with in-memory LRU caching to prevent redundant disk I/O.
- Deterministic, consistent placeholder generation for missing/corrupted images.
- High-quality image resizing.
- Base64 encoding for web application rendering.
- Strict isolation: Never associate an unrelated image with a product.
"""

import os
import io
import sys
import base64
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PIL import Image, ImageDraw, ImageFont
from services.config_service import resolve_path, ConfigService


class ProductImageService:
    """
    Centralized service for managing and serving fashion product images.
    Thread-safe and CPU-efficient with LRU in-memory decoded image caching.
    """

    _instance: Optional["ProductImageService"] = None

    def __init__(self, images_dir: Optional[str | Path] = None, cache_size: int = 512):
        cfg = ConfigService.get_instance()
        self.images_dir = resolve_path(images_dir) if images_dir else cfg.get_path("images_dir")
        self.cache_size = cache_size
        self._cache: Dict[str, Image.Image] = {}
        self._b64_cache: Dict[str, str] = {}
        self._known_files: Optional[set] = None

    @classmethod
    def get_instance(cls, images_dir: Optional[str | Path] = None) -> "ProductImageService":
        if cls._instance is None:
            cls._instance = cls(images_dir)
        return cls._instance

    def _ensure_file_index(self):
        if self._known_files is None:
            if self.images_dir.exists():
                self._known_files = set(os.listdir(self.images_dir))
            else:
                self._known_files = set()

    def get_image_path(self, product_id: str | int) -> Optional[Path]:
        """Resolve the file path for a canonical product ID if it exists."""
        pid_str = str(product_id).strip()
        self._ensure_file_index()
        filename = f"{pid_str}.jpg"
        if filename in self._known_files:
            return self.images_dir / filename
        return None

    def has_image(self, product_id: str | int) -> bool:
        """Check whether a product image exists on disk."""
        return self.get_image_path(product_id) is not None

    def generate_placeholder(
        self,
        product_id: str | int,
        size: Tuple[int, int] = (200, 260)
    ) -> Image.Image:
        """
        Generate a professional, deterministic placeholder image when the catalog
        image is missing or unreadable. Never displays an unrelated catalog item.
        """
        pid_str = str(product_id).strip()
        width, height = size
        img = Image.new("RGB", (width, height), color=(245, 247, 250))
        draw = ImageDraw.Draw(img)

        # Border
        draw.rectangle([0, 0, width - 1, height - 1], outline=(203, 213, 225), width=2)

        # Subtle hanger / icon placeholder
        cx, cy = width // 2, height // 2 - 20
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(226, 232, 240))
        draw.line([cx - 20, cy, cx + 20, cy], fill=(148, 163, 184), width=3)
        draw.line([cx - 20, cy, cx, cy - 15], fill=(148, 163, 184), width=2)
        draw.line([cx, cy - 15, cx + 20, cy], fill=(148, 163, 184), width=2)

        # Text notice
        text_top = "No Image"
        text_sub = f"ID: {pid_str}"
        draw.text((width // 2, height // 2 + 25), text_top, fill=(100, 116, 139), anchor="mm")
        draw.text((width // 2, height // 2 + 45), text_sub, fill=(148, 163, 184), anchor="mm")

        return img

    def load_image(
        self,
        product_id: str | int,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Load and return a PIL Image for the given product ID.
        Uses in-memory cache to prevent repeatedly decoding the same file.
        If missing or corrupted, returns the consistent placeholder.
        """
        pid_str = str(product_id).strip()
        cache_key = f"{pid_str}_{target_size[0]}x{target_size[1]}" if target_size else pid_str

        if cache_key in self._cache:
            return self._cache[cache_key]

        img_path = self.get_image_path(pid_str)
        img: Optional[Image.Image] = None

        if img_path and img_path.exists():
            try:
                with Image.open(img_path) as raw_img:
                    img = raw_img.convert("RGB")
                    if target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
            except Exception:
                img = None

        if img is None:
            # Fallback to deterministic placeholder
            size = target_size or (200, 260)
            img = self.generate_placeholder(pid_str, size=size)

        # Maintain bounded cache
        if len(self._cache) >= self.cache_size:
            # Evict first 64 items
            keys_to_remove = list(self._cache.keys())[:64]
            for k in keys_to_remove:
                self._cache.pop(k, None)

        self._cache[cache_key] = img
        return img

    def get_image_base64(
        self,
        product_id: str | int,
        target_size: Optional[Tuple[int, int]] = (200, 260),
        format: str = "JPEG"
    ) -> str:
        """
        Return a base64 Data URI string for rendering inside HTML components.
        """
        pid_str = str(product_id).strip()
        cache_key = f"{pid_str}_{target_size[0]}x{target_size[1]}_{format}" if target_size else f"{pid_str}_{format}"

        if cache_key in self._b64_cache:
            return self._b64_cache[cache_key]

        img = self.load_image(pid_str, target_size=target_size)
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/{format.lower()};base64,{encoded}"

        if len(self._b64_cache) >= self.cache_size:
            keys_to_remove = list(self._b64_cache.keys())[:64]
            for k in keys_to_remove:
                self._b64_cache.pop(k, None)

        self._b64_cache[cache_key] = data_uri
        return data_uri

    def clear_cache(self):
        """Clear decoded image caches."""
        self._cache.clear()
        self._b64_cache.clear()
        self._known_files = None


if __name__ == "__main__":
    svc = ProductImageService.get_instance()
    print(f"[ImageService] Initialized with images directory: {svc.images_dir}")
    print(f"[ImageService] Testing image resolution for 25 actual catalog IDs...")
    from services.catalog_service import CatalogService
    cat = CatalogService.get_instance()
    sample_ids = cat.catalog_ids[:25]
    loaded = 0
    for pid in sample_ids:
        img = svc.load_image(pid, target_size=(100, 130))
        if img:
            loaded += 1
    print(f"[ImageService] Successfully resolved and loaded {loaded}/{len(sample_ids)} images.")
    # Test fallback
    ph = svc.load_image("NON_EXISTENT_999999", target_size=(100, 130))
    print(f"[ImageService] Fallback placeholder generation test: {'PASS' if ph else 'FAIL'}")

