"""
services/clip_service.py
CLIP Visual Retrieval Foundation & Embedding Management
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Separate visual retrieval service using CLIP (ViT-B/32).
- Zero re-computation loading of precomputed image embeddings (10,000 x 512).
- Offline preprocessing enforcement: never downloads or indexes images at application startup.
- Graceful text-only fallback when visual embeddings are not yet generated.
- Lazy, safe encoder initialization with memory capacity verification.
- Cosine similarity visual retrieval from uploaded image or catalog product image.
- Safe upload validation (file size, format, Pillow decodability).
"""

import io
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from PIL import Image

from services.config_service import resolve_path, ConfigService
from services.catalog_service import CatalogService


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


class CLIPService:
    """
    Service managing CLIP visual embeddings and image-based product retrieval.
    """

    _instance: Optional["CLIPService"] = None

    def __init__(
        self,
        embeddings_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        model_name: str = "sentence-transformers/clip-ViT-B-32",
        dimension: int = 512,
        device: Optional[str] = None
    ):
        cfg = ConfigService.get_instance()
        clip_cfg = cfg.config.get("clip_model", {})
        self.embeddings_path = resolve_path(embeddings_path) if embeddings_path else cfg.get_path("clip_embeddings_npy")
        self.manifest_path = resolve_path(manifest_path) if manifest_path else cfg.get_path("clip_manifest_json")
        self.model_name = clip_cfg.get("name", model_name)
        self.dimension = clip_cfg.get("dimension", dimension)
        self.device = device or clip_cfg.get("device", "cpu")

        self._embeddings: Optional[np.ndarray] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._model = None

        self._try_load_embeddings()

    @classmethod
    def get_instance(cls) -> "CLIPService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _try_load_embeddings(self):
        """Load embeddings if artifact exists; do not fail if not yet generated."""
        self.reload_embeddings()

    def reload_embeddings(self) -> bool:
        """Attempt to (re)load precomputed CLIP embeddings from disk."""
        cfg = ConfigService.get_instance()
        candidates = [
            self.embeddings_path,
            cfg.artifacts_dir / "clip_image_embeddings.npy",
            cfg.artifacts_dir / "clip_embeddings.npy"
        ]
        loaded = False
        for p in candidates:
            if p and p.exists():
                try:
                    matrix = np.load(str(p))
                    if matrix.ndim == 2 and matrix.shape[1] == self.dimension and np.all(np.isfinite(matrix)):
                        norms = np.linalg.norm(matrix, axis=1)
                        if np.allclose(norms, 1.0, atol=1e-3):
                            self._embeddings = matrix.astype(np.float32)
                            self.embeddings_path = p
                            loaded = True
                            break
                except Exception as e:
                    print(f"[WARN] Failed to load CLIP embeddings from {p}: {e}")

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            except Exception:
                pass

        return loaded

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Expose precomputed CLIP visual embeddings matrix."""
        return self._embeddings

    def is_visual_retrieval_ready(self) -> bool:
        """Return True only if verified visual embeddings exist on disk."""
        cat_svc = CatalogService.get_instance()
        if self._embeddings is None:
            self.reload_embeddings()
        return (
            self._embeddings is not None and
            len(self._embeddings) == len(cat_svc) and
            self._embeddings.shape[1] == self.dimension
        )

    def is_encoder_available(self) -> bool:
        """Check if live sentence-transformers / PyTorch CLIP encoder can be loaded."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Report accurate operational status of CLIP visual retrieval."""
        ready = self.is_visual_retrieval_ready()
        encoder_avail = self.is_encoder_available()
        return {
            "ready": ready,
            "encoder_available": encoder_avail,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "artifact_exists": self.embeddings_path.exists(),
            "manifest_exists": self.manifest_path.exists(),
            "indexed_image_count": len(self._embeddings) if self._embeddings is not None else 0,
            "device": self.device,
            "message": (
                "Visual search operational (Encoder & precomputed index active)" if (ready and encoder_avail)
                else ("Visual precomputed index operational (Lightweight CPU mode, live encoder disabled)" if ready
                else "Visual embeddings not yet indexed (offline generation required)")
            )
        }

    @staticmethod
    def validate_uploaded_image(
        file_or_bytes: Any,
        max_bytes: int = 10 * 1024 * 1024,
        allowed_formats: Tuple[str, ...] = ("JPEG", "JPG", "PNG", "WEBP")
    ) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
        """
        Validate and safely decode an uploaded clothing image.
        Returns (is_valid, PIL Image in RGB mode, error_message).
        """
        if file_or_bytes is None:
            return False, None, "No image file provided."

        try:
            # Handle Streamlit UploadedFile, bytes, or file path
            if hasattr(file_or_bytes, "read"):
                raw_bytes = file_or_bytes.read()
                if hasattr(file_or_bytes, "seek"):
                    file_or_bytes.seek(0)
            elif isinstance(file_or_bytes, bytes):
                raw_bytes = file_or_bytes
            elif isinstance(file_or_bytes, (str, Path)):
                with open(file_or_bytes, "rb") as f:
                    raw_bytes = f.read()
            else:
                return False, None, f"Unsupported file input type: {type(file_or_bytes)}"

            # 1. File size check
            if len(raw_bytes) > max_bytes:
                mb = max_bytes / (1024 * 1024)
                return False, None, f"Image file exceeds maximum allowed size ({mb:.1f} MB)."
            if len(raw_bytes) == 0:
                return False, None, "Uploaded file is empty (0 bytes)."

            # 2. Safe Pillow decode
            stream = io.BytesIO(raw_bytes)
            img = Image.open(stream)
            img.verify()  # Verify image integrity

            # Reopen after verify
            stream.seek(0)
            img = Image.open(stream)

            # 3. Format check
            img_format = (img.format or "").upper()
            if img_format not in allowed_formats and img_format != "MPO":
                return False, None, f"Unsupported image format '{img_format}'. Supported formats: {', '.join(allowed_formats)}."

            # Convert to RGB (handles RGBA, Palette, Grayscale)
            rgb_img = img.convert("RGB")
            return True, rgb_img, None

        except Exception as e:
            return False, None, f"Failed to decode image: {str(e)}"

    def _get_encoder(self):
        """Lazy model loading with safety guard against simultaneous heavy GPU allocations."""
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "PyTorch and sentence-transformers are required to encode new images. "
                    "In lightweight CPU deployment mode, live image encoding is disabled while "
                    "precomputed catalog visual vectors and text search remain fully active."
                ) from e

            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
            if avail_gb < 1.0:
                raise MemoryError(f"Insufficient RAM to load CLIP encoder ({avail_gb:.2f} GB available).")

            print(f"[INFO] Initializing CLIP model '{self.model_name}' on device '{self.device}'...")
            model = SentenceTransformer(self.model_name, device=self.device)
            model.eval()
            self._model = model
        return self._model

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image into a unit-normalized 512-d vector."""
        model = self._get_encoder()
        vec = model.encode(image, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        if vec.ndim > 1:
            vec = vec.squeeze()
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query into CLIP joint embedding space."""
        model = self._get_encoder()
        vec = model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        if vec.ndim > 1:
            vec = vec.squeeze()
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def get_visual_similarities(
        self,
        query_image: Image.Image,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate cosine visual similarity for a set of candidate product IDs against a query image.
        Returns a mapping {product_id: visual_similarity_score}.
        """
        if not self.is_visual_retrieval_ready():
            raise RuntimeError("Visual search is not operational. Precomputed CLIP embeddings are missing.")

        cat_svc = CatalogService.get_instance()
        id_to_idx = {pid: i for i, pid in enumerate(cat_svc.catalog_ids)}

        q_vec = self.encode_image(query_image)
        results = {}

        valid_indices = []
        valid_pids = []
        for pid in candidate_ids:
            pid_str = str(pid).strip()
            if pid_str in id_to_idx:
                valid_indices.append(id_to_idx[pid_str])
                valid_pids.append(pid_str)

        if not valid_indices:
            return {}

        sub_matrix = self._embeddings[valid_indices]
        sims = np.dot(sub_matrix, q_vec)

        for pid_str, s in zip(valid_pids, sims):
            # Calibrate cosine similarity to [0.0, 1.0] range
            results[pid_str] = float(max(0.0, min(1.0, float(s))))

        return results

    def search_by_image(
        self,
        query_image: Image.Image,
        top_k: int = 5,
        eligible_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve visually similar products using a query PIL image.
        """
        if not self.is_visual_retrieval_ready():
            raise RuntimeError(
                "Visual search is not operational. Precomputed CLIP embeddings are missing."
            )

        cat_svc = CatalogService.get_instance()
        id_to_idx = {pid: i for i, pid in enumerate(cat_svc.catalog_ids)}

        # Encode query image
        q_vec = self.encode_image(query_image)

        # Slice candidates
        if eligible_ids is not None:
            valid_indices = [id_to_idx[pid] for pid in eligible_ids if pid in id_to_idx]
            if not valid_indices:
                return []
            sub_matrix = self._embeddings[valid_indices]
            sub_ids = [cat_svc.catalog_ids[i] for i in valid_indices]
        else:
            sub_matrix = self._embeddings
            sub_ids = cat_svc.catalog_ids

        # Dot product cosine similarity
        sims = np.dot(sub_matrix, q_vec)

        # Sort descending by visual similarity with deterministic tie-breaking
        items = [
            (float(sims[i]), sub_ids[i])
            for i in range(len(sub_ids))
        ]
        items.sort(key=lambda x: (-round(x[0], 7), str(x[1])))

        results = []
        for score, pid in items[:top_k]:
            prod = cat_svc.get_product(pid)
            if prod:
                rec = dict(prod)
                rec["visual_similarity"] = float(score)
                rec["score"] = float(score)
                results.append(rec)

        return results
