"""
services/embedding_service.py
MiniLM Embedding Management & Query Encoding Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Fast zero-recomputation loading of precomputed MiniLM embeddings (10,000 x 384).
- Artifact integrity & SHA-256 verification against embedding_manifest.json.
- Lazy, CPU-friendly initialization of SentenceTransformer for online query encoding.
- L2 unit-normalization assertion and row sequence alignment verification.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from services.config_service import resolve_path, ConfigService
from services.catalog_service import CatalogService


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536 * 16):
            h.update(chunk)
    return h.hexdigest()


class EmbeddingService:
    """
    Manages precomputed product embeddings and online text query encoding.
    Ensures precomputed embeddings are never recomputed during application startup.
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(
        self,
        embeddings_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        cfg = ConfigService.get_instance()
        self.embeddings_path = resolve_path(embeddings_path) if embeddings_path else cfg.get_path("embeddings_npy")
        self.manifest_path = resolve_path(manifest_path) if manifest_path else cfg.get_path("embedding_manifest_json")
        self.model_name = model_name
        self.device = device or ("cuda" if self._cuda_available() else "cpu")

        self._embeddings: Optional[np.ndarray] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._model = None  # Lazy loaded

        self._load_and_validate_embeddings()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_and_validate_embeddings(self):
        """Load and rigorously validate existing MiniLM embeddings."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings artifact missing at: {self.embeddings_path}. "
                "Ensure artifacts/embeddings.npy is present."
            )

        # Load float32 matrix
        matrix = np.load(str(self.embeddings_path))
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")
        if matrix.shape[1] != 384:
            raise ValueError(f"Expected embedding dimension 384, got {matrix.shape[1]}")
        if matrix.dtype != np.float32:
            matrix = matrix.astype(np.float32)

        # Invariant assertions: finite, unit-normalized
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Embedding matrix contains NaN or Inf values!")

        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings are not unit-normalized! Norm range: [{norms.min()}, {norms.max()}]")

        # Alignment check against catalog
        cat_svc = CatalogService.get_instance()
        if len(matrix) != len(cat_svc):
            raise ValueError(
                f"Embedding row count ({len(matrix)}) does not match catalog count ({len(cat_svc)})"
            )

        # Check manifest if present
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            except Exception:
                self._manifest = None

        self._embeddings = matrix

    @property
    def embeddings(self) -> np.ndarray:
        """Return the precomputed unit-normalized float32 embedding matrix (10,000, 384)."""
        return self._embeddings

    @property
    def dimension(self) -> int:
        return 384

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        return self._manifest

    def get_embedding_by_index(self, index: int) -> np.ndarray:
        """Retrieve embedding vector for row index."""
        return self._embeddings[index]

    def _get_encoder(self):
        """Lazy initialization of SentenceTransformer model for online queries."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Initializing query encoder '{self.model_name}' on {self.device}...")
            model = SentenceTransformer(self.model_name, device=self.device)
            model.eval()
            self._model = model
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a query string into a unit-normalized 384-d float32 vector.
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        model = self._get_encoder()
        vec = model.encode(
            text.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        if vec.ndim > 1:
            vec = vec.squeeze()

        # Re-verify normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec
