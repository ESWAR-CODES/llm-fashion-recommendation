"""
services/config_service.py
Centralized Configuration & Portable Path Management
LLM-Assisted Personalized Fashion Recommendation System
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def find_project_root() -> Path:
    """
    Locate the canonical 'fashion_project' root directory reliably across runtimes.
    Works whether running from the workspace root or inside fashion_project/.
    """
    # 1. From current file location
    current_file = Path(__file__).resolve()
    candidate = current_file.parent.parent
    if (candidate / "artifacts").exists() and (candidate / "config.json").exists():
        return candidate

    # 2. From current working directory
    cwd = Path.cwd().resolve()
    if (cwd / "fashion_project" / "config.json").exists():
        return cwd / "fashion_project"
    if (cwd / "config.json").exists() and (cwd / "artifacts").exists():
        return cwd

    # Fallback to candidate
    return candidate


PROJECT_ROOT = find_project_root()


def resolve_path(relative_or_absolute: str | Path) -> Path:
    """
    Resolve a path portably relative to PROJECT_ROOT.
    Prevents duplicate 'fashion_project/fashion_project' paths and normalizes separators.
    """
    if isinstance(relative_or_absolute, Path):
        p = relative_or_absolute
    else:
        # Normalize slashes
        clean_str = str(relative_or_absolute).replace("\\", "/")
        p = Path(clean_str)

    if p.is_absolute():
        return p

    # If already starts with fashion_project, strip it when joining to PROJECT_ROOT
    parts = p.parts
    if parts and parts[0] == "fashion_project":
        p = Path(*parts[1:])

    resolved = (PROJECT_ROOT / p).resolve()
    return resolved


class ConfigService:
    """Singleton service to load, validate, and manage application configuration."""

    _instance: Optional["ConfigService"] = None

    def __init__(self, config_path: Optional[str | Path] = None):
        if config_path:
            self.config_file = resolve_path(config_path)
        else:
            self.config_file = resolve_path("config.json")
        self._config: Dict[str, Any] = self._load_config()

    @classmethod
    def get_instance(cls, config_path: Optional[str | Path] = None) -> "ConfigService":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            return self._default_config()
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "project_title": "LLM-Assisted Personalized Fashion Recommendation System",
            "random_seed": 42,
            "dataset": {
                "source_url": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
                "styles_csv": "data/raw/styles.csv",
                "images_dir": "data/raw/images",
                "archive_preserved": "data/raw/fashion-product-images-small.zip"
            },
            "paths": {
                "catalog_csv": "artifacts/catalog.csv",
                "catalog_ids_json": "artifacts/catalog_ids.json",
                "data_audit_json": "artifacts/data_audit.json",
                "catalog_manifest_json": "artifacts/catalog_manifest.json",
                "product_descriptions_csv": "artifacts/product_descriptions.csv",
                "embeddings_npy": "artifacts/embeddings.npy",
                "embedding_manifest_json": "artifacts/embedding_manifest.json",
                "clip_embeddings_npy": "artifacts/clip_embeddings.npy",
                "clip_manifest_json": "artifacts/clip_manifest.json",
                "profiles_sqlite": "backups/profiles.sqlite",
                "backups_dir": "backups",
                "logs_dir": "logs"
            },
            "embedding_model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "normalized": true,
                "batch_size": 64
            },
            "clip_model": {
                "name": "sentence-transformers/clip-ViT-B-32",
                "dimension": 512,
                "normalized": true,
                "batch_size": 64,
                "device": "cpu"
            },
            "language_model": {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "tokenizer_vocab_size": 151643,
                "chat_template": "official",
                "decoding": "deterministic",
                "max_new_tokens": 256,
                "retry_limit": 1
            },
            "phase3_personalization": {
                "query_weight": 0.8,
                "profile_weight": 0.2,
                "top_k": 5
            }
        }

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a top-level key from configuration."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def artifacts_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts"

    def get_path(self, key: str) -> Path:
        """Resolve a path configured under 'paths' or 'dataset'."""
        paths_section = self._config.get("paths", {})
        if key in paths_section:
            return resolve_path(paths_section[key])
        dataset_section = self._config.get("dataset", {})
        if key in dataset_section:
            return resolve_path(dataset_section[key])
        return resolve_path(key)

    def get_hardware_info(self) -> Dict[str, Any]:
        """Detect system resources without importing heavy ML packages prematurely."""
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        ram_avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)
        disk_total, disk_used, disk_free = shutil.disk_usage(str(PROJECT_ROOT))

        cuda_available = False
        device_name = "CPU"
        vram_gb = 0.0
        torch_version = "Not Installed"
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except ImportError:
            pass

        return {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram_gb": vram_gb,
            "torch_version": torch_version,
            "system_ram_gb": ram_gb,
            "system_ram_available_gb": ram_avail_gb,
            "disk_free_gb": round(disk_free / (1024**3), 2),
            "project_root": str(PROJECT_ROOT)
        }
