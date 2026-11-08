"""
services/catalog_service.py
Frozen Catalog Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Ingestion & validation of the frozen 10,000-item catalog.
- O(1) product record lookup by canonical string ID.
- Discrete attribute domain discovery (categories, article types, colors, usages, genders).
- Candidate subset slicing without altering catalog ordering.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
from services.config_service import resolve_path, ConfigService


class CatalogService:
    """
    Service managing access to the frozen 10,000-product fashion catalog.
    Thread-safe and read-only.
    """

    _instance: Optional["CatalogService"] = None

    def __init__(
        self,
        catalog_csv_path: Optional[str | Path] = None,
        catalog_ids_path: Optional[str | Path] = None
    ):
        cfg = ConfigService.get_instance()
        self.catalog_path = resolve_path(catalog_csv_path) if catalog_csv_path else cfg.get_path("catalog_csv")
        self.ids_path = resolve_path(catalog_ids_path) if catalog_ids_path else cfg.get_path("catalog_ids_json")

        self._catalog_df: Optional[pd.DataFrame] = None
        self._catalog_ids: Optional[List[str]] = None
        self._product_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._load_catalog()

    @classmethod
    def get_instance(
        cls,
        catalog_csv_path: Optional[str | Path] = None,
        catalog_ids_path: Optional[str | Path] = None
    ) -> "CatalogService":
        if cls._instance is None:
            cls._instance = cls(catalog_csv_path, catalog_ids_path)
        return cls._instance

    def _load_catalog(self):
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog file missing: {self.catalog_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"Catalog IDs file missing: {self.ids_path}")

        df = pd.read_csv(self.catalog_path, dtype={"id": str})
        df["id"] = df["id"].astype(str).str.strip()

        with open(self.ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)

        ids = [str(x).strip() for x in ids]

        if len(df) != 10000:
            raise ValueError(f"Catalog size mismatch: expected 10,000 rows, found {len(df)}")
        if df["id"].tolist() != ids:
            raise ValueError("Catalog DataFrame row sequence does not match catalog_ids.json!")

        self._catalog_df = df
        self._catalog_ids = ids

        # Fast O(1) dict lookup mapping
        self._product_map = {row["id"]: row.to_dict() for _, row in df.iterrows()}

    @property
    def catalog_df(self) -> pd.DataFrame:
        """Return a copy of the frozen catalog DataFrame."""
        return self._catalog_df.copy()

    @property
    def catalog_ids(self) -> List[str]:
        """Return the immutable ordered list of product IDs."""
        return list(self._catalog_ids)

    def __len__(self) -> int:
        return len(self._catalog_ids) if self._catalog_ids else 0

    def get_product(self, product_id: str | int) -> Optional[Dict[str, Any]]:
        """Retrieve a product record in O(1) time by canonical ID."""
        pid_str = str(product_id).strip()
        return self._product_map.get(pid_str)

    def contains(self, product_id: str | int) -> bool:
        """Check if a product ID exists in the frozen catalog."""
        return str(product_id).strip() in self._product_map

    def get_unique_values(self, column: str) -> List[str]:
        """Return sorted unique non-null values for a metadata column."""
        if column not in self._catalog_df.columns:
            return []
        vals = self._catalog_df[column].dropna().astype(str).str.strip().unique()
        return sorted([v for v in vals if v and v.lower() not in ("nan", "none")])

    def get_categories(self) -> List[str]:
        return self.get_unique_values("masterCategory")

    def get_article_types(self, master_category: Optional[str] = None) -> List[str]:
        if master_category:
            sub = self._catalog_df[self._catalog_df["masterCategory"] == master_category]
            vals = sub["articleType"].dropna().astype(str).str.strip().unique()
            return sorted([v for v in vals if v and v.lower() not in ("nan", "none")])
        return self.get_unique_values("articleType")

    def get_colors(self) -> List[str]:
        return self.get_unique_values("baseColour")

    def get_genders(self) -> List[str]:
        return self.get_unique_values("gender")

    def get_subcategories(self, master_category: Optional[str] = None) -> List[str]:
        if master_category:
            sub = self._catalog_df[self._catalog_df["masterCategory"] == master_category]
            vals = sub["subCategory"].dropna().astype(str).str.strip().unique()
            return sorted([v for v in vals if v and v.lower() not in ("nan", "none")])
        return self.get_unique_values("subCategory")

    def get_seasons(self) -> List[str]:
        return self.get_unique_values("season")

    def get_usages(self) -> List[str]:
        return self.get_unique_values("usage")
