"""
profile_manager.py
Phase 3: User Profiles, Explicit Constraints & Product Feedback
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Steps 5 & 7)
"""

import os
import sys
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set, Union

import numpy as np
import pandas as pd

# Default paths relative to workspace root or fashion_project
DEFAULT_DB_PATH = "fashion_project/backups/profiles.sqlite"
DEFAULT_ARTIFACTS_DIR = "fashion_project/artifacts"

# In-memory cache for catalog metadata validation domain
_CATALOG_DOMAIN_CACHE = None


def resolve_path(path_str: str) -> Path:
    """Resolve a path considering current working directory and fashion_project root."""
    p = Path(path_str)
    if p.exists():
        return p
    # Try stripping or prepending 'fashion_project'
    if path_str.startswith("fashion_project/"):
        alt = Path(path_str[len("fashion_project/"):])
        if alt.exists():
            return alt
    else:
        alt = Path("fashion_project") / p
        if alt.exists():
            return alt
    return p


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establish SQLite connection with foreign keys enabled and row_factory configured.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    p = resolve_path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database schema for Phase 3:
    - profiles: profile_id, preferences_json, created_at, updated_at
    - feedback: profile_id, item_id, action, timestamp
    - active_states: profile_id, constraints_json, updated_at
    """
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                preferences_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                profile_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('like', 'dismiss')),
                timestamp TEXT NOT NULL,
                PRIMARY KEY (profile_id, item_id),
                FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_states (
                profile_id TEXT PRIMARY KEY,
                constraints_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_profile_action 
            ON feedback(profile_id, action);
        """)
    conn.close()


def get_catalog_domain(catalog_or_path: Optional[Union[pd.DataFrame, str]] = None) -> Dict[str, Any]:
    """
    Load or retrieve cached catalog attribute domains for rigorous validation.
    """
    global _CATALOG_DOMAIN_CACHE
    if _CATALOG_DOMAIN_CACHE is not None and catalog_or_path is None:
        return _CATALOG_DOMAIN_CACHE

    if isinstance(catalog_or_path, pd.DataFrame):
        df = catalog_or_path
    else:
        path = catalog_or_path or f"{DEFAULT_ARTIFACTS_DIR}/catalog.csv"
        p = resolve_path(str(path))
        if not p.exists():
            raise FileNotFoundError(f"Catalog file not found at: {p}")
        df = pd.read_csv(p, dtype={"id": str})

    domain = {
        "masterCategory": set(str(v).strip() for v in df["masterCategory"].dropna().unique()),
        "subCategory": set(str(v).strip() for v in df["subCategory"].dropna().unique()),
        "articleType": set(str(v).strip() for v in df["articleType"].dropna().unique()),
        "baseColour": set(str(v).strip() for v in df["baseColour"].dropna().unique()),
        "usage": set(str(v).strip() for v in df["usage"].dropna().unique()),
        "product_ids": set(str(v).strip() for v in df["id"].dropna().unique())
    }

    # Case-insensitive lookup dictionaries mapping lowercase to canonical title-case
    domain["masterCategory_lower"] = {k.lower(): k for k in domain["masterCategory"]}
    domain["subCategory_lower"] = {k.lower(): k for k in domain["subCategory"]}
    domain["articleType_lower"] = {k.lower(): k for k in domain["articleType"]}
    domain["baseColour_lower"] = {k.lower(): k for k in domain["baseColour"]}
    domain["usage_lower"] = {k.lower(): k for k in domain["usage"]}

    if catalog_or_path is None:
        _CATALOG_DOMAIN_CACHE = domain
    return domain


def validate_preferences(
    preferences: Dict[str, Any],
    domain: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate saved explicit preferences against actual catalog values.
    Supported fields:
    - preferred_categories: List[str] matching catalog masterCategory or subCategory
    - preferred_colors: List[str] matching catalog baseColour
    """
    if not isinstance(preferences, dict):
        raise ValueError(f"Preferences must be a dictionary, got {type(preferences).__name__}")

    if domain is None:
        domain = get_catalog_domain()

    valid_cats_lower = {**domain["masterCategory_lower"], **domain["subCategory_lower"]}
    valid_colors_lower = domain["baseColour_lower"]

    cleaned = {}
    for key, val in preferences.items():
        if key in ("preferred_categories", "preferred_category", "categories"):
            if isinstance(val, str):
                val = [val]
            if not isinstance(val, (list, tuple)):
                raise ValueError(f"'{key}' must be a list of category names.")
            cleaned_cats = []
            for c in val:
                c_str = str(c).strip()
                c_lower = c_str.lower()
                if c_lower not in valid_cats_lower:
                    raise ValueError(f"Invalid category '{c}'. Not found in catalog masterCategory or subCategory.")
                cleaned_cats.append(valid_cats_lower[c_lower])
            cleaned["preferred_categories"] = cleaned_cats

        elif key in ("preferred_colors", "preferred_colours", "preferred_color", "colors"):
            if isinstance(val, str):
                val = [val]
            if not isinstance(val, (list, tuple)):
                raise ValueError(f"'{key}' must be a list of color names.")
            cleaned_cols = []
            for c in val:
                c_str = str(c).strip()
                c_lower = c_str.lower()
                if c_lower not in valid_colors_lower:
                    raise ValueError(f"Invalid color '{c}'. Not found in catalog baseColour.")
                cleaned_cols.append(valid_colors_lower[c_lower])
            cleaned["preferred_colors"] = cleaned_cols
        else:
            raise ValueError(f"Unsupported preference field '{key}'. Supported: preferred_categories, preferred_colors")

    return cleaned


SUPPORTED_CONSTRAINT_KEYS = {"articleType", "baseColour", "usage", "masterCategory", "exclude_colors"}


def validate_constraints(
    constraints: Dict[str, Any],
    domain: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    Validate active request constraints against catalog domain.
    Supported fields:
    - articleType: List[str]
    - baseColour: List[str]
    - usage: List[str]
    - masterCategory: List[str]
    - exclude_colors: List[str]
    """
    if not isinstance(constraints, dict):
        raise ValueError(f"Constraints must be a dictionary, got {type(constraints).__name__}")

    if domain is None:
        domain = get_catalog_domain()

    cleaned = {}
    for key, val in constraints.items():
        if key not in SUPPORTED_CONSTRAINT_KEYS:
            raise ValueError(f"Unsupported constraint field '{key}'. Supported fields: {sorted(SUPPORTED_CONSTRAINT_KEYS)}")

        if val is None:
            continue
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"Value for constraint '{key}' must be a list of strings.")

        cleaned_list = []
        for item in val:
            item_str = str(item).strip()
            item_lower = item_str.lower()

            if key == "articleType":
                if item_lower not in domain["articleType_lower"]:
                    raise ValueError(f"Invalid articleType '{item}'. Not found in catalog.")
                cleaned_list.append(domain["articleType_lower"][item_lower])
            elif key in ("baseColour", "exclude_colors"):
                if item_lower not in domain["baseColour_lower"]:
                    raise ValueError(f"Invalid color '{item}' for '{key}'. Not found in catalog.")
                cleaned_list.append(domain["baseColour_lower"][item_lower])
            elif key == "usage":
                if item_lower not in domain["usage_lower"]:
                    raise ValueError(f"Invalid usage '{item}'. Not found in catalog.")
                cleaned_list.append(domain["usage_lower"][item_lower])
            elif key == "masterCategory":
                if item_lower not in domain["masterCategory_lower"]:
                    raise ValueError(f"Invalid masterCategory '{item}'. Not found in catalog.")
                cleaned_list.append(domain["masterCategory_lower"][item_lower])

        cleaned[key] = cleaned_list

    return cleaned


# ==============================================================================
# PROFILE MANAGEMENT (CRUD)
# ==============================================================================

def create_profile(
    preferences: Optional[Dict[str, Any]] = None,
    profile_id: Optional[str] = None,
    db_path: Optional[str] = None,
    catalog_or_path: Optional[Any] = None
) -> str:
    """
    Create a new anonymous user profile with optional explicit preferences.
    """
    init_db(db_path)
    clean_prefs = {}
    if preferences:
        domain = get_catalog_domain(catalog_or_path)
        clean_prefs = validate_preferences(preferences, domain=domain)

    if profile_id is None:
        profile_id = f"usr_{uuid.uuid4().hex[:12]}"
    else:
        profile_id = str(profile_id).strip()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO profiles (profile_id, preferences_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (profile_id, json.dumps(clean_prefs), now, now)
        )
        conn.execute(
            "INSERT INTO active_states (profile_id, constraints_json, updated_at) VALUES (?, ?, ?)",
            (profile_id, "{}", now)
        )
    conn.close()
    return profile_id


def get_profile(
    profile_id: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve user profile by profile_id. Returns None if not found.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT profile_id, preferences_json, created_at, updated_at FROM profiles WHERE profile_id = ?",
        (str(profile_id).strip(),)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "profile_id": row["profile_id"],
        "preferences": json.loads(row["preferences_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def update_profile(
    profile_id: str,
    preferences: Dict[str, Any],
    db_path: Optional[str] = None,
    catalog_or_path: Optional[Any] = None
) -> bool:
    """
    Update explicit preferences for an existing profile.
    Preserves previous valid state if validation fails.
    """
    init_db(db_path)
    profile = get_profile(profile_id, db_path=db_path)
    if profile is None:
        raise ValueError(f"Profile '{profile_id}' does not exist.")

    # Validate before executing database write
    domain = get_catalog_domain(catalog_or_path)
    clean_prefs = validate_preferences(preferences, domain=domain)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE profiles SET preferences_json = ?, updated_at = ? WHERE profile_id = ?",
            (json.dumps(clean_prefs), now, str(profile_id).strip())
        )
    conn.close()
    return True


def delete_profile(
    profile_id: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Delete a profile and cascade-delete its feedback and active states.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            "DELETE FROM profiles WHERE profile_id = ?",
            (str(profile_id).strip(),)
        )
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def list_profiles(db_path: Optional[str] = None) -> List[str]:
    """List all registered profile IDs in database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    rows = conn.execute("SELECT profile_id FROM profiles ORDER BY created_at ASC").fetchall()
    conn.close()
    return [row["profile_id"] for row in rows]


# ==============================================================================
# ACTIVE CONSTRAINT MANAGEMENT
# ==============================================================================

def get_active_constraints(
    profile_id: str,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve active constraints dictionary for profile. Returns {} if none set."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT constraints_json FROM active_states WHERE profile_id = ?",
        (str(profile_id).strip(),)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {}
    return json.loads(row["constraints_json"])


def set_active_constraints(
    profile_id: str,
    constraints: Dict[str, Any],
    db_path: Optional[str] = None,
    catalog_or_path: Optional[Any] = None
) -> bool:
    """
    Validate and persist active request constraints for the given profile.
    """
    init_db(db_path)
    profile = get_profile(profile_id, db_path=db_path)
    if profile is None:
        raise ValueError(f"Profile '{profile_id}' does not exist.")

    domain = get_catalog_domain(catalog_or_path)
    clean_cons = validate_constraints(constraints, domain=domain)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO active_states (profile_id, constraints_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                constraints_json = excluded.constraints_json,
                updated_at = excluded.updated_at
        """, (str(profile_id).strip(), json.dumps(clean_cons), now))
    conn.close()
    return True


def clear_active_constraints(
    profile_id: str,
    db_path: Optional[str] = None
) -> bool:
    """Reset active constraints for profile to empty dict."""
    return set_active_constraints(profile_id, {}, db_path=db_path)


# ==============================================================================
# PRODUCT FEEDBACK MANAGEMENT
# ==============================================================================

def save_feedback(
    profile_id: str,
    item_id: str,
    action: str,
    db_path: Optional[str] = None,
    catalog_or_path: Optional[Any] = None
) -> bool:
    """
    Persist feedback ('like' or 'dismiss') for a given product and profile.
    - Repeated likes are idempotent (never duplicate rows).
    - Latest feedback action replaces previous action for (profile_id, item_id).
    - Validates profile exists and product exists in catalog.
    """
    init_db(db_path)
    action_clean = str(action).strip().lower()
    if action_clean not in ("like", "dismiss"):
        raise ValueError(f"Invalid feedback action '{action}'. Must be 'like' or 'dismiss'.")

    pid_clean = str(profile_id).strip()
    item_clean = str(item_id).strip()

    profile = get_profile(pid_clean, db_path=db_path)
    if profile is None:
        raise ValueError(f"Profile '{profile_id}' does not exist.")

    domain = get_catalog_domain(catalog_or_path)
    if item_clean not in domain["product_ids"]:
        raise ValueError(f"Product ID '{item_id}' not found in frozen catalog.")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO feedback (profile_id, item_id, action, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id, item_id) DO UPDATE SET
                action = excluded.action,
                timestamp = excluded.timestamp
        """, (pid_clean, item_clean, action_clean, now))
    conn.close()
    return True


def get_feedback(
    profile_id: str,
    db_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Retrieve liked and dismissed item IDs for the given profile.
    Returns: {"likes": [...], "dismissals": [...]}
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    pid_clean = str(profile_id).strip()
    rows = conn.execute(
        "SELECT item_id, action FROM feedback WHERE profile_id = ? ORDER BY timestamp ASC",
        (pid_clean,)
    ).fetchall()
    conn.close()

    likes = [row["item_id"] for row in rows if row["action"] == "like"]
    dismissals = [row["item_id"] for row in rows if row["action"] == "dismiss"]
    return {"likes": likes, "dismissals": dismissals}


def remove_feedback(
    profile_id: str,
    item_id: str,
    db_path: Optional[str] = None
) -> bool:
    """Remove feedback entry for (profile_id, item_id)."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            "DELETE FROM feedback WHERE profile_id = ? AND item_id = ?",
            (str(profile_id).strip(), str(item_id).strip())
        )
        removed = cursor.rowcount > 0
    conn.close()
    return removed


# ==============================================================================
# USER PREFERENCE VECTOR
# ==============================================================================

def build_user_vector(
    profile_id: str,
    embeddings: Optional[np.ndarray] = None,
    catalog_ids: Optional[List[str]] = None,
    catalog: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    db_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Construct the normalized 384-dimensional user preference vector following Step 5 of the pipeline.
    
    Methodology:
    1. Retrieve the user's liked product IDs.
    2. Map them to their corresponding embedding rows in embeddings.npy.
    3. Retrieve their normalized embeddings.
    4. Calculate the mean embedding.
    5. Normalize the resulting mean vector.
    
    Fallback:
    - If no likes exist, explicit saved preferences (preferred categories / colors) are used.
    - If neither likes nor explicit preferences exist, returns None (query-only fallback).
    """
    profile = get_profile(profile_id, db_path=db_path)
    if profile is None:
        return None

    fb = get_feedback(profile_id, db_path=db_path)
    liked_ids = fb.get("likes", [])

    art_dir = Path(artifacts_dir or DEFAULT_ARTIFACTS_DIR)

    # Load embeddings and catalog_ids if not passed in
    if embeddings is None or catalog_ids is None:
        emb_p = resolve_path(str(art_dir / "embeddings.npy"))
        ids_p = resolve_path(str(art_dir / "catalog_ids.json"))
        embeddings = np.load(str(emb_p))
        with open(ids_p, "r", encoding="utf-8") as f:
            catalog_ids = json.load(f)

    id_to_idx = {str(pid): i for i, pid in enumerate(catalog_ids)}

    # 1. Primary Path: Mean of Liked Item Embeddings
    if liked_ids:
        indices = [id_to_idx[str(pid)] for pid in liked_ids if str(pid) in id_to_idx]
        if indices:
            unique_indices = sorted(set(indices))
            selected_embs = embeddings[unique_indices]
            mean_vec = np.mean(selected_embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-9:
                return (mean_vec / norm).astype(np.float32)

    # 2. Secondary Path: Explicit Saved Preferences Fallback
    prefs = profile.get("preferences", {})
    pref_cats = prefs.get("preferred_categories", [])
    pref_cols = prefs.get("preferred_colors", [])

    if pref_cats or pref_cols:
        if model is not None:
            parts = []
            if pref_cats:
                parts.append(f"Category: {' '.join(pref_cats)}")
            if pref_cols:
                parts.append(f"Color: {' '.join(pref_cols)}")
            pref_text = " | ".join(parts)
            pref_vec = model.encode(pref_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            if pref_vec.ndim > 1:
                pref_vec = pref_vec.squeeze()
            norm = np.linalg.norm(pref_vec)
            if norm > 1e-9:
                return (pref_vec / norm).astype(np.float32)
        else:
            # Direct retrieval from catalog embeddings without model
            if catalog is None:
                cat_p = resolve_path(str(art_dir / "catalog.csv"))
                catalog = pd.read_csv(cat_p, dtype={"id": str})

            mask = pd.Series(False, index=catalog.index)
            if pref_cats:
                mask = mask | catalog["masterCategory"].isin(pref_cats) | catalog["subCategory"].isin(pref_cats)
            if pref_cols:
                mask = mask | catalog["baseColour"].isin(pref_cols)

            matching_indices = mask[mask].index.tolist()
            if matching_indices:
                selected_embs = embeddings[matching_indices]
                mean_vec = np.mean(selected_embs, axis=0).astype(np.float32)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-9:
                    return (mean_vec / norm).astype(np.float32)

    # 3. No usable preference data
    return None
