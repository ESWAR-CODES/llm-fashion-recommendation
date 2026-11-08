"""
services/persistence_service.py
Persistence Abstraction & SQLite Profile Repository
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Abstract persistence interface allowing future migration to PostgreSQL or Cloud SQL.
- SQLite local implementation with ACID compliance and foreign key enforcement.
- Anonymous user profile lifecycle, taste preferences, active constraint states.
- Idempotent product feedback tracking (likes and dismissals).
- Synthesis of unit-normalized user preference vector u from liked products or preferences.
- Native SQLite Online Backup API snapshot management.
"""

import sys
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from services.config_service import resolve_path, ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService


class PersistenceService:
    """
    Persistence service managing SQLite profile storage and preference vectors.
    """

    _instance: Optional["PersistenceService"] = None

    def __init__(self, db_path: Optional[str | Path] = None):
        cfg = ConfigService.get_instance()
        self.db_path = resolve_path(db_path) if db_path else cfg.get_path("profiles_sqlite")
        self.init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[str | Path] = None) -> "PersistenceService":
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize database schema idempotently."""
        conn = self._get_connection()
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

    def create_profile(
        self,
        profile_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new anonymous user profile."""
        pid = profile_id or f"usr_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prefs_json = json.dumps(preferences or {})

        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT INTO profiles (profile_id, preferences_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (pid, prefs_json, now, now)
            )
            conn.execute(
                "INSERT OR REPLACE INTO active_states (profile_id, constraints_json, updated_at) VALUES (?, '{}', ?)",
                (pid, now)
            )
        conn.close()
        return pid

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve profile record by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT profile_id, preferences_json, created_at, updated_at FROM profiles WHERE profile_id = ?",
            (profile_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "profile_id": row["profile_id"],
            "preferences": json.loads(row["preferences_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def update_profile(self, profile_id: str, preferences: Dict[str, Any]):
        """Update explicit taste preferences for a profile."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = self._get_connection()
        with conn:
            cur = conn.execute(
                "UPDATE profiles SET preferences_json = ?, updated_at = ? WHERE profile_id = ?",
                (json.dumps(preferences), now, profile_id)
            )
            if cur.rowcount == 0:
                conn.close()
                raise ValueError(f"Profile {profile_id} does not exist.")
        conn.close()

    def delete_profile(self, profile_id: str) -> bool:
        """Delete profile and cascade associated state."""
        conn = self._get_connection()
        with conn:
            cur = conn.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
            deleted = cur.rowcount > 0
        conn.close()
        return deleted

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all registered anonymous profiles."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT profile_id, preferences_json, created_at, updated_at FROM profiles ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "profile_id": r["profile_id"],
                "preferences": json.loads(r["preferences_json"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"]
            }
            for r in rows
        ]

    def save_feedback(self, profile_id: str, item_id: str, action: str):
        """Save idempotent product feedback ('like' or 'dismiss')."""
        if action not in ("like", "dismiss"):
            raise ValueError(f"Invalid feedback action: '{action}'. Must be 'like' or 'dismiss'.")

        pid_str = str(profile_id).strip()
        item_str = str(item_id).strip()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            prof = conn.execute("SELECT 1 FROM profiles WHERE profile_id = ?", (pid_str,)).fetchone()
            if not prof:
                conn.close()
                raise ValueError(f"Profile '{pid_str}' does not exist.")

            conn.execute(
                """
                INSERT INTO feedback (profile_id, item_id, action, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(profile_id, item_id) DO UPDATE SET
                    action = excluded.action,
                    timestamp = excluded.timestamp
                """,
                (pid_str, item_str, action, now)
            )
        conn.close()

    def get_feedback(self, profile_id: str) -> Dict[str, List[str]]:
        """Retrieve all recorded likes and dismissals for a profile."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT item_id, action FROM feedback WHERE profile_id = ? ORDER BY timestamp ASC",
            (profile_id,)
        ).fetchall()
        conn.close()

        likes = []
        dismissals = []
        for r in rows:
            if r["action"] == "like":
                likes.append(r["item_id"])
            elif r["action"] == "dismiss":
                dismissals.append(r["item_id"])

        return {"likes": likes, "dismissals": dismissals}

    def get_active_constraints(self, profile_id: str) -> Dict[str, Any]:
        """Retrieve current conversation active constraints."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT constraints_json FROM active_states WHERE profile_id = ?",
            (profile_id,)
        ).fetchone()
        conn.close()
        if row and row["constraints_json"]:
            return json.loads(row["constraints_json"])
        return {}

    def set_active_constraints(self, profile_id: str, constraints: Dict[str, Any]):
        """Persist current conversation active constraints."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO active_states (profile_id, constraints_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    constraints_json = excluded.constraints_json,
                    updated_at = excluded.updated_at
                """,
                (profile_id, json.dumps(constraints), now)
            )
        conn.close()

    def clear_active_constraints(self, profile_id: str):
        """Reset conversation active constraints to empty."""
        self.set_active_constraints(profile_id, {})

    def build_user_vector(self, profile_id: str) -> Optional[np.ndarray]:
        """
        Synthesize unit-normalized user preference vector u.
        Formula:
        u = L2_normalize(mean(v_liked_1, v_liked_2, ...))
        If no likes, falls back to explicit preferences encoded by MiniLM.
        If empty profile, returns None.
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        feedback = self.get_feedback(profile_id)
        liked_ids = feedback.get("likes", [])

        emb_svc = EmbeddingService.get_instance()
        cat_svc = CatalogService.get_instance()
        id_to_idx = {pid: i for i, pid in enumerate(cat_svc.catalog_ids)}

        # 1. Primary: synthesize from liked product embeddings
        liked_vectors = []
        for lid in liked_ids:
            if lid in id_to_idx:
                liked_vectors.append(emb_svc.get_embedding_by_index(id_to_idx[lid]))

        if liked_vectors:
            mean_vec = np.mean(liked_vectors, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-9:
                return mean_vec / norm
            return mean_vec

        # 2. Secondary: synthesize from explicit preferences
        prefs = profile.get("preferences", {})
        pref_parts = []
        if prefs.get("preferred_categories"):
            pref_parts.append(f"Categories: {', '.join(prefs['preferred_categories'])}")
        if prefs.get("preferred_colors"):
            pref_parts.append(f"Colors: {', '.join(prefs['preferred_colors'])}")
        if prefs.get("preferred_article_types"):
            pref_parts.append(f"Types: {', '.join(prefs['preferred_article_types'])}")

        if pref_parts:
            text = " | ".join(pref_parts)
            return emb_svc.encode_text(text)

        return None

    def create_database_backup(self, snapshot_path: Optional[str | Path] = None) -> Tuple[bool, str]:
        """
        Create an atomic snapshot using SQLite's native Online Backup API.
        """
        snap_p = resolve_path(snapshot_path) if snapshot_path else resolve_path("backups/profiles_snapshot.sqlite")
        snap_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            source_conn = self._get_connection()
            dest_conn = sqlite3.connect(str(snap_p))
            source_conn.backup(dest_conn)
            dest_conn.close()
            source_conn.close()
            return True, f"Snapshot successfully created at: {snap_p}"
        except Exception as e:
            return False, f"Backup error: {e}"

    def restore_database_backup(self, snapshot_path: Optional[str | Path] = None) -> Tuple[bool, str]:
        """
        Restore SQLite database from an atomic snapshot.
        """
        snap_p = resolve_path(snapshot_path) if snapshot_path else resolve_path("backups/profiles_snapshot.sqlite")
        if not snap_p.exists():
            return False, f"Snapshot file not found at: {snap_p}"

        try:
            source_conn = sqlite3.connect(str(snap_p))
            dest_conn = self._get_connection()
            source_conn.backup(dest_conn)
            source_conn.close()
            dest_conn.close()
            return True, f"Database restored successfully from: {snap_p}"
        except Exception as e:
            return False, f"Restore error: {e}"
