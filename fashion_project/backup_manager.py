"""
backup_manager.py
Phase 5: SQLite Database Snapshot Backup, Integrity Verification & Runtime Recovery
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Step 9)
"Back up the database with SQLite's backup API to a snapshot, then copy that snapshot to Drive.
Save catalog artifacts and the notebook too. On restart, restore files, verify the catalog and
embedding mapping, load models, and initialize the controls. Colab runtime files are temporary."
"""

import os
import sys
import shutil
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


def resolve_path(path_str: str) -> Path:
    """Resolve path relative to current working directory or fashion_project root."""
    p = Path(path_str)
    if p.exists():
        return p
    if path_str.startswith("fashion_project/"):
        alt = Path(path_str[len("fashion_project/"):])
        if alt.exists():
            return alt
    else:
        alt = Path("fashion_project") / p
        if alt.exists():
            return alt
    return p


def create_database_backup(
    source_db_path: Optional[str] = None,
    backup_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Create an atomic SQLite snapshot using SQLite's native Online Backup API (conn.backup).
    This guarantees consistency even if writes are concurrently attempted.
    """
    if source_db_path is None:
        source_db_path = "fashion_project/backups/profiles.sqlite"
    if backup_path is None:
        backup_path = "fashion_project/backups/profiles_snapshot.sqlite"

    src_p = resolve_path(source_db_path)
    bck_p = Path(backup_path) if not Path(backup_path).is_absolute() else Path(backup_path)

    # Ensure source exists
    if not src_p.exists():
        # Initialize an empty source database if it doesn't exist yet
        from profile_manager import init_db
        init_db(str(src_p))

    bck_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        src_conn = sqlite3.connect(str(src_p))
        bck_conn = sqlite3.connect(str(bck_p))

        with bck_conn:
            src_conn.backup(bck_conn)

        src_conn.close()
        bck_conn.close()

        # Verify integrity of backup
        is_valid, msg = verify_database_integrity(str(bck_p))
        if not is_valid:
            return False, f"Backup created but integrity check failed: {msg}"

        return True, f"Snapshot created successfully at {bck_p} ({bck_p.stat().st_size} bytes)"
    except Exception as e:
        return False, f"Failed to create backup: {str(e)}"


def restore_database_backup(
    backup_path: Optional[str] = None,
    target_db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Restore a snapshot into the runtime SQLite database using SQLite's backup API.
    """
    if backup_path is None:
        backup_path = "fashion_project/backups/profiles_snapshot.sqlite"
    if target_db_path is None:
        target_db_path = "fashion_project/backups/profiles.sqlite"

    bck_p = resolve_path(backup_path)
    tgt_p = Path(target_db_path)

    if not bck_p.exists():
        return False, f"Backup snapshot not found at {bck_p}"

    # Verify backup integrity before restoring
    is_valid, msg = verify_database_integrity(str(bck_p))
    if not is_valid:
        return False, f"Refusing to restore corrupt backup: {msg}"

    tgt_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        bck_conn = sqlite3.connect(str(bck_p))
        tgt_conn = sqlite3.connect(str(tgt_p))

        with tgt_conn:
            bck_conn.backup(tgt_conn)

        bck_conn.close()
        tgt_conn.close()

        return True, f"Restored snapshot from {bck_p} into {tgt_p}"
    except Exception as e:
        return False, f"Failed to restore backup: {str(e)}"


def verify_database_integrity(db_path: str) -> Tuple[bool, str]:
    """
    Verify SQLite file integrity using PRAGMA integrity_check and table validation.
    """
    p = resolve_path(db_path)
    if not p.exists():
        return False, f"Database file does not exist: {p}"

    try:
        conn = sqlite3.connect(str(p))
        cursor = conn.cursor()

        # Run integrity check
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        if not res or res[0] != "ok":
            conn.close()
            return False, f"Integrity check failed: {res}"

        # Verify expected tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        required = {"profiles", "feedback", "active_states"}
        missing = required - tables
        if missing:
            conn.close()
            return False, f"Missing required tables: {missing}"

        # Count records
        cursor.execute("SELECT COUNT(*) FROM profiles;")
        num_profiles = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM feedback;")
        num_feedback = cursor.fetchone()[0]

        conn.close()
        return True, f"Integrity OK. Tables: {list(tables)}, Profiles: {num_profiles}, Feedback: {num_feedback}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def get_persistence_status() -> Dict[str, Any]:
    """
    Return comprehensive diagnostics on runtime database and snapshot backup.
    """
    db_path = resolve_path("fashion_project/backups/profiles.sqlite")
    snapshot_path = resolve_path("fashion_project/backups/profiles_snapshot.sqlite")

    db_ok, db_msg = verify_database_integrity(str(db_path)) if db_path.exists() else (False, "Database not found")
    snap_ok, snap_msg = verify_database_integrity(str(snapshot_path)) if snapshot_path.exists() else (False, "Snapshot not found")

    return {
        "runtime_db": {
            "path": str(db_path),
            "exists": db_path.exists(),
            "valid": db_ok,
            "details": db_msg,
            "size_bytes": db_path.stat().st_size if db_path.exists() else 0
        },
        "snapshot_backup": {
            "path": str(snapshot_path),
            "exists": snapshot_path.exists(),
            "valid": snap_ok,
            "details": snap_msg,
            "size_bytes": snapshot_path.stat().st_size if snapshot_path.exists() else 0
        }
    }


if __name__ == "__main__":
    print("[BACKUP MANAGER] Testing backup and restore workflows...")
    ok, msg = create_database_backup()
    print(f"Backup: {'PASS' if ok else 'FAIL'} - {msg}")
    st = get_persistence_status()
    print(f"Status: {st}")
