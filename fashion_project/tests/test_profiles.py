"""
test_profiles.py
Test Suite for Phase 3: Anonymous Profiles, Explicit Constraints & Feedback Persistence
Authoritative Specification: End_to_End PipeLine.pdf (Steps 5 & 9)
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
import unittest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_manager import (
    resolve_path,
    init_db,
    get_db_connection,
    create_profile,
    update_profile,
    delete_profile,
    get_profile,
    list_profiles,
    save_feedback,
    get_feedback,
    get_active_constraints,
    set_active_constraints,
    clear_active_constraints
)


class TestProfiles(unittest.TestCase):

    def setUp(self):
        self.test_db = str(PROJECT_ROOT / "backups" / "test_unit_profiles.sqlite")
        if Path(self.test_db).exists():
            try:
                Path(self.test_db).unlink()
            except Exception:
                pass
        init_db(self.test_db)

    def tearDown(self):
        if Path(self.test_db).exists():
            try:
                Path(self.test_db).unlink()
            except Exception:
                pass

    def test_01_profile_creation_and_retrieval(self):
        pid = create_profile(db_path=self.test_db)
        self.assertTrue(pid.startswith("usr_"), "Profile ID must start with usr_")
        prof = get_profile(pid, db_path=self.test_db)
        self.assertIsNotNone(prof)
        self.assertEqual(prof["profile_id"], pid)
        self.assertEqual(prof["preferences"], {})

    def test_02_preferences_persistence(self):
        pid = create_profile(db_path=self.test_db)
        prefs = {
            "preferred_categories": ["Apparel"],
            "preferred_colors": ["Navy Blue", "Black"]
        }
        update_profile(pid, prefs, db_path=self.test_db)
        prof = get_profile(pid, db_path=self.test_db)
        self.assertEqual(prof["preferences"], prefs)

    def test_03_active_constraints_lifecycle(self):
        pid = create_profile(db_path=self.test_db)
        constraints = {"articleType": ["Shirts"], "exclude_colors": ["Red"]}
        set_active_constraints(pid, constraints, db_path=self.test_db)

        retrieved = get_active_constraints(pid, db_path=self.test_db)
        self.assertEqual(retrieved, constraints)

        clear_active_constraints(pid, db_path=self.test_db)
        cleared = get_active_constraints(pid, db_path=self.test_db)
        self.assertEqual(cleared, {})

    def test_04_feedback_likes_idempotency(self):
        pid = create_profile(db_path=self.test_db)
        # Record like twice
        save_feedback(pid, "10051", "like", db_path=self.test_db)
        save_feedback(pid, "10051", "like", db_path=self.test_db)

        feedback = get_feedback(pid, db_path=self.test_db)
        self.assertEqual(feedback["likes"], ["10051"], "Repeated likes must be idempotent (1 entry)")

    def test_05_feedback_dismissals(self):
        pid = create_profile(db_path=self.test_db)
        save_feedback(pid, "10128", "dismiss", db_path=self.test_db)

        feedback = get_feedback(pid, db_path=self.test_db)
        self.assertIn("10128", feedback["dismissals"])

    def test_06_profile_isolation(self):
        pid_a = create_profile(db_path=self.test_db)
        pid_b = create_profile(db_path=self.test_db)

        set_active_constraints(pid_a, {"articleType": ["Shirts"]}, db_path=self.test_db)
        set_active_constraints(pid_b, {"articleType": ["Formal Shoes"]}, db_path=self.test_db)

        save_feedback(pid_a, "10051", "like", db_path=self.test_db)
        save_feedback(pid_b, "10128", "like", db_path=self.test_db)

        self.assertEqual(get_active_constraints(pid_a, db_path=self.test_db), {"articleType": ["Shirts"]})
        self.assertEqual(get_active_constraints(pid_b, db_path=self.test_db), {"articleType": ["Formal Shoes"]})
        self.assertEqual(get_feedback(pid_a, db_path=self.test_db)["likes"], ["10051"])
        self.assertEqual(get_feedback(pid_b, db_path=self.test_db)["likes"], ["10128"])


if __name__ == "__main__":
    unittest.main()
