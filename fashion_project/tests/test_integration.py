"""
test_integration.py
Test Suite for Phase 5: Full End-to-End Pipeline Integration, Evaluation & Backup
Authoritative Specification: End_to_End PipeLine.pdf (Steps 9, 10, 11, 12)
"""

import os
import sys
import json
import zipfile
from pathlib import Path
import unittest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_manager import resolve_path, init_db, create_profile, save_feedback
from backup_manager import (
    create_database_backup,
    restore_database_backup,
    verify_database_integrity,
    get_persistence_status
)
from evaluation.run_evaluation import run_full_evaluation, format_evaluation_summary_table
from interactive_ui import FashionRecommenderApp


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db = str(PROJECT_ROOT / "backups" / "test_integ_profiles.sqlite")
        cls.snapshot_db = str(PROJECT_ROOT / "backups" / "test_integ_snapshot.sqlite")
        init_db(cls.test_db)
        pid = create_profile(db_path=cls.test_db)
        save_feedback(pid, "10051", "like", db_path=cls.test_db)

    @classmethod
    def tearDownClass(cls):
        for p in [cls.test_db, cls.snapshot_db]:
            if Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass

    def test_01_sqlite_online_backup_api(self):
        # Test atomic snapshot creation
        ok, msg = create_database_backup(
            source_db_path=self.test_db,
            backup_path=self.snapshot_db
        )
        self.assertTrue(ok, f"Snapshot creation failed: {msg}")
        self.assertTrue(Path(self.snapshot_db).exists())

        # Test integrity check on snapshot
        is_valid, v_msg = verify_database_integrity(self.snapshot_db)
        self.assertTrue(is_valid, f"Snapshot integrity check failed: {v_msg}")

    def test_02_sqlite_snapshot_restore(self):
        restored_db = str(PROJECT_ROOT / "backups" / "test_integ_restored.sqlite")
        try:
            ok, msg = restore_database_backup(
                backup_path=self.snapshot_db,
                target_db_path=restored_db
            )
            self.assertTrue(ok, f"Snapshot restore failed: {msg}")
            is_valid, _ = verify_database_integrity(restored_db)
            self.assertTrue(is_valid, "Restored database integrity failed")
        finally:
            if Path(restored_db).exists():
                try:
                    Path(restored_db).unlink()
                except Exception:
                    pass

    def test_03_evaluation_benchmark_outputs_exist_and_valid(self):
        eval_dir = PROJECT_ROOT / "evaluation"
        pred_p = eval_dir / "predictions.jsonl"
        judge_p = eval_dir / "human_judgements.csv"
        metrics_p = eval_dir / "metrics.json"

        self.assertTrue(pred_p.exists(), "predictions.jsonl not found")
        self.assertTrue(judge_p.exists(), "human_judgements.csv not found")
        self.assertTrue(metrics_p.exists(), "metrics.json not found")

        with open(metrics_p, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertEqual(metrics["evaluation_dataset"]["total_requests"], 40)
        self.assertIn("relevance_precision_at_5", metrics)
        self.assertIn("constraint_satisfaction", metrics)
        # All constraint violation rates must be 0.0%
        for sys_name, rate in metrics["constraint_satisfaction"]["violation_rates_pct"].items():
            self.assertEqual(rate, 0.0, f"{sys_name} had non-zero constraint violation rate: {rate}")

    def test_04_interactive_ui_instantiation(self):
        app = FashionRecommenderApp()
        self.assertIsNotNone(app.current_profile_id)
        self.assertIsNotNone(app.embeddings_matrix)
        self.assertEqual(len(app.catalog_df), 10000)


if __name__ == "__main__":
    unittest.main()
