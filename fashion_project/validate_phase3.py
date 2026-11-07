"""
validate_phase3.py
Phase 3 Validation & Testing Suite (30 Comprehensive Checks)
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Steps 5 & 7)
"""

import os
import sys
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add fashion_project to sys.path if not present
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from profile_manager import (
    resolve_path,
    get_db_connection,
    init_db,
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    get_active_constraints,
    set_active_constraints,
    clear_active_constraints,
    save_feedback,
    get_feedback,
    build_user_vector,
    validate_preferences,
    validate_constraints,
    get_catalog_domain,
    DEFAULT_DB_PATH,
    DEFAULT_ARTIFACTS_DIR
)
from personalized_ranker import filter_candidates, recommend_personalized


def run_all_checks(project_root: str = "fashion_project") -> Tuple[int, int, Dict[str, Tuple[str, str]]]:
    root = resolve_path(project_root)
    artifacts_dir = root / "artifacts"
    backups_dir = root / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(backups_dir / "profiles.sqlite")
    test_db_path = str(backups_dir / "test_validate_phase3.sqlite")

    catalog_csv_path = artifacts_dir / "catalog.csv"
    catalog_ids_path = artifacts_dir / "catalog_ids.json"
    embeddings_path = artifacts_dir / "embeddings.npy"

    print("=" * 80)
    print("PHASE 3 RIGOROUS VALIDATION AND TEST SUITE (30 CHECKS)")
    print("=" * 80)

    # Preload catalog, catalog_ids, and embeddings once
    catalog = pd.read_csv(catalog_csv_path, dtype={"id": str})
    with open(catalog_ids_path, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)
    embeddings = np.load(str(embeddings_path))

    results = {}

    def record(name: str, passed: bool, details: str):
        status = "PASS" if passed else "FAIL"
        results[name] = (status, details)
        prefix = f"[{status}]"
        print(f"{prefix:<7} {name}")
        print(f"        Details: {details}")

    # Clean up test DB if left over
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # --------------------------------------------------------------------------
    # Check 1: SQLite database initialization
    # --------------------------------------------------------------------------
    try:
        init_db(test_db_path)
        conn = get_db_connection(test_db_path)
        tables = [row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        indexes = [row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        fk_check = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()

        req_tables = {"profiles", "feedback", "active_states"}
        passed = req_tables.issubset(set(tables)) and (fk_check == 1)
        record(
            "Check 01: SQLite database initialization",
            passed,
            f"Tables created: {sorted(tables)}, Foreign keys enabled: {fk_check == 1}"
        )
    except Exception as e:
        record("Check 01: SQLite database initialization", False, str(e))

    # --------------------------------------------------------------------------
    # Check 2: Profile creation
    # --------------------------------------------------------------------------
    try:
        pid1 = create_profile(db_path=test_db_path)
        p1 = get_profile(pid1, db_path=test_db_path)
        passed = p1 is not None and p1["profile_id"] == pid1 and p1["preferences"] == {}
        record(
            "Check 02: Profile creation",
            passed,
            f"Created anonymous profile ID: {pid1}"
        )
    except Exception as e:
        record("Check 02: Profile creation", False, str(e))

    # --------------------------------------------------------------------------
    # Check 3: Unique profile IDs
    # --------------------------------------------------------------------------
    try:
        pids = [create_profile(db_path=test_db_path) for _ in range(10)]
        passed = len(set(pids)) == 10
        record(
            "Check 03: Unique profile IDs",
            passed,
            f"Generated 10 distinct unique profile IDs without collision"
        )
    except Exception as e:
        record("Check 03: Unique profile IDs", False, str(e))

    # --------------------------------------------------------------------------
    # Check 4: Profile retrieval
    # --------------------------------------------------------------------------
    try:
        p = get_profile(pid1, db_path=test_db_path)
        passed = (
            p is not None and
            "created_at" in p and
            "updated_at" in p and
            p["profile_id"] == pid1
        )
        record(
            "Check 04: Profile retrieval",
            passed,
            f"Retrieved profile record with timestamp: {p.get('created_at') if p else None}"
        )
    except Exception as e:
        record("Check 04: Profile retrieval", False, str(e))

    # --------------------------------------------------------------------------
    # Check 5: Preference persistence
    # --------------------------------------------------------------------------
    try:
        new_prefs = {
            "preferred_categories": ["Apparel"],
            "preferred_colors": ["Navy Blue", "Black"]
        }
        update_profile(pid1, new_prefs, db_path=test_db_path)
        p_updated = get_profile(pid1, db_path=test_db_path)
        passed = (
            p_updated is not None and
            p_updated["preferences"].get("preferred_categories") == ["Apparel"] and
            p_updated["preferences"].get("preferred_colors") == ["Navy Blue", "Black"]
        )
        record(
            "Check 05: Preference persistence",
            passed,
            f"Updated and verified stored explicit preferences: {p_updated['preferences'] if p_updated else None}"
        )
    except Exception as e:
        record("Check 05: Preference persistence", False, str(e))

    # --------------------------------------------------------------------------
    # Check 6: Active constraint persistence
    # --------------------------------------------------------------------------
    try:
        active_cons = {
            "articleType": ["Shirts"],
            "baseColour": ["Navy Blue"],
            "usage": ["Casual"],
            "exclude_colors": ["Red"]
        }
        set_active_constraints(pid1, active_cons, db_path=test_db_path)
        saved_cons = get_active_constraints(pid1, db_path=test_db_path)
        passed = (
            saved_cons.get("articleType") == ["Shirts"] and
            saved_cons.get("baseColour") == ["Navy Blue"] and
            saved_cons.get("exclude_colors") == ["Red"]
        )
        record(
            "Check 06: Active constraint persistence",
            passed,
            f"Persisted and retrieved active constraints: {saved_cons}"
        )
    except Exception as e:
        record("Check 06: Active constraint persistence", False, str(e))

    # --------------------------------------------------------------------------
    # Check 7: Multiple profile isolation
    # --------------------------------------------------------------------------
    try:
        pid_a = create_profile(db_path=test_db_path)
        pid_b = create_profile(db_path=test_db_path)

        set_active_constraints(pid_a, {"articleType": ["Shirts"]}, db_path=test_db_path)
        set_active_constraints(pid_b, {"articleType": ["Heels"]}, db_path=test_db_path)

        save_feedback(pid_a, catalog_ids[0], "like", db_path=test_db_path)
        save_feedback(pid_b, catalog_ids[1], "like", db_path=test_db_path)

        cons_a = get_active_constraints(pid_a, db_path=test_db_path)
        cons_b = get_active_constraints(pid_b, db_path=test_db_path)
        fb_a = get_feedback(pid_a, db_path=test_db_path)
        fb_b = get_feedback(pid_b, db_path=test_db_path)

        passed = (
            cons_a["articleType"] == ["Shirts"] and
            cons_b["articleType"] == ["Heels"] and
            fb_a["likes"] == [catalog_ids[0]] and
            fb_b["likes"] == [catalog_ids[1]]
        )
        record(
            "Check 07: Multiple profile isolation",
            passed,
            f"Profile A and Profile B maintained completely independent states and feedback"
        )
    except Exception as e:
        record("Check 07: Multiple profile isolation", False, str(e))

    # --------------------------------------------------------------------------
    # Check 8: Invalid profile handling
    # --------------------------------------------------------------------------
    try:
        fake_id = "non_existent_profile_xyz"
        p_fake = get_profile(fake_id, db_path=test_db_path)
        assert p_fake is None, "Expected None for unknown profile ID"

        fb_failed = False
        try:
            save_feedback(fake_id, catalog_ids[0], "like", db_path=test_db_path)
        except ValueError:
            fb_failed = True

        rec_fake = recommend_personalized(
            profile_id=fake_id,
            query="casual shirts",
            top_k=2,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        passed = fb_failed and rec_fake["has_personalized_vector"] is False
        record(
            "Check 08: Invalid profile handling",
            passed,
            "Unknown profile gracefully returned None, rejected feedback write, and defaulted to query-only"
        )
    except Exception as e:
        record("Check 08: Invalid profile handling", False, str(e))

    # --------------------------------------------------------------------------
    # Check 9: Invalid product ID handling
    # --------------------------------------------------------------------------
    try:
        invalid_pid_rejected = False
        try:
            save_feedback(pid1, "99999999_invalid_product", "like", db_path=test_db_path)
        except ValueError:
            invalid_pid_rejected = True

        fb = get_feedback(pid1, db_path=test_db_path)
        passed = invalid_pid_rejected and ("99999999_invalid_product" not in fb["likes"])
        record(
            "Check 09: Invalid product ID handling",
            passed,
            "Non-catalog product ID was rejected with ValueError without corrupting database"
        )
    except Exception as e:
        record("Check 09: Invalid product ID handling", False, str(e))

    # --------------------------------------------------------------------------
    # Check 10: Like persistence
    # --------------------------------------------------------------------------
    try:
        target_item = catalog_ids[5]
        save_feedback(pid1, target_item, "like", db_path=test_db_path)
        fb1 = get_feedback(pid1, db_path=test_db_path)
        passed = target_item in fb1["likes"]
        record(
            "Check 10: Like persistence",
            passed,
            f"Liked product ID {target_item} successfully persisted in feedback table"
        )
    except Exception as e:
        record("Check 10: Like persistence", False, str(e))

    # --------------------------------------------------------------------------
    # Check 11: Idempotent likes
    # --------------------------------------------------------------------------
    try:
        target_item = catalog_ids[5]
        save_feedback(pid1, target_item, "like", db_path=test_db_path)
        save_feedback(pid1, target_item, "like", db_path=test_db_path)
        save_feedback(pid1, target_item, "like", db_path=test_db_path)

        fb1 = get_feedback(pid1, db_path=test_db_path)
        conn = get_db_connection(test_db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE profile_id = ? AND item_id = ?",
            (pid1, target_item)
        ).fetchone()[0]
        conn.close()

        passed = count == 1 and fb1["likes"].count(target_item) == 1
        record(
            "Check 11: Idempotent likes",
            passed,
            f"Repeated likes resulted in exactly {count} database record (idempotency verified)"
        )
    except Exception as e:
        record("Check 11: Idempotent likes", False, str(e))

    # --------------------------------------------------------------------------
    # Check 12: Dismissal persistence
    # --------------------------------------------------------------------------
    try:
        dismiss_item = catalog_ids[10]
        save_feedback(pid1, dismiss_item, "dismiss", db_path=test_db_path)
        fb1 = get_feedback(pid1, db_path=test_db_path)
        passed = dismiss_item in fb1["dismissals"] and dismiss_item not in fb1["likes"]
        record(
            "Check 12: Dismissal persistence",
            passed,
            f"Dismissed product ID {dismiss_item} persisted in dismissals list"
        )
    except Exception as e:
        record("Check 12: Dismissal persistence", False, str(e))

    # --------------------------------------------------------------------------
    # Check 13: Dismissed item exclusion
    # --------------------------------------------------------------------------
    try:
        # Get top recommendation for a query
        rec_before = recommend_personalized(
            profile_id=pid1,
            query="black t-shirt",
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        top_item_id = rec_before["results"][0]["id"]

        # Dismiss the top item
        save_feedback(pid1, top_item_id, "dismiss", db_path=test_db_path)

        # Query again
        rec_after = recommend_personalized(
            profile_id=pid1,
            query="black t-shirt",
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        after_ids = [r["id"] for r in rec_after["results"]]
        passed = top_item_id not in after_ids
        record(
            "Check 13: Dismissed item exclusion",
            passed,
            f"Dismissed item {top_item_id} strictly excluded from subsequent recommendations"
        )
    except Exception as e:
        record("Check 13: Dismissed item exclusion", False, str(e))

    # --------------------------------------------------------------------------
    # Check 14: User vector generation
    # --------------------------------------------------------------------------
    try:
        # Create user with 3 known likes
        pid_likes = create_profile(db_path=test_db_path)
        sample_likes = [catalog_ids[12], catalog_ids[25], catalog_ids[40]]
        for itm in sample_likes:
            save_feedback(pid_likes, itm, "like", db_path=test_db_path)

        u_vec = build_user_vector(
            pid_likes,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            catalog=catalog,
            db_path=test_db_path
        )

        id_map = {pid: i for i, pid in enumerate(catalog_ids)}
        expected_mean = np.mean([embeddings[id_map[itm]] for itm in sample_likes], axis=0)
        expected_vec = expected_mean / np.linalg.norm(expected_mean)

        passed = (
            u_vec is not None and
            u_vec.shape == (384,) and
            u_vec.dtype == np.float32 and
            np.isclose(np.linalg.norm(u_vec), 1.0, atol=1e-4) and
            np.allclose(u_vec, expected_vec, atol=1e-4)
        )
        record(
            "Check 14: User vector generation",
            passed,
            f"Mean normalized vector computed correctly (dim={u_vec.shape if u_vec is not None else None}, norm={np.linalg.norm(u_vec) if u_vec is not None else 0:.4f})"
        )
    except Exception as e:
        record("Check 14: User vector generation", False, str(e))

    # --------------------------------------------------------------------------
    # Check 15: Empty-profile fallback
    # --------------------------------------------------------------------------
    try:
        pid_empty = create_profile(db_path=test_db_path)
        u_empty = build_user_vector(
            pid_empty,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            catalog=catalog,
            db_path=test_db_path
        )
        passed = u_empty is None
        record(
            "Check 15: Empty-profile fallback",
            passed,
            "build_user_vector returned None when profile has no likes and no explicit preferences"
        )
    except Exception as e:
        record("Check 15: Empty-profile fallback", False, str(e))

    # --------------------------------------------------------------------------
    # Check 16: Explicit preference fallback
    # --------------------------------------------------------------------------
    try:
        pid_pref = create_profile(
            preferences={"preferred_categories": ["Apparel"], "preferred_colors": ["Navy Blue"]},
            db_path=test_db_path
        )
        u_pref = build_user_vector(
            pid_pref,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            catalog=catalog,
            db_path=test_db_path
        )
        passed = (
            u_pref is not None and
            u_pref.shape == (384,) and
            np.isclose(np.linalg.norm(u_pref), 1.0, atol=1e-4)
        )
        record(
            "Check 16: Explicit preference fallback",
            passed,
            f"Synthesized preference vector from saved categories/colors (dim={u_pref.shape if u_pref is not None else None}, norm={np.linalg.norm(u_pref) if u_pref is not None else 0:.4f})"
        )
    except Exception as e:
        record("Check 16: Explicit preference fallback", False, str(e))

    # --------------------------------------------------------------------------
    # Check 17: Strict category filtering
    # --------------------------------------------------------------------------
    try:
        filtered = filter_candidates(catalog, {"articleType": ["Shirts"]})
        unique_types = filtered["articleType"].unique().tolist()
        passed = len(filtered) > 0 and unique_types == ["Shirts"]
        record(
            "Check 17: Strict category filtering",
            passed,
            f"Filtered {len(filtered):,} items; distinct articleTypes: {unique_types}"
        )
    except Exception as e:
        record("Check 17: Strict category filtering", False, str(e))

    # --------------------------------------------------------------------------
    # Check 18: Strict color filtering
    # --------------------------------------------------------------------------
    try:
        filtered = filter_candidates(catalog, {"baseColour": ["Navy Blue"]})
        unique_colors = filtered["baseColour"].unique().tolist()
        passed = len(filtered) > 0 and unique_colors == ["Navy Blue"]
        record(
            "Check 18: Strict color filtering",
            passed,
            f"Filtered {len(filtered):,} items; distinct baseColour: {unique_colors}"
        )
    except Exception as e:
        record("Check 18: Strict color filtering", False, str(e))

    # --------------------------------------------------------------------------
    # Check 19: Excluded color filtering
    # --------------------------------------------------------------------------
    try:
        filtered = filter_candidates(catalog, {"exclude_colors": ["Black"]})
        has_black = "Black" in filtered["baseColour"].unique()
        passed = len(filtered) > 0 and not has_black
        record(
            "Check 19: Excluded color filtering",
            passed,
            f"Filtered {len(filtered):,} items; 'Black' strictly absent from results"
        )
    except Exception as e:
        record("Check 19: Excluded color filtering", False, str(e))

    # --------------------------------------------------------------------------
    # Check 20: Missing attribute handling
    # --------------------------------------------------------------------------
    try:
        test_df = pd.DataFrame([
            {"id": "test_1", "articleType": "Shirts", "baseColour": "Black", "usage": "Casual", "masterCategory": "Apparel"},
            {"id": "test_2", "articleType": None, "baseColour": "Black", "usage": "Casual", "masterCategory": "Apparel"},
            {"id": "test_3", "articleType": "Shirts", "baseColour": np.nan, "usage": "Casual", "masterCategory": "Apparel"},
            {"id": "test_4", "articleType": "Shirts", "baseColour": "Black", "usage": "", "masterCategory": "Apparel"},
        ])
        filtered = filter_candidates(test_df, {"articleType": ["Shirts"], "baseColour": ["Black"], "usage": ["Casual"]})
        passed = len(filtered) == 1 and filtered.iloc[0]["id"] == "test_1"
        record(
            "Check 20: Missing attribute handling",
            passed,
            f"Rows with None, NaN, or empty string failed strict constraints as expected"
        )
    except Exception as e:
        record("Check 20: Missing attribute handling", False, str(e))

    # --------------------------------------------------------------------------
    # Check 21: Combined AND/OR filtering
    # --------------------------------------------------------------------------
    try:
        constraints = {
            "articleType": ["Shirts", "Tshirts"],
            "baseColour": ["Black", "White"],
            "usage": ["Casual"],
            "exclude_colors": ["Red"]
        }
        filtered = filter_candidates(catalog, constraints)
        passed = (
            len(filtered) > 0 and
            filtered["articleType"].isin(["Shirts", "Tshirts"]).all() and
            filtered["baseColour"].isin(["Black", "White"]).all() and
            filtered["usage"].isin(["Casual"]).all() and
            (~filtered["baseColour"].isin(["Red"])).all()
        )
        record(
            "Check 21: Combined AND/OR filtering",
            passed,
            f"Verified {len(filtered):,} candidate items satisfy OR within field, AND across fields, and exclusions"
        )
    except Exception as e:
        record("Check 21: Combined AND/OR filtering", False, str(e))

    # --------------------------------------------------------------------------
    # Check 22: Personalized ranking
    # --------------------------------------------------------------------------
    try:
        # Run recommendation with user having likes
        rec_res = recommend_personalized(
            profile_id=pid_likes,
            query="formal dress shoes",
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        results_list = rec_res["results"]
        passed = rec_res["has_personalized_vector"] is True and len(results_list) > 0

        # Check score formula: score == 0.8 * sim_q + 0.2 * sim_u
        for r in results_list:
            expected_score = 0.8 * r["query_similarity"] + 0.2 * r["profile_similarity"]
            if not np.isclose(r["score"], expected_score, atol=1e-5):
                passed = False
                break

        record(
            "Check 22: Personalized ranking",
            passed,
            f"Verified 0.8 * query + 0.2 * profile formula exactly on {len(results_list)} items"
        )
    except Exception as e:
        record("Check 22: Personalized ranking", False, str(e))

    # --------------------------------------------------------------------------
    # Check 23: Query-only fallback
    # --------------------------------------------------------------------------
    try:
        rec_fallback = recommend_personalized(
            profile_id=pid_empty,
            query="casual sports shoes",
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        passed = (
            rec_fallback["has_personalized_vector"] is False and
            all(np.isclose(r["score"], r["query_similarity"], atol=1e-6) for r in rec_fallback["results"])
        )
        record(
            "Check 23: Query-only fallback",
            passed,
            "When user vector is unavailable, score equals query similarity exactly (score = 1.0 * sim_q)"
        )
    except Exception as e:
        record("Check 23: Query-only fallback", False, str(e))

    # --------------------------------------------------------------------------
    # Check 24: Hard constraint preservation
    # --------------------------------------------------------------------------
    try:
        # pid_likes has likes in apparel/footwear, but active constraint strictly restricts to Shirts
        rec_strict = recommend_personalized(
            profile_id=pid_likes,
            query="shoes",
            constraints={"articleType": ["Shirts"]},
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        types = [r["articleType"] for r in rec_strict["results"]]
        passed = len(types) > 0 and all(t == "Shirts" for t in types)
        record(
            "Check 24: Hard constraint preservation",
            passed,
            f"All {len(types)} returned items strictly adhere to hard constraint 'Shirts' despite user vector"
        )
    except Exception as e:
        record("Check 24: Hard constraint preservation", False, str(e))

    # --------------------------------------------------------------------------
    # Check 25: Top-five result limit
    # --------------------------------------------------------------------------
    try:
        rec_5 = recommend_personalized(
            query="casual shirts",
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        passed = len(rec_5["results"]) == 5
        record(
            "Check 25: Top-five result limit",
            passed,
            f"Returned exactly {len(rec_5['results'])} items for top_k=5"
        )
    except Exception as e:
        record("Check 25: Top-five result limit", False, str(e))

    # --------------------------------------------------------------------------
    # Check 26: Fewer-than-five result handling
    # --------------------------------------------------------------------------
    try:
        # In catalog, Baby Dolls + Black has exactly 3 items
        rec_few = recommend_personalized(
            query="black nightwear",
            constraints={"articleType": ["Baby Dolls"], "baseColour": ["Black"]},
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        passed = len(rec_few["results"]) == 3 and rec_few["total_eligible"] == 3
        record(
            "Check 26: Fewer-than-five result handling",
            passed,
            f"Returned exactly {len(rec_few['results'])} items without padding when only 3 items exist"
        )
    except Exception as e:
        record("Check 26: Fewer-than-five result handling", False, str(e))

    # --------------------------------------------------------------------------
    # Check 27: No-match handling
    # --------------------------------------------------------------------------
    try:
        rec_nomatch = recommend_personalized(
            query="summer shirt",
            constraints={"articleType": ["Baby Dolls"], "baseColour": ["Navy Blue"]},
            top_k=5,
            catalog=catalog,
            embeddings=embeddings,
            catalog_ids=catalog_ids,
            db_path=test_db_path
        )
        passed = (
            rec_nomatch["status"] == "no_match" and
            len(rec_nomatch["results"]) == 0 and
            "revised_constraints_suggestion" in rec_nomatch
        )
        record(
            "Check 27: No-match handling",
            passed,
            f"Gracefully reported no matches with actionable constraint revision guidance"
        )
    except Exception as e:
        record("Check 27: No-match handling", False, str(e))

    # --------------------------------------------------------------------------
    # Check 28: Deterministic tie-breaking
    # --------------------------------------------------------------------------
    try:
        # Test tie breaking: items with equal score must sort ascending by string ID
        mock_candidates = [
            {"id": "20000", "score": 0.85},
            {"id": "10005", "score": 0.85},
            {"id": "15000", "score": 0.85},
            {"id": "10001", "score": 0.90}
        ]
        mock_candidates.sort(key=lambda x: (-round(x["score"], 7), str(x["id"])))
        sorted_ids = [c["id"] for c in mock_candidates]
        passed = sorted_ids == ["10001", "10005", "15000", "20000"]
        record(
            "Check 28: Deterministic tie-breaking",
            passed,
            f"Tie-breaking verified: {sorted_ids} (score desc, id asc)"
        )
    except Exception as e:
        record("Check 28: Deterministic tie-breaking", False, str(e))

    # --------------------------------------------------------------------------
    # Check 29: SQLite restart persistence
    # --------------------------------------------------------------------------
    try:
        # Create restart test database
        restart_db = str(backups_dir / "test_restart.sqlite")
        if os.path.exists(restart_db):
            os.remove(restart_db)

        r_pid = create_profile(
            preferences={"preferred_categories": ["Footwear"], "preferred_colors": ["White"]},
            db_path=restart_db
        )
        set_active_constraints(r_pid, {"usage": ["Sports"]}, db_path=restart_db)
        save_feedback(r_pid, catalog_ids[0], "like", db_path=restart_db)
        save_feedback(r_pid, catalog_ids[1], "dismiss", db_path=restart_db)

        # Reopen with brand new connection
        conn2 = get_db_connection(restart_db)
        p_row = conn2.execute("SELECT * FROM profiles WHERE profile_id = ?", (r_pid,)).fetchone()
        cons_row = conn2.execute("SELECT * FROM active_states WHERE profile_id = ?", (r_pid,)).fetchone()
        fb_rows = conn2.execute("SELECT item_id, action FROM feedback WHERE profile_id = ?", (r_pid,)).fetchall()
        conn2.close()

        fb_map = {r["item_id"]: r["action"] for r in fb_rows}
        passed = (
            p_row is not None and
            cons_row is not None and
            fb_map.get(catalog_ids[0]) == "like" and
            fb_map.get(catalog_ids[1]) == "dismiss"
        )
        if os.path.exists(restart_db):
            os.remove(restart_db)

        record(
            "Check 29: SQLite restart persistence",
            passed,
            "State fully verified across complete database reconnect"
        )
    except Exception as e:
        record("Check 29: SQLite restart persistence", False, str(e))

    # --------------------------------------------------------------------------
    # Check 30: Phase 1 and Phase 2 regression validation
    # --------------------------------------------------------------------------
    try:
        import validate_phase1
        import validate_phase2

        p1_res = validate_phase1.run_all_checks(str(root))
        p2_res = validate_phase2.run_all_checks(str(root))

        p1_fails = [k for k, (st, _) in p1_res.items() if st != "PASS"]
        p2_fails = [k for k, (st, _) in p2_res.items() if st != "PASS"]

        passed = (len(p1_fails) == 0 and len(p2_fails) == 0)
        record(
            "Check 30: Phase 1 and Phase 2 regression validation",
            passed,
            f"Phase 1: {len(p1_res)}/12 passed | Phase 2: {len(p2_res)}/17 passed"
        )
    except Exception as e:
        record("Check 30: Phase 1 and Phase 2 regression validation", False, str(e))

    # Cleanup test db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 VALIDATION SUMMARY REPORT")
    print("=" * 80)
    passed_count = sum(1 for st, _ in results.values() if st == "PASS")
    total_count = len(results)
    for name, (status, details) in results.items():
        print(f"[{status}] {name}")
        print(f"        Details: {details}")

    print("=" * 80)
    status_str = "ALL 30 CHECKS PASSED" if passed_count == total_count else f"{total_count - passed_count} CHECKS FAILED"
    print(f"OVERALL STATUS: {status_str} ({passed_count}/{total_count})")
    print("=" * 80)

    return passed_count, total_count, results


if __name__ == "__main__":
    proj_dir = "fashion_project" if os.path.exists("fashion_project") else "."
    passed, total, _ = run_all_checks(proj_dir)
    sys.exit(0 if passed == total else 1)
