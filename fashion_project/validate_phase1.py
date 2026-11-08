"""
validate_phase1.py
Phase 1 Validation and Testing Suite
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Tests all 12 mandatory requirements from Section 11.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_all_checks(project_root: str = "fashion_project"):
    root = Path(project_root)
    if not root.exists():
        root = Path(".")
    artifacts_dir = root / "artifacts"
    raw_dir = root / "data" / "raw"
    styles_csv = raw_dir / "styles.csv"
    images_dir = raw_dir / "images"
    catalog_csv = artifacts_dir / "catalog.csv"
    catalog_ids_json = artifacts_dir / "catalog_ids.json"
    data_audit_json = artifacts_dir / "data_audit.json"
    manifest_json = artifacts_dir / "catalog_manifest.json"
    config_json = root / "config.json"
    
    results = {}
    
    # Check 1: Environment setup completes successfully
    try:
        assert config_json.exists(), "config.json does not exist"
        with open(config_json) as f:
            cfg = json.load(f)
        assert "environment" in cfg
        results["Check 1: Environment setup completes successfully"] = ("PASS", "config.json valid and loaded")
    except Exception as e:
        results["Check 1: Environment setup completes successfully"] = ("FAIL", str(e))
        
    # Check 2: Required dependencies import correctly
    try:
        import pandas as _pd
        import numpy as _np
        import PIL as _PIL
        import pydantic as _pydantic
        results["Check 2: Required dependencies import correctly"] = ("PASS", f"pd={_pd.__version__}, np={_np.__version__}, pil={_PIL.__version__}, pydantic={_pydantic.__version__}")
    except Exception as e:
        results["Check 2: Required dependencies import correctly"] = ("FAIL", str(e))
        
    # Check 3: The dataset loads correctly
    try:
        assert styles_csv.exists(), "styles.csv missing"
        assert images_dir.exists(), "images/ missing"
        df_raw = pd.read_csv(styles_csv, on_bad_lines="skip")
        assert len(df_raw) > 0
        results["Check 3: The dataset loads correctly"] = ("PASS", f"Loaded {len(df_raw)} raw rows")
    except Exception as e:
        results["Check 3: The dataset loads correctly"] = ("FAIL", str(e))
        
    # Check 4: All ten expected columns exist
    expected_cols = [
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName"
    ]
    try:
        catalog_df = pd.read_csv(catalog_csv, dtype={"id": str})
        assert list(catalog_df.columns) == expected_cols, f"Columns mismatch: {list(catalog_df.columns)}"
        results["Check 4: All ten expected columns exist"] = ("PASS", f"All 10 columns present in catalog: {expected_cols}")
    except Exception as e:
        results["Check 4: All ten expected columns exist"] = ("FAIL", str(e))
        
    # Check 5: Product IDs are stored as strings
    try:
        with open(catalog_ids_json) as f:
            ids = json.load(f)
        all_str = all(isinstance(x, str) and len(x) > 0 for x in ids)
        assert all_str, "Not all IDs in catalog_ids.json are non-empty strings"
        assert catalog_df["id"].apply(lambda x: isinstance(x, str)).all(), "catalog.csv id column contains non-strings"
        results["Check 5: Product IDs are stored as strings"] = ("PASS", f"Verified all {len(ids)} IDs are strings")
    except Exception as e:
        results["Check 5: Product IDs are stored as strings"] = ("FAIL", str(e))
        
    # Check 6: No duplicate IDs exist in the final catalog
    try:
        unique_ids = set(ids)
        assert len(unique_ids) == len(ids), f"Duplicate IDs detected in catalog_ids.json: {len(ids) - len(unique_ids)}"
        assert catalog_df["id"].duplicated().sum() == 0, "Duplicate IDs detected in catalog.csv"
        results["Check 6: No duplicate IDs exist in the final catalog"] = ("PASS", f"0 duplicates among {len(ids)} products")
    except Exception as e:
        results["Check 6: No duplicate IDs exist in the final catalog"] = ("FAIL", str(e))
        
    # Check 7: Every selected product has a valid image
    try:
        bad_imgs = []
        for pid in ids:
            img_path = images_dir / f"{pid}.jpg"
            if not img_path.exists():
                bad_imgs.append((pid, "missing"))
            else:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as ex:
                    bad_imgs.append((pid, str(ex)))
        assert len(bad_imgs) == 0, f"Found invalid images: {bad_imgs[:5]}"
        results["Check 7: Every selected product has a valid image"] = ("PASS", f"Verified {len(ids)} product images exist and are decodable")
    except Exception as e:
        results["Check 7: Every selected product has a valid image"] = ("FAIL", str(e))
        
    # Check 8: Every selected record contains sufficient metadata
    try:
        na_count = catalog_df.isna().sum().sum()
        assert na_count == 0, f"Catalog contains {na_count} missing values"
        assert catalog_df["masterCategory"].isin(["Apparel", "Footwear"]).all(), "Catalog contains non-Apparel/Footwear items"
        results["Check 8: Every selected record contains sufficient metadata"] = ("PASS", "0 missing values across all 10 columns; all in Apparel/Footwear")
    except Exception as e:
        results["Check 8: Every selected record contains sufficient metadata"] = ("FAIL", str(e))
        
    # Check 9: The catalog contains no more than 10,000 products
    try:
        assert len(catalog_df) <= 10000, f"Catalog size exceeds 10,000: {len(catalog_df)}"
        assert len(catalog_df) == 10000, f"Expected 10,000 items, got {len(catalog_df)}"
        results["Check 9: The catalog contains no more than 10,000 products"] = ("PASS", f"Catalog size is exactly {len(catalog_df)}")
    except Exception as e:
        results["Check 9: The catalog contains no more than 10,000 products"] = ("FAIL", str(e))
        
    # Check 10: The catalog ID list matches the catalog CSV ordering
    try:
        csv_ids = catalog_df["id"].tolist()
        assert csv_ids == ids, "Mismatch between catalog.csv IDs and catalog_ids.json ordering"
        results["Check 10: The catalog ID list matches the catalog CSV ordering"] = ("PASS", f"Exact match for all {len(ids)} positions")
    except Exception as e:
        results["Check 10: The catalog ID list matches the catalog CSV ordering"] = ("FAIL", str(e))
        
    # Check 11: The manifest hashes match the generated files
    try:
        with open(manifest_json) as f:
            manifest = json.load(f)
        hashes = manifest["sha256_hashes"]
        assert hashes["catalog_csv"] == compute_sha256(str(catalog_csv)), "catalog.csv hash mismatch"
        assert hashes["catalog_ids_json"] == compute_sha256(str(catalog_ids_json)), "catalog_ids.json hash mismatch"
        assert hashes["styles_csv"] == compute_sha256(str(styles_csv)), "styles.csv hash mismatch"
        assert hashes["data_audit_json"] == compute_sha256(str(data_audit_json)), "data_audit.json hash mismatch"
        results["Check 11: The manifest hashes match the generated files"] = ("PASS", "All 4 SHA-256 hashes verified identically")
    except Exception as e:
        results["Check 11: The manifest hashes match the generated files"] = ("FAIL", str(e))
        
    # Check 12: Re-running catalog generation produces the exact same selected product IDs
    try:
        import numpy as np
        # Simulate re-running sampling on same clean candidates
        from prepare_catalog import inspect_and_load_csv, validate_images
        df_re, _, _ = inspect_and_load_csv(str(styles_csv))
        af_re = df_re[df_re["masterCategory"].isin(["Apparel", "Footwear"])].copy()
        valid_img_re, _, _ = validate_images(df_re, str(images_dir))
        clean_re = af_re[af_re["id"].isin(valid_img_re) & ~af_re.isna().any(axis=1)].copy()
        clean_re = clean_re.drop_duplicates(subset=["id"]).sort_values(by="id").reset_index(drop=True)
        
        target_size = 10000
        groups = clean_re.groupby("articleType")
        allocations = {}
        for name, group in groups:
            share = int(np.floor(len(group) * target_size / len(clean_re)))
            allocations[name] = max(1, share)
        diff = target_size - sum(allocations.values())
        sorted_groups = sorted(allocations.keys(), key=lambda k: len(groups.get_group(k)), reverse=True)
        for i in range(abs(diff)):
            k = sorted_groups[i % len(sorted_groups)]
            if diff > 0:
                allocations[k] += 1
            else:
                allocations[k] -= 1
        sampled = []
        for name, group in groups:
            s = group.sample(n=allocations[name], random_state=42)
            sampled.append(s)
        rerun_catalog = pd.concat(sampled, ignore_index=True).sort_values(by="id").reset_index(drop=True)
        rerun_ids = rerun_catalog["id"].tolist()
        assert rerun_ids == ids, "Re-running sampling did not yield identical product IDs!"
        results["Check 12: Re-running catalog generation produces the same selected product IDs"] = ("PASS", f"100% deterministic reproducibility ({len(ids)} identical IDs)")
    except Exception as e:
        results["Check 12: Re-running catalog generation produces the same selected product IDs"] = ("FAIL", str(e))
        
    print("\n" + "="*80)
    print("PHASE 1 RIGOROUS VALIDATION REPORT")
    print("="*80)
    all_passed = True
    for test_name, (status, detail) in results.items():
        print(f"[{status}] {test_name}\n        Details: {detail}")
        if status != "PASS":
            all_passed = False
    print("="*80)
    print(f"OVERALL STATUS: {'ALL 12 CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("="*80 + "\n")
    return results

if __name__ == "__main__":
    run_all_checks()
