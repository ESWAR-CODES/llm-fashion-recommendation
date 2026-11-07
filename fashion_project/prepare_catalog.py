"""
prepare_catalog.py
Phase 1: Environment Setup, Dataset Acquisition & Catalog Preparation
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf
"""

import os
import sys
import csv
import json
import time
import shutil
import hashlib
import datetime
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

NA_STRINGS = {"", "NA", "NaN", "null", "none"}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_system_ram_gb() -> float:
    """Detect available system RAM cross-platform."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024**3), 2)
    elif os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    return round(int(line.split()[1]) / (1024**2), 2)
    return 0.0


def detect_environment():
    """Detect runtime platform, hardware, python, torch and storage."""
    is_colab = "google.colab" in sys.modules
    python_ver = sys.version.split()[0]
    
    # PyTorch and GPU detection
    torch_ver = "Not Installed"
    cuda_avail = False
    gpu_name = "None"
    gpu_mem_gb = 0.0
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        pass

    ram_gb = get_system_ram_gb()
    disk_total, disk_used, disk_free = shutil.disk_usage(".")
    
    env_info = {
        "is_colab": is_colab,
        "python_version": python_ver,
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "system_ram_gb": ram_gb,
        "disk_total_gb": round(disk_total / (1024**3), 2),
        "disk_free_gb": round(disk_free / (1024**3), 2),
        "os_platform": platform.platform()
    }
    return env_info


def inspect_and_load_csv(csv_path: str):
    """
    Carefully parse styles.csv to detect malformed lines,
    validate field counts and column headers without silently skipping rows.
    Empty strings and NA markers are converted to None so missing attributes are correctly detected.
    """
    expected_columns = [
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName"
    ]
    
    malformed_records = []
    valid_rows = []
    
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != expected_columns:
            raise ValueError(f"Column header mismatch! Expected: {expected_columns}, Found: {header}")
        
        for line_idx, row in enumerate(reader, start=2):
            if len(row) != 10:
                malformed_records.append({
                    "line_number": line_idx,
                    "field_count": len(row),
                    "raw_tokens": row,
                    "id": row[0] if len(row) > 0 else f"unknown_line_{line_idx}",
                    "reason": "malformed_csv_token_count_mismatch"
                })
            else:
                cleaned_row = [val.strip() if val.strip() not in NA_STRINGS else None for val in row]
                valid_rows.append(cleaned_row)
                
    df = pd.DataFrame(valid_rows, columns=expected_columns)
    df["id"] = df["id"].astype(str).str.strip()
    return df, malformed_records, len(valid_rows) + len(malformed_records)


def validate_images(df: pd.DataFrame, images_dir: str):
    """
    Validate product images using Pillow.
    Checks file existence, readability and full decompression.
    """
    existing_files = set(os.listdir(images_dir))
    product_ids = df["id"].tolist()
    
    missing_image_ids = []
    existing_image_ids = []
    
    for pid in product_ids:
        filename = f"{pid}.jpg"
        if filename not in existing_files:
            missing_image_ids.append(pid)
        else:
            existing_image_ids.append(pid)
            
    # Verify and load each existing image
    corrupted_images = []
    
    def check_file(pid):
        path = os.path.join(images_dir, f"{pid}.jpg")
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
            return pid, True, None
        except Exception as e:
            return pid, False, str(e)

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(check_file, existing_image_ids))
        
    valid_image_ids = []
    for pid, ok, err in results:
        if ok:
            valid_image_ids.append(pid)
        else:
            corrupted_images.append({"id": pid, "error": err, "reason": "corrupted_image_file"})
            
    return set(valid_image_ids), missing_image_ids, corrupted_images


def run_pipeline(project_root: str = "fashion_project"):
    """
    Execute Phase 1 data pipeline end-to-end:
    - Ingest & inspect raw data
    - Validate images
    - Filter Apparel and Footwear
    - Exclude invalid & missing metadata records
    - Stratified sampling to 10,000 items with seed 42
    - Export catalog.csv, catalog_ids.json, data_audit.json, catalog_manifest.json, config.json
    """
    root = Path(project_root)
    raw_dir = root / "data" / "raw"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    styles_csv = raw_dir / "styles.csv"
    images_dir = raw_dir / "images"
    
    if not styles_csv.exists():
        raise FileNotFoundError(f"styles.csv not found at {styles_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found at {images_dir}")
        
    print("[1/6] Detecting environment configuration...")
    env_info = detect_environment()
    
    print("[2/6] Inspecting and parsing raw dataset...")
    df, malformed_records, total_raw_rows = inspect_and_load_csv(str(styles_csv))
    print(f"  Total raw rows in styles.csv: {total_raw_rows}")
    print(f"  Valid 10-field rows: {len(df)}")
    print(f"  Malformed rows detected: {len(malformed_records)}")
    
    # Dataset statistics before cleaning
    duplicate_ids_count = int(df["id"].duplicated().sum())
    duplicate_rows_count = int(df.duplicated().sum())
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}
    print(f"  Missing attributes by column: {missing_by_col}")
    
    print("[3/6] Auditing and validating image files with Pillow...")
    valid_img_ids, missing_img_ids, corrupted_images = validate_images(df, str(images_dir))
    print(f"  Valid images: {len(valid_img_ids)}")
    print(f"  Missing images: {len(missing_img_ids)}")
    print(f"  Corrupted images: {len(corrupted_images)}")
    
    print("[4/6] Cleaning data according to Phase 1 rules...")
    # Exclusion tracking
    excluded_records = []
    
    # 1. Log malformed rows
    for m in malformed_records:
        excluded_records.append({
            "id": m["id"],
            "line_number": m["line_number"],
            "reason": m["reason"]
        })
        
    # 2. Filter masterCategory in Apparel & Footwear
    apparel_footwear_mask = df["masterCategory"].isin(["Apparel", "Footwear"])
    out_of_scope_df = df[~apparel_footwear_mask]
    for _, row in out_of_scope_df.iterrows():
        excluded_records.append({
            "id": row["id"],
            "category": row["masterCategory"],
            "reason": "out_of_scope_master_category"
        })
        
    af_df = df[apparel_footwear_mask].copy()
    apparel_count = int((af_df["masterCategory"] == "Apparel").sum())
    footwear_count = int((af_df["masterCategory"] == "Footwear").sum())
    total_af_count = len(af_df)
    
    # 3. Check missing images in Apparel & Footwear
    missing_img_set = set(missing_img_ids)
    corrupted_img_set = set(c["id"] for c in corrupted_images)
    
    missing_img_af = af_df[af_df["id"].isin(missing_img_set)]
    for _, row in missing_img_af.iterrows():
        excluded_records.append({
            "id": row["id"],
            "reason": "missing_image_file"
        })
        
    corrupted_img_af = af_df[af_df["id"].isin(corrupted_img_set)]
    for _, row in corrupted_img_af.iterrows():
        excluded_records.append({
            "id": row["id"],
            "reason": "corrupted_image_file"
        })
        
    # 4. Check missing attributes in required fields
    clean_image_mask = af_df["id"].isin(valid_img_ids)
    af_with_images = af_df[clean_image_mask].copy()
    
    missing_attr_mask = af_with_images.isna().any(axis=1)
    missing_attr_df = af_with_images[missing_attr_mask]
    for _, row in missing_attr_df.iterrows():
        missing_fields = [c for c in af_with_images.columns if pd.isna(row[c])]
        excluded_records.append({
            "id": row["id"],
            "missing_fields": missing_fields,
            "reason": "missing_attribute_values"
        })
        
    # Filter candidates with complete metadata and valid images
    candidates = af_with_images[~missing_attr_mask].copy()
    candidates = candidates.drop_duplicates(subset=["id"]).reset_index(drop=True)
    candidates = candidates.sort_values(by="id").reset_index(drop=True)
    total_candidates = len(candidates)
    print(f"  Apparel items: {apparel_count}")
    print(f"  Footwear items: {footwear_count}")
    print(f"  Apparel + Footwear clean candidates: {total_candidates}")
    
    print("[5/6] Performing reproducible stratified sampling across articleType...")
    target_catalog_size = min(10000, total_candidates)
    groups = candidates.groupby("articleType")
    
    allocations = {}
    for name, group in groups:
        share = int(np.floor(len(group) * target_catalog_size / total_candidates))
        allocations[name] = max(1, share)
        
    current_sum = sum(allocations.values())
    diff = target_catalog_size - current_sum
    sorted_groups = sorted(allocations.keys(), key=lambda k: len(groups.get_group(k)), reverse=True)
    for i in range(abs(diff)):
        k = sorted_groups[i % len(sorted_groups)]
        if diff > 0:
            allocations[k] += 1
        else:
            allocations[k] -= 1
            
    sampled_subsets = []
    for name, group in groups:
        n_sample = allocations[name]
        sampled = group.sample(n=n_sample, random_state=42)
        sampled_subsets.append(sampled)
        
    catalog_df = pd.concat(sampled_subsets, ignore_index=True)
    catalog_df = catalog_df.sort_values(by="id").reset_index(drop=True)
    catalog_ids = catalog_df["id"].tolist()
    
    # Track unselected candidates in audit
    selected_id_set = set(catalog_ids)
    unselected_df = candidates[~candidates["id"].isin(selected_id_set)]
    for _, row in unselected_df.iterrows():
        excluded_records.append({
            "id": row["id"],
            "reason": "sampling_unselected"
        })
        
    print(f"  Final frozen catalog size: {len(catalog_df)} products")
    print(f"  Represented article types: {catalog_df['articleType'].nunique()}")
    
    print("[6/6] Generating and hashing artifacts...")
    catalog_csv_path = artifacts_dir / "catalog.csv"
    catalog_ids_path = artifacts_dir / "catalog_ids.json"
    data_audit_path = artifacts_dir / "data_audit.json"
    manifest_path = artifacts_dir / "catalog_manifest.json"
    config_path = root / "config.json"
    requirements_lock_path = root / "requirements-lock.txt"
    
    # Save catalog.csv
    catalog_df.to_csv(catalog_csv_path, index=False)
    
    # Save catalog_ids.json
    with open(catalog_ids_path, "w", encoding="utf-8") as f:
        json.dump(catalog_ids, f, indent=2)
        
    # Distributions
    article_type_dist = catalog_df["articleType"].value_counts().to_dict()
    gender_dist = catalog_df["gender"].value_counts().to_dict()
    color_dist = catalog_df["baseColour"].value_counts().to_dict()
    usage_dist = catalog_df["usage"].value_counts().to_dict()
    season_dist = catalog_df["season"].value_counts().to_dict()
    
    # Build data_audit.json
    audit_data = {
        "dataset_name": "Fashion Product Images Small",
        "dataset_source": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
        "audit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_dataset_row_count": total_raw_rows,
        "original_valid_field_count_rows": len(df),
        "malformed_rows_count": len(malformed_records),
        "original_unique_product_count": df["id"].nunique(),
        "duplicate_ids_count": duplicate_ids_count,
        "duplicate_records_count": duplicate_rows_count,
        "missing_attributes_by_column": missing_by_col,
        "missing_image_count": len(missing_img_ids),
        "missing_image_ids": missing_img_ids,
        "corrupted_image_count": len(corrupted_images),
        "corrupted_image_details": corrupted_images,
        "apparel_total_count": apparel_count,
        "footwear_total_count": footwear_count,
        "apparel_and_footwear_total": total_af_count,
        "eligible_candidates_after_cleaning": total_candidates,
        "final_usable_catalog_count": len(catalog_df),
        "final_article_type_distribution": article_type_dist,
        "final_gender_distribution": gender_dist,
        "final_color_distribution": color_dist,
        "final_usage_distribution": usage_dist,
        "final_season_distribution": season_dist,
        "exclusion_summary": {
            "malformed_csv_token_count_mismatch": len(malformed_records),
            "out_of_scope_master_category": len(out_of_scope_df),
            "missing_image_file": len(missing_img_af),
            "corrupted_image_file": len(corrupted_img_af),
            "missing_attribute_values": len(missing_attr_df),
            "sampling_unselected": len(unselected_df)
        },
        "excluded_records": excluded_records
    }
    with open(data_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    # Compute SHA-256 hashes
    styles_hash = compute_sha256(str(styles_csv))
    catalog_csv_hash = compute_sha256(str(catalog_csv_path))
    catalog_ids_hash = compute_sha256(str(catalog_ids_path))
    data_audit_hash = compute_sha256(str(data_audit_path))
    
    # Build catalog_manifest.json
    manifest_data = {
        "project_title": "LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval",
        "phase": 1,
        "manifest_version": "1.0.0",
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_source_url": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
        "dataset_download_date": "2026-11-07",
        "dataset_file_info": {
            "styles_csv_path": str(styles_csv),
            "styles_csv_size_bytes": os.path.getsize(styles_csv),
            "styles_csv_sha256": styles_hash,
            "images_directory": str(images_dir),
            "total_raw_images": len(os.listdir(images_dir))
        },
        "random_seed": 42,
        "selection_rules": (
            "Keep Apparel and Footwear products only. Remove duplicate product IDs. "
            "Exclude records with missing or corrupted image files. Exclude records with missing required "
            "attributes (gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName). "
            "Perform reproducible stratified sampling across articleType groups to select a frozen subset of 10,000 items. "
            "Sort catalog strictly ascending by string product ID."
        ),
        "catalog_item_count": len(catalog_df),
        "catalog_column_names": list(catalog_df.columns),
        "product_id_ordering_information": "Ascending alphanumeric string sort",
        "sha256_hashes": {
            "catalog_csv": catalog_csv_hash,
            "catalog_ids_json": catalog_ids_hash,
            "styles_csv": styles_hash,
            "data_audit_json": data_audit_hash
        },
        "cleaning_configuration": {
            "allowed_master_categories": ["Apparel", "Footwear"],
            "max_catalog_size": 10000,
            "random_seed": 42,
            "require_image_load_verification": True,
            "strict_metadata_fields": [
                "id", "gender", "masterCategory", "subCategory", "articleType",
                "baseColour", "season", "year", "usage", "productDisplayName"
            ]
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        
    # Build config.json
    config_data = {
        "project_title": "LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval",
        "current_phase": 1,
        "random_seed": 42,
        "environment": env_info,
        "dataset": {
            "source_url": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
            "download_date": "2026-11-07",
            "styles_csv": str(styles_csv),
            "images_dir": str(images_dir),
            "archive_preserved": "fashion_project/data/raw/fashion-product-images-small.zip"
        },
        "paths": {
            "catalog_csv": str(catalog_csv_path),
            "catalog_ids_json": str(catalog_ids_path),
            "data_audit_json": str(data_audit_path),
            "catalog_manifest_json": str(manifest_path),
            "backups_dir": str(root / "backups"),
            "logs_dir": str(root / "logs")
        },
        "planned_models": {
            "language_model": "Qwen/Qwen2.5-3B-Instruct",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        
    # Build requirements-lock.txt
    requirements = [
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"pillow=={Image.__version__}",
        "pydantic>=2.10.0",
        "ipywidgets>=8.1.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "sentence-transformers>=3.1.0",
        "torch>=2.4.0"
    ]
    with open(requirements_lock_path, "w", encoding="utf-8") as f:
        f.write("\n".join(requirements) + "\n")
        
    print("\n[SUCCESS] Phase 1 Catalog Preparation Pipeline Completed Successfully!")
    print(f"  Artifacts generated in: {artifacts_dir}")
    print(f"  Configuration saved to: {config_path}")
    print(f"  Requirements lock saved to: {requirements_lock_path}")
    return manifest_data


if __name__ == "__main__":
    run_pipeline()
