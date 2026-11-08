# Dataset Documentation: Fashion Product Images Small

## 1. Overview and Source Reference

- **Dataset Name:** Fashion Product Images (Small)
- **Author:** Param Aggarwal
- **Source URL:** [Kaggle Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- **Local Preserved Archive:** `fashion_project/data/raw/fashion-product-images-small.zip` (592.6 MB)

---

## 2. Directory Layout and File Organization

The dataset maintains a clean separation between raw, unprocessed source files and deterministic processed artifacts:

```
fashion_project/
├── data/
│   └── raw/
│       ├── styles.csv                         # Raw catalog metadata (44,446 lines)
│       └── images/                            # Raw JPEG product images (44,441 files)
│           ├── 10000.jpg
│           ├── 10001.jpg
│           └── ...
└── artifacts/
    ├── catalog.csv                            # 10,000-item stratified frozen catalog
    ├── catalog_ids.json                       # Canonical row sequence mapping
    ├── catalog_manifest.json                  # SHA-256 hashes and audit metadata
    ├── data_audit.json                        # Ingestion audit log
    ├── product_descriptions.csv               # Standardized text descriptions
    ├── embeddings.npy                         # (10000, 384) MiniLM embedding matrix
    ├── embedding_manifest.json                # Sentence-transformers configuration
    ├── clip_embeddings.npy                    # (10000, 512) CLIP ViT-B/32 image embedding matrix
    ├── clip_image_embeddings.npy              # Canonical duplicate sync of CLIP vectors
    ├── clip_catalog_ids.json                  # Product ID sequence mapping for CLIP
    └── clip_manifest.json                     # CLIP ViT-B/32 offline indexing manifest
```

---

## 3. Metadata Schema Specification

The dataset contains exactly 10 metadata columns per valid record:

| Column Name | Data Type | Description | Ingestion Constraint |
|---|---|---|---|
| `id` | `str` | Canonical product identifier | Must be string, non-null, unique |
| `gender` | `str` | Target demographic (`Men`, `Women`, `Boys`, `Girls`, `Unisex`) | Non-null |
| `masterCategory` | `str` | High-level category | Restricted to `Apparel` and `Footwear` |
| `subCategory` | `str` | Mid-level category (`Topwear`, `Bottomwear`, `Shoes`, etc.) | Recorded value |
| `articleType` | `str` | Fine-grained garment type (`Tshirts`, `Shirts`, `Casual Shoes`, etc.) | Critical filter field |
| `baseColour` | `str` | Primary dominant color (`Navy Blue`, `Black`, `Red`, etc.) | Critical filter field |
| `season` | `str` | Seasonal designation (`Summer`, `Fall`, `Winter`, `Spring`) | Recorded value |
| `year` | `float/int`| Release year | Recorded value |
| `usage` | `str` | Intended occasion (`Casual`, `Formal`, `Sports`, `Ethnic`, `Party`) | Critical filter field |
| `productDisplayName`| `str` | Full descriptive title | Critical search text |

---

## 4. Rigorous Ingestion & Data Audit Statistics

The dataset loader (`services/dataset_service.py`) audited all raw records:

- **Total CSV Lines Read:** 44,446
- **Syntactically Valid 10-Field Rows:** 44,424
- **Malformed Rows (Preserved in Audit Log):** 22 (delimiters or token mismatches; never silently discarded)
- **Unique Product IDs in Valid Rows:** 44,424 (0 duplicate IDs)
- **Missing Attributes Breakdown:**
  - `baseColour`: 15 nulls
  - `season`: 21 nulls
  - `year`: 1 null
  - `usage`: 317 nulls
  - `productDisplayName`: 7 nulls
  - `id`, `gender`, `masterCategory`, `subCategory`, `articleType`: 0 nulls
- **Disk Images on Local System:** 44,441 JPEG files
- **Missing Images Across Raw Dataset:** 5 product IDs (`39403`, `39410`, `39401`, `39425`, `12347`)
- **Corrupt / Unreadable Images:** 0 (all present images pass Pillow verification and decompression)
- **Apparel Total:** 21,397 items
- **Footwear Total:** 9,219 items
- **Eligible Clean Subset:** 30,558 products
- **Frozen Balanced Catalog:** Exactly 10,000 items (deterministic stratified sampling with random seed 42)

---

## 5. Frozen Catalog Integrity Rules

1. **Exact Size:** Exactly 10,000 products.
2. **Category Boundaries:** Strictly restricted to `Apparel` and `Footwear`.
3. **Zero Nulls in Critical Attributes:** 0 missing values in `id`, `articleType`, `baseColour`, `usage`, `productDisplayName`.
4. **Valid Image for Every Product:** Every single selected item ID maps to a verified, decodable JPEG image in `data/raw/images/<id>.jpg`.
5. **Exact Row Ordering:** Row $j$ in `artifacts/catalog.csv` corresponds to index $j$ in `artifacts/catalog_ids.json` and row $j$ in `artifacts/embeddings.npy`.
6. **No Data Drift:** SHA-256 hashes are verified against `artifacts/catalog_manifest.json`.

---

## 6. Dataset Installation & Validation Command

To verify the dataset installation and check artifact integrity:

```bash
python fashion_project/scripts/validate_dataset.py
```
