# LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Academic Level:** AI / Machine Learning B.Tech Capstone / Research Project  
**Authoritative Reference:** `End_to_End PipeLine.pdf`  
**Current Milestone:** Phase 1 — Environment Setup, Dataset Acquisition & Catalog Preparation

---

## 1. Project Objective
This project builds an end-to-end conversational fashion recommendation system capable of:
1. Parsing natural language user clothing requests and explicit constraint updates using a localized Large Language Model (**Qwen2.5-3B-Instruct**).
2. Maintaining persistent user style profiles and multi-turn conversational preferences using SQLite.
3. Conducting semantic retrieval and personalized ranking using dense sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**).
4. Providing interactive image recommendations with grounded, evidence-based natural language explanations.

> [!IMPORTANT]
> **Phase 1 Scope:** This phase strictly implements the environment configuration, dataset acquisition, full image decoding validation, rigorous data auditing, and reproducible stratified catalog selection. Embeddings, user profiling, and recommendation pipelines are reserved for future phases.

---

## 2. Dataset Specifications

- **Source:** Kaggle — Fashion Product Images (Small)  
- **URL:** [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)  
- **Dataset Files:**
  - `styles.csv` (44,446 raw records, 44,424 valid 10-token rows, 22 malformed rows)
  - `images/` (44,441 JPEG images matching `<id>.jpg`)
- **Canonical Columns (10):**
  1. `id`: Stable string unique identifier
  2. `gender`: Target gender (`Men`, `Women`, `Boys`, `Girls`, `Unisex`)
  3. `masterCategory`: Broad category (`Apparel`, `Footwear`, `Accessories`, etc.)
  4. `subCategory`: Sub-grouping (`Topwear`, `Shoes`, `Bottomwear`, etc.)
  5. `articleType`: Specific garment type (`Tshirts`, `Casual Shoes`, `Kurtas`, etc.)
  6. `baseColour`: Primary color (`Black`, `White`, `Blue`, `Navy Blue`, etc.)
  7. `season`: Seasonality (`Summer`, `Fall`, `Winter`, `Spring`)
  8. `year`: Year of release
  9. `usage`: Occasion (`Casual`, `Sports`, `Ethnic`, `Formal`, etc.)
  10. `productDisplayName`: Complete descriptive product title

---

## 3. Project Directory Hierarchy

```
fashion_project/
├── Fashion_Recommender.ipynb    # Main 10-section interactive notebook
├── prepare_catalog.py           # Production-grade dataset preparation & audit script
├── validate_phase1.py           # Comprehensive 12-point validation test suite
├── README.md                    # Detailed documentation and reproduction guide
├── config.json                  # Environment, paths, model revisions, and seed config
├── requirements-lock.txt        # Frozen dependency specifications
│
├── data/
│   └── raw/
│       ├── styles.csv           # Original dataset metadata (44,446 raw rows)
│       ├── images/              # Original product images (44,441 JPEG files)
│       └── fashion-product-images-small.zip # Preserved original archive
│
├── artifacts/
│   ├── catalog.csv              # Frozen stratified catalog (exactly 10,000 items)
│   ├── catalog_ids.json         # Ordered list of 10,000 string product IDs
│   ├── data_audit.json          # Granular dataset audit, counts, distributions & exclusions
│   └── catalog_manifest.json    # SHA-256 hashes, selection rules & frozen metadata
│
├── backups/                     # Dedicated directory for runtime database snapshots
└── logs/                        # Dedicated directory for runtime execution logs
```

---

## 4. Environment & Dependency Setup

### 4.1 Prerequisites
- **Python:** 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- **PyTorch:** Preserves Colab's default PyTorch installation (`torch>=2.4.0`)
- **GPU Runtime:** Optional for Phase 1 (CPU runtime works seamlessly); recommended for Phases 2–4 (Colab T4, L4, or A100).

### 4.2 Installation
To install the initial required dependencies without disturbing PyTorch:
```bash
pip install -r fashion_project/requirements-lock.txt
```

---

## 5. Google Colab & Google Drive Setup

### 5.1 Google Drive Persistence
When executing in Google Colab:
1. Open `fashion_project/Fashion_Recommender.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Cell `1.1` automatically mounts Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Project artifacts and backup snapshots persist under `/content/drive/MyDrive/fashion_project`.

### 5.2 Kaggle Authentication via Colab Secrets
To download the dataset automatically using the Kaggle API in Colab:
1. Obtain your Kaggle username and API key from [kaggle.com/settings](https://www.kaggle.com/settings) (Generate New Token).
2. In Google Colab, click the **Secrets** icon (the key symbol on the left toolbar).
3. Add two secrets with notebook access enabled:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API token
4. The notebook securely accesses these credentials via `google.colab.userdata` without hardcoding them in the notebook or logs.

### 5.3 Manual Fallback Upload
If Kaggle API credentials are not configured:
1. Download `fashion-product-images-small.zip` from Kaggle manually.
2. Upload `fashion-product-images-small.zip` directly to `data/raw/`.
3. The pipeline detects the local archive, preserves it, and extracts `styles.csv` and `images/` automatically.

---

## 6. Data Cleaning & Catalog Methodology

### 6.1 Cleaning Rules
1. **Scope Restriction:** Filter `masterCategory` to `["Apparel", "Footwear"]` only. Out-of-scope categories (`Accessories`, `Personal Care`, `Free Items`, `Sporting Goods`, `Home`) are excluded with explicit audit logging.
2. **Malformed Row Handling:** 22 rows in `styles.csv` contain unquoted commas within `productDisplayName`, resulting in token counts of 11 or 12. These are recorded in `data_audit.json` with line numbers and raw tokens, and excluded with reason `malformed_csv_token_count_mismatch`.
3. **Pillow Image Validation:** All 44,441 images are verified and fully decoded using `PIL.Image.open().verify()` and `im.load()`.
   - Missing images across entire dataset: exactly 5 (`39403`, `39410`, `39401`, `39425`, `12347`). All 5 fall under Apparel/Footwear and are excluded with reason `missing_image_file`.
   - Corrupted images across entire dataset: 0.
4. **Metadata Completeness:** All 10 canonical columns must be non-empty and well-formed. Empty strings and `NA` sentinel tokens in `baseColour` (6 in AF), `season` (21 in AF), `year` (1 in AF), and `usage` (26 in AF) totaling 53 rows are excluded with reason `missing_attribute_values`.
5. **Deduplication:** Product IDs are checked for duplicates (0 duplicates found).
6. **Clean Candidate Pool:** Yields exactly **30,558** fully qualified candidates.

### 6.2 Stratified Sampling Strategy
- **Sample Cap:** Exactly **10,000** items.
- **Random Seed:** Fixed to **`42`**.
- **Stratification:** Grouped by `articleType` (64 distinct categories). Each category receives a proportional allocation with a guaranteed minimum of 1 item per category. Remainder slots are allocated deterministically to the largest categories.
- **Ordering:** The final catalog is sorted strictly ascending by string product ID, guaranteeing 100% deterministic reproducibility across runs and platforms.

---

## 7. Generated Artifact Specifications

All artifacts are saved to `fashion_project/artifacts/`:

| Artifact | File Format | Size / Count | Purpose |
|---|---|---|---|
| `catalog.csv` | CSV | 10,000 rows, 10 cols | Frozen retrieval catalog containing all 10 canonical attributes |
| `catalog_ids.json` | JSON | 10,000 strings | Frozen string product IDs matching the exact order of `catalog.csv` |
| `data_audit.json` | JSON | Complete audit trail | Raw stats, missing values, exclusions, and attribute distributions |
| `catalog_manifest.json` | JSON | Manifest & SHA-256 | Immutable audit manifest with file hashes, creation date, and rules |
| `config.json` | JSON | Project config | Environment specs, hardware info, paths, and seed `42` |
| `requirements-lock.txt` | TXT | Locked versions | Exact package versions for reproducibility |

---

## 8. Step-by-Step Reproduction Guide

### Option A: Command Line / Automated Pipeline
From the project workspace root:
```bash
# 1. Execute the catalog preparation and data audit pipeline
python fashion_project/prepare_catalog.py

# 2. Execute the comprehensive 12-point validation suite
python fashion_project/validate_phase1.py
```

### Option B: Interactive Jupyter Notebook
1. Open `fashion_project/Fashion_Recommender.ipynb` in VS Code, JupyterLab, or Google Colab.
2. Execute **Section 1: Setup** to initialize the environment and inspect hardware.
3. Execute **Section 2: Catalog** to run the pipeline, view distributions, and render visual image verification cards for 5 random items.
4. Sections 3–10 display clearly labeled Phase placeholders.

---

## 9. Validation Results

Running `python fashion_project/validate_phase1.py` executes all 12 rigorous checks:

| # | Check Description | Status | Verification Detail |
|---|---|---|---|
| 1 | Environment setup completes successfully | **PASS** | `config.json` valid, paths and environment recorded |
| 2 | Required dependencies import correctly | **PASS** | `pandas`, `numpy`, `PIL`, `pydantic` verified |
| 3 | The dataset loads correctly | **PASS** | 44,424 valid raw rows loaded from `styles.csv` |
| 4 | All ten expected columns exist | **PASS** | Exact match with canonical 10-column schema |
| 5 | Product IDs are stored as strings | **PASS** | All 10,000 IDs are non-empty strings |
| 6 | No duplicate IDs exist in the final catalog | **PASS** | 0 duplicates across all 10,000 entries |
| 7 | Every selected product has a valid image | **PASS** | 10,000 images verified and decodable via Pillow |
| 8 | Every selected record contains sufficient metadata | **PASS** | 0 missing values; 100% in Apparel and Footwear |
| 9 | Catalog contains no more than 10,000 products | **PASS** | Exact count: 10,000 items |
| 10 | Catalog ID list matches CSV row ordering | **PASS** | 100% 1-to-1 index alignment |
| 11 | Manifest hashes match generated files | **PASS** | All 4 SHA-256 hashes matched identically |
| 12 | Deterministic re-run reproducibility | **PASS** | Re-run produces 100% identical product IDs |

---

## 10. Common Errors and Recovery Steps

| Issue | Cause | Recovery Step |
|---|---|---|
| `FileNotFoundError: styles.csv` | Raw dataset not extracted to `data/raw/` | Place `archive (5).zip` in `data/raw/` or run `prepare_catalog.py` which unzips preserved archive automatically |
| `Colab Secrets Kaggle 401 Unauthorized` | Invalid Kaggle token in Colab Secrets | Verify username and token at `kaggle.com/settings`; update secrets in Colab sidebar |
| `Missing Image Warnings` | Missing 5 specific Kaggle dataset images | Expected behavior; pipeline automatically logs them in `data_audit.json` and excludes them from the catalog |
| `Malformed CSV Rows (ParserWarning)` | Unescaped commas in 22 raw display names | Expected behavior; `csv.reader` catches and logs them into `malformed_csv_token_count_mismatch` |
| `CUDA Not Available` | Running on CPU runtime | Normal for Phase 1. For later LLM phases, switch runtime in Colab to T4 or L4 GPU |
