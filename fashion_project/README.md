# LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Academic Level:** AI / Machine Learning B.Tech Capstone / Research Project  
**Authoritative Reference:** `End_to_End PipeLine.pdf`  
**Current Milestone:** Phase 2 — Product Text Embeddings & Semantic Retrieval

---

## 1. Project Objective
This project builds an end-to-end conversational fashion recommendation system capable of:
1. Parsing natural language user clothing requests and explicit constraint updates using a localized Large Language Model (**Qwen2.5-3B-Instruct**).
2. Maintaining persistent user style profiles and multi-turn conversational preferences using SQLite.
3. Conducting semantic retrieval and personalized ranking using dense sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**).
4. Providing interactive image recommendations with grounded, evidence-based natural language explanations.

> [!IMPORTANT]
> **Phase Status:**
> - **Phase 1 (Complete & Verified):** Dataset acquisition, 10-field inspection, Pillow decoding validation, data cleaning, and 10,000-item stratified catalog creation.
> - **Phase 2 (Complete & Verified):** Product text description generation, `all-MiniLM-L6-v2` embedding generation, persistence, manifest hashing, and pure NumPy semantic retrieval.
> - **Phases 3–7 (Reserved):** User profiles, conversational LLM state extraction, ranking weights, UI widgets, and evaluation are reserved for future phases.

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
  3. `masterCategory`: Broad category (`Apparel`, `Footwear`)
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
├── Fashion_Recommender.ipynb    # Main 10-section interactive notebook (Sections 1, 2, 3 active)
├── prepare_catalog.py           # Phase 1 data engineering & Pillow image audit pipeline
├── validate_phase1.py           # Phase 1 automated test suite (12 checks)
├── embedding_engine.py          # Phase 2 core engine (descriptions, encoding, persistence, search)
├── validate_phase2.py           # Phase 2 automated test suite (17 checks)
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
│   ├── catalog_manifest.json    # SHA-256 hashes, selection rules & frozen metadata
│   ├── product_descriptions.csv # Structured text representation for all 10,000 products
│   ├── embeddings.npy           # Normalized float32 embedding matrix (10000, 384)
│   └── embedding_manifest.json  # Model revision, dimension, hashes & audit trail
│
├── backups/                     # Dedicated directory for runtime database snapshots
└── logs/                        # Dedicated directory for runtime execution logs
```

---

## 4. Environment & Dependency Setup

### 4.1 Prerequisites
- **Python:** 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- **PyTorch:** `torch>=2.4.0` (preserves existing Colab installation)
- **Sentence-Transformers:** `sentence-transformers>=3.1.0`
- **GPU Runtime:** Optional for Phase 1 & 2 (runs in under 50s on CPU); recommended for later LLM phases.

### 4.2 Installation
```bash
pip install -r fashion_project/requirements-lock.txt
```

---

## 5. Google Colab & Google Drive Setup

1. Open `fashion_project/Fashion_Recommender.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Cell `1.1` mounts Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Persistent files are stored under `/content/drive/MyDrive/fashion_project`.
5. Kaggle credentials can be stored securely in Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).

---

## 6. Phase 2: Product Text Embeddings & Semantic Retrieval

### 6.1 Product Text Representation
To ensure grounded semantic representations without hallucinating unsupported information, product descriptions are constructed strictly from recorded metadata:
```
Name: {productDisplayName}
Type: {articleType} | Color: {baseColour} | Usage: {usage}
Category: {masterCategory} | Season: {season}
```
- Example:
  ```
  Name: Turtle Check Men Navy Blue Shirt
  Type: Shirts | Color: Navy Blue | Usage: Casual
  Category: Apparel | Season: Fall
  ```
- Descriptions for all 10,000 catalog products are saved to `artifacts/product_descriptions.csv` preserving exact catalog row ordering.

### 6.2 Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture:** 6-layer MiniLM producing 384-dimensional dense vectors.
- **Inference Mode:** The model runs in `eval()` mode with hardware auto-detection (CUDA if available, otherwise CPU).

### 6.3 Batch Encoding & Normalization
- Encodings are generated in batches of 64.
- Vectors are L2 unit-normalized (`||v||_2 = 1.0`).
- The resulting matrix is saved as **NumPy float32** of shape `(10000, 384)`.
- Recomputing is bypassed automatically when valid persisted artifacts already exist on disk.

### 6.4 Embedding Persistence & Manifest
Saved files:
- `artifacts/embeddings.npy`: Binary matrix (15,360,128 bytes).
- `artifacts/embedding_manifest.json`: Stores model name, dimension (384), normalization, catalog manifest hash linkage, and file SHA-256 checksums.

### 6.5 Semantic Retrieval Engine
- Queries are encoded with the identical `all-MiniLM-L6-v2` model and normalized.
- Similarity scores are computed via pure NumPy dot product matrix multiplication:
  $$\text{score}(i) = \mathbf{e}_i \cdot \mathbf{q}$$
  (Equivalent to cosine similarity since both $\mathbf{e}_i$ and $\mathbf{q}$ are unit vectors).
- Products are ranked in descending order, returning the top 5 candidates with metadata, scores, and image paths.
- No external vector databases (FAISS, Chroma, Pinecone) are used.

---

## 7. Execution and Reproduction Guide

### Run via Command Line:
```bash
# Execute Phase 2 embedding engine pipeline
python fashion_project/embedding_engine.py

# Execute Phase 2 automated validation suite (17 checks)
python fashion_project/validate_phase2.py

# Regression check: execute Phase 1 validation suite (12 checks)
python fashion_project/validate_phase1.py
```

### Run via Interactive Notebook:
1. Open `fashion_project/Fashion_Recommender.ipynb`.
2. Execute **Section 1: Setup** to initialize environment and directories.
3. Execute **Section 2: Catalog** to load and verify the frozen catalog.
4. Execute **Section 3: Embeddings** to load the model, generate/reload embeddings, and run interactive semantic retrieval queries with visual product image cards.

---

## 8. Validation Test Suites Summary

### Phase 2 Validation Suite (`validate_phase2.py` - 17 Checks)
| # | Check Description | Status | Verification Detail |
|---|---|---|---|
| 1 | Product descriptions generated for all catalog rows | **PASS** | 10,000 non-empty descriptions in `product_descriptions.csv` |
| 2 | Product IDs remain correctly associated | **PASS** | Exact 1-to-1 row sequence alignment with `catalog_ids.json` |
| 3 | Missing attributes handled without fabricated values | **PASS** | Sparse records format safely using recorded values only |
| 4 | Embedding model loads successfully | **PASS** | Loaded `all-MiniLM-L6-v2` in inference mode |
| 5 | Embedding dimension is exactly 384 | **PASS** | Output dimension verified: 384 |
| 6 | Embedding dtype is float32 | **PASS** | `dtype: float32` verified |
| 7 | Embeddings contain no NaN or infinite values | **PASS** | All 3,840,000 values are finite |
| 8 | Embeddings are normalized | **PASS** | L2 norms unit-normalized within $10^{-3}$ tolerance |
| 9 | Embedding row count matches catalog row count | **PASS** | Row count 10,000 matches catalog |
| 10 | Product ID ordering remains unchanged | **PASS** | Row $j$ corresponds strictly to catalog row $j$ |
| 11 | Saved embeddings can be reloaded | **PASS** | Reloaded exact matrix without recomputation |
| 12 | Manifest and hash validation succeeds | **PASS** | SHA-256 verified against `embedding_manifest.json` |
| 13 | Semantic retrieval returns no more than five products | **PASS** | Returned exactly 5 items for top_k=5 |
| 14 | Retrieved product IDs exist in frozen catalog | **PASS** | All retrieved IDs verified in catalog |
| 15 | Retrieved images correspond to returned product IDs | **PASS** | All retrieved images exist and decode via Pillow |
| 16 | Empty queries are handled correctly | **PASS** | Empty & whitespace-only queries return `[]` gracefully |
| 17 | Retrieval works after loading without recomputation | **PASS** | Cosine similarity scores ordered monotonically |

### Phase 1 Regression Suite (`validate_phase1.py` - 12 Checks)
- **ALL 12 CHECKS PASSED** (Catalog integrity, image decodability, 10 columns, 0 duplicates, and manifest checksums verified intact).
