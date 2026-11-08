# LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Academic Level:** AI / Machine Learning B.Tech Capstone / Research Project  
**Authoritative Reference:** `End_to_End PipeLine.pdf`  
**Current Milestone:** Phase 5 — Final Integration, Interactive Interface, Evaluation, Repository Cleanup & Deployment Readiness (**Complete & Fully Operational**)  
**Repository:** https://github.com/ESWAR-CODES/llm-fashion-recommendation  

---

## 1. Project Objective

This project implements an end-to-end, reproducible, lightweight fashion recommendation system designed for execution in Google Colab and local Python runtimes:
1. **Conversational Preference Extraction:** Parses natural language clothing requests, multi-turn updates, and negative constraints using an open-weights LLM (**Qwen2.5-3B-Instruct**) with official ChatML formatting and Pydantic validation.
2. **Persistent User Taste Profiles:** Manages anonymous user profiles, explicit style preferences, and idempotent feedback using SQLite (`backups/profiles.sqlite`) with native online backup snapshot recovery.
3. **Candidate Filtering & Personalized Ranking:** Applies strict hard constraints (AND/OR/NOT) before computing hybrid personalized ranking ($0.8 \times \text{query} + 0.2 \times \text{profile}$) via dense sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**).
4. **Evidence-Grounded Explanations:** Explains recommendations using only verified recorded catalog attributes, rejecting hallucinated claims (e.g. ungrounded colors, subjective fit or material claims).
5. **Interactive Notebook Interface:** Delivers an interactive `ipywidgets` application with product cards, images, and real-time feedback controls.
6. **Formal 4-System Comparative Evaluation:** Benchmarks Keyword Baseline, Semantic Retrieval, Personalized Retrieval, and the Complete Pipeline across 40 frozen test requests.

---

## 2. Dataset Specifications

- **Dataset Source:** Kaggle — Fashion Product Images (Small)  
- **Dataset URL:** [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)  
- **Raw Files:**
  - `styles.csv` (44,446 raw records, 44,424 valid rows, 22 malformed rows cleanly excluded)
  - `images/` (44,441 JPEG product images matching `<id>.jpg`)
- **Frozen Retrieval Catalog (`artifacts/catalog.csv`):**
  - Exactly **10,000 products** sampled across article types using fixed seed 42.
  - Restricted strictly to `masterCategory` $\in \{\text{"Apparel"}, \text{"Footwear"}\}$.
  - 100% of images verified and decodable via Pillow.
  - Exactly 0 null or missing values across all ten metadata columns:
    `id`, `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `year`, `usage`, `productDisplayName`.

---

## 3. Directory Layout

```
fashion_project/
├── Fashion_Recommender.ipynb      # Main interactive 10-section notebook
├── README.md                      # Comprehensive project manual
├── config.json                    # Configuration, seeds, model revisions & paths
├── requirements-lock.txt          # Frozen dependencies
│
├── interactive_ui.py              # ipywidgets application implementation (Step 9)
├── backup_manager.py              # SQLite Online Backup API snapshot manager
├── prepare_catalog.py             # Phase 1 dataset ingestion & Pillow image audit
├── embedding_engine.py            # Phase 2 MiniLM embeddings & pure NumPy search
├── profile_manager.py             # Phase 3 SQLite profile, feedback & user vector engine
├── personalized_ranker.py         # Phase 3 candidate filter & personalized ranker
├── preference_extractor.py        # Phase 4 top-level export bridge
├── explanation_engine.py          # Phase 4 top-level export bridge
│
├── artifacts/
│   ├── catalog.csv                # 10,000-item stratified frozen catalog
│   ├── catalog_ids.json           # Exact row ordering matching catalog.csv
│   ├── catalog_manifest.json      # SHA-256 hashes, random seed 42, dataset audit
│   ├── data_audit.json            # Ingestion audit log
│   ├── product_descriptions.csv   # Standardized template text descriptions
│   ├── embeddings.npy             # (10000, 384) float32 unit-normalized matrix
│   └── embedding_manifest.json    # Embedding model configuration and checksums
│
├── evaluation/
│   ├── development_requests.json  # 10 development tuning queries
│   ├── test_requests.json         # 40 formal test benchmark queries (4 categories)
│   ├── predictions.jsonl          # Detailed per-query outputs for Systems A-D
│   ├── human_judgements.csv       # 696 blinded protocol records (trial template)
│   ├── metrics.json               # Formal evaluation metrics (accuracy, latency, etc.)
│   └── run_evaluation.py          # Step 10 comparative benchmark script
│
├── backups/
│   ├── .gitkeep                   # Directory placeholder
│   ├── profiles.sqlite            # Active runtime SQLite database (gitignored)
│   └── profiles_snapshot.sqlite   # Verified atomic snapshot backup (gitignored)
│
├── engines/
│   ├── preference_extractor.py    # Qwen ChatML preference parser & Pydantic validator
│   └── explanation_engine.py      # Factual attribute grounding & template fallback
│
├── tests/
│   ├── run_all_tests.py           # Master unified test suite runner (38 tests)
│   ├── test_catalog.py            # Phase 1 catalog verification
│   ├── test_embeddings.py         # Phase 2 embedding & retrieval tests
│   ├── test_profiles.py           # Phase 3 profile & feedback tests
│   ├── test_ranking.py            # Phase 3 ranking formula tests
│   ├── test_llm_extraction.py     # Phase 4 extraction & Pydantic tests
│   ├── test_explanations.py       # Phase 4 explanation grounding tests
│   └── test_integration.py        # Phase 5 integration & backup tests
│
└── docs/
    ├── End_to_End_PipeLine.pdf    # Authoritative project specification
    ├── final_report/
    │   └── FINAL_ACADEMIC_REPORT.md # 11-section comprehensive report
    ├── demonstration/
    │   └── DEMONSTRATION_PLAN.md   # Step 12 demonstration protocol
    └── phase_reports/
        ├── PHASE_1_COMPLETION_REPORT.md
        ├── PHASE_2_COMPLETION_REPORT.md
        ├── PHASE_3_COMPLETION_REPORT.md
        ├── PHASE_4_COMPLETION_REPORT.md
        ├── PHASE_5_CLEANUP_REPORT.md
        ├── DEPLOYMENT_READINESS_REPORT.md
        └── PHASE_5_COMPLETION_REPORT.md
```

---

## 4. Installation & Environment Setup

### 4.1 Local Environment
```bash
# Clone the repository
git clone https://github.com/ESWAR-CODES/llm-fashion-recommendation.git
cd llm-fashion-recommendation/fashion_project

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies from frozen requirements
pip install -r requirements-lock.txt
```

### 4.2 Google Colab Environment
Open `Fashion_Recommender.ipynb` in Google Colab. In Cell 1, run:
```python
%pip install pandas numpy pillow sentence-transformers
%pip install transformers accelerate pydantic ipywidgets
```
Mount Google Drive for persistent storage:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 5. Running the Interactive Interface (Step 9)

Inside `Fashion_Recommender.ipynb`, navigate to **Section 8: Interactive Interface** and run:
```python
from interactive_ui import FashionRecommenderApp

app = FashionRecommenderApp()
app.render()
```

### Features:
- **Profile Controls:** Create an anonymous profile, select preferred categories and colors, or explicitly wipe profile data.
- **Request Controls:** Enter free-text queries (e.g., *"navy blue casual shirts"* or *"black instead but avoid red"*).
- **Product Display:** View up to 5 recommended items with thumbnail images, metadata badges, evidence-grounded explanations, and `👍 Like` / `✕ Dismiss` buttons.
- **Constraints Display:** Real-time indicator displaying active conversation constraints.
- **No-Match Recovery:** Clear, actionable advice on relaxing filters when 0 items match.

---

## 6. Formal Evaluation Benchmark (Step 10)

To run the full 4-system evaluation comparing Keyword Baseline, Semantic Retrieval, Personalized Retrieval, and the Complete Pipeline:
```bash
python fashion_project/evaluation/run_evaluation.py
```

### 4-System Comparative Study Results:

| System Variant | Architecture / Approach | Precision@5 (Constraint Compliant) | Constraint Violations (%) | Median Latency (ms) | P95 Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|
| **System A** | Keyword Baseline (Token Overlap + Gold Filters) | 100.0% | 0.0% | 36.41 ms | 202.06 ms |
| **System B** | Semantic Retrieval (all-MiniLM-L6-v2 + Gold Filters) | 100.0% | 0.0% | 26.86 ms | 44.06 ms |
| **System C** | Personalized Retrieval ($0.8\mathbf{q} + 0.2\mathbf{u}$ + Gold Filters) | 100.0% | 0.0% | 29.73 ms | 42.49 ms |
| **System D** | Complete Pipeline (Extracted Constraints + Ranking + Explanations) | 100.0% | 0.0% | 27.13 ms | 36.57 ms |

- **Constraint Violation Rate:** 0.0% across all systems (onboarding likes strictly excluded)
- **Extraction Valid JSON Rate:** 100.0%
- **Unsupported Attribute Rejection:** 100.0% (4/4 test cases accurately detected)
- **Explanation Grounding Compliance:** 100.0% (0 ungrounded claims across 174 explanations)
- **Evaluation Methodology:** `human_judgements.csv` (696 records) is a blinded trial protocol template (`uncollected`); no human subjects were recruited.
- **Hardware & Runtime Note:** Benchmark executed on local Windows 11 CPU in deterministic fallback mode. Autoregressive Qwen2.5-3B-Instruct GPU execution runs on Google Colab T4. Google Colab fresh runtime execution is marked as **NOT TESTED** in this local session.
- **Constraint Violations:** 0.0% (Target: 0.0%)

---

## 7. Running the Automated Test Suite

To verify all 38 unit and integration tests:
```bash
python fashion_project/tests/run_all_tests.py
```
**Output:**
```
================================================================================
CONSOLIDATED TEST SUITE EXECUTION
================================================================================
Ran 38 tests in 15.82s

OK
================================================================================
TESTS RUN: 38 | FAILURES: 0 | ERRORS: 0 | TIME: 21.12s
OVERALL STATUS: ALL TESTS PASSED [PASS]
================================================================================
```

---

## 8. Persistence & Recovery Workflow

- **Snapshot Creation:**
  ```python
  from backup_manager import create_database_backup
  create_database_backup()  # Creates backups/profiles_snapshot.sqlite using SQLite's backup API
  ```
- **Runtime Recovery:**
  ```python
  from backup_manager import restore_database_backup
  restore_database_backup()  # Restores from snapshot into runtime database
  ```
- **Integrity Verification:**
  ```python
  from backup_manager import verify_database_integrity
  verify_database_integrity("backups/profiles_snapshot.sqlite")
  ```

---

## 9. Academic Reference

Base Paper Reference:  
> Wang, X., Li, Y., Zhou, K., and Tu, W. (2025). *LLM-SCRec: LLM-Powered Style-Context Insights for Fashion Recommendation.* IEEE Access, 13, 192143–192154. DOI: 10.1109/ACCESS.2025.3628684.
