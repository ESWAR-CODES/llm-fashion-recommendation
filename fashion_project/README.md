# LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Academic Level:** AI / Machine Learning B.Tech Capstone / Research Project  
**Authoritative Reference:** `End_to_End PipeLine.pdf`  
**Current Milestone:** Phase 3 — User Profiles, Explicit Constraints & Personalized Ranking

---

## 1. Project Objective
This project builds an end-to-end conversational fashion recommendation system capable of:
1. Parsing natural language user clothing requests and explicit constraint updates using a localized Large Language Model (**Qwen2.5-3B-Instruct**).
2. Maintaining persistent user style profiles and multi-turn conversational preferences using SQLite (`backups/profiles.sqlite`).
3. Conducting semantic retrieval and personalized ranking using dense sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**).
4. Providing interactive image recommendations with grounded, evidence-based natural language explanations.

> [!IMPORTANT]
> **Phase Status:**
> - **Phase 1 (Complete & Verified):** Dataset acquisition, 10-field inspection, Pillow decoding validation, data cleaning, and 10,000-item stratified catalog creation.
> - **Phase 2 (Complete & Verified):** Product text description generation, `all-MiniLM-L6-v2` embedding generation, persistence, manifest hashing, and pure NumPy semantic retrieval.
> - **Phase 3 (Complete & Verified):** SQLite user profile management, explicit preference persistence, active request constraints, product feedback (likes/dismissals), user preference vector construction, candidate filtering, and hybrid personalized ranking ($0.8 \times \text{query} + 0.2 \times \text{profile}$).
> - **Phases 4–7 (Reserved):** Qwen preference extraction, conversational interpretation, LLM-generated explanations, ipywidgets interface, and benchmark evaluation are strictly reserved for future phases.

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
├── Fashion_Recommender.ipynb     # Main interactive notebook (Sections 1, 2, 3, 4, 6 active)
├── prepare_catalog.py            # Phase 1 data engineering & Pillow image audit pipeline
├── validate_phase1.py            # Phase 1 automated test suite (12 checks)
├── embedding_engine.py           # Phase 2 core engine (descriptions, encoding, persistence, search)
├── validate_phase2.py            # Phase 2 automated test suite (17 checks)
├── profile_manager.py            # Phase 3 profile CRUD, constraint validation, feedback & vectors
├── personalized_ranker.py        # Phase 3 candidate filtering & personalized ranking engine
├── validate_phase3.py            # Phase 3 automated test suite (30 checks)
├── PHASE_1_COMPLETION_REPORT.md  # Phase 1 signoff and verification report
├── PHASE_2_COMPLETION_REPORT.md  # Phase 2 signoff and verification report
├── PHASE_3_COMPLETION_REPORT.md  # Phase 3 signoff and verification report
├── README.md                     # Comprehensive project documentation
├── config.json                   # Environment, paths, model revisions, seeds, and ranking weights
├── requirements-lock.txt         # Frozen dependency specifications
│
├── data/
│   └── raw/
│       ├── styles.csv            # Original dataset metadata (44,446 raw rows)
│       ├── images/               # Original product images (44,441 JPEG files)
│       └── fashion-product-images-small.zip # Preserved original archive
│
├── artifacts/
│   ├── catalog.csv               # Frozen stratified catalog (exactly 10,000 items)
│   ├── catalog_ids.json          # Ordered list of 10,000 string product IDs
│   ├── data_audit.json           # Granular dataset audit, counts, distributions & exclusions
│   ├── catalog_manifest.json     # SHA-256 hashes, selection rules & frozen metadata
│   ├── product_descriptions.csv  # Structured text representation for all 10,000 products
│   ├── embeddings.npy            # Normalized float32 embedding matrix (10000, 384)
│   └── embedding_manifest.json   # Model revision, dimension, hashes & audit trail
│
├── backups/
│   └── profiles.sqlite           # SQLite relational database for profiles, feedback, & active states
└── logs/                         # Dedicated directory for runtime execution logs
```

---

## 4. Phase 3: User Profiles, Explicit Constraints & Personalized Ranking

### 4.1 SQLite Database Schema (`backups/profiles.sqlite`)
The system uses Python's standard `sqlite3` library with explicit foreign key constraints (`PRAGMA foreign_keys = ON;`) and parameterized queries:

1. **`profiles` Table:**
   - `profile_id` (TEXT PRIMARY KEY): Unique anonymous identifier (e.g. `usr_...`).
   - `preferences_json` (TEXT NOT NULL): Stored explicit preferences (`preferred_categories`, `preferred_colors`).
   - `created_at` (TEXT NOT NULL): ISO 8601 UTC timestamp.
   - `updated_at` (TEXT NOT NULL): ISO 8601 UTC timestamp.

2. **`feedback` Table:**
   - `profile_id` (TEXT NOT NULL): References `profiles(profile_id)` ON DELETE CASCADE.
   - `item_id` (TEXT NOT NULL): Validated string product ID from frozen catalog.
   - `action` (TEXT NOT NULL): `'like'` or `'dismiss'`.
   - `timestamp` (TEXT NOT NULL): ISO 8601 UTC timestamp.
   - `PRIMARY KEY (profile_id, item_id)`: Enforces idempotency (repeated likes update timestamp, never duplicate rows).

3. **`active_states` Table:**
   - `profile_id` (TEXT PRIMARY KEY): References `profiles(profile_id)` ON DELETE CASCADE.
   - `constraints_json` (TEXT NOT NULL): Current request filters (`articleType`, `baseColour`, `usage`, `masterCategory`, `exclude_colors`).
   - `updated_at` (TEXT NOT NULL): ISO 8601 UTC timestamp.

4. **Indexes:**
   - `idx_feedback_profile_action ON feedback(profile_id, action)` for fast feedback retrieval.

### 4.2 Decoupling Active Constraints from Saved Preferences
- **Active Request Constraints:** Represent the user's immediate search requirement (e.g., currently seeking "casual shirts" in "navy blue"). Active constraints are applied as strict hard filters *before* ranking.
- **Saved User Preferences:** Represent long-term tastes (e.g., user generally likes shoes or black apparel). Saved preferences modulate the user preference vector but *never* override current active constraints.

### 4.3 Candidate Filtering (`filter_candidates`)
Applied strictly before ranking scores are computed:
- **OR within values of the same field:** `articleType: ["Shirts", "Tshirts"]` matches Shirts or Tshirts.
- **AND across different fields:** Product must satisfy all specified attribute criteria simultaneously.
- **Excluded colors:** Products with `baseColour` in `exclude_colors` strictly fail.
- **Missing attributes:** Any product with missing/null attributes for a constrained field strictly fails.
- **Dismissed items:** Any product dismissed by the active profile is strictly removed.
- **ID deduplication:** Eliminates duplicate candidates.

### 4.4 User Preference Vector (`build_user_vector`)
Computes a normalized 384-dimensional representation:
1. **Primary Methodology (Liked Products):**
   - Retrieves all product IDs marked as `'like'` for the profile.
   - Maps IDs to rows in `artifacts/embeddings.npy`.
   - Computes the mean embedding vector.
   - Normalizes the mean vector to unit L2 norm: $\mathbf{u} = \frac{\bar{\mathbf{e}}}{\|\bar{\mathbf{e}}\|_2}$.
2. **Secondary Fallback (Saved Explicit Preferences):**
   - If no liked products exist, encodes saved categories/colors using `all-MiniLM-L6-v2` or computes the normalized mean of matching catalog items.
3. **Empty Profile Fallback:**
   - If neither likes nor saved preferences exist, returns `None` (falling back to query-only similarity).

### 4.5 Personalized Hybrid Ranking Formula
For each eligible candidate item $i$:
$$\text{score}(i) = 0.8 \times \cos(\mathbf{q}, \mathbf{v}_i) + 0.2 \times \cos(\mathbf{u}, \mathbf{v}_i)$$

Where:
- $\mathbf{q}$ is the unit-normalized query vector.
- $\mathbf{u}$ is the unit-normalized user preference vector.
- $\mathbf{v}_i$ is the unit-normalized product embedding vector.
- When $\mathbf{u}$ is unavailable: $\text{score}(i) = \cos(\mathbf{q}, \mathbf{v}_i)$.

**Tie-Breaking:** Stable sort descending by `score`, breaking ties ascending by string `id`.  
**Result Limit:** Up to `top_k=5` items (or fewer if fewer eligible items exist).  
**No-Match Handling:** Returns `status: "no_match"` with actionable suggestions on relaxing constraints.

---

## 5. Execution and Reproduction Guide

### Run via Command Line:
```bash
# Execute Phase 3 automated validation suite (30 checks)
python fashion_project/validate_phase3.py

# Execute Phase 2 automated validation suite (17 checks)
python fashion_project/validate_phase2.py

# Execute Phase 1 automated validation suite (12 checks)
python fashion_project/validate_phase1.py
```

### Run via Interactive Notebook:
1. Open `fashion_project/Fashion_Recommender.ipynb` in Colab or Jupyter.
2. Execute **Section 1: Setup** to verify environment and dependencies.
3. Execute **Section 2: Catalog** to inspect the 10,000-item frozen catalog.
4. Execute **Section 3: Embeddings** to inspect product descriptions and embeddings.
5. Execute **Section 4: Profiles** to create anonymous profiles, save preferences, record likes, and build user preference vectors.
6. Execute **Section 6: Ranking** to run personalized recommendations ($0.8 \times \text{query} + 0.2 \times \text{profile}$), dismiss products, verify exclusion, and demonstrate multi-user profile isolation.

---

## 6. Comprehensive Validation Results

### Phase 3 Validation Suite (`validate_phase3.py` - 30/30 Checks Passed)

| # | Check Description | Status | Verification Detail |
|---|---|---|---|
| 01 | SQLite database initialization | **PASS** | Tables `profiles`, `feedback`, `active_states` created; Foreign keys enabled |
| 02 | Profile creation | **PASS** | Created anonymous profile ID `usr_...` with default empty preferences |
| 03 | Unique profile IDs | **PASS** | Generated 10 distinct unique profile IDs without collision |
| 04 | Profile retrieval | **PASS** | Retrieved profile record with UTC ISO timestamps |
| 05 | Preference persistence | **PASS** | Stored and retrieved explicit categories and colors |
| 06 | Active constraint persistence | **PASS** | Persisted and retrieved active constraints in `active_states` table |
| 07 | Multiple profile isolation | **PASS** | Profile A and B maintain completely independent preferences, constraints, & likes |
| 08 | Invalid profile handling | **PASS** | Unknown profile gracefully returns None, rejects feedback write, defaults to query-only |
| 09 | Invalid product ID handling | **PASS** | Non-catalog product ID rejected with ValueError; database remains uncorrupted |
| 10 | Like persistence | **PASS** | Liked product ID persisted in `feedback` table |
| 11 | Idempotent likes | **PASS** | Repeated likes result in exactly 1 database record (idempotency verified) |
| 12 | Dismissal persistence | **PASS** | Dismissed product ID persisted in dismissals list |
| 13 | Dismissed item exclusion | **PASS** | Dismissed item strictly excluded from subsequent recommendations |
| 14 | User vector generation | **PASS** | Mean normalized vector computed correctly ($\text{dim}=384, \|\mathbf{u}\|_2 = 1.0$) |
| 15 | Empty-profile fallback | **PASS** | Returns `None` when profile has no likes and no preferences |
| 16 | Explicit preference fallback | **PASS** | Synthesizes normalized 384-d vector from saved explicit categories/colors |
| 17 | Strict category filtering | **PASS** | 100% of returned items match requested `articleType` |
| 18 | Strict color filtering | **PASS** | 100% of returned items match requested `baseColour` |
| 19 | Excluded color filtering | **PASS** | Excluded color strictly absent from all returned items |
| 20 | Missing attribute handling | **PASS** | Rows with None, NaN, or empty string fail strict constraints |
| 21 | Combined AND/OR filtering | **PASS** | Satisfies OR within field, AND across fields, and exclusions |
| 22 | Personalized ranking | **PASS** | Verified $0.8 \times \text{query} + 0.2 \times \text{profile}$ formula to within float tolerance |
| 23 | Query-only fallback | **PASS** | When user vector is unavailable, score equals query similarity exactly |
| 24 | Hard constraint preservation | **PASS** | Hard constraints strictly override user preference vector |
| 25 | Top-five result limit | **PASS** | Returns at most 5 items for `top_k=5` |
| 26 | Fewer-than-five result handling | **PASS** | Returns exactly 3 items without padding when only 3 matching items exist |
| 27 | No-match handling | **PASS** | Gracefully reports no matches with actionable constraint revision guidance |
| 28 | Deterministic tie-breaking | **PASS** | Tie-breaking verified (score descending, ID ascending) |
| 29 | SQLite restart persistence | **PASS** | State fully verified across complete database reconnect |
| 30 | Phase 1 & 2 regression validation | **PASS** | Phase 1: 12/12 passed \| Phase 2: 17/17 passed |

---

## 7. Phase Progression Summary

- **Phase 1:** Complete & Verified (12/12 Checks Passed)
- **Phase 2:** Complete & Verified (17/17 Checks Passed)
- **Phase 3:** Complete & Verified (30/30 Checks Passed)
- **Phase 4:** *Planned* — Qwen2.5-3B-Instruct Preference Extraction & LLM Explanations
- **Phase 5:** *Planned* — Interactive UI Widgets
- **Phase 6:** *Planned* — Benchmark Evaluation
- **Phase 7:** *Planned* — Packaging & Final Submission
