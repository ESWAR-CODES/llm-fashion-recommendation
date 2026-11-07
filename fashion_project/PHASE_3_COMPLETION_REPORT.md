# PHASE 3 COMPLETION REPORT
## LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Project Title:** LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval  
**Authoritative Reference:** `End_to_End PipeLine.pdf` (Steps 5 & 7)  
**Phase:** 3 — User Profiles, Explicit Constraints & Personalized Ranking  
**Status:** COMPLETE & RIGOROUSLY VALIDATED (30/30 Checks Passed, 100% Pass Rate)  
**Date of Completion:** 2026-11-08  

---

## 1. Executive Summary

Phase 3 introduces persistent relational user profile management, explicit preference storage, short-term active request constraints, idempotent product feedback (likes and dismissals), user preference vector synthesis, and candidate filtering with personalized recommendation ranking ($0.8 \times \text{query} + 0.2 \times \text{profile}$).

All Phase 3 requirements have been implemented strictly following `End_to_End PipeLine.pdf`. Zero unrecorded attributes were hallucinated, no external vector databases (FAISS, Chroma, Pinecone) or future-phase dependencies (Qwen LLM, ipywidgets) were prematurely introduced, and all 30 automated tests in `validate_phase3.py` passed cleanly alongside full regressions for Phase 1 (12/12) and Phase 2 (17/17).

---

## 2. Implemented Modules and Files

| File Path | Role | Description |
|---|---|---|
| `fashion_project/profile_manager.py` | Profile & Feedback Engine | SQLite CRUD, schema migration, preference validation, constraint parsing, idempotent feedback, and user preference vector synthesis. |
| `fashion_project/personalized_ranker.py` | Recommendation Ranker | Hard constraint filtering (AND/OR, exclusions, dismissals), vectorized cosine similarity scoring ($0.8 \times q + 0.2 \times u$), and deterministic tie-breaking. |
| `fashion_project/validate_phase3.py` | Automated Test Suite | Comprehensive 30-check test suite verifying SQLite persistence, profile isolation, feedback idempotency, vector math, ranking, edge cases, and regressions. |
| `fashion_project/backups/profiles.sqlite` | Relational Storage | SQLite database with foreign keys enabled, containing `profiles`, `feedback`, and `active_states` tables. |
| `fashion_project/Fashion_Recommender.ipynb` | Capstone Notebook | Updated Section 4 (Profiles) and Section 6 (Ranking), demonstrating end-to-end profile creation, likes, vector construction, ranking, dismissals, and profile isolation. |
| `fashion_project/config.json` | Configuration | Updated to `current_phase: 3`, recording database paths, supported constraint keys, and 80/20 ranking weights. |
| `fashion_project/README.md` | Documentation | Comprehensive architectural and reproduction documentation for Phase 3. |

---

## 3. Database Schema (`backups/profiles.sqlite`)

The database is built using Python's standard `sqlite3` library with explicit foreign key enforcement (`PRAGMA foreign_keys = ON;`), parameterized SQL queries, and transaction safety.

### 3.1 Logical Tables

#### 1. `profiles`
Tracks registered anonymous users and their long-term explicit preferences.
```sql
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    preferences_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```
- `profile_id`: Unique string identifier generated as `usr_<uuid4_hex[:12]>`.
- `preferences_json`: JSON string storing validated long-term preferences (e.g. `{"preferred_categories": ["Apparel"], "preferred_colors": ["Navy Blue", "Black"]}`).
- `created_at` / `updated_at`: UTC ISO 8601 timestamps.

#### 2. `feedback`
Tracks user interactions (likes and dismissals) on specific catalog products.
```sql
CREATE TABLE IF NOT EXISTS feedback (
    profile_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('like', 'dismiss')),
    timestamp TEXT NOT NULL,
    PRIMARY KEY (profile_id, item_id),
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);
```
- Composite Primary Key `(profile_id, item_id)` enforces idempotency: repeatedly liking the same product updates the timestamp but never creates duplicate rows.
- Switching feedback (e.g. from `'like'` to `'dismiss'`) cleanly updates the existing record.
- Foreign key cascading deletes all feedback upon profile deletion.

#### 3. `active_states`
Tracks current session request constraints per profile.
```sql
CREATE TABLE IF NOT EXISTS active_states (
    profile_id TEXT PRIMARY KEY,
    constraints_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);
```
- `constraints_json`: Validated request constraints (e.g. `{"articleType": ["Shirts"], "usage": ["Casual"], "exclude_colors": ["Red"]}`).

#### 4. Indexes
```sql
CREATE INDEX IF NOT EXISTS idx_feedback_profile_action 
ON feedback(profile_id, action);
```
Enables instantaneous retrieval of likes and dismissals per profile without full table scans.

---

## 4. Separation of Active Constraints from Saved Preferences

A foundational architectural requirement in Step 5 of `End_to_End PipeLine.pdf` is strictly maintaining two independent states:

1. **Active Request Constraints:**
   - Define immediate search criteria for the current turn: `articleType`, `baseColour`, `usage`, `masterCategory`, and `exclude_colors`.
   - Applied as strict hard filters *before* candidate scoring.
   - Example: A user requesting casual shirts will only be shown casual shirts.
2. **Saved User Preferences:**
   - Define long-term affinities: `preferred_categories`, `preferred_colors`.
   - Used exclusively to compute the user preference vector $\mathbf{u}$.
   - **Never override explicit active constraints:** If a user with saved preference for "Footwear" actively requests "Shirts", the system will never suggest footwear; it will recommend eligible shirts with stylistic modulation aligned with their profile.

---

## 5. Candidate Filtering Logic (`filter_candidates`)

Candidate filtering executes before any similarity scoring, eliminating non-qualifying products from the 10,000-item catalog:

1. **Dismissed Product Exclusion:** Any product ID in the profile's dismissal set is immediately pruned.
2. **Excluded Colors:** Any product whose `baseColour` matches `exclude_colors` strictly fails.
3. **OR within same field:** If `articleType: ["Shirts", "Tshirts"]`, items matching either Shirts or Tshirts pass.
4. **AND across different fields:** Items must satisfy all constrained fields simultaneously (e.g. `articleType == "Shirts"` AND `usage == "Casual"`).
5. **Missing Attribute Safety:** Any candidate record with a missing, null, NaN, or empty attribute for a constrained field strictly fails.
6. **Deduplication:** Ensures candidate IDs are distinct.

---

## 6. User Preference Vector Methodology (`build_user_vector`)

The user vector $\mathbf{u} \in \mathbb{R}^{384}$ is constructed following Step 5 of the pipeline:

1. **Primary Strategy (Liked Products):**
   - Retrieve all product IDs with `action == 'like'` for the profile.
   - Look up each product's row index in `artifacts/catalog_ids.json`.
   - Retrieve corresponding 384-dimensional normalized vectors from `artifacts/embeddings.npy`.
   - Calculate the arithmetic mean vector:
     $$\bar{\mathbf{e}} = \frac{1}{|L|} \sum_{i \in L} \mathbf{e}_i$$
   - Normalize the result to unit L2 norm:
     $$\mathbf{u} = \frac{\bar{\mathbf{e}}}{\|\bar{\mathbf{e}}\|_2}$$
2. **Secondary Strategy (Explicit Preference Fallback):**
   - If no liked items exist, retrieve saved `preferred_categories` and `preferred_colors`.
   - Construct a preference representation and encode with `all-MiniLM-L6-v2` or calculate the normalized mean of catalog items matching the saved preferences.
3. **Query-Only Fallback:**
   - If neither likes nor explicit preferences exist, $\mathbf{u} = \text{None}$.

---

## 7. Personalized Recommendation Ranking Formula

Following Step 7 of `End_to_End PipeLine.pdf`:

For each candidate item $i$ passing all active hard constraints:
$$\text{score}(i) = 0.8 \times \cos(\mathbf{q}, \mathbf{v}_i) + 0.2 \times \cos(\mathbf{u}, \mathbf{v}_i)$$

Where:
- $\cos(\mathbf{q}, \mathbf{v}_i) = \mathbf{q} \cdot \mathbf{v}_i$ (since $\|\mathbf{q}\|_2 = \|\mathbf{v}_i\|_2 = 1$).
- $\cos(\mathbf{u}, \mathbf{v}_i) = \mathbf{u} \cdot \mathbf{v}_i$ (since $\|\mathbf{u}\|_2 = \|\mathbf{v}_i\|_2 = 1$).
- When $\mathbf{u}$ is unavailable: $\text{score}(i) = \cos(\mathbf{q}, \mathbf{v}_i)$.

**Vectorized Scoring:** Scoring is implemented as high-performance NumPy matrix multiplication:
$$\mathbf{s} = 0.8 \times (\mathbf{E}_{\text{eligible}} \mathbf{q}) + 0.2 \times (\mathbf{E}_{\text{eligible}} \mathbf{u})$$
Executing in under 3 milliseconds across the full candidate pool.

**Deterministic Tie-Breaking:** Candidate items are stably sorted descending by score, breaking any score ties ascending by string `id`.  
**Result Clamping:** Up to `top_k=5` items are returned. If fewer matches exist (e.g. 2 or 3), exactly that number is returned without padding.

---

## 8. Validation Test Suite Results (`validate_phase3.py`)

All 30 automated tests execute deterministically and verify runtime behavior against the frozen 10,000-item catalog and embeddings:

| Check | Name | Result | Details |
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
| 17 | Strict category filtering | **PASS** | 100% of returned items match requested `articleType` (1,052 shirts) |
| 18 | Strict color filtering | **PASS** | 100% of returned items match requested `baseColour` (502 navy blue items) |
| 19 | Excluded color filtering | **PASS** | Excluded color strictly absent from all returned items (7,992 items) |
| 20 | Missing attribute handling | **PASS** | Rows with None, NaN, or empty string fail strict constraints |
| 21 | Combined AND/OR filtering | **PASS** | Satisfies OR within field, AND across fields, and exclusions (777 items) |
| 22 | Personalized ranking | **PASS** | Verified $0.8 \times \text{query} + 0.2 \times \text{profile}$ formula to within float tolerance |
| 23 | Query-only fallback | **PASS** | When user vector is unavailable, score equals query similarity exactly |
| 24 | Hard constraint preservation | **PASS** | Hard constraints strictly override user preference vector |
| 25 | Top-five result limit | **PASS** | Returns at most 5 items for `top_k=5` |
| 26 | Fewer-than-five result handling | **PASS** | Returns exactly 3 items without padding when only 3 matching items exist |
| 27 | No-match handling | **PASS** | Gracefully reports no matches with actionable constraint revision guidance |
| 28 | Deterministic tie-breaking | **PASS** | Tie-breaking verified (score descending, ID ascending) |
| 29 | SQLite restart persistence | **PASS** | State fully verified across complete database reconnect |
| 30 | Phase 1 & 2 regression validation | **PASS** | Phase 1: 12/12 passed \| Phase 2: 17/17 passed |

**Overall Result: ALL 30 CHECKS PASSED (30/30)**

---

## 9. Notebook Verification (`Fashion_Recommender.ipynb`)

Sections 4 and 6 have been fully integrated and verified:
1. **Section 4: Profiles:** Demonstrates SQLite initialization, anonymous profile creation, explicit preferences, active constraints, product likes, and user vector construction.
2. **Section 6: Ranking:** Demonstrates personalized recommendation ($0.8 \times \text{query} + 0.2 \times \text{profile}$), product dismissal and negative filtering, and multi-user profile switching.
3. **Placeholders Preserved:** Sections 5 (LLM), 7 (Explanations), 8 (Interface), 9 (Evaluation), and 10 (Export) remain clearly designated placeholders for subsequent phases.

---

## 10. Reproduction Commands

To independently reproduce and verify Phase 3 from terminal:
```bash
# Execute Phase 3 automated validation suite (30 checks)
python fashion_project/validate_phase3.py

# Execute Phase 2 regression suite (17 checks)
python fashion_project/validate_phase2.py

# Execute Phase 1 regression suite (12 checks)
python fashion_project/validate_phase1.py
```

---

## 11. Known Limitations & Technical Notes

1. **Initial 80/20 Weights:** The 0.8 query / 0.2 profile weights represent initial pipeline design choices specified in `End_to_End PipeLine.pdf` and have intentionally not been hyperparameter-tuned in this phase.
2. **No LLM State Parsing:** Active request constraints in Phase 3 are managed deterministically via Python dictionaries and SQLite state. LLM-based conversational constraint extraction belongs to Phase 4 (Qwen2.5-3B-Instruct).
3. **No UI Widgets:** Interactive recommendation widgets belong to Phase 5 (`ipywidgets`). Visualizations in Phase 3 are rendered using standard Pillow and Matplotlib figures.

---

## 12. Scope Boundaries & Next Phase Readiness

- **Completed Phases:**
  - [x] Phase 1: Environment Setup, Dataset Acquisition & Catalog Preparation (12/12 Checks Passed)
  - [x] Phase 2: Product Text Embeddings & Semantic Retrieval (17/17 Checks Passed)
  - [x] Phase 3: User Profiles, Explicit Constraints & Personalized Ranking (30/30 Checks Passed)
- **Pending Phases (Awaiting Authorization):**
  - [ ] Phase 4: Localized LLM Preference Extraction (Qwen2.5-3B-Instruct) & Evidence-Based Explanations
  - [ ] Phase 5: Interactive Interface (`ipywidgets` Conversational UI)
  - [ ] Phase 6: System Evaluation & 40-Query Benchmark Suite
  - [ ] Phase 7: Export, Google Drive Backup & Packaging
