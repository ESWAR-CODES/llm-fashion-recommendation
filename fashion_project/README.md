# LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

**Academic Level:** AI / Machine Learning B.Tech Capstone / Research Project  
**Authoritative Reference:** `End_to_End PipeLine.pdf`  
**Current Milestone:** Phase 4 — Qwen-Powered Preference Extraction & Evidence-Based Explanations

---

## 1. Project Objective
This project builds an end-to-end conversational fashion recommendation system capable of:
1. Parsing natural language user clothing requests and conversational constraint updates using a localized Large Language Model (**Qwen2.5-3B-Instruct**).
2. Maintaining persistent user style profiles and multi-turn conversational preferences using SQLite (`backups/profiles.sqlite`).
3. Conducting semantic retrieval and personalized ranking using dense sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**).
4. Providing interactive image recommendations with grounded, evidence-based natural language explanations.

> [!IMPORTANT]
> **Phase Status:**
> - **Phase 1 (Complete & Verified):** Dataset acquisition, 10-field inspection, Pillow decoding validation, data cleaning, and 10,000-item stratified catalog creation.
> - **Phase 2 (Complete & Verified):** Product text description generation, `all-MiniLM-L6-v2` embedding generation, persistence, manifest hashing, and pure NumPy semantic retrieval.
> - **Phase 3 (Complete & Verified):** SQLite user profile management, explicit preference persistence, active request constraints, product feedback (likes/dismissals), user preference vector construction, candidate filtering, and hybrid personalized ranking ($0.8 \times \text{query} + 0.2 \times \text{profile}$).
> - **Phase 4 (Complete & Verified):** Qwen2.5-3B-Instruct preference extraction, Pydantic update validation, multi-turn state updates (`set`, `add`, `remove`, `clear`, `reset`), unsupported attribute isolation, and evidence-based explanation generation with factual attribute grounding and claim rejection.
> - **Phases 5–7 (Reserved):** ipywidgets interactive interface, 40-query benchmark evaluation, and packaging are strictly reserved for subsequent phases.

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
├── Fashion_Recommender.ipynb     # Main interactive notebook (Sections 1–7 operational)
├── README.md                     # Comprehensive project documentation
├── config.json                   # Environment, paths, model revisions, seeds, and ranking weights
├── requirements-lock.txt         # Frozen dependency specifications
│
├── prepare_catalog.py            # Phase 1 data engineering & Pillow image audit pipeline
├── embedding_engine.py           # Phase 2 core engine (descriptions, encoding, persistence, search)
├── profile_manager.py            # Phase 3 profile CRUD, constraint validation, feedback & vectors
├── personalized_ranker.py        # Phase 3 candidate filtering & personalized ranking engine
├── preference_extractor.py       # Phase 4 top-level export bridge
├── explanation_engine.py         # Phase 4 top-level export bridge
│
├── validate_phase1.py            # Phase 1 automated test suite (12 checks)
├── validate_phase2.py            # Phase 2 automated test suite (17 checks)
├── validate_phase3.py            # Phase 3 automated test suite (30 checks)
├── validate_phase4.py            # Phase 4 top-level runner (28 checks)
│
├── engines/
│   ├── preference_extractor.py   # Phase 4 Qwen extraction, Pydantic schemas, & state updater
│   └── explanation_engine.py     # Phase 4 evidence-based explanations & claim grounding
│
├── tests/
│   └── validate_phase4.py        # Phase 4 complete 28-check validation suite
│
├── docs/
│   ├── End_to_End_PipeLine.pdf   # Authoritative project specification
│   ├── phase_reports/
│   │   ├── PHASE_1_COMPLETION_REPORT.md
│   │   ├── PHASE_2_COMPLETION_REPORT.md
│   │   ├── PHASE_3_COMPLETION_REPORT.md
│   │   └── PHASE_4_COMPLETION_REPORT.md
│   └── plans/
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

## 4. Phase 4: Qwen Preference Extraction & Evidence-Based Explanations

### 4.1 Language Model Configuration
- **Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Tokenizer:** Official Qwen tokenizer (Vocabulary size: 151,643) with ChatML template (`<|im_start|>` / `<|im_end|>`).
- **Inference Mode:** `eval()` mode with deterministic decoding (`do_sample=False`, `max_new_tokens=256`).
- **Hardware Auto-Detection:** Automatically configures CUDA with `torch.bfloat16`/`float16` and `device_map="auto"` when a GPU is available; supports CPU execution with memory inspection and graceful fallbacks.

### 4.2 Preference Extraction & Pydantic Schema
The model returns strictly structured JSON validating against Pydantic schemas:
```json
{
  "reset": false,
  "unsupported": [],
  "updates": [
    {
      "field": "articleType",
      "op": "set",
      "values": ["Shirts"]
    },
    {
      "field": "exclude_colors",
      "op": "add",
      "values": ["Red"]
    }
  ]
}
```

**Supported Operations:**
- `set`: Replaces the current values of the field.
- `add`: Unions new values into the existing field list.
- `remove`: Subtracts specified values from the field list.
- `clear`: Empties the field list.
- `reset`: Resets active session constraints to `{}` without touching saved profile preferences.

**Supported Fields:**
`articleType`, `baseColour`, `usage`, `masterCategory`, `exclude_colors`.

**Unsupported Attributes:**
Non-catalog attributes (e.g. size "Large", price "under $50", brand "Nike") are parsed and safely isolated into the `unsupported` list rather than mapped to fictitious catalog fields.

### 4.3 Catalog Domain Validation & Synonym Resolution
Attribute values are checked against the frozen catalog metadata domain. Deterministic synonym mapping resolves common colloquial terms to canonical catalog categories (e.g., `"navy"` $\rightarrow$ `"Navy Blue"`, `"tee"` $\rightarrow$ `"Tshirts"`). Invented values are strictly rejected, preserving the previous valid state.

### 4.4 Evidence-Based LLM Explanations
Explanations are strictly grounded in verified metadata:
- **Scope Restriction:** Sends *only* the top-5 recommended products and their recorded attributes (`articleType`, `baseColour`, `usage`, `masterCategory`, `season`) to Qwen. The entire catalog is never transmitted.
- **Factual Attribute Grounding:** Any mentioned color, garment type, or usage is validated against the product's catalog metadata.
- **Unsupported Claim Rejection:** Explanations containing unverified claims regarding fabric materials ("cotton", "silk"), fit ("perfect fit", "slim fit"), quality ("premium", "luxury"), price ("cheap"), or stock ("in stock") are strictly rejected.
- **Deterministic Template Fallback:** If an LLM explanation fails or is rejected, a grounded template explanation is assigned (e.g., *"This is a Navy Blue Shirts marked Casual, matching your request."*). Product recommendations and rankings are 100% preserved.

---

## 5. Execution and Reproduction Guide

### Run via Command Line:
```bash
# Execute Phase 4 automated validation suite (28 checks)
python fashion_project/validate_phase4.py

# Execute Phase 3 regression suite (30 checks)
python fashion_project/validate_phase3.py

# Execute Phase 2 regression suite (17 checks)
python fashion_project/validate_phase2.py

# Execute Phase 1 regression suite (12 checks)
python fashion_project/validate_phase1.py
```

### Run via Interactive Notebook:
1. Open `fashion_project/Fashion_Recommender.ipynb` in Colab or Jupyter.
2. Execute **Section 1: Setup** through **Section 4: Profiles**.
3. Execute **Section 5: LLM Preference Extraction** to load Qwen tokenizer, inspect hardware, extract multi-turn constraints, and validate state transitions.
4. Execute **Section 6: Ranking** to generate candidate recommendations.
5. Execute **Section 7: Explanations** to generate grounded product explanations, test unsupported claim rejection, and verify deterministic template fallbacks.

---

## 6. Comprehensive Validation Results

### Phase 4 Validation Suite (`validate_phase4.py` - 28/28 Checks Passed)

| # | Check Description | Status | Verification Detail |
|---|---|---|---|
| 01 | Model initialization & Hardware detection | **PASS** | Detected CPU/CUDA, VRAM, available RAM, and PyTorch environment |
| 02 | Tokenizer loading | **PASS** | Loaded Qwen2.5-3B-Instruct tokenizer (Vocab size: 151,643) |
| 03 | Chat template usage | **PASS** | Official Qwen ChatML template (`<|im_start|>` / `<|im_end|>`) verified |
| 04 | JSON extraction | **PASS** | Parsed JSON from markdown fences and raw text strings |
| 05 | Pydantic schema validation | **PASS** | Validated `ProposedUpdate` schema structure and operation types |
| 06 | Supported fields | **PASS** | Accepted approved fields; rejected unauthorized field names |
| 07 | Supported operations | **PASS** | Accepted `set`, `add`, `remove`, `clear`; rejected invalid operations |
| 08 | Catalog value validation | **PASS** | Valid catalog attributes and synonyms accepted; invented values rejected |
| 09 | Set operation | **PASS** | Replaced previous field values with proposed list |
| 10 | Add operation | **PASS** | Unioned new values into existing field list |
| 11 | Remove operation | **PASS** | Subtracted specified values from existing list |
| 12 | Clear operation | **PASS** | Cleared field while preserving remaining active constraints |
| 13 | Reset behavior | **PASS** | Reset active session constraints to empty without altering saved profile |
| 14 | Unsupported attribute handling | **PASS** | Size, price, and brand requests safely isolated into `unsupported` list |
| 15 | Multi-turn correction | **PASS** | 4-turn sequence preserved compatible context and applied corrections |
| 16 | Previous state preservation | **PASS** | Failed update proposal preserved active state completely intact |
| 17 | Malformed JSON handling | **PASS** | Caught malformed model output syntax error without unhandled crash |
| 18 | Retry limit | **PASS** | Enforced bounded single corrective retry limit |
| 19 | Manual fallback | **PASS** | Supported direct manual preference update when extraction fails |
| 20 | Explanation generation | **PASS** | Generated concise factual explanations for recommended items |
| 21 | Product ID validation | **PASS** | Strictly pruned product IDs outside returned recommendation set |
| 22 | Attribute grounding | **PASS** | Verified factual color/usage matching; rejected ungrounded color claim |
| 23 | Unsupported claim rejection | **PASS** | Rejected claims containing 'premium', 'perfect fit', 'cheap', 'in stock' |
| 24 | Deterministic explanation fallback | **PASS** | Produced grounded template explanation when LLM is unavailable |
| 25 | Recommendation preservation | **PASS** | 100% of product recommendations preserved regardless of explanation failure |
| 26 | Phase 1 regression validation | **PASS** | Phase 1: 12/12 checks passed (0 failures) |
| 27 | Phase 2 regression validation | **PASS** | Phase 2: 17/17 checks passed (0 failures) |
| 28 | Phase 3 regression validation | **PASS** | Phase 3: 30/30 checks passed (0 failures) |

---

## 7. Phase Progression Summary

- **Phase 1:** Complete & Verified (12/12 Checks Passed)
- **Phase 2:** Complete & Verified (17/17 Checks Passed)
- **Phase 3:** Complete & Verified (30/30 Checks Passed)
- **Phase 4:** Complete & Verified (28/28 Checks Passed)
- **Phase 5:** *Planned* — Interactive UI Widgets (`ipywidgets`)
- **Phase 6:** *Planned* — System Evaluation & 40-Query Benchmark Suite
- **Phase 7:** *Planned* — Google Drive Synchronization & Export
