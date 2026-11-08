# System Architecture: LLM-Assisted Personalized Fashion Recommendation

## 1. Architectural Overview

The application is structured into decoupled, independently testable Python services and a modern Streamlit user interface designed for CPU-first deployment.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Application                           │
│     app.py  │  ui/sidebar.py  │  ui/styles.py  │  ui/views/*.py        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Independent ML Services                         │
│                                                                        │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │     CatalogService     │  │  ProductImageService   │                │
│  │  - 10k Frozen Catalog  │  │  - LRU In-Memory Cache │                │
│  │  - O(1) Record Lookup  │  │  - Safe Placeholders   │                │
│  └───────────┬────────────┘  └────────────────────────┘                │
│              │                                                         │
│              ▼                                                         │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │    EmbeddingService    │  │    RetrievalService    │                │
│  │  - all-MiniLM-L6-v2    │  │  - Pure NumPy Matrix   │                │
│  │  - (10000, 384) Matrix │  │    Dot Product Cosine  │                │
│  │  - L2 Normalized       │  │  - Sub-millisecond Res │                │
│  └───────────┬────────────┘  └───────────┬────────────┘                │
│              │                           │                             │
│              ▼                           ▼                             │
│  ┌────────────────────────────────────────────────────┐                │
│  │                   RankingService                   │                │
│  │  - Strict Hard Constraints (AND / OR / NOT)        │                │
│  │  - score = 0.8*cos(q,v) + 0.2*cos(u,v)             │                │
│  │  - Dismissal & Onboarding Likes Exclusion          │                │
│  │  - Deterministic Tie-Breaking (Score Desc, ID Asc) │                │
│  └───────────┬───────────────────────────┬────────────┘                │
│              │                           │                             │
│              ▼                           ▼                             │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │   PreferenceService    │  │   ExplanationService   │                │
│  │  - Provider Interface  │  │  - Metadata Grounding  │                │
│  │  - Local Qwen / Fallback│ │  - Anti-Hallucination  │                │
│  │  - Pydantic Validation │  │  - Factual Verification│                │
│  └────────────────────────┘  └────────────────────────┘                │
│                                                                        │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │   PersistenceService   │  │      CLIPService       │                │
│  │  - SQLite Profiles     │  │  - ViT-B/32 Foundation │                │
│  │  - Feedback & States   │  │  - Offline Indexing    │                │
│  │  - User Vector Synth   │  │  - Text-Only Fallback  │                │
│  └────────────────────────┘  └────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Service Responsibilities

### A. Catalog Service (`services/catalog_service.py`)
- Loads frozen catalog artifacts (`artifacts/catalog.csv`, `artifacts/catalog_ids.json`).
- Validates 10,000 product rows, ensuring string IDs and zero nulls in critical metadata.
- Provides $O(1)$ dictionary lookups for product records by string ID.
- Exposes discrete metadata domains (categories, article types, colors, usages, genders).

### B. Product Image Service (`services/image_service.py`)
- Resolves product JPEG images using canonical product IDs (`data/raw/images/<id>.jpg`).
- Maintains an in-memory decoded image cache (LRU) to eliminate redundant disk reads.
- Provides a consistent, deterministic placeholder image when an image is unreadable or missing.
- Enforces strict isolation: never displays an unrelated product image.

### C. Embedding Service (`services/embedding_service.py`)
- Zero-recomputation loader for precomputed MiniLM embeddings (`artifacts/embeddings.npy`).
- Asserts matrix dimensions `(10000, 384)`, `float32` precision, and $L_2 = 1.0$ unit norms.
- Lazy loads `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` only for query encoding.

### D. Retrieval Service (`services/retrieval_service.py`)
- Executes vector dot product cosine similarity over unit-normalized embeddings.
- Executes in under 3 milliseconds over 10,000 items in pure NumPy.
- Strict terminology guardrail: Cosine similarity represents semantic proximity, never "recommendation accuracy".

### E. Ranking Service (`services/ranking_service.py`)
- Stage 1: Candidate Filtering (`filter_candidates`):
  * OR within multiple values of same field.
  * AND across different constraint fields.
  * Excluded colors strictly eliminated.
  * Dismissed products and onboarding likes excluded.
  * Zero silent relaxation.
- Stage 2: Personalized Scoring:
  $$\text{score}(i) = 0.8 \cdot \cos(\mathbf{q}, \mathbf{v}_i) + 0.2 \cdot \cos(\mathbf{u}, \mathbf{v}_i)$$
- Stage 3: Deterministic Sorting:
  * Sort descending by final score.
  * Tie-breaking ascending by canonical string ID.

### F. Preference Extraction Service (`services/preference_service.py`)
- Employs a decoupled `LLMProvider` interface:
  * `LocalQwenProvider`: Uses local Qwen2.5-3B-Instruct if hardware supports it.
  * `RuleBasedFallbackProvider`: Deterministic regex and catalog synonym parser (0ms CPU latency).
- Validates output using Pydantic `ProposedUpdate` schema (`set`, `add`, `remove`, `clear`, `reset`).
- Rejects unsupported attributes (sizing, pricing, fabric, live stock) into an `unsupported` list.
- Preserves prior valid conversation state upon failure.

### G. Explanation Service (`services/explanation_service.py`)
- Generates evidence-based justifications grounded solely in recorded metadata attributes.
- Validates explanation text against catalog metadata with `validate_explanation()`.
- Prohibits marketing fabrications (*"premium"*, *"best quality"*, *"fits perfectly"*, *"cheap"*).

### H. Persistence Service (`services/persistence_service.py`)
- SQLite storage with foreign key constraints and ACID transaction boundaries.
- Anonymous user profiles, explicit preferences, active constraints, and idempotent feedback (`like`, `dismiss`).
- Synthesizes user vector $\mathbf{u}$ as the $L_2$-normalized mean of liked items or explicit preferences.
- Manages atomic snapshot backups via the SQLite Online Backup API (`conn.backup()`).

### I. CLIP Visual Retrieval Foundation (`services/clip_service.py`)
- Implements visual retrieval using `sentence-transformers/clip-ViT-B-32`.
- Offline image indexing foundation (`scripts/generate_clip_embeddings.py`).
- Normalizes image embeddings to 512 dimensions.
- Text-only fallback: The application remains fully functional if visual embeddings have not yet been generated.

---

## 3. CPU-First Deployment Architecture

To support low-cost hosting without requiring expensive continuous GPUs:
1. **Zero Runtime Catalog Recomputation:** Precomputed MiniLM embeddings (15.36 MB) are loaded directly into RAM via `np.load()`.
2. **Lazy Model Initialization:** Models are never loaded during application startup.
3. **Deterministic Fallbacks:** The preference extraction service runs with zero GPU overhead using the rule-based fallback provider on CPU-only machines.
4. **Sub-50ms Latency:** Vectorized NumPy operations deliver instant interactive recommendations on standard CPUs.
