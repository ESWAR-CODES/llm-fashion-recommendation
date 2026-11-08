# System Architecture
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Modular Multi-Tier Layer Architecture

The application is structured into four decoupled, independently testable architectural tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT PRESENTATION TIER                  │
│  Dashboard  │  Fashion Discovery  │  Assistant  │  Insights     │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │ REST / Python API
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVICE & ORCHESTRATION TIER                │
│  CatalogService   │  RetrievalService   │  RankingService       │
│  CLIPService      │  PreferenceService  │  ExplanationService   │
│  ProductImageService                    │  PersistenceService   │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CORE ML ENGINE & DATA TIER                  │
│  all-MiniLM-L6-v2 (384-d)         │  CLIP ViT-B/32 (512-d)      │
│  NumPy Matrix Vector Search       │  SQLite Profile Storage     │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PERSISTENT STORAGE ARTIFACTS                │
│  catalog.csv (10k) │ embeddings.npy (384) │ clip_embeddings.npy │
│  catalog_images/ (10k JPEGs, 107MB)       │ profiles.sqlite     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Service Responsibilities

### 2.1 `CatalogService`
- In-memory index of 10,000 frozen products.
- Provides $O(1)$ product attribute lookup by canonical string product ID.
- Supplies facet domain listings (genders, master categories, subcategories, article types, colors, usages, seasons).

### 2.2 `EmbeddingService`
- Loads precomputed text embedding matrix `embeddings.npy` of shape `(10000, 384)`.
- Handles on-demand query string encoding via `all-MiniLM-L6-v2`.
- Verifies matrix integrity ($L_2$ norm $\approx 1.0$, zero NaNs/Infs).

### 2.3 `CLIPService`
- Loads precomputed visual embedding matrix `clip_embeddings.npy` of shape `(10000, 512)`.
- Preprocesses and encodes uploaded query images into 512-dimensional unit vectors.
- Validates uploaded images (formats: JPEG, PNG, WebP; size $<10$ MB; RGB decoding).

### 2.4 `RankingService`
- Executes candidate filtering against hard catalog constraints.
- Calculates pure NumPy dot products between query vectors and candidate matrix subsets.
- Synthesizes user taste vectors $\mathbf{u}$ from liked product IDs.
- Calculates dynamic weights and outputs deterministic top-$K$ ranked recommendations.

### 2.5 `ProductImageService`
- Multi-tier image resolution pipeline:
  1. Primary: `data/catalog_images/{id}.jpg` (10,000 isolated images).
  2. Secondary: `data/raw/images/{id}.jpg`.
  3. Remote CDN: Fetches and caches via HTTP when configured.
  4. Fallback: Deterministic vector placeholder.
- LRU decoded memory cache to minimize redundant disk decodes.

### 2.6 `PreferenceExtractionService`
- Three-tier provider interface:
  - `ExternalAPIProvider` (optional OpenAI / Hugging Face credentials).
  - `RuleBasedFallbackProvider` (deterministic, zero latency, 100% reliable).
- Enforces Pydantic schema validation for state mutations (`set`, `add`, `remove`, `clear`, `reset`).
- Rejects unsupported attributes (pricing, sizing, material composition, stock).

### 2.7 `ExplanationService`
- Verifies that any explanation generated for a recommended product is factually grounded in catalog metadata attributes.
- Automatically falls back to deterministic template explanations if a claim cannot be verified against the catalog.

### 2.8 `PersistenceService`
- Thread-safe SQLite backend (`profiles.sqlite`) with connection pooling and WAL mode.
- Manages user profiles, liked products, dismissed products, and active conversational constraints.
- Supports atomic SQLite online snapshot backups.
