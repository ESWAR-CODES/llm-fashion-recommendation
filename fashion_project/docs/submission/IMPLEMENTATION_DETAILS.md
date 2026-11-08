# Implementation Details
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Technology Stack & Software Versions

| Component | Library / Framework | Version | Purpose |
|---|---|:---:|---|
| **Language Runtime** | Python | 3.10 / 3.11 / 3.12 / 3.14 | Core execution environment |
| **Web UI** | Streamlit | $\ge 1.30.0$ | Interactive web application & component rendering |
| **Vector Algebra** | NumPy | $\ge 1.24.0$ | High-speed matrix dot-product similarity search |
| **Data Processing** | Pandas | $\ge 2.0.0$ | Metadata catalog slicing, filtering, and indexing |
| **Image Processing** | Pillow (PIL) | $\ge 10.0.0$ | Image loading, resizing, validation, and Base64 encoding |
| **Text Embedding** | Sentence-Transformers | $\ge 2.2.0$ | `all-MiniLM-L6-v2` dense text representations |
| **Visual Embedding**| Transformers / CLIP | $\ge 4.36.0$ | `clip-ViT-B-32` visual aesthetic representations |
| **Schema Validation**| Pydantic | $\ge 2.5.0$ | Structured state mutation schemas (`ProposedUpdate`) |
| **Database** | SQLite3 | Built-in | Thread-safe profile and feedback persistence |
| **System Diagnostics**| psutil | $\ge 5.9.0$ | Process RSS memory and hardware resource telemetry |

---

## 2. Service Architecture & Design Patterns

### 2.1 Singleton Pattern with Lazy Initialization
All primary service classes (`CatalogService`, `EmbeddingService`, `CLIPService`, `RankingService`, `ProductImageService`, `PersistenceService`, `PreferenceExtractionService`, `ExplanationService`) implement thread-safe Singleton patterns via `get_instance()`. In Streamlit, services are wrapped in `@st.cache_resource`, ensuring that heavy catalog CSV files and embedding matrices are loaded into memory exactly once per worker process.

### 2.2 In-Memory Decoded Image LRU Cache
Disk I/O and JPEG decoding are significant bottlenecks during web serving. `ProductImageService` maintains an in-memory decoded PIL Image cache bounded at 512 entries with FIFO/LRU eviction:
- Decoded images are cached alongside their target thumbnail dimensions (e.g., `180x220`).
- Base64 Data URI strings (`data:image/jpeg;base64,...`) are cached independently to prevent repeated Base64 serialization on page re-runs.

### 2.3 Decoupled Provider Strategy
The conversational extraction interface uses the Strategy Pattern:
```python
class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def extract(self, query: str, state: Dict, catalog_path: str) -> Tuple[ProposedUpdate, str]:
        pass
```
The active provider can be switched dynamically between `ExternalAPIProvider` and `RuleBasedFallbackProvider` without altering downstream state management or UI code.

---

## 3. SQLite Database Schema (`profiles.sqlite`)

```sql
-- User Profiles
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Active Conversational Constraints
CREATE TABLE IF NOT EXISTS constraints (
    profile_id TEXT PRIMARY KEY,
    constraints_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
);

-- User Feedback (Likes & Dismissals)
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    feedback_type TEXT CHECK(feedback_type IN ('like', 'dismiss')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, product_id),
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
);

-- Explicit Profile Preferences
CREATE TABLE IF NOT EXISTS preferences (
    profile_id TEXT PRIMARY KEY,
    preferences_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
);
```

WAL (*Write-Ahead Logging*) mode is enabled on initialization: `PRAGMA journal_mode=WAL;`, ensuring non-blocking concurrent reads and writes.
