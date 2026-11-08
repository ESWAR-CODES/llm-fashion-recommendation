# System Limitations
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Catalog & Attribute Boundaries

### 1.1 Lack of Inventory & Sizing Data
The Kaggle *Fashion Product Images Small* dataset does not contain real-time stock levels, inventory SKUs, or size availability matrices (e.g., S, M, L, XL, Shoe Size 10).
- **Impact:** The system cannot recommend products based on user sizing constraints or in-stock status.
- **Mitigation:** The preference extraction service explicitly identifies sizing attributes (e.g., "size medium", "size 9") as `unsupported` and politely notifies the user rather than hallucinating availability.

### 1.2 Lack of Pricing & Transaction Logs
The catalog lacks monetary price tags, currency indicators, and historical purchasing transaction logs.
- **Impact:** Recommendation cannot rank by price-to-performance ratio or budget thresholds.
- **Mitigation:** Pricing requests (e.g., "under \$50", "cheap") are isolated as unsupported constraints.

---

## 2. Model & Representation Boundaries

### 2.1 Fixed CLIP ViT-B/32 Visual Resolution
CLIP ViT-B/32 utilizes an input image resolution of $224 \times 224$ pixels.
- **Impact:** Fine-grained micro-patterns, intricate fabric weaves (e.g., herringbone vs. houndstooth), and small stitch details may be compressed during downsampling.
- **Mitigation:** Bi-encoder text embeddings complement visual search by incorporating explicit descriptive metadata (`articleType`, `baseColour`, `productDisplayName`).

### 2.2 Unimodal Text Vector Profiling
The synthetic user taste profile vector $\mathbf{u}$ is currently synthesized from the text embeddings of liked items:
$$\mathbf{u} = \frac{\sum_{p \in \mathcal{L}_u} \mathbf{v}_{\text{text}, p}}{\|\cdot\|_2}$$
- **Impact:** If a user likes an item solely for its visual texture, that aesthetic nuance is captured indirectly via the item's textual description rather than a dedicated visual user profile vector.

---

## 3. Operational & Cloud Infrastructure Boundaries

### 3.1 Single-Node SQLite Concurrency
User profile persistence relies on SQLite. While WAL (*Write-Ahead Logging*) mode supports high concurrent read volume, write operations are serialized via database locks.
- **Impact:** Sufficient for hundreds of concurrent users, but enterprise multi-million user deployments would require PostgreSQL or a distributed relational database.

### 3.2 Ephemeral Cloud Container Storage
Streamlit Community Cloud containers are ephemeral. If an application restarts or is redeployed to a new container, runtime modifications to `profiles.sqlite` are reset unless backed up to external object storage (e.g., AWS S3).
- **Mitigation:** The application includes automated SQLite snapshot backup routines (`PersistenceService.backup()`) ready for external synchronization.
