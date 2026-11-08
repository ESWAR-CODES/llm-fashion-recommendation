# Project Objectives
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Primary Academic & Engineering Goals

The project is structured around four primary engineering objectives:

### Objective 1: Frozen Catalog Integrity & Data Pipeline
- Curate and validate a frozen 10,000-product subset from the Kaggle *Fashion Product Images Small* dataset, restricted exclusively to the `Apparel` and `Footwear` master categories.
- Ensure 100% data integrity with zero null values in critical attributes (`gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `usage`, `productDisplayName`).
- Build reproducible SHA-256 manifests that verify artifact determinism across restarts.

### Objective 2: Dual-Encoder Multimodal Vector Space
- **Semantic Text Embeddings:** Compute, normalize (L2 unit norm), and persist 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` for all 10,000 catalog descriptions.
- **Visual Image Embeddings:** Compute, normalize (L2 unit norm), and persist 512-dimensional dense visual representations using `sentence-transformers/clip-ViT-B-32` across all 10,000 product images.
- Ensure row-for-row alignment between catalog metadata, text embedding indices, and visual vector indices.

### Objective 3: Multi-Stage Personalized Ranking with Hard Filtering
- Formalize a transparent, mathematically grounded linear score fusion formula:
  $$\text{Score}(p) = W_t \cdot \cos(\mathbf{q}_{\text{text}}, \mathbf{v}_{\text{text}, p}) + W_v \cdot \cos(\mathbf{q}_{\text{img}}, \mathbf{v}_{\text{img}, p}) + W_p \cdot \cos(\mathbf{u}, \mathbf{v}_{\text{text}, p})$$
- Dynamically normalize weights such that $W_t + W_v + W_p = 1.0$ under all modality combinations (text-only, image-only, multimodal, with/without user profile).
- Enforce strict hard constraint filtering prior to similarity ranking with 0% silent relaxation.
- Synthesize user taste profile vectors $\mathbf{u}$ from liked products with decay-free normalization.

### Objective 4: Grounded Conversational Extraction & Explanations
- Implement a three-tier language model provider interface supporting external LLM inference, local execution, and a deterministic rule-based engine.
- Isolate and reject unsupported attributes (pricing, sizing, material composition, stock inventory) during multi-turn preference extraction.
- Guarantee that all recommendation explanations cite exclusively factual, verified attributes present in catalog metadata.

### Objective 5: Cloud Deployment & Production Latency
- Package the system into a modular, production-ready Streamlit application targeted for Streamlit Community Cloud.
- Maintain total system memory consumption under 500 MB (well within 1 GB cloud container limits).
- Achieve sub-100 ms end-to-end CPU recommendation latency and $>2,000$ QPS vector search throughput.
- Deliver isolated, anonymous guest sessions to ensure seamless public evaluation.
