# Future Enhancements & Roadmap
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Algorithmic Enhancements

### 1.1 Dual-Modality User Taste Profiles
Extend the user profile vector synthesis from text-only centroids to dual multimodal centroids:
$$\mathbf{u}_{\text{text}} \in \mathbb{R}^{384}, \quad \mathbf{u}_{\text{vis}} \in \mathbb{R}^{512}$$
This will allow direct visual taste modeling, capturing aesthetic silhouette preferences even when textual descriptions are identical.

### 1.2 Temporal Decay & Reinforcement Weighting
Incorporate an exponential time-decay factor $\lambda$ on user likes:
$$\mathbf{u}(t) = \sum_{i=1}^k e^{-\lambda(t - t_i)} \mathbf{v}_i$$
This ensures that recent browsing behavior exerts greater influence than preferences registered weeks prior, dynamically adapting to shifting seasonal wardrobes.

### 1.3 Approximate Nearest Neighbors (ANN) Indexing
For catalogs scaling beyond 100,000 items, replace pure NumPy matrix dot products with an HNSW (*Hierarchical Navigable Small World*) index (e.g., via FAISS or USearch):
- Reduces search complexity from $O(N \cdot D)$ to $O(\log N \cdot D)$.
- Retains sub-2 ms query latency on million-item inventories.

---

## 2. Model & Vision Enhancements

### 2.1 Fashion-Specific Vision Foundations
Fine-tune CLIP using domain-specific fashion datasets (such as Fashion-Gen or DeepFashion) using Low-Rank Adaptation (LoRA) or adopt specialized models such as Fashion-CLIP (Xu et al., 2021). This improves sensitivity to garment attributes like collar cuts, sleeve lengths, and hemlines.

### 2.2 Multi-Garment Outfit Generation
Extend single-product recommendation to complementary outfit synthesis:
- Given a selected topwear item (e.g., navy blue shirt), recommend compatible bottomwear (e.g., khaki trousers) and footwear (e.g., brown brogues) using graph neural networks (GNNs) trained on aesthetic outfit compatibility.

---

## 3. Infrastructure & Deployment Enhancements

### 3.1 Distributed Vector Database & Cache
Transition from SQLite to PostgreSQL with `pgvector` or a dedicated vector database (Milvus / Qdrant) combined with a Redis caching layer for enterprise-grade horizontal scaling.

### 3.2 Mobile Progressive Web App (PWA)
Package the Streamlit web client into a PWA or React Native mobile interface with direct camera integration, allowing users to snap clothing photos in real-time and retrieve instant catalog matches.
