# Literature Review
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Evolution of Recommendation Systems

### 1.1 Collaborative Filtering & Matrix Factorization
Early e-commerce recommendation systems relied extensively on Collaborative Filtering (CF) and Matrix Factorization (MF) techniques, such as Singular Value Decomposition (SVD) and Alternating Least Squares (ALS) (Koren et al., 2009). While effective at capturing latent user-item interaction matrices in high-density environments, collaborative filtering exhibits severe vulnerabilities:
- **Cold-Start Problem:** Inability to recommend newly cataloged items without prior interaction history.
- **Content Blindness:** Inability to reason about stylistic nuances, visual colors, and seasonal changes.
- **Explainability Deficit:** Latent factor representations cannot provide human-interpretable justifications.

### 1.2 Content-Based & Semantic Information Retrieval
To mitigate cold-start limitations, content-based recommendation systems map items into explicit feature spaces (Lops et al., 2011). In recent years, Transformer-based Sentence Embeddings (Reimers & Gurevych, 2019) have revolutionized semantic retrieval. Models such as `all-MiniLM-L6-v2` distill large cross-encoder networks into compact 384-dimensional bi-encoders capable of semantic similarity search via dot products:
$$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
Unit-normalizing embeddings transforms cosine calculation into a pure inner product, enabling millisecond retrieval over thousands of products.

---

## 2. Multimodal Fashion Representation

Fashion is intrinsically visual. Text descriptions like "casual cotton t-shirt" underdetermine the item's cut, print, and silhouette.

### 2.1 Contrastive Language-Image Pretraining (CLIP)
Radford et al. (2021) introduced CLIP (*Contrastive Language-Image Pretraining*), training dual Vision and Text Transformers across 400 million image-text pairs via InfoNCE loss:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{z}_i \cdot \mathbf{t}_i / \tau)}{\sum_j \exp(\mathbf{z}_i \cdot \mathbf{t}_j / \tau)}$$
In fashion retrieval, CLIP ViT-B/32 generates a 512-dimensional continuous aesthetic representation that allows zero-shot visual similarity retrieval and cross-modal querying (image-to-image and text-to-image).

### 2.2 Hybrid Score Fusion in Information Retrieval
Multimodal retrieval requires combining disparate signal spaces. Prior literature distinguishes between *early fusion* (concatenating raw feature vectors prior to classification) and *late fusion* (computing modality-specific similarity scores and combining them via linear weighting) (Snoek et al., 2005). Late linear score fusion provides strict interpretability, runtime efficiency, and resilience against missing modalities.

---

## 3. Large Language Models in Conversational Recommendation

Conversational Recommender Systems (CRS) have shifted from rule-based slot filling toward Large Language Models (LLMs) (Gao et al., 2021; Wang et al., 2023). However, generative LLMs introduce significant risks when applied directly:
1. **Hallucination of Catalog SKUs:** Without explicit grounding, generative decoders fabricate product metadata (Ji et al., 2023).
2. **Computational Overhead:** Running 7B+ parameter autoregressive models on CPUs incurs prohibitive latency (5–15 seconds per turn), rendering them impractical for interactive web interfaces.
3. **Decoupled Architecture Solution:** Modern consensus advocates decoupling intent extraction and constraint parsing (handled by conversational models or deterministic grammars) from candidate ranking (handled by dense vector indices and relational databases) (Shuster et al., 2022).

This project adopts this decoupled paradigm, ensuring mathematically bounded latency, deterministic grounding, and zero risk of catalog fabrication.
