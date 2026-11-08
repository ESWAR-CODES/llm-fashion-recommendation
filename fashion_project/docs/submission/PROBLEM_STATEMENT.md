# Problem Statement
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Context & Motivation

E-commerce fashion search is fundamentally multimodal and personal. Consumers rarely describe clothing using standardized catalog taxonomies (e.g., `articleType: Track Pants`, `masterCategory: Apparel`). Instead, users express intent through:
- Natural-language descriptions with subjective stylistic qualifiers ("breezy summer outfit", "retro minimal sneakers").
- Visual references (photographs taken on mobile phones or screenshots from social media).
- Conversational refinements across multi-turn interactions ("show me casual blue shirts, but avoid striped patterns and red").
- Dynamic personal taste built from browsing history and historical preferences.

## 2. Core Deficiencies in Existing Systems

Conventional e-commerce recommendation systems suffer from several critical shortcomings:

### A. Vocabulary Mismatch in Lexical Keyword Search
Traditional inverted index search engines (e.g., BM25, Elasticsearch) depend on exact string matching. If a catalog item is tagged as "Navy Blue Cotton Tee" and a user searches for "dark blue casual t-shirt", lexical search either misses relevant products or requires brittle, labor-intensive synonym dictionaries.

### B. Modality Blindness
Text descriptions alone fail to capture fine-grained visual aesthetics such as silhouette, texture, neckline cut, collar style, and pattern density. Conversely, purely visual search engines disregard functional constraints such as gender, season, and event appropriateness.

### C. LLM Hallucination and Unbounded Latency
Deploying raw Large Language Models (LLMs) directly for recommendation causes catastrophic failure modes:
1. **Catalog Hallucinations:** Generative models frequently fabricate non-existent SKUs, incorrect product titles, and inaccurate prices.
2. **Unsupported Inferences:** Unconstrained models promise attributes that the product database cannot fulfill (e.g., asserting a cotton shirt is "waterproof" or promising specific sizing availability when sizes are absent).
3. **Severe Hardware & Latency Costs:** Generating text tokens autoregressively for tens of thousands of items is infeasible in low-cost or CPU-only cloud environments ($>10$ seconds per request).

### D. Silent Constraint Violation
Standard collaborative filtering algorithms optimize for statistical correlation and frequently violate hard user constraints (e.g., recommending footwear when the user explicitly filtered for formal shirts, or recommending red dresses when the user said "no red").

## 3. The Research & Engineering Challenge

To address these vulnerabilities, this project formalizes the following core challenge:

> *How can an academic, CPU-compatible machine learning architecture achieve real-time ($<100$ ms) personalized fashion retrieval over a 10,000-item catalog by harmonizing dual dense embeddings (text + image), multi-turn conversational state extraction, and strict constraint filtering without hallucinating product attributes or exceeding cloud memory limits?*
