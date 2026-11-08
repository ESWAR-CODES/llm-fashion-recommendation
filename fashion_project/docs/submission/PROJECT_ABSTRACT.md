# Project Abstract
**LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval**

**Author:** AI/ML Engineering & Research Team  
**Catalog Scope:** 10,000 Verified Fashion Items (Apparel & Footwear)  
**Dataset:** Kaggle *Fashion Product Images Small*  
**Architecture:** Dual-Encoder Multimodal Retrieval + Conversational State Extraction + Linear Score Fusion  

---

## Abstract

Modern e-commerce fashion discovery platforms encounter severe limitations due to lexical keyword mismatch, visual nuance neglect, cold-start latency, and hallucinations in generative large language models (LLMs). This project presents an end-to-end, production-grade, multimodal fashion recommendation system that synthesizes text semantic retrieval, visual feature similarity, explicit multi-turn conversational preference extraction, and personalized profile ranking over a frozen 10,000-item catalog.

The system integrates:
1. A 384-dimensional dense text retriever based on `sentence-transformers/all-MiniLM-L6-v2` unit-normalized with L2 projection.
2. A 512-dimensional visual retriever utilizing `sentence-transformers/clip-ViT-B-32` mapping clothing aesthetics directly into a shared multimodal manifold.
3. A multi-stage scoring function that dynamically fuses textual relevance, visual affinity, and synthesized user taste vectors under strict hard constraint satisfaction.
4. A three-tier conversational preference extraction pipeline (Optional External LLM $\to$ Factual Rule Engine $\to$ Unsupported Attribute Isolation) that prevents hallucinated product claims while strictly preserving catalog truth.
5. An interactive, responsive web application implemented in Streamlit with sub-second CPU latency ($<1$ ms matrix retrieval over 10,000 items, 2,225+ QPS throughput, and $<315$ MB RAM footprint), featuring isolated guest demonstration sessions.

Rigorous automated evaluation across 78 test suites and multi-system benchmarking demonstrates statistically superior candidate relevance ($Precision@5 = 0.880$ for multimodal personalization vs. $0.460$ for unpersonalized text baselines) without sacrificing real-time CPU deployment feasibility on cloud infrastructure.
