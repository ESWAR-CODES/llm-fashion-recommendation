# Project Presentation & Defense Outline
**LLM-Assisted Personalized Fashion Recommendation System**

**Recommended Duration:** 15–20 minutes + 5 minutes Q&A  
**Audience:** Academic Evaluation Committee, AI/ML Faculty, Engineering Peers  

---

## Slide 1: Title & Overview
- **Title:** LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval
- **Presenter:** AI/ML Engineering & Research Team
- **Key Takeaway:** An end-to-end multimodal recommender harmonizing dual dense vectors (MiniLM + CLIP), conversational intent extraction, and strict hard constraint satisfaction over 10,000 fashion items on low-cost CPU infrastructure.

## Slide 2: The Core Problem in Fashion Discovery
- Lexical keyword search fails on stylistic phrasing ("breezy summer outfit").
- Text-only retrieval is blind to fabric, cut, and visual aesthetics.
- Raw generative LLMs hallucinate non-existent products and suffer from prohibitive latency.
- Traditional collaborative filtering suffers from cold-start and silently ignores user negative constraints.

## Slide 3: Research Objectives
- Curate a frozen 10,000-item clean catalog (Apparel & Footwear).
- Build dual-encoder vector representations: 384-d text (MiniLM) and 512-d visual (CLIP).
- Formalize transparent linear late fusion with dynamic weight re-normalization.
- Guarantee 100% hard constraint compliance and grounded explanations.
- Deploy a responsive Streamlit application with sub-100 ms latency and $<315$ MB RAM usage.

## Slide 4: System Architecture & Data Flow
- Diagram of the 4-tier decoupled architecture:
  - Streamlit UI Tier
  - Service & Orchestration Tier
  - Core ML & Vector Algebra Tier (Pure NumPy)
  - Persistent SQLite & Artifact Tier

## Slide 5: Mathematical Formulation
- Text projection: $\mathbf{v}_{\text{text}} \in \mathbb{R}^{384}$
- Visual projection: $\mathbf{v}_{\text{img}} \in \mathbb{R}^{512}$
- User taste centroid: $\mathbf{u} = \frac{\sum_{p \in \mathcal{L}_u} \mathbf{v}_{\text{text}, p}}{\|\cdot\|_2}$
- Scoring function: $\text{Score}(p) = W_t S_t(p) + W_v S_v(p) + W_p S_p(p)$
- Proof that weights sum to 1.0 under all modality configurations.

## Slide 6: Candidate Pruning & Constraint Integrity
- Zero silent relaxation: $\mathcal{E} = \{p \in \text{Catalog} \mid \forall c \in \mathcal{C}, \text{satisfies}(p, c) \land p \notin \mathcal{D}_u\}$.
- Boolean masking over DataFrame facets prior to dot products.
- Explaining the `no_match` fallback mechanism with helpful facet relaxation guidance.

## Slide 7: Conversational Intent & Hallucination Defense
- Three-tier provider interface: External LLM $\to$ Rule Engine Fallback $\to$ Unsupported Isolation.
- Pydantic schema validation for state operations (`set`, `add`, `remove`, `clear`, `reset`).
- Explicit rejection of sizing and pricing queries to preserve catalog truth.
- Factual verification of all recommendation justifications.

## Slide 8: Experimental Evaluation & Results
- Benchmarking 4 system configurations:
  - Text-Only Baseline: $P@5 = 0.540$
  - Personalized Text: $P@5 = 0.720$
  - Multimodal Unpersonalized: $P@5 = 0.780$
  - Full Multimodal Personalized: $P@5 = 0.880$ ($+34.0$ percentage point overall lift)
- Constraint Satisfaction Rate: 100.0% across all scenarios.

## Slide 9: Performance Benchmarking & CPU Viability
- Cold startup time: 5.08 seconds.
- Process resident RAM: 312.57 MB (69.5% headroom on 1 GB cloud containers).
- 10,000-item vector dot product: 0.53 ms.
- Total end-to-end recommendation latency: 68.37 ms.
- Search throughput: 2,225+ QPS.

## Slide 10: Production Cloud Deployment & Demonstration
- Target: Streamlit Community Cloud (`share.streamlit.io`).
- Multi-tier image resolution (10,000 isolated images in 107 MB + CDN resolver + fallback).
- Isolated guest demonstration mode for anonymous evaluators.
- Live demonstration of the 5 views: Dashboard, Fashion Discovery, AI Assistant, My Collection, Model Insights.

## Slide 11: Limitations & Future Roadmap
- Limitations: Lack of sizing and inventory data; unimodal text profiling.
- Roadmap: Dual-modality user centroids; temporal time decay; fashion-specific LoRA vision models.

## Slide 12: Conclusion & Q&A
- Summary of academic and engineering contributions.
- 78 automated test suites passing with 100% pass rate.
- Open floor for questions from the evaluation committee.
