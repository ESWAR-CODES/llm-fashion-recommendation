# AGENTS.md: Developer & Agent Codebase Guide

## 1. Project Purpose & Core Rules

This repository houses an **LLM-Assisted Personalized Fashion Recommendation System** combining:
1. Natural language search and conversational preference refinement.
2. Semantic product retrieval via `all-MiniLM-L6-v2`.
3. Personalized ranking ($0.8\mathbf{q} + 0.2\mathbf{u}$) with strict candidate constraints.
4. Visual product similarity foundation using CLIP.
5. Evidence-grounded explanations with metadata verification.
6. A professional Streamlit web application.

---

## 2. Invariants Agents Must NEVER Break

1. **Frozen Catalog Invariants:**
   - Catalog size is strictly **10,000 items**.
   - Categories are strictly restricted to `Apparel` and `Footwear`.
   - Never regenerate `artifacts/catalog.csv` or change random seed 42.
   - Row $j$ in `artifacts/catalog.csv` must match index $j$ in `artifacts/catalog_ids.json` and row $j$ in `artifacts/embeddings.npy`.

2. **Zero Fabrication Policy:**
   - Never invent missing product attributes.
   - Never create fabricated product records or mock products as real catalog items.
   - Never associate an unrelated image with a product.
   - Never describe cosine similarity as "recommendation accuracy".

3. **No Silent Relaxation:**
   - Candidate filtering must strictly enforce hard constraints (AND across distinct fields, OR within same field, excluded colors, dismissed products).
   - If 0 items match, return zero results with actionable suggestions rather than relaxing filters.

4. **CPU-First & Low Memory Footprint:**
   - Never load heavy models (Qwen or CLIP) at application startup.
   - Precomputed embeddings must be loaded without recomputing.
   - Support graceful fallback to `RuleBasedFallbackProvider` on CPU environments.

5. **Secrets & Git Boundaries:**
   - Exclude all `*.sqlite`, `*.db`, `*.zip`, `data/raw/` from Git tracking.
   - Use `services/config_service.py:resolve_path()` for portable path resolution.

---

## 3. Standard Verification Commands

```bash
# 1. Verify Dataset Installation
python fashion_project/scripts/validate_dataset.py

# 2. Run Consolidated Master Test Suite
python fashion_project/tests/run_all_tests.py

# 3. Launch Streamlit Application
streamlit run fashion_project/app.py
```
