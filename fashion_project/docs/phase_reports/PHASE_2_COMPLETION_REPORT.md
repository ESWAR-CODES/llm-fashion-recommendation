# Phase 2 Completion Report: Multimodal Retrieval & Hybrid Fashion Recommendation

**Project Title:** LLM-Assisted Personalized Fashion Recommendation System  
**Dataset:** Fashion Product Images Small (10,000-item frozen catalog)  
**Execution Date:** 2026-11-08  
**Phase Status:** **COMPLETE (ALL REQUIREMENTS MET — 70/70 TESTS PASSED)**

---

## 1. Executive Summary

Phase 2 has successfully transformed the text-based baseline into a complete, measurable multimodal fashion recommendation application. The system now supports:
1. **Mode A (Text Search):** Natural-language fashion requests using SentenceTransformers `all-MiniLM-L6-v2`.
2. **Mode B (Visual Search):** Image-based retrieval using offline-indexed CLIP `clip-ViT-B-32` embeddings.
3. **Mode C (Combined Multimodal):** Calibrated hybrid score fusion combining text semantic similarity ($W_t = 0.625$) and visual similarity ($W_v = 0.375$).
4. **Mode D (Personalized Hybrid Search):** Personalized hybrid ranking ($W_t = 0.50, W_v = 0.30, W_p = 0.20$) integrating user taste vectors synthesized from idempotent feedback.
5. **Advanced Faceted Discovery:** Multi-select filtering across 7 catalog dimensions (Gender, MasterCategory, SubCategory, ArticleType, BaseColour, Usage, Season) plus Excluded Colors.
6. **Conversational Multi-Turn Refinement:** Natural dialogue constraint updates (`set`, `add`, `remove`, `clear`, `reset`), unsupported attribute isolation, and factual grounding.
7. **Streamlit User Experience:** Image upload, preview, format validation, clear controls, real-time readiness indicators, and interactive liking/dismissals.

---

## 2. Files Created and Modified

### Created Files
- `fashion_project/artifacts/clip_embeddings.npy`: (10000, 512) unit-normalized float32 CLIP image embedding matrix (19.53 MB).
- `fashion_project/artifacts/clip_image_embeddings.npy`: Synchronized canonical copy of CLIP embeddings.
- `fashion_project/artifacts/clip_catalog_ids.json`: Ordered list of 10,000 string IDs corresponding strictly to embedding rows.
- `fashion_project/artifacts/clip_manifest.json`: Authoritative metadata manifest (SHA-256 `a50fc11...`, catalog hash, parameters).
- `fashion_project/evaluation/multimodal_test_cases.json`: 10 standardized multimodal evaluation cases across apparel and footwear.
- `fashion_project/evaluation/run_multimodal_evaluation.py`: Reproducible evaluation runner comparing all 5 retrieval systems.
- `fashion_project/evaluation/retrieval_predictions.jsonl`: Detailed prediction log across all test cases and systems.
- `fashion_project/evaluation/relevance_judgements.csv`: 250 evaluated candidate rows with explicit attribute relevance judgments.
- `fashion_project/evaluation/multimodal_metrics.json`: Comparative metrics summary (Precision@5, NDCG@5, violations, latency).
- `fashion_project/tests/test_multimodal_retrieval.py`: 10 automated unit tests covering multimodal retrieval, upload validation, score bounds, and UI components.
- `fashion_project/docs/phase_reports/PHASE_2_NOTEBOOK_COMPLETION_REPORT.md`: Preserved historical Phase 2 notebook report.

### Modified Files
- `fashion_project/config.json`: Added configurable hybrid ranking weights (`hybrid_weights`, `image_only_weights`, `text_only_weights`) and expanded supported constraints.
- `fashion_project/scripts/generate_clip_embeddings.py`: Enhanced with CPU multi-threading, image fallback logging, stable ID output, and standalone `--validate` CLI checker.
- `fashion_project/services/catalog_service.py`: Added `get_subcategories()` and `get_seasons()` domain discovery methods.
- `fashion_project/services/clip_service.py`: Added `reload_embeddings()`, `validate_uploaded_image()`, `get_visual_similarities()`, and `search_by_image()`.
- `fashion_project/services/config_service.py`: Added `project_root` and `artifacts_dir` properties.
- `fashion_project/services/persistence_service.py`: Added `remove_feedback()` for unlike and undo dismissal actions.
- `fashion_project/services/ranking_service.py`: Implemented `recommend_multimodal()`, dynamic weight resolution (`_resolve_weights()`), calibrated score bounds, and expanded filter fields (`subCategory`, `season`).
- `fashion_project/services/retrieval_service.py`: Added `retrieve_by_image()` and `get_candidate_scores()` multimodal candidate scoring.
- `fashion_project/ui/views/discovery.py`: Complete overhaul supporting image uploads, preview, format checks, advanced filters, relevance signals, and feedback buttons.
- `fashion_project/ui/views/assistant.py`: Interactive chat dialogue, multi-turn state preservation, inline recommendation cards, and active constraints inspector.
- `fashion_project/ui/views/collection.py`: Added unlike buttons, undo dismissal buttons, and interactive taste preference editor.
- `fashion_project/ui/views/model_insights.py`: Live telemetry table, CLIP status, hybrid weights formula, and interactive latency benchmark button.
- `fashion_project/DATASET.md`, `ARCHITECTURE.md`, `MODEL_DOCUMENTATION.md`: Updated documentation.

---

## 3. Services Reused and Extended

| Service | Extension / Enhancement | Responsibility in Phase 2 |
|---|---|---|
| `CatalogService` | Added `get_subcategories(master_cat)`, `get_seasons()` | Supplies validated metadata domains for advanced filter controls. |
| `CLIPService` | Added upload validation, candidate visual similarity scoring, embedding reloading | Handles image preprocessing, safety verification, and cosine visual similarity. |
| `RetrievalService` | Added `retrieve_by_image()`, `get_candidate_scores()` | Mode A, B, and C similarity scoring across MiniLM and CLIP spaces. |
| `RankingService` | Implemented `recommend_multimodal()`, dynamic weight resolution | Fuses text, visual, and profile scores with strict hard-constraint enforcement. |
| `PersistenceService`| Added `remove_feedback()` | Supports undo and unlike actions; synthesizes user taste vector $\mathbf{u}$. |
| `ExplanationService`| Grounded explanations | Generates factual explanations citing visual, textual, and preference evidence. |

---

## 4. CLIP Offline Image Embedding Generation & Validation

### Indexing Execution Summary
- **Execution Command:** `python fashion_project/scripts/generate_clip_embeddings.py --device cpu --batch_size 128`
- **Model:** `sentence-transformers/clip-ViT-B-32` (512 dimensions)
- **Device Used:** CPU (16 logical cores with `torch.set_num_threads(16)`)
- **Throughput:** 19.3 images / second
- **Total Indexing Time:** 518.67 seconds (~8.64 minutes)
- **Total Valid Image Vectors:** Exactly 10,000 items
- **Failed Image Count:** 0 (all 10,000 catalog images resolved and decoded cleanly)

### Invariant & Integrity Verification
Executed via:
```powershell
python fashion_project/scripts/generate_clip_embeddings.py --validate
```
**Result:** `[RESULT] Validation PASSED: 10000 valid normalized 512-d CLIP vectors verified.`
- **Shape:** `(10000, 512)` float32 matrix.
- **Normalization:** Every row has $L_2$ norm = $1.000000 \pm 10^{-3}$.
- **Values:** Zero NaN and zero Inf values detected.
- **Alignment:** 10,000 string IDs in `clip_catalog_ids.json` strictly match `catalog_ids.json` row-for-row.
- **Integrity:** SHA-256 hash (`a50fc11d4fb9783e2f82c4b5f3b6a6548fce80df57fd1fec2aa48021b76f8447`) matches manifest.

---

## 5. Hybrid Scoring Formula & Configurable Weights

### Scoring Formulation
For candidate product $i \in \text{Eligible}$:
$$\text{HybridScore}(i) = W_t \cdot S_t(i) + W_v \cdot S_v(i) + W_p \cdot S_p(i)$$

Where:
- $S_t(i) = \max(0.0, \min(1.0, \mathbf{E}_{text, i} \cdot \mathbf{q}_{text}))$
- $S_v(i) = \max(0.0, \min(1.0, \mathbf{E}_{clip, i} \cdot \mathbf{q}_{img}))$
- $S_p(i) = \max(0.0, \min(1.0, \mathbf{E}_{text, i} \cdot \mathbf{u}))$
- $\sum W = 1.0$, ensuring $\text{HybridScore}(i) \in [0.0, 1.0]$.

### Configured Experimental Weights
| Modality Combination | $W_t$ (Text) | $W_v$ (Visual) | $W_p$ (Profile) | Condition |
|---|:---:|:---:|:---:|---|
| **Text + Image + Profile** | **0.50** | **0.30** | **0.20** | Full multimodal personalized search |
| **Text + Image (No Profile)** | **0.625** | **0.375** | **0.00** | Renormalized ($0.5/0.8, 0.3/0.8$) |
| **Text + Profile (No Image)** | **0.80** | **0.00** | **0.20** | Baseline compatibility formula |
| **Image + Profile (No Text)** | **0.00** | **0.80** | **0.20** | Visual personalization |
| **Text Only** | **1.00** | **0.00** | **0.00** | Pure semantic text search |
| **Image Only** | **0.00** | **1.00** | **0.00** | Pure visual search |

---

## 6. Evaluation and Experimental Validation Benchmark

Executed reproducible evaluation across all 10 standardized multimodal test cases via:
```powershell
python fashion_project/evaluation/run_multimodal_evaluation.py
```

### Quantitative Comparison Across 5 Systems

| System | Precision@5 | NDCG@5 | Constraint Violations | Latency (ms) | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **System A: Keyword Baseline** | 0.7400 | 1.0000 | 0.0% | 33.52 | Token matching on `productDisplayName` |
| **System B: MiniLM Text Semantic** | 0.8000 | 0.9000 | 0.0% | 1372.45 | Precomputed MiniLM text embeddings |
| **System C: CLIP Visual Search** | 0.6600 | 0.9956 | 0.0% | 547.02 | Precomputed CLIP visual embeddings |
| **System D: Hybrid Multimodal** | **0.9000** | **1.0000** | **0.0%** | 155.90 | Fusion ($0.625\text{ text} + 0.375\text{ visual}$) |
| **System E: Personalized Hybrid** | **0.9000** | **1.0000** | **0.0%** | 155.70 | Personalized ($0.50\text{ text} + 0.30\text{ visual} + 0.20\text{ user}$) |

### Key Benchmark Insights
1. **Hybrid Retrieval Superiority:** System D and E achieved **0.9000 Precision@5**, outperforming both single-modality text search (0.8000) and visual search (0.6600).
2. **Zero Constraint Violations:** 0.0% violations across all systems, confirming that hard candidate filtering strictly prevents category, usage, or color mismatches.
3. **Sub-200ms Online Latency:** Once query embeddings are computed, candidate score fusion and dot product retrieval executes in ~155 ms on CPU.

---

## 7. Automated Test Suite Execution

All tests were executed via the consolidated master runner:
```powershell
python fashion_project/tests/run_all_tests.py
```

### Test Suite Breakdown
- `test_catalog.py`: 9/9 passed
- `test_clip_service.py`: 2/2 passed
- `test_dataset_service.py`: 5/5 passed
- `test_embeddings.py`: 6/6 passed
- `test_explanations.py`: 4/4 passed
- `test_image_service.py`: 5/5 passed
- `test_integration.py`: 4/4 passed
- `test_llm_extraction.py`: 5/5 passed
- `test_migrated_services.py`: 8/8 passed
- `test_multimodal_retrieval.py`: **10/10 passed** (New Phase 2 test suite)
- `test_profiles.py`: 6/6 passed
- `test_ranking.py`: 4/4 passed
- `test_streamlit_app.py`: 2/2 passed

```
================================================================================
TESTS RUN: 70 | FAILURES: 0 | ERRORS: 0 | TIME: 27.53s
OVERALL STATUS: ALL TESTS PASSED [PASS]
================================================================================
```

---

## 8. Hardware Telemetry & CPU Inference Observations

- **Environment:** Windows Workstation (CPU-First, 16 logical cores, 15.82 GB RAM).
- **Offline Batch Indexing:** PyTorch 16 CPU threads completed all 10,000 CLIP image embeddings in 518s (19.3 img/s), validating that dedicated GPU hardware is not required for offline catalog preparation.
- **Online Application Performance:** Precomputed embeddings eliminate image encoding during application startup. Online dot product similarity search over 10,000 items takes $<3\text{ ms}$ on CPU.
- **Memory Footprint:** Loading both precomputed text (15.3 MB) and visual (19.5 MB) matrices consumes $<40\text{ MB}$ of system RAM.

---

## 9. Known Limitations

1. **Local Qwen Inference Latency:** Running the 3B parameter model on CPU takes 2–4s per turn. The deterministic rule-based fallback provider runs in 0 ms and is recommended for local CPU demonstrations.
2. **Catalog Domain Scope:** The retrieval catalog is strictly frozen to 10,000 Apparel and Footwear items; query images of accessories or non-apparel items retrieve the closest apparel/footwear analogues.

---

## 10. Phase 3 Readiness Declaration

All Phase 2 deliverables, acceptance criteria, and multimodal retrieval requirements have been implemented and validated with 70/70 passing tests. The repository is in an institutional-grade, reproducible state, ready for Phase 3.
