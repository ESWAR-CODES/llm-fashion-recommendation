# LLM-Assisted Personalized Fashion Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Optimized-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-78%2F78%20Passing-brightgreen.svg)](tests/)
[![Catalog](https://img.shields.io/badge/Catalog-10%2C000%20Items%20(Frozen)-purple.svg)](artifacts/catalog.csv)
[![Deployment](https://img.shields.io/badge/Deploy-Streamlit%20Cloud%20Ready-success.svg)](DEPLOYMENT.md)

An end-to-end, production-grade multimodal fashion recommendation web application. Combines natural language semantic search (`sentence-transformers/all-MiniLM-L6-v2`), visual clothing similarity retrieval (`sentence-transformers/clip-ViT-B-32`), conversational preference extraction with unsupported attribute isolation, dynamic late linear score fusion, and factual explanation grounding.

Targeted for deployment on **Streamlit Community Cloud** with sub-100 ms CPU latency, 2,225+ QPS search throughput, and $<315$ MB resident memory footprint.

---

## 🚀 Quickstart

### 1. Local Execution
```bash
# Clone the repository
git clone https://github.com/ESWAR-CODES/llm-fashion-recommendation.git
cd llm-fashion-recommendation

# Install production dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Master Test Suite (78 Tests)
```bash
python tests/run_all_tests.py
```
Output: `TESTS RUN: 78 | FAILURES: 0 | ERRORS: 0 | OVERALL STATUS: ALL TESTS PASSED [PASS]`

### 3. Run Performance Benchmark Suite
```bash
python scripts/benchmark_performance.py
```

---

## 🌐 Public Cloud Deployment

The repository is pre-configured for one-click deployment on **[Streamlit Community Cloud](https://share.streamlit.io/)**:
- **Main file path:** `app.py` (or `fashion_project/app.py`)
- **Theme & Port:** Configured via `.streamlit/config.toml`
- **Dependencies:** Pinned in `requirements.txt`
- **Guide:** Detailed step-by-step instructions in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📐 System Architecture & Scoring Formulation

### Dynamic Multimodal Linear Late Score Fusion
$$\text{Score}(p) = W_t \cdot \cos(\mathbf{q}_{\text{text}}, \mathbf{v}_{\text{text}, p}) + W_v \cdot \cos(\mathbf{q}_{\text{img}}, \mathbf{v}_{\text{img}, p}) + W_p \cdot \cos(\mathbf{u}, \mathbf{v}_{\text{text}, p})$$

Weights dynamically normalize to $1.0$ across active modalities:
- **Multimodal Personalized:** $W_t = 0.50, W_v = 0.30, W_p = 0.20$
- **Multimodal Unpersonalized:** $W_t = 0.625, W_v = 0.375, W_p = 0.0$
- **Text Personalized:** $W_t = 0.80, W_v = 0.0, W_p = 0.20$
- **Text-Only Baseline:** $W_t = 1.0, W_v = 0.0, W_p = 0.0$

Candidate products are strictly filtered against user hard constraints prior to ranking, guaranteeing **0% silent constraint relaxation**.

---

## ⚡ Performance Benchmarks (Verified on CPU)

| Metric | Result | Benchmark Description |
|---|:---:|---|
| **Cold Startup Latency** | **5.08 s** | Ingests 10,000 items, MiniLM & CLIP matrices |
| **Total Resident Memory (RSS)**| **312.57 MB** | Leaves >68% headroom on 1 GB cloud containers |
| **MiniLM Query Encoding** | **14.09 ms** | Average text query encoding latency |
| **10,000 Text Vector Search** | **0.53 ms** | Pure NumPy dot product across all 10,000 items |
| **10,000 Visual Vector Search**| **0.69 ms** | Pure NumPy dot product across 10,000 512-d CLIP vectors |
| **End-to-End Pipeline Latency**| **68.37 ms** | Query encode + hard pruning + hybrid rank + top-5 |
| **Search Throughput** | **2,225+ QPS**| Concurrent queries per second on CPU |

---

## 📊 Experimental Evaluation

| Configuration | Modalities | Precision@5 | MRR | Constraint Satisfaction |
|---|---|:---:|:---:|:---:|
| **Sys 1: Text Baseline** | Text Only | 0.540 | 0.620 | 100.0% |
| **Sys 2: Personalized Text**| Text + Profile | 0.720 | 0.790 | 100.0% |
| **Sys 3: Multimodal Unpers**| Text + Visual (CLIP)| 0.780 | 0.830 | 100.0% |
| **Sys 4: Full Multimodal** | Text + Visual + Profile| **0.880** | **0.910** | **100.0%** |

---

## 📚 Academic Submission Documentation

Complete publication-quality documentation is available in [`docs/submission/`](docs/submission/):
1. [Project Abstract](docs/submission/PROJECT_ABSTRACT.md)
2. [Problem Statement](docs/submission/PROBLEM_STATEMENT.md)
3. [Objectives](docs/submission/OBJECTIVES.md)
4. [Literature Review](docs/submission/LITERATURE_REVIEW.md)
5. [Proposed Methodology](docs/submission/PROPOSED_METHODOLOGY.md)
6. [System Architecture](docs/submission/SYSTEM_ARCHITECTURE.md)
7. [Dataset Description & Audit](docs/submission/DATASET_DESCRIPTION.md)
8. [Algorithm Design & Pseudocode](docs/submission/ALGORITHM_DESIGN.md)
9. [Implementation Details](docs/submission/IMPLEMENTATION_DETAILS.md)
10. [Experimental Results & Evaluation](docs/submission/EXPERIMENTAL_RESULTS.md)
11. [Performance Analysis & Resource Profiling](docs/submission/PERFORMANCE_ANALYSIS.md)
12. [System Limitations](docs/submission/LIMITATIONS.md)
13. [Future Enhancements & Roadmap](docs/submission/FUTURE_ENHANCEMENTS.md)
14. [References & Bibliography](docs/submission/REFERENCES.md)
15. [User Guide & Operation Manual](docs/submission/USER_GUIDE.md)
16. [Project Presentation Outline](docs/submission/PROJECT_PRESENTATION_OUTLINE.md)
17. [Final Comprehensive Project Report](docs/submission/FINAL_PROJECT_REPORT.md)
