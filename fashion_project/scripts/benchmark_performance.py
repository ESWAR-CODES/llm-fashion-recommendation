"""
scripts/benchmark_performance.py
Production Performance & Latency Benchmarking Suite
LLM-Assisted Personalized Fashion Recommendation System

Measures:
1. Cold startup & model loading latency.
2. Memory footprint (RAM RSS before and after loading 10k catalog & embedding matrices).
3. Text query encoding & dot-product retrieval latency across 10,000 vectors.
4. Visual embedding dot-product retrieval latency across 10,000 vectors.
5. Hybrid multimodal scoring & constraint filtering latency.
6. Warm throughput (Queries Per Second / QPS).
7. JSON report emission for academic documentation.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Ensure fashion_project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil

from services.config_service import ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
from services.ranking_service import RankingService
from services.image_service import ProductImageService


def get_current_rss_mb() -> float:
    """Return process resident set size in megabytes."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("PRODUCTION PERFORMANCE & RESOURCE BENCHMARKING SUITE")
    print(f"Platform: {sys.platform} | Python: {sys.version.split()[0]}")
    print("=" * 80)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "hardware": {},
        "memory": {},
        "latency_ms": {},
        "throughput_qps": {}
    }

    initial_ram = get_current_rss_mb()
    results["memory"]["initial_rss_mb"] = initial_ram
    print(f"[MEMORY] Initial process RSS: {initial_ram:.2f} MB")

    # 1. Measure Startup & Service Loading
    print("\n--- 1. Cold Startup & Service Initialization ---")
    t0 = time.perf_counter()
    cfg = ConfigService.get_instance()
    cat_svc = CatalogService.get_instance()
    t_cat = (time.perf_counter() - t0) * 1000
    cat_ram = get_current_rss_mb()
    print(f"  - Catalog loaded ({len(cat_svc):,} items): {t_cat:.2f} ms (RSS: {cat_ram:.2f} MB)")

    t1 = time.perf_counter()
    emb_svc = EmbeddingService.get_instance()
    t_emb = (time.perf_counter() - t1) * 1000
    emb_ram = get_current_rss_mb()
    print(f"  - MiniLM embeddings loaded (shape {emb_svc.embeddings.shape}): {t_emb:.2f} ms (RSS: {emb_ram:.2f} MB)")

    t2 = time.perf_counter()
    clip_svc = CLIPService.get_instance()
    t_clip = (time.perf_counter() - t2) * 1000
    clip_ram = get_current_rss_mb()
    print(f"  - CLIP embeddings loaded (shape {clip_svc.embeddings.shape if clip_svc.embeddings is not None else 'None'}): {t_clip:.2f} ms (RSS: {clip_ram:.2f} MB)")

    rank_svc = RankingService.get_instance()
    img_svc = ProductImageService.get_instance()
    total_startup_time = (time.perf_counter() - t0) * 1000
    peak_startup_ram = get_current_rss_mb()

    results["memory"]["catalog_rss_mb"] = cat_ram
    results["memory"]["embedding_rss_mb"] = emb_ram
    results["memory"]["clip_rss_mb"] = clip_ram
    results["memory"]["loaded_services_rss_mb"] = peak_startup_ram
    results["memory"]["memory_growth_mb"] = round(peak_startup_ram - initial_ram, 2)
    results["latency_ms"]["catalog_load"] = round(t_cat, 2)
    results["latency_ms"]["minilm_load"] = round(t_emb, 2)
    results["latency_ms"]["clip_load"] = round(t_clip, 2)
    results["latency_ms"]["total_startup"] = round(total_startup_time, 2)

    hw = cfg.get_hardware_info()
    results["hardware"] = hw
    print(f"[SUMMARY] Total cold startup: {total_startup_time:.2f} ms | Net memory delta: +{results['memory']['memory_growth_mb']} MB")

    # 2. MiniLM Text Query Encoding Latency
    print("\n--- 2. MiniLM Text Query Encoding Latency ---")
    sample_queries = [
        "casual blue cotton shirt",
        "black running sports shoes",
        "red summer formal dress",
        "white sneakers with laces",
        "leather jacket for winter"
    ]
    # Warmup
    _ = emb_svc.encode_text(sample_queries[0])

    enc_times: List[float] = []
    for q in sample_queries * 4:  # 20 runs
        start = time.perf_counter()
        _ = emb_svc.encode_text(q)
        enc_times.append((time.perf_counter() - start) * 1000)

    avg_enc = float(np.mean(enc_times))
    p95_enc = float(np.percentile(enc_times, 95))
    min_enc = float(np.min(enc_times))
    print(f"  - MiniLM Query Encoding: Avg {avg_enc:.2f} ms | p95 {p95_enc:.2f} ms | Min {min_enc:.2f} ms")
    results["latency_ms"]["text_encoding_avg"] = round(avg_enc, 2)
    results["latency_ms"]["text_encoding_p95"] = round(p95_enc, 2)

    # 3. Vector Similarity Search (10,000 Items)
    print("\n--- 3. Vector Dot-Product Retrieval Latency (10,000 Catalog Items) ---")
    q_vec = emb_svc.encode_text("navy blue casual shirt")
    mat_text = emb_svc.embeddings

    search_times: List[float] = []
    for _ in range(50):
        start = time.perf_counter()
        sims = np.dot(mat_text, q_vec)
        top5 = np.argpartition(sims, -5)[-5:]
        search_times.append((time.perf_counter() - start) * 1000)

    avg_search = float(np.mean(search_times))
    p95_search = float(np.percentile(search_times, 95))
    print(f"  - Text Similarity (10k x 384): Avg {avg_search:.2f} ms | p95 {p95_search:.2f} ms")
    results["latency_ms"]["text_search_10k_avg"] = round(avg_search, 2)
    results["latency_ms"]["text_search_10k_p95"] = round(p95_search, 2)

    # 4. Visual Vector Retrieval (10,000 CLIP vectors)
    if clip_svc.embeddings is not None:
        mat_clip = clip_svc.embeddings
        dummy_clip_q = mat_clip[0]  # sample 512-d unit vector

        clip_search_times: List[float] = []
        for _ in range(50):
            start = time.perf_counter()
            sims = np.dot(mat_clip, dummy_clip_q)
            top5 = np.argpartition(sims, -5)[-5:]
            clip_search_times.append((time.perf_counter() - start) * 1000)

        avg_clip_search = float(np.mean(clip_search_times))
        p95_clip_search = float(np.percentile(clip_search_times, 95))
        print(f"  - Visual Similarity (10k x 512): Avg {avg_clip_search:.2f} ms | p95 {p95_clip_search:.2f} ms")
        results["latency_ms"]["visual_search_10k_avg"] = round(avg_clip_search, 2)
        results["latency_ms"]["visual_search_10k_p95"] = round(p95_clip_search, 2)

    # 5. End-to-End Multimodal Recommendation Pipeline
    print("\n--- 5. End-to-End Multimodal Recommendation Pipeline ---")
    pipeline_times: List[float] = []
    for _ in range(25):
        start = time.perf_counter()
        res = rank_svc.recommend_multimodal(
            query_text="casual blue cotton shirt",
            constraints={"masterCategory": ["Apparel"], "gender": ["Men"]},
            top_k=5
        )
        pipeline_times.append((time.perf_counter() - start) * 1000)

    avg_pipeline = float(np.mean(pipeline_times))
    p95_pipeline = float(np.percentile(pipeline_times, 95))
    print(f"  - End-to-End Recommendation (Text + Filters + Rank): Avg {avg_pipeline:.2f} ms | p95 {p95_pipeline:.2f} ms")
    results["latency_ms"]["end_to_end_pipeline_avg"] = round(avg_pipeline, 2)
    results["latency_ms"]["end_to_end_pipeline_p95"] = round(p95_pipeline, 2)

    # 6. Throughput / QPS Simulation
    print("\n--- 6. Search Throughput (Queries Per Second) ---")
    n_queries = 200
    t_start = time.perf_counter()
    for i in range(n_queries):
        _ = np.dot(mat_text, q_vec)
    total_batch_time = time.perf_counter() - t_start
    qps = round(n_queries / total_batch_time, 2)
    print(f"  - 10,000-Item Vector Search Throughput: {qps:,} QPS ({total_batch_time*1000:.2f} ms for {n_queries} queries)")
    results["throughput_qps"]["vector_search_qps"] = qps

    # Save benchmark report to docs/
    report_path = PROJECT_ROOT / "docs" / "performance_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Benchmark report saved to: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_benchmarks()
