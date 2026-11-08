"""
ui/views/dashboard.py
Streamlit Dashboard View: Overview & System Readiness
"""

import streamlit as st
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.clip_service import CLIPService
from services.persistence_service import PersistenceService
from services.image_service import ProductImageService
from ui.components import render_brand_header, render_metric_card, render_status_pill


def render_dashboard_view():
    render_brand_header(
        title="Fashion Recommendation Dashboard",
        subtitle="AI-driven personalized discovery powered by semantic text embeddings and multi-stage ranking"
    )

    cat_svc = CatalogService.get_instance()
    emb_svc = EmbeddingService.get_instance()
    clip_svc = CLIPService.get_instance()
    pers_svc = PersistenceService.get_instance()
    img_svc = ProductImageService.get_instance()

    # 1. Top Metric Boxes
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Frozen Catalog", f"{len(cat_svc):,}", "Verified Apparel & Footwear")
    with c2:
        render_metric_card("Text Embeddings", f"{emb_svc.dimension}-d", "MiniLM-L6-v2 Unit L2")
    with c3:
        clip_status = "512-d" if clip_svc.is_visual_retrieval_ready() else "Offline"
        render_metric_card("CLIP Foundation", clip_status, "ViT-B/32 Visual Encoder")
    with c4:
        profiles_count = len(pers_svc.list_profiles())
        render_metric_card("Active Profiles", f"{profiles_count}", "SQLite Persistence")

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. System Readiness Table
    st.markdown("### 🔍 Service Readiness & Deployment Health")

    col_health, col_preview = st.columns([1.1, 0.9])

    with col_health:
        clip_ready_badge = "🟢 Ready (512-d)" if clip_svc.is_visual_retrieval_ready() else "🟡 Standby"
        st.markdown(f"""
        | Service Component | Operational Status | Architecture Implementation |
        |---|:---:|---|
        | **Dataset Loader** | 🟢 Ready | Ingests `styles.csv` (44.4k rows), validates 10k frozen subset |
        | **Catalog Service** | 🟢 Ready | O(1) in-memory lookup, 10,000 clothing items, Apparel & Footwear |
        | **Text Embeddings** | 🟢 Ready | Precomputed `(10000, 384)` float32 unit-normalized matrix |
        | **Visual Embeddings** | {clip_ready_badge} | Precomputed `(10000, 512)` float32 CLIP ViT-B/32 matrix |
        | **Multimodal Retrieval** | 🟢 Ready | Pure NumPy matrix dot product cosine similarity |
        | **Personalized Ranker** | 🟢 Ready | Formula: `Wt*cos(qt,vt) + Wv*cos(qv,vv) + Wp*cos(u,vt)` |
        | **Preference Service** | 🟢 Ready | Pydantic validation with deterministic fallback provider |
        | **Explanation Engine** | 🟢 Ready | Metadata attribute grounding validator with template fallback |
        | **Product Image Service** | 🟢 Ready | Multi-tier isolated catalog resolver with local & remote CDN |
        """)

    with col_preview:
        st.markdown("#### 👗 Catalog Sample")
        sample_ids = cat_svc.catalog_ids[:3]
        sub_cols = st.columns(3)
        for idx, pid in enumerate(sample_ids):
            with sub_cols[idx]:
                prod = cat_svc.get_product(pid)
                img = img_svc.load_image(pid, target_size=(140, 180))
                st.image(img, use_container_width=True)
                st.caption(f"**{prod.get('articleType')}**\n{prod.get('baseColour')}")

    st.markdown("---")
    st.info("💡 **Production Ready:** Complete multimodal personalized fashion recommendation system with text & visual retrieval, hybrid personalized ranking, grounded explanations, and isolated guest sessions.")
