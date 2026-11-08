"""
ui/views/discovery.py
Fashion Discovery View (Phase 1 Operational Shell)
"""

import streamlit as st
from services.catalog_service import CatalogService
from services.ranking_service import RankingService
from services.explanation_service import ExplanationService
from services.image_service import ProductImageService
from ui.components import render_brand_header, render_product_card_html, render_placeholder_view


def render_discovery_view():
    render_brand_header(
        title="Fashion Discovery",
        subtitle="Semantic natural language search and personalized fashion catalog browsing"
    )

    cat_svc = CatalogService.get_instance()
    rank_svc = RankingService.get_instance()
    expl_svc = ExplanationService.get_instance()
    img_svc = ProductImageService.get_instance()

    profile_id = st.session_state.get("active_profile_id")

    # 1. Search Bar & Filters
    with st.container():
        query_text = st.text_input(
            "Natural Language Search Query",
            placeholder="e.g. casual blue cotton shirt or summer footwear...",
            key="discovery_query_input"
        )

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            categories = ["All Categories"] + cat_svc.get_categories()
            sel_cat = st.selectbox("Category", categories, index=0)
        with f_col2:
            colors = ["All Colors"] + cat_svc.get_colors()
            sel_color = st.selectbox("Color", colors, index=0)
        with f_col3:
            usages = ["All Usages"] + cat_svc.get_usages()
            sel_usage = st.selectbox("Usage", usages, index=0)

        search_btn = st.button("🔍 Search Catalog", type="primary", use_container_width=True)

    st.markdown("---")

    # 2. Results or Initial Guidance
    if search_btn or query_text:
        # Build hard constraints
        constraints = {}
        if sel_cat != "All Categories":
            constraints["masterCategory"] = [sel_cat]
        if sel_color != "All Colors":
            constraints["baseColour"] = [sel_color]
        if sel_usage != "All Usages":
            constraints["usage"] = [sel_usage]

        with st.spinner("Scoring catalog via pure NumPy semantic similarity..."):
            res = rank_svc.recommend(
                query=query_text or "casual fashion",
                profile_id=profile_id,
                constraints=constraints,
                top_k=5
            )

        if res["status"] == "no_match":
            st.warning(f"⚠️ {res['message']}")
            st.info(f"💡 Suggestion: {res.get('revised_constraints_suggestion')}")
        else:
            products = res["results"]
            st.success(f"Found {res['total_eligible']} matching products ({len(products)} displayed).")

            # Generate explanations
            expl_map = expl_svc.explain_batch(products, active_constraints=constraints)

            # Render 5-column product card grid
            cols = st.columns(min(len(products), 5))
            for i, p in enumerate(products):
                with cols[i]:
                    pid = str(p.get("id"))
                    b64 = img_svc.get_image_base64(pid, target_size=(180, 220))
                    card_html = render_product_card_html(
                        product=p,
                        image_b64=b64,
                        score=p.get("score"),
                        explanation=expl_map.get(pid)
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
    else:
        render_placeholder_view(
            title="Semantic Fashion Discovery",
            description="Enter a natural-language search query above to retrieve catalog items using MiniLM vector similarity and personalized taste weighting.",
            icon="🔎"
        )
