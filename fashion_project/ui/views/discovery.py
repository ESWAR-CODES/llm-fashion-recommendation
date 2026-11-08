"""
ui/views/discovery.py
Fashion Discovery View (Multimodal Search, Advanced Filters & Personalization)
LLM-Assisted Personalized Fashion Recommendation System

Capabilities:
- Mode A (Text Search): Natural language search with MiniLM.
- Mode B (Image Search): Clothing photo upload with CLIP ViT-B/32.
- Mode C (Combined Multimodal): Joint text + visual candidate scoring.
- Mode D (Personalized Retrieval): User taste weighting ($0.8q + 0.2u$ or hybrid $0.5q + 0.3v + 0.2u$).
- Advanced Filters: Gender, MasterCategory, SubCategory, ArticleType, BaseColour, Usage, Season, ExcludeColors.
- Interactive Feedback: Idempotent likes, dismissals, and undo actions.
- Real-time Visual Search readiness indicator.
"""

import streamlit as st
from PIL import Image

from services.catalog_service import CatalogService
from services.ranking_service import RankingService
from services.explanation_service import ExplanationService
from services.image_service import ProductImageService
from services.clip_service import CLIPService
from services.persistence_service import PersistenceService
from ui.components import render_brand_header, render_product_card_html, render_placeholder_view


def render_discovery_view():
    render_brand_header(
        title="Fashion Discovery",
        subtitle="Multimodal semantic retrieval, visual clothing similarity, and personalized recommendation"
    )

    cat_svc = CatalogService.get_instance()
    rank_svc = RankingService.get_instance()
    expl_svc = ExplanationService.get_instance()
    img_svc = ProductImageService.get_instance()
    clip_svc = CLIPService.get_instance()
    pers_svc = PersistenceService.get_instance()

    profile_id = st.session_state.get("active_profile_id")
    clip_status = clip_svc.get_status()

    # 1. Operational Readiness Banner
    status_col1, status_col2 = st.columns([1.5, 1])
    with status_col1:
        if clip_status["ready"] and clip_status.get("encoder_available", False):
            st.success("🟢 **Visual Search Operational**: CLIP ViT-B/32 encoder & precomputed index active.")
        elif clip_status["ready"]:
            st.success("🟢 **Catalog & Text Operational**: Precomputed embeddings active (Lightweight CPU mode).")
        else:
            st.info("🟡 **Text-Only Operational**: CLIP visual embeddings offline (Text search fully functional).")
    with status_col2:
        if profile_id:
            fb = pers_svc.get_feedback(profile_id)
            likes_cnt = len(fb.get("likes", []))
            dism_cnt = len(fb.get("dismissals", []))
            st.caption(f"Active Profile: `{profile_id}` | ❤️ {likes_cnt} Likes | 🚫 {dism_cnt} Dismissed")

    # 2. Multimodal Search Inputs (Text + Image Upload)
    with st.expander("🔍 **Search Input & Clothing Image Upload**", expanded=True):
        input_col1, input_col2 = st.columns([1.2, 0.8])

        with input_col1:
            query_text = st.text_input(
                "Natural Language Fashion Query",
                placeholder="e.g. 'navy blue casual shirt' or 'black running shoes'...",
                key="discovery_text_query"
            )

        with input_col2:
            uploaded_file = st.file_uploader(
                "Upload Clothing Image (Optional)",
                type=["jpg", "jpeg", "png", "webp"],
                key="discovery_image_upload",
                help="Upload a photo of clothing to search for visually similar items in the catalog."
            )

        # Validate uploaded image if provided
        query_image = None
        if uploaded_file is not None:
            is_valid, decoded_img, err_msg = clip_svc.validate_uploaded_image(uploaded_file)
            if is_valid and decoded_img:
                p_col1, p_col2 = st.columns([0.3, 0.7])
                with p_col1:
                    st.image(decoded_img, caption="Query Preview", width=110)
                with p_col2:
                    if not clip_svc.is_encoder_available():
                        st.info("ℹ️ Live image query extraction requires PyTorch. In lightweight CPU mode, search utilizes text retrieval and catalog filters.")
                        query_image = None
                    elif not clip_status["ready"]:
                        st.warning("⚠️ CLIP product index is not yet loaded on disk. Search will gracefully fall back to text modality.")
                        query_image = None
                    else:
                        st.caption("✅ Image verified and preprocessed for CLIP inference.")
                        query_image = decoded_img
            else:
                st.error(f"❌ Invalid image: {err_msg}")

    # 3. Advanced Catalog Facet Filters
    with st.expander("🎛️ **Advanced Catalog Filters**", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            all_genders = cat_svc.get_genders()
            sel_gender = st.multiselect("Gender", all_genders, key="fil_gender")
            all_masters = cat_svc.get_categories()
            sel_master = st.selectbox("Master Category", ["All"] + all_masters, index=0, key="fil_master")
        with c2:
            sub_master = None if sel_master == "All" else sel_master
            all_subcats = cat_svc.get_subcategories(sub_master)
            sel_subcat = st.multiselect("SubCategory", all_subcats, key="fil_subcat")
            all_articles = cat_svc.get_article_types(sub_master)
            sel_article = st.multiselect("Article Type", all_articles, key="fil_article")
        with c3:
            all_colors = cat_svc.get_colors()
            sel_colors = st.multiselect("Base Color", all_colors, key="fil_colors")
            all_usages = cat_svc.get_usages()
            sel_usages = st.multiselect("Usage", all_usages, key="fil_usages")

        c4, c5 = st.columns([1, 1])
        with c4:
            all_seasons = cat_svc.get_seasons()
            sel_seasons = st.multiselect("Season", all_seasons, key="fil_seasons")
        with c5:
            sel_exclude = st.multiselect("🚫 Exclude Colors", all_colors, key="fil_exclude")

        # Active filter chips & Reset button
        chip_col, reset_col = st.columns([0.8, 0.2])
        with reset_col:
            if st.button("🔄 Reset Filters", use_container_width=True):
                for k in ["fil_gender", "fil_master", "fil_subcat", "fil_article", "fil_colors", "fil_usages", "fil_seasons", "fil_exclude"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Search trigger button
    search_btn = st.button("🚀 Search Recommendations", type="primary", use_container_width=True)

    st.markdown("---")

    # 4. Search Execution
    has_input = bool((query_text and query_text.strip()) or query_image is not None or search_btn)

    if has_input:
        # Build strict constraints dictionary
        constraints = {}
        if sel_gender:
            constraints["gender"] = sel_gender
        if sel_master and sel_master != "All":
            constraints["masterCategory"] = [sel_master]
        if sel_subcat:
            constraints["subCategory"] = sel_subcat
        if sel_article:
            constraints["articleType"] = sel_article
        if sel_colors:
            constraints["baseColour"] = sel_colors
        if sel_usages:
            constraints["usage"] = sel_usages
        if sel_seasons:
            constraints["season"] = sel_seasons
        if sel_exclude:
            constraints["exclude_colors"] = sel_exclude

        with st.spinner("Executing multimodal candidate retrieval & hybrid scoring..."):
            res = rank_svc.recommend_multimodal(
                query_text=query_text or (None if query_image else "casual"),
                query_image=query_image,
                profile_id=profile_id,
                constraints=constraints,
                top_k=5
            )

        if res["status"] == "no_match":
            st.warning(f"⚠️ {res['message']}")
            st.info(f"💡 Suggestion: {res.get('revised_constraints_suggestion')}")
        else:
            products = res["results"]
            mode_badge = res.get("retrieval_mode", "multimodal").upper().replace("_", " ")
            weights = res.get("weights_applied", {})

            w_summary = ", ".join([f"{k.split('_')[0].capitalize()}: {v:.2f}" for k, v in weights.items() if v > 0])
            st.success(
                f"**Retrieval Mode:** `{mode_badge}` | Found {res['total_eligible']} matching products "
                f"({len(products)} displayed) | Weights: `{w_summary}`"
            )

            if res.get("visual_fallback_triggered"):
                st.info("ℹ️ Visual search fell back to text semantic scoring because precomputed CLIP embeddings are not yet generated.")

            # Generate explanations
            expl_map = expl_svc.explain_batch(products, active_constraints=constraints)

            # Get user's current feedback to render active like/dismiss states
            current_feedback = pers_svc.get_feedback(profile_id) if profile_id else {"likes": [], "dismissals": []}
            liked_set = set(current_feedback.get("likes", []))
            dismissed_set = set(current_feedback.get("dismissals", []))

            # Render 5-column product grid
            cols = st.columns(min(len(products), 5))
            for i, p in enumerate(products):
                with cols[i]:
                    pid = str(p.get("id"))
                    b64 = img_svc.get_image_base64(pid, target_size=(180, 220))

                    # Render card HTML
                    card_html = render_product_card_html(
                        product=p,
                        image_b64=b64,
                        score=p.get("score"),
                        explanation=expl_map.get(pid)
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Multimodal relevance signals display
                    st.markdown(
                        f"<div style='font-size:0.75rem; color:#888; margin-top:4px; text-align:center;'>"
                        f"Txt: <b>{p.get('text_similarity', 0.0):.2f}</b> | "
                        f"Vis: <b>{p.get('visual_similarity', 0.0):.2f}</b> | "
                        f"Pref: <b>{p.get('preference_similarity', 0.0):.2f}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Interactive Feedback Buttons (Like, Dismiss, Undo)
                    if profile_id:
                        fb_col1, fb_col2 = st.columns(2)
                        with fb_col1:
                            if pid in liked_set:
                                if st.button("❤️ Liked", key=f"unlike_{pid}", use_container_width=True):
                                    pers_svc.remove_feedback(profile_id, pid)
                                    st.rerun()
                            else:
                                if st.button("👍 Like", key=f"like_{pid}", use_container_width=True):
                                    pers_svc.save_feedback(profile_id, pid, "like")
                                    st.rerun()

                        with fb_col2:
                            if pid in dismissed_set:
                                if st.button("↩️ Undo", key=f"undismiss_{pid}", use_container_width=True):
                                    pers_svc.remove_feedback(profile_id, pid)
                                    st.rerun()
                            else:
                                if st.button("👎 Pass", key=f"dism_{pid}", use_container_width=True):
                                    pers_svc.save_feedback(profile_id, pid, "dismiss")
                                    st.rerun()

    else:
        render_placeholder_view(
            title="Multimodal Fashion Discovery",
            description="Enter a natural-language fashion request or upload a clothing photograph above to search the 10,000-item catalog using hybrid vector retrieval and personalized preference weighting.",
            icon="👗"
        )
