"""
ui/views/collection.py
My Collection & User Preferences View (Phase 1 Placeholder Shell)
"""

import streamlit as st
from services.persistence_service import PersistenceService
from services.catalog_service import CatalogService
from services.image_service import ProductImageService
from ui.components import render_brand_header, render_placeholder_view


def render_collection_view():
    render_brand_header(
        title="My Collection & Preferences",
        subtitle="Saved taste preferences, product likes, and personalized profile history"
    )

    pers_svc = PersistenceService.get_instance()
    cat_svc = CatalogService.get_instance()
    img_svc = ProductImageService.get_instance()

    profile_id = st.session_state.get("active_profile_id")
    if not profile_id:
        st.warning("No active profile selected.")
        return

    profile = pers_svc.get_profile(profile_id)
    feedback = pers_svc.get_feedback(profile_id)
    likes = feedback.get("likes", [])
    dismissals = feedback.get("dismissals", [])

    st.markdown(f"#### 👤 Profile: `{profile_id}`")
    st.caption(f"Created: {profile.get('created_at')} · Last Updated: {profile.get('updated_at')}")

    tab_likes, tab_prefs, tab_dismiss = st.tabs(["❤️ Liked Products", "⚙️ Taste Preferences", "🚫 Dismissals"])

    with tab_likes:
        if likes:
            st.write(f"You have liked **{len(likes)}** items.")
            cols = st.columns(min(len(likes), 4))
            for idx, pid in enumerate(likes[:8]):
                prod = cat_svc.get_product(pid)
                with cols[idx % 4]:
                    img = img_svc.load_image(pid, target_size=(140, 180))
                    st.image(img, use_container_width=True)
                    st.caption(f"**{prod.get('productDisplayName', pid)}**")
        else:
            render_placeholder_view(
                title="No Liked Products Yet",
                description="Browse products in Fashion Discovery and click 'Like' to build your personalized user taste vector.",
                icon="❤️"
            )

    with tab_prefs:
        st.markdown("##### Explicit Category & Color Preferences")
        prefs = profile.get("preferences", {})
        st.json(prefs)

    with tab_dismiss:
        if dismissals:
            st.write(f"Dismissed items count: {len(dismissals)}")
            st.write(", ".join([f"#{d}" for d in dismissals]))
        else:
            st.caption("No products dismissed for this profile.")
