"""
ui/views/collection.py
My Collection & User Preferences View (Personalization Integration)
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Liked products gallery with unlike / remove feedback capability.
- Dismissed products management with undo capability.
- Explicit taste preferences editor (preferred colors, categories, styles).
- Profile isolation and persistence verification.
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
        st.warning("⚠️ No active profile selected. Please select or create an anonymous profile in the sidebar.")
        return

    profile = pers_svc.get_profile(profile_id)
    if not profile:
        st.error(f"Profile `{profile_id}` not found in database.")
        return

    feedback = pers_svc.get_feedback(profile_id)
    likes = feedback.get("likes", [])
    dismissals = feedback.get("dismissals", [])
    prefs = profile.get("preferences", {})

    st.markdown(f"#### 👤 Profile: `{profile_id}`")
    st.caption(f"Created: {profile.get('created_at')} · Last Updated: {profile.get('updated_at')}")

    tab_likes, tab_prefs, tab_dismiss = st.tabs(["❤️ Liked Products", "⚙️ Taste Preferences Editor", "🚫 Dismissals & Exclusions"])

    with tab_likes:
        if likes:
            st.write(f"You have liked **{len(likes)}** items. These products shape your personalized user taste vector $\\mathbf{{u}}$.")
            cols = st.columns(min(len(likes), 4))
            for idx, pid in enumerate(likes):
                prod = cat_svc.get_product(pid)
                col_idx = idx % min(len(likes), 4)
                with cols[col_idx]:
                    b64 = img_svc.get_image_base64(pid, target_size=(160, 200))
                    title = prod.get('productDisplayName', pid) if prod else pid
                    st.markdown(
                        f"<div style='border:1px solid #333; border-radius:8px; padding:8px; text-align:center; background:#1a1a1a; margin-bottom:12px;'>"
                        f"<img src='{b64}' style='width:100%; border-radius:4px; max-height:160px; object-fit:contain;'/>"
                        f"<div style='font-size:0.8rem; font-weight:600; margin-top:6px; height:32px; overflow:hidden;'>{title}</div>"
                        f"<div style='font-size:0.7rem; color:#888;'>#{pid} · {prod.get('baseColour', '')} {prod.get('articleType', '')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if st.button("💔 Unlike", key=f"coll_unlike_{pid}", use_container_width=True):
                        pers_svc.remove_feedback(profile_id, pid)
                        st.rerun()
        else:
            render_placeholder_view(
                title="No Liked Products Yet",
                description="Browse products in Fashion Discovery or AI Assistant and click 'Like' to build your personalized user taste vector.",
                icon="❤️"
            )

    with tab_prefs:
        st.markdown("##### 🎨 Explicit Profile Taste Preferences")
        st.caption("These explicit attributes influence personalized candidate weighting alongside your liked products.")

        all_colors = cat_svc.get_colors()
        all_categories = cat_svc.get_categories()
        all_genders = cat_svc.get_genders()

        current_pref_colors = prefs.get("preferred_colors", [])
        current_pref_cats = prefs.get("preferred_categories", [])
        current_pref_gender = prefs.get("preferred_gender", "All")

        with st.form("edit_preferences_form"):
            new_pref_colors = st.multiselect(
                "Preferred Colors",
                options=all_colors,
                default=[c for c in current_pref_colors if c in all_colors]
            )
            new_pref_cats = st.multiselect(
                "Preferred Categories",
                options=all_categories,
                default=[c for c in current_pref_cats if c in all_categories]
            )
            new_pref_gender = st.selectbox(
                "Preferred Gender Focus",
                options=["All"] + all_genders,
                index=(all_genders.index(current_pref_gender) + 1) if current_pref_gender in all_genders else 0
            )

            save_btn = st.form_submit_button("💾 Save Profile Preferences", type="primary")

            if save_btn:
                updated_prefs = {
                    "preferred_colors": new_pref_colors,
                    "preferred_categories": new_pref_cats,
                    "preferred_gender": new_pref_gender if new_pref_gender != "All" else None
                }
                pers_svc.update_profile(profile_id, updated_prefs)
                st.success("✅ Taste preferences updated successfully!")
                st.rerun()

        st.markdown("##### 📄 Raw Stored Profile JSON")
        st.json(profile)

    with tab_dismiss:
        if dismissals:
            st.write(f"You have dismissed **{len(dismissals)}** items. These products are strictly excluded from all recommendation rankings.")
            for pid in dismissals:
                prod = cat_svc.get_product(pid)
                title = prod.get('productDisplayName', pid) if prod else pid
                d_col1, d_col2 = st.columns([0.8, 0.2])
                with d_col1:
                    st.write(f"🚫 **#{pid}**: {title}")
                with d_col2:
                    if st.button("↩️ Undo", key=f"undo_dism_{pid}", use_container_width=True):
                        pers_svc.remove_feedback(profile_id, pid)
                        st.rerun()
        else:
            st.caption("No products currently dismissed for this profile.")
