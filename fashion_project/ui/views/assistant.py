"""
ui/views/assistant.py
Conversational AI Fashion Assistant (Phase 1 Placeholder Shell)
"""

import streamlit as st
from services.preference_service import PreferenceExtractionService
from services.persistence_service import PersistenceService
from ui.components import render_brand_header, render_placeholder_view


def render_assistant_view():
    render_brand_header(
        title="AI Fashion Assistant",
        subtitle="Conversational preference extraction and multi-turn constraint refinement"
    )

    pers_svc = PersistenceService.get_instance()
    pref_svc = PreferenceExtractionService.get_instance()
    profile_id = st.session_state.get("active_profile_id")

    current_cons = pers_svc.get_active_constraints(profile_id) if profile_id else {}

    col_chat, col_state = st.columns([1.2, 0.8])

    with col_chat:
        st.markdown("#### 💬 Conversational Refinement")
        user_msg = st.text_input(
            "Talk to the assistant",
            placeholder="e.g. 'Show me black instead but avoid red' or 'Reset search'...",
            key="assistant_chat_input"
        )
        send_btn = st.button("Send Message", use_container_width=True)

        if send_btn and user_msg:
            new_state, meta = pref_svc.extract_and_apply(user_msg, current_cons)
            if profile_id:
                pers_svc.set_active_constraints(profile_id, new_state)
            st.success(f"Constraints updated via provider: `{meta['provider_name']}`")
            if meta.get("unsupported"):
                st.warning(f"⚠️ Unsupported attributes noted: {', '.join(meta['unsupported'])}")
            st.rerun()

        render_placeholder_view(
            title="Interactive Conversational Journey",
            description="The conversational preference extraction service is fully connected to the Pydantic validator. Multi-turn dialogue history, interactive feedback bubbles, and speech controls will be expanded in Phase 2.",
            icon="💬"
        )

    with col_state:
        st.markdown("#### 📋 Active Constraints Inspector")
        st.caption(f"Profile ID: `{profile_id}`")
        if current_cons:
            st.json(current_cons)
        else:
            st.info("No active constraints applied for this profile.")
