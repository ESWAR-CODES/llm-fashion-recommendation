"""
ui/sidebar.py
Sidebar Navigation & Profile Management Component
"""

import streamlit as st
from services.persistence_service import PersistenceService
from services.config_service import ConfigService


def render_sidebar() -> str:
    """
    Render sidebar navigation and return selected view name.
    """
    with st.sidebar:
        st.markdown("### 🛍️ Fashion AI")
        st.caption("Personalized Semantic Recommender")
        st.markdown("---")

        # 1. Navigation
        views = [
            "Dashboard",
            "Fashion Discovery",
            "AI Assistant",
            "My Collection",
            "Model Insights"
        ]
        
        current_idx = 0
        if "navigation" in st.session_state and st.session_state.navigation in views:
            current_idx = views.index(st.session_state.navigation)

        selected_view = st.radio(
            "Navigation",
            views,
            index=current_idx,
            label_visibility="collapsed"
        )
        st.session_state.navigation = selected_view

        st.markdown("---")

        # 2. Profile Management
        st.markdown("#### 👤 User Profile")
        ps = PersistenceService.get_instance()
        profiles = ps.list_profiles()

        # Isolate anonymous sessions: assign a fresh dedicated guest profile if none set
        if "active_profile_id" not in st.session_state:
            guest_pid = ps.create_profile()
            st.session_state.active_profile_id = guest_pid
            profiles = ps.list_profiles()

        if not profiles:
            default_id = ps.create_profile()
            profiles = ps.list_profiles()
            st.session_state.active_profile_id = default_id

        profile_ids = [p["profile_id"] for p in profiles]

        if st.session_state.active_profile_id not in profile_ids:
            st.session_state.active_profile_id = profile_ids[0]

        curr_prof_idx = profile_ids.index(st.session_state.active_profile_id)
        chosen_pid = st.selectbox(
            "Active Profile",
            profile_ids,
            index=curr_prof_idx,
            key="profile_selector"
        )
        st.session_state.active_profile_id = chosen_pid

        col_new, col_reset = st.columns(2)
        with col_new:
            if st.button("➕ New", use_container_width=True, help="Create a fresh isolated guest session"):
                new_pid = ps.create_profile()
                st.session_state.active_profile_id = new_pid
                st.rerun()

        with col_reset:
            if st.button("🧹 Clear", use_container_width=True, help="Reset preferences & constraints for current session"):
                ps.clear_active_constraints(st.session_state.active_profile_id)
                st.success("Session reset!")

        st.markdown("---")

        # 3. System Readiness Badge
        st.markdown("#### ⚙️ System Status")
        hw = ConfigService.get_instance().get_hardware_info()
        st.caption(f"**Runtime:** {hw['platform']} ({hw['device_name']})")
        st.caption(f"**Memory Avail:** {hw['system_ram_available_gb']} GB")
        torch_status = hw.get("torch_version", "Not Installed")
        if torch_status == "Not Installed":
            st.caption("**ML Runtime:** Pure CPU (Zero PyTorch)")
        else:
            st.caption(f"**PyTorch:** {torch_status}")
        st.caption("**Catalog:** 10,000 Items (Frozen)")
        st.caption("**Text Retrieval:** all-MiniLM-L6-v2")
        st.caption("**Visual Retrieval:** CLIP ViT-B/32")
        st.caption("**Extraction Engine:** Hybrid Grounded / LLM")

        st.markdown("---")
        st.caption("Phase 3: Production & Submission Ready")

    return selected_view
