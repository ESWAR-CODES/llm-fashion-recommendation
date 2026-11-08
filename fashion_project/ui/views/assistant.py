"""
ui/views/assistant.py
Conversational AI Fashion Assistant (Multi-turn Constraint Refinement & Recommendations)
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Multi-turn conversational constraint refinement.
- Grounded state operations: set, add, remove, clear, reset.
- Unsupported attribute detection (sizing, pricing, fabric, stock).
- Transparent provider reporting (local Qwen vs deterministic rule fallback).
- Integrated product recommendation stream with grounded explanations.
- Interactive liking and dismissals within the conversation flow.
"""

import streamlit as st
from services.preference_service import PreferenceExtractionService
from services.persistence_service import PersistenceService
from services.ranking_service import RankingService
from services.explanation_service import ExplanationService
from services.image_service import ProductImageService
from ui.components import render_brand_header, render_product_card_html


def render_assistant_view():
    render_brand_header(
        title="AI Fashion Assistant",
        subtitle="Conversational fashion discovery, multi-turn preference extraction, and constraint refinement"
    )

    pers_svc = PersistenceService.get_instance()
    pref_svc = PreferenceExtractionService.get_instance()
    rank_svc = RankingService.get_instance()
    expl_svc = ExplanationService.get_instance()
    img_svc = ProductImageService.get_instance()

    profile_id = st.session_state.get("active_profile_id")
    current_cons = pers_svc.get_active_constraints(profile_id) if profile_id else {}

    # Initialize dialogue history in session state
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "text": "Hello! I am your AI Fashion Assistant. What kind of apparel or footwear are you looking for today? (e.g. 'Show me casual blue shirts' or 'Find sports shoes, but avoid red')",
                "provider": "system"
            }
        ]

    col_chat, col_state = st.columns([1.3, 0.7])

    with col_chat:
        st.markdown("#### 💬 Conversational Dialogue")

        # Render message stream
        for idx, msg in enumerate(st.session_state["assistant_messages"]):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["text"])
            else:
                with st.chat_message("assistant", avatar="✨"):
                    st.write(msg["text"])
                    if "unsupported" in msg and msg["unsupported"]:
                        st.caption(f"⚠️ Note: Catalog constraints do not support: *{', '.join(msg['unsupported'])}*")
                    if "provider" in msg and msg["provider"] != "system":
                        st.caption(f"Provider: `{msg['provider']}`")

        # Chat Input
        user_msg = st.chat_input("Type your fashion preference or refinement here...")

        if user_msg:
            # Append user message
            st.session_state["assistant_messages"].append({"role": "user", "text": user_msg})

            # Extract constraints
            new_state, meta = pref_svc.extract_and_apply(user_msg, current_cons)

            if profile_id:
                pers_svc.set_active_constraints(profile_id, new_state)

            # Build assistant response text
            reply_text = f"Updated search filters."
            if meta.get("reset_triggered"):
                reply_text = "Reset all active search constraints."
            elif meta.get("applied_operations"):
                ops_summary = []
                for op in meta["applied_operations"]:
                    ops_summary.append(f"{op.get('op', 'set')} {op.get('field', 'filter')} to {op.get('values', [])}")
                reply_text = f"Applied: {'; '.join(ops_summary)}."

            st.session_state["assistant_messages"].append({
                "role": "assistant",
                "text": reply_text,
                "provider": meta.get("provider_name", "rule_fallback"),
                "unsupported": meta.get("unsupported", [])
            })

            st.rerun()

        # Display recommendations based on active conversational state
        st.markdown("---")
        st.markdown("#### 🛍️ Refined Catalog Recommendations")

        with st.spinner("Scoring catalog against active conversational constraints..."):
            res = rank_svc.recommend_multimodal(
                query_text=user_msg if user_msg else (st.session_state["assistant_messages"][-1]["text"] if len(st.session_state["assistant_messages"]) > 1 else "casual"),
                profile_id=profile_id,
                constraints=current_cons,
                top_k=5
            )

        if res["status"] == "no_match":
            st.warning(f"⚠️ {res['message']}")
            st.info(f"💡 Suggestion: {res.get('revised_constraints_suggestion')}")
        else:
            products = res["results"]
            expl_map = expl_svc.explain_batch(products, active_constraints=current_cons)
            current_fb = pers_svc.get_feedback(profile_id) if profile_id else {"likes": [], "dismissals": []}
            liked_set = set(current_fb.get("likes", []))
            dismissed_set = set(current_fb.get("dismissals", []))

            cols = st.columns(min(len(products), 5))
            for i, p in enumerate(products):
                with cols[i]:
                    pid = str(p.get("id"))
                    b64 = img_svc.get_image_base64(pid, target_size=(160, 200))
                    card_html = render_product_card_html(
                        product=p,
                        image_b64=b64,
                        score=p.get("score"),
                        explanation=expl_map.get(pid)
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    if profile_id:
                        f1, f2 = st.columns(2)
                        with f1:
                            if pid in liked_set:
                                if st.button("❤️", key=f"asst_unlike_{pid}_{i}", help="Unlike"):
                                    pers_svc.remove_feedback(profile_id, pid)
                                    st.rerun()
                            else:
                                if st.button("👍", key=f"asst_like_{pid}_{i}", help="Like"):
                                    pers_svc.save_feedback(profile_id, pid, "like")
                                    st.rerun()
                        with f2:
                            if pid in dismissed_set:
                                if st.button("↩️", key=f"asst_undism_{pid}_{i}", help="Undo dismissal"):
                                    pers_svc.remove_feedback(profile_id, pid)
                                    st.rerun()
                            else:
                                if st.button("👎", key=f"asst_dism_{pid}_{i}", help="Dismiss"):
                                    pers_svc.save_feedback(profile_id, pid, "dismiss")
                                    st.rerun()

    with col_state:
        st.markdown("#### 📋 Active Constraints Inspector")
        st.caption(f"Profile: `{profile_id}`")

        if current_cons:
            for k, v in current_cons.items():
                st.markdown(f"**{k}**: `{', '.join(v) if isinstance(v, list) else v}`")

            if st.button("🗑️ Reset All Constraints", use_container_width=True):
                if profile_id:
                    pers_svc.clear_active_constraints(profile_id)
                st.session_state["assistant_messages"].append({
                    "role": "assistant",
                    "text": "All active conversation constraints have been cleared.",
                    "provider": "system"
                })
                st.rerun()
        else:
            st.info("No active filters applied. All 10,000 catalog products are currently eligible.")

        st.markdown("---")
        st.markdown("#### 💡 Try Saying:")
        examples = [
            "Find casual blue shirts",
            "Only black footwear",
            "Avoid red and yellow",
            "Show formal shirts instead",
            "Any color is fine",
            "Clear all filters"
        ]
        for eg in examples:
            if st.button(f"👉 \"{eg}\"", key=f"eg_{eg}", use_container_width=True):
                st.session_state["assistant_messages"].append({"role": "user", "text": eg})
                new_state, meta = pref_svc.extract_and_apply(eg, current_cons)
                if profile_id:
                    pers_svc.set_active_constraints(profile_id, new_state)
                st.session_state["assistant_messages"].append({
                    "role": "assistant",
                    "text": f"Applied: {meta.get('applied_operations', eg)}",
                    "provider": meta.get("provider_name", "rule_fallback"),
                    "unsupported": meta.get("unsupported", [])
                })
                st.rerun()
