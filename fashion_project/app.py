"""
app.py
Streamlit Web Application Entry Point
LLM-Assisted Personalized Fashion Recommendation System

Phase 1: Architecture Migration & ML Foundation
"""

import sys
from pathlib import Path

# Add fashion_project to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Fashion AI | Personalized Recommendation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from services.config_service import ConfigService
from services.catalog_service import CatalogService
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.ranking_service import RankingService
from services.image_service import ProductImageService
from services.persistence_service import PersistenceService
from services.preference_service import PreferenceExtractionService
from services.explanation_service import ExplanationService
from services.clip_service import CLIPService

from ui.styles import CUSTOM_CSS
from ui.sidebar import render_sidebar
from ui.views.dashboard import render_dashboard_view
from ui.views.discovery import render_discovery_view
from ui.views.assistant import render_assistant_view
from ui.views.collection import render_collection_view
from ui.views.model_insights import render_model_insights_view


@st.cache_resource(show_spinner="Initializing Fashion ML Services...")
def init_services():
    """
    Centralized service initialization with caching.
    Ensures precomputed embeddings and catalogs are loaded once into memory.
    """
    cfg = ConfigService.get_instance()
    cat_svc = CatalogService.get_instance()
    emb_svc = EmbeddingService.get_instance()
    ret_svc = RetrievalService.get_instance()
    pers_svc = PersistenceService.get_instance()
    rank_svc = RankingService.get_instance()
    img_svc = ProductImageService.get_instance()
    pref_svc = PreferenceExtractionService.get_instance()
    expl_svc = ExplanationService.get_instance()
    clip_svc = CLIPService.get_instance()

    return {
        "config": cfg,
        "catalog": cat_svc,
        "embeddings": emb_svc,
        "retrieval": ret_svc,
        "persistence": pers_svc,
        "ranking": rank_svc,
        "images": img_svc,
        "preference": pref_svc,
        "explanation": expl_svc,
        "clip": clip_svc
    }


def main():
    # 1. Apply global custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 2. Centralized service initialization
    services = init_services()

    # 3. Sidebar Navigation
    selected_view = render_sidebar()

    # 4. View Routing
    if selected_view == "Dashboard":
        render_dashboard_view()
    elif selected_view == "Fashion Discovery":
        render_discovery_view()
    elif selected_view == "AI Assistant":
        render_assistant_view()
    elif selected_view == "My Collection":
        render_collection_view()
    elif selected_view == "Model Insights":
        render_model_insights_view()
    else:
        render_dashboard_view()


if __name__ == "__main__":
    main()
