"""
ui/components.py
Reusable Streamlit UI Components & HTML Renderers
"""

from typing import Dict, Any, Optional
import streamlit as st


def render_brand_header(title: str, subtitle: str):
    """Render top application branding banner."""
    html = f"""
    <div class="fashion-brand-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(title: str, value: Any, subtext: str = ""):
    """Render an individual dashboard metric card."""
    html = f"""
    <div class="metric-box">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_pill(label: str, status_type: str = "ready"):
    """
    Render a status badge.
    status_type: 'ready', 'notice', 'offline'
    """
    cls_map = {
        "ready": "status-pill-ready",
        "notice": "status-pill-notice",
        "offline": "status-pill-offline"
    }
    css_class = cls_map.get(status_type, "status-pill-notice")
    return f'<span class="status-pill {css_class}">{label}</span>'


def render_product_card_html(
    product: Dict[str, Any],
    image_b64: str,
    score: Optional[float] = None,
    explanation: Optional[str] = None
) -> str:
    """Generate HTML for a styled product card."""
    pid = product.get("id", "Unknown")
    name = product.get("productDisplayName", "Fashion Item")
    color = product.get("baseColour", "Neutral")
    usage = product.get("usage", "Casual")
    article = product.get("articleType", "Apparel")

    score_badge = ""
    if score is not None:
        score_badge = f'<span style="float: right; font-weight: 600; color: #0284c7;">{score:.3f}</span>'

    expl_html = ""
    if explanation:
        expl_html = f'<div class="product-explanation">💡 {explanation}</div>'

    html = f"""
    <div class="product-card">
        <div class="product-img-container">
            <img src="{image_b64}" alt="{name}" />
        </div>
        <div class="product-title" title="{name}">{name}</div>
        <div class="product-meta">
            <span>{color} · {article} · {usage}</span>
            {score_badge}
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8;">ID: #{pid}</div>
        {expl_html}
    </div>
    """
    return html


def render_placeholder_view(title: str, description: str, icon: str = "🎨"):
    """Render an informative Phase 1 placeholder view."""
    html = f"""
    <div class="placeholder-view">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
        <span class="status-pill status-pill-notice">Architecture Migrated · Full Journey in Phase 2</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
