"""
ui/styles.py
Custom Styling and Branding CSS for Streamlit Fashion Application
"""

CUSTOM_CSS = """
<style>
/* Global App Typography & Theme */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Header Styling */
.fashion-brand-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 1.75rem 2rem;
    border-radius: 12px;
    color: #f8fafc;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #38bdf8;
}

.fashion-brand-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.025em;
}

.fashion-brand-header p {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0.35rem 0 0 0;
}

/* Status Cards & Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
}

.status-pill-ready {
    background-color: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.status-pill-notice {
    background-color: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
}

.status-pill-offline {
    background-color: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}

/* Metric Summary Card */
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Product Card Grid */
.product-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
    transition: box-shadow 0.2s ease;
}

.product-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.product-img-container {
    width: 100%;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f8fafc;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.75rem;
}

.product-img-container img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
}

.product-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.product-meta {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 0.5rem;
}

.product-explanation {
    font-size: 0.8rem;
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    padding: 0.5rem;
    border-radius: 4px;
    color: #334155;
    margin-top: 0.5rem;
    line-height: 1.4;
}

/* Placeholder Card */
.placeholder-view {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    margin-top: 1.5rem;
}

.placeholder-view h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
}

.placeholder-view p {
    font-size: 0.9rem;
    color: #64748b;
    max-width: 520px;
    margin: 0 auto 1.5rem auto;
}
</style>
"""
