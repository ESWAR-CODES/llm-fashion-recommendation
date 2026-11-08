# Streamlit Community Cloud Production Deployment Guide
**LLM-Assisted Personalized Fashion Recommendation System**

This guide provides end-to-end instructions for deploying the Multimodal Fashion Recommendation System to [Streamlit Community Cloud](https://share.streamlit.io/).

---

## 1. Architecture & Resource Profile

- **Application Framework:** Streamlit 1.30+
- **Primary Runtime:** Python 3.10 / 3.11 / 3.12 / 3.14 (CPU-Only)
- **Active Catalog:** 10,000 Frozen Apparel & Footwear Items
- **Text Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, precomputed matrix `embeddings.npy`)
- **Visual Encoder:** `sentence-transformers/clip-ViT-B-32` (512-d, precomputed matrix `clip_embeddings.npy`)
- **Language Model:** Three-Tier Provider (Optional External API -> Deterministic Rule Engine Fallback)
- **Memory Footprint:** ~312 MB RSS after loading catalog and both embedding matrices (Leaves >68% headroom within Streamlit Cloud's 1 GB container limit)
- **Cold Startup Latency:** ~5.08 seconds
- **Vector Search Throughput:** 2,225+ Queries Per Second (pure NumPy dot products)

---

## 2. Pre-Deployment Repository Audit

Ensure the repository contains the following deployment configuration files:

```
├── .streamlit/
│   └── config.toml               # Dark theme, headless execution, port 8501
├── requirements.txt              # Pinned production runtime dependencies
├── .env.example                  # Environment configuration template
├── app.py                        # Root entry point launcher
├── fashion_project/
│   ├── app.py                    # Modular Streamlit application
│   ├── .streamlit/config.toml    # Inner Streamlit config
│   ├── requirements.txt          # Inner requirements
│   ├── config.json               # System configuration & path definitions
│   ├── artifacts/
│   │   ├── catalog.csv           # 10,000 frozen products
│   │   ├── catalog_ids.json      # Row index mapping
│   │   ├── embeddings.npy        # (10000, 384) text vector matrix
│   │   └── clip_embeddings.npy   # (10000, 512) visual vector matrix
│   └── data/
│       └── catalog_images/       # 10,000 isolated images (107.39 MB)
```

---

## 3. Product Image Hosting Strategy

The system implements a **multi-tier image resolution pipeline**:

1. **Tier 1 (Bundled Local):** `fashion_project/data/catalog_images/{id}.jpg` (10,000 items, 107.39 MB total).
2. **Tier 2 (Remote CDN / GitHub Releases / Cloudflare):** If `IMAGE_CDN_URL` is set, the system dynamically fetches missing images and caches them in `.cache/images/`.
3. **Tier 3 (Deterministic Fallback):** For any corrupted or unreadable images, the system renders an elegant PIL vector placeholder with product ID metadata.

### Hosting Options:
- **Option A (All-in-One):** Commit `fashion_project/data/catalog_images/` directly to your GitHub repository. Since the entire directory is only 107 MB and individual files are ~10 KB, Git handles them smoothly without Git LFS.
- **Option B (External CDN):** Host the image files in an AWS S3 bucket, Cloudflare R2, or GitHub Release asset, and set `IMAGE_CDN_URL=https://your-bucket.com/catalog_images` in Streamlit Cloud Secrets.

---

## 4. Step-by-Step Deployment Instructions

### Step 1: Push Repository to GitHub
Ensure all code and artifacts are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "feat: complete phase 3 production polish and deployment readiness"
git push origin main
```

### Step 2: Sign In to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with your GitHub account.

### Step 3: Create New App
Click **"New app"** and fill in the deployment fields:
- **Repository:** `YourGitHubUsername/llm-fashion-recommendation` (or your repository name)
- **Branch:** `main`
- **Main file path:** `app.py` (or `fashion_project/app.py`)
- **App URL (optional):** Customize your subdomain (e.g., `fashion-ai-recommendation.streamlit.app`)

### Step 4: Configure App Secrets (Optional)
Under **Advanced settings** -> **Secrets**, paste your configuration in TOML format:

```toml
# Optional External LLM Provider (falls back to rule engine if omitted)
OPENAI_API_KEY = "sk-..."
HF_TOKEN = "hf_..."

# Optional External CDN Base URL for Product Images
IMAGE_CDN_URL = ""

# Hardware Overrides
DEVICE_OVERRIDE = "cpu"
```

### Step 5: Deploy
Click **"Deploy!"**. Streamlit Cloud will:
1. Provision a Linux container.
2. Install Python packages from `requirements.txt`.
3. Launch `app.py`.
4. Render the application at your public URL.

---

## 5. Session Isolation & Multi-User Concurrency

- **Guest Demonstration Mode:** Every new visitor is automatically allocated an isolated guest profile (`guest_<id>`) in Streamlit's `st.session_state`.
- **Zero Cross-Talk:** Likes, passes, and conversational filters applied by one user are stored against their session profile and do not leak into another user's session.
- **Ephemeral State Reset:** Users can click **"🧹 Clear"** or **"➕ New"** in the sidebar at any time to purge their session state.

---

## 6. Security & Privacy Guarantees

1. **Zero User PII:** The system does not collect names, email addresses, payment information, or passwords.
2. **Local Session DB:** Profiles and feedback are stored in SQLite; no user-generated data is transmitted to third-party databases.
3. **Safe File Uploads:** Uploaded clothing images are strictly validated (Pillow decode check, size limit <10 MB, accepted formats: JPEG, PNG, WebP) and never written to public directories.
4. **Error Masking:** Database and model exceptions are caught and masked with polite UI notices; internal stack traces and environment credentials are never exposed in user-facing views.

---

## 7. Performance & Health Telemetry

Once deployed, visit the **Model Insights** view within the app to monitor live metrics:
- Real-time CPU, RAM, and Disk metrics.
- Active embedding matrix shapes and normalization checks.
- Interactive latency benchmark executing live 10,000-product dot-product vector search in <1 ms.
