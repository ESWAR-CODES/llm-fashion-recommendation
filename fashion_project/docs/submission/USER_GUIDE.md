# User Guide & Operation Manual
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Quick Launch & Local Execution

### Prerequisites:
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, and 3.14)
- Pip package manager

### Launch Commands:
```bash
# 1. Clone repository and navigate to root
cd llm-fashion

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 2. Walkthrough of Application Views

### 2.1 📊 Dashboard View
- **Overview:** Displays system health, frozen catalog size (10,000 items), active vector dimensions (384-d text, 512-d visual), and active SQLite profiles.
- **Service Status:** Shows readiness badges for all backend engines.
- **Catalog Sample:** Displays a random sample of verified catalog clothing with image thumbnails.

### 2.2 👗 Fashion Discovery View
- **Natural Language Search:** Type fashion requests like `"navy blue casual shirt"` or `"running sneakers"`.
- **Image Upload:** Drag and drop clothing photos (JPEG, PNG, WebP) to retrieve visually similar garments.
- **Combined Multimodal:** Enter text AND upload an image simultaneously for joint aesthetic and semantic scoring.
- **Advanced Facet Filters:** Filter by Gender, Master Category, SubCategory, Article Type, Color, Usage, Season, or Exclude specific colors.
- **Candidate Cards:** View top-5 candidates with similarity breakdowns ($S_{\text{text}}, S_{\text{vis}}, S_{\text{prof}}$), grounded justifications, and interactive feedback buttons (**❤️ Like** and **👎 Pass**).

### 2.3 💬 AI Fashion Assistant View
- **Conversational Chat:** Natural-language multi-turn dialogue.
- **Constraint Refinement:** Speak naturally (e.g., *"Find formal trousers, but avoid black"* followed by *"Now show me only blue ones"*).
- **Unsupported Attribute Guards:** Inquiries about sizing or pricing are recognized and isolated with a helpful advisory notice.
- **Inline Recommendations:** Browse products recommended directly in response to your conversation.

### 2.4 ❤️ My Collection & Preferences View
- **Liked Products Gallery:** View all items you have liked. Clicking **💔 Unlike** instantly updates your user taste vector $\mathbf{u}$.
- **Taste Preferences Editor:** Manually set preferred colors, categories, and gender preferences.
- **Dismissals & Exclusions:** View passed items and click **↩️ Undo** to restore them to search results.

### 2.5 🔬 Model Insights View
- **Hardware Telemetry:** Live CPU, RAM, Disk, and Python runtime metrics.
- **Model Readiness:** Dimensionality, L2 normalization status, and device allocation for MiniLM and CLIP.
- **Live Latency Benchmark:** Click **"⚡ Run Live Latency Benchmark"** to measure actual microsecond dot-product vector search across 10,000 items in real-time.

---

## 3. Profile Management & Guest Session Isolation

- **Sidebar Profile Selector:** Switch between different user profiles or click **"➕ New"** to create a fresh isolated guest session.
- **Reset State:** Click **"🧹 Clear"** to purge active constraints and reset conversation filters.
