# Demonstration Sequence Plan

**Project Title:** LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval  
**Specification:** `End_to_End PipeLine.pdf` (Step 12, page 8)  
**Target Outcome:** An interactive prototype demonstration verifying preference extraction, conversational refinement, personalized ranking, no-match handling, profile switching, and comparative evaluation.  

---

## 1. Demonstration Protocol Overview

This document specifies the exact 6-step walkthrough protocol required for academic demonstration, evaluation, and screencast presentation. All product IDs and requests utilize actual frozen catalog records verified in `artifacts/catalog.csv`.

---

## 2. Step-by-Step Walkthrough Sequence

### Step 1: Onboarding & Profile Initialization
- **Action:** Open `Fashion_Recommender.ipynb`, run Section 8, and click **`+ New Profile`**.
- **System State:** Creates a unique anonymous user profile (e.g. `usr_demo_01`).
- **Interaction:**
  - Select preferred categories: `Apparel`, `Footwear`.
  - Select favorite colors: `Navy Blue`, `Black`.
  - Click **`Save Preferences`**.
- **Verification:** The active profile status indicator reflects the newly registered taste vector.

### Step 2: Cold-Start Request & Positive Feedback
- **Action:** In the search input, type:  
  `"Show me navy blue shirts for casual wear"` and click **`Recommend Products`**.
- **System Response:**
  - Active constraints update to: `articleType: ['Shirts']`, `baseColour: ['Navy Blue']`, `usage: ['Casual']`.
  - Up to five product cards are displayed with images, metadata, and grounded explanations (e.g. *"This is a Navy Blue Shirts marked Casual, matching your request for Shirts, Navy Blue."*).
- **Interaction:** Click **`👍 Like`** on Product ID `10051` (*Puma Men Motorsport Navy Blue Shirts*).
- **Verification:** Button updates to *"Liked!"* and disables. SQLite `feedback` table records the action idempotently. The profile vector $\mathbf{u}$ incorporates the liked item representation.

### Step 3: Conversational Refinement & Constraint Updates
- **Action:** In the search input, submit a conversational refinement:  
  `"Black instead but avoid red"` and click **`Recommend Products`**.
- **System Response:**
  - Extraction updates: `baseColour` changes from `Navy Blue` to `Black` (`op: set`), and `Red` is added to `exclude_colors` (`op: add`).
  - Active constraints indicator updates in real-time.
  - Returned items dynamically re-rank to show black shirts while strictly filtering out any red apparel.
- **Verification:** Hard constraints take priority over user vector; zero red or non-black items appear.

### Step 4: No-Match Handling & Graceful Recovery
- **Action:** In the search input, submit an incompatible constraint combination:  
  `"Formal flip flops in black"` or set manual dropdowns to `Flip Flops` + `Formal`.
- **System Response:**
  - Candidate filtering identifies 0 matching products in the frozen catalog.
  - System cleanly presents the **No Matching Products Found** warning card.
  - Actionable guidance is displayed: *"Try clearing excluded colors or broadening category / usage filters."*
- **Verification:** The pipeline **never silently relaxes constraints** or crashes with an unhandled exception.

### Step 5: Profile Switching & Taste Isolation
- **Action:** In the profile dropdown, switch to an alternative profile (e.g. `usr_demo_02` with preferred footwear taste) or click **`+ New Profile`**.
- **Interaction:** Submit `"casual wear"` under the new profile.
- **Verification:**
  - The recommendations rank completely differently based on the new profile's independent taste vector.
  - Previous likes from `usr_demo_01` do not leak into `usr_demo_02`.

### Step 6: Step 10 Comparative Evaluation Review
- **Action:** Navigate to Section 9 of the notebook.
- **Display:** Review the 4-system benchmark comparison table generated from the 40 test queries:
  - **System A (Keyword baseline):** 100.0% constraint-compliant precision on filtered items, 36.4 ms median latency.
  - **System B (Semantic retrieval):** 100.0% precision, 26.9 ms median latency.
  - **System C (Personalized retrieval):** 100.0% precision, 29.7 ms median latency (demonstrates personalization shift).
  - **System D (Full pipeline):** 100.0% precision, 0.0% constraint violations, 27.1 ms median latency (deterministic CPU fallback), 100.0% explanation grounding compliance.
  - **Relevance Protocol:** `human_judgements.csv` (696 rows) formatted as a blinded human trial protocol template; no volunteer subjects recruited during development.

---

## 3. Reviewer Verification Guide

Reviewers can reproduce this exact sequence inside `Fashion_Recommender.ipynb` or via automated test assertions by running:
```bash
python fashion_project/tests/run_all_tests.py
```
All 38 test assertions will execute and pass in approximately 20 seconds.
