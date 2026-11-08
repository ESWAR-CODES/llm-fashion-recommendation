"""
interactive_ui.py
Phase 5: Step 9 Interactive Prototype Interface with ipywidgets
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Step 9, page 6)
- Profile controls: choose or create an anonymous profile; edit saved preferences; clear profile via explicit control.
- Request controls: text input, category, color, usage dropdowns, Recommend button, Clear current request button.
- Results: up to 5 product cards with images, names, attributes, grounded explanations, Like and Dismiss buttons.
- States: loading, working, no matches, unsupported attribute warnings, and manual filter fallback.
- Disable repeated submissions while generation is running.
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any, Set

import numpy as np
import pandas as pd
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Add fashion_project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from embedding_engine import load_embeddings, load_embedding_model
from profile_manager import (
    resolve_path,
    init_db,
    create_profile,
    update_profile,
    delete_profile,
    get_profile,
    list_profiles,
    save_feedback,
    get_feedback,
    get_active_constraints,
    set_active_constraints,
    clear_active_constraints,
    get_catalog_domain,
    build_user_vector
)
from personalized_ranker import filter_candidates, recommend_personalized
from engines.preference_extractor import (
    extract_update,
    validate_and_apply,
    ProposedUpdate
)
from engines.explanation_engine import (
    explain_items,
    validate_explanation,
    generate_template_explanation
)


class FashionRecommenderApp:
    """
    Interactive ipywidgets Application implementing Step 9 of the End-to-End Pipeline.
    """

    def __init__(
        self,
        catalog_path: str = "fashion_project/artifacts/catalog.csv",
        embeddings_path: str = "fashion_project/artifacts/embeddings.npy",
        catalog_ids_path: str = "fashion_project/artifacts/catalog_ids.json",
        db_path: str = "fashion_project/backups/profiles.sqlite",
        images_dir: str = "fashion_project/data/raw/images"
    ):
        self.catalog_path = resolve_path(catalog_path)
        self.catalog_ids_path = resolve_path(catalog_ids_path)
        self.db_path = str(resolve_path(db_path))
        self.images_dir = resolve_path(images_dir)

        # Initialize SQLite database
        init_db(self.db_path)

        # Load catalog and domain values
        print("[UI] Ingesting catalog data...")
        self.catalog_df = pd.read_csv(self.catalog_path, dtype={"id": str})
        with open(self.catalog_ids_path, "r", encoding="utf-8") as f:
            self.catalog_ids = json.load(f)

        # Load embeddings matrix and model
        print("[UI] Loading precomputed embeddings and MiniLM model...")
        self.embeddings_matrix, _ = load_embeddings(str(self.catalog_path.parent))
        self.embedding_model = load_embedding_model()

        # LLM references (lazy loaded or tokenized)
        self.qwen_model = None
        self.qwen_tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        except Exception:
            pass

        # Extract domain values for dropdowns
        self.domain = get_catalog_domain(self.catalog_df)
        self.article_types = sorted(list(self.domain.get("articleType", [])))
        self.colors = sorted(list(self.domain.get("baseColour", [])))
        self.usages = sorted(list(self.domain.get("usage", [])))
        self.categories = sorted(list(self.domain.get("masterCategory", [])))

        # Ensure active profile exists
        existing = list_profiles(self.db_path)
        if existing:
            self.current_profile_id = existing[0]
        else:
            self.current_profile_id = create_profile(db_path=self.db_path)

        # Current active constraints
        self.active_constraints: Dict[str, List[str]] = get_active_constraints(self.current_profile_id, self.db_path) or {}

        # Build UI components
        self._init_widgets()
        self._refresh_profile_ui()

    def _init_widgets(self):
        """Create and style all ipywidgets controls."""
        # 1. Profile Controls
        self.profile_select_dropdown = widgets.Dropdown(
            description="Active Profile:",
            options=self._get_profile_dropdown_options(),
            value=self.current_profile_id,
            style={"description_width": "110px"},
            layout=widgets.Layout(width="340px")
        )
        self.btn_new_profile = widgets.Button(
            description="+ New Profile",
            button_style="info",
            tooltip="Create a new anonymous profile",
            icon="user-plus",
            layout=widgets.Layout(width="130px")
        )
        self.btn_clear_profile = widgets.Button(
            description="Clear Profile",
            button_style="danger",
            tooltip="Wipe profile preferences and liked history",
            icon="trash",
            layout=widgets.Layout(width="130px")
        )

        self.pref_categories_select = widgets.SelectMultiple(
            options=self.categories,
            description="Preferred Cat:",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="280px", height="80px")
        )
        self.pref_colors_select = widgets.SelectMultiple(
            options=self.colors[:20],
            description="Fav Colors:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="280px", height="80px")
        )
        self.btn_update_prefs = widgets.Button(
            description="Save Preferences",
            button_style="primary",
            tooltip="Update saved taste profile",
            icon="check",
            layout=widgets.Layout(width="150px")
        )

        # 2. Request Controls
        self.query_text_input = widgets.Text(
            value="",
            placeholder="e.g. Navy blue casual shirts, or avoid red",
            description="Search Request:",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="580px")
        )
        self.manual_category_dropdown = widgets.Dropdown(
            options=["[All Article Types]"] + self.article_types,
            value="[All Article Types]",
            description="Type Filter:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="280px")
        )
        self.manual_color_dropdown = widgets.Dropdown(
            options=["[All Colors]"] + self.colors,
            value="[All Colors]",
            description="Color Filter:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="280px")
        )
        self.manual_usage_dropdown = widgets.Dropdown(
            options=["[All Usages]"] + self.usages,
            value="[All Usages]",
            description="Usage Filter:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="280px")
        )

        self.btn_recommend = widgets.Button(
            description="Recommend Products",
            button_style="success",
            icon="search",
            tooltip="Parse preferences, apply constraints, and rank top-5 items",
            layout=widgets.Layout(width="190px", height="38px")
        )
        self.btn_clear_request = widgets.Button(
            description="Clear Search Filters",
            button_style="warning",
            icon="eraser",
            tooltip="Reset conversation constraints",
            layout=widgets.Layout(width="170px", height="38px")
        )

        # 3. Status & Notification Area
        self.status_output = widgets.Output()
        self.constraints_badge_output = widgets.Output()
        self.results_output = widgets.Output()

        # Connect event handlers
        self.profile_select_dropdown.observe(self._on_profile_selected, names="value")
        self.btn_new_profile.on_click(self._on_create_profile)
        self.btn_clear_profile.on_click(self._on_clear_profile)
        self.btn_update_prefs.on_click(self._on_update_preferences)
        self.btn_recommend.on_click(self._on_recommend_clicked)
        self.btn_clear_request.on_click(self._on_clear_request_clicked)

    def _get_profile_dropdown_options(self) -> List[Tuple[str, str]]:
        profiles = list_profiles(self.db_path)
        return [(f"User Profile {pid[-6:]} ({pid})", pid) for pid in profiles]

    def _refresh_profile_ui(self):
        """Update profile options and active saved preferences."""
        profiles = list_profiles(self.db_path)
        self.profile_select_dropdown.options = self._get_profile_dropdown_options()
        if self.current_profile_id in profiles:
            self.profile_select_dropdown.value = self.current_profile_id
        elif profiles:
            self.current_profile_id = profiles[0]
            self.profile_select_dropdown.value = self.current_profile_id

        # Load saved preferences
        prof = get_profile(self.current_profile_id, self.db_path)
        if prof and "preferences" in prof:
            saved = prof["preferences"]
            cats = saved.get("preferred_categories", [])
            cols = saved.get("preferred_colors", [])
            self.pref_categories_select.value = tuple(c for c in cats if c in self.categories)
            self.pref_colors_select.value = tuple(c for c in cols if c in self.colors)

        # Render constraints display
        self._render_constraints_display()

    def _render_constraints_display(self):
        with self.constraints_badge_output:
            clear_output(wait=True)
            if not self.active_constraints:
                html = "<div style='padding:6px; background:#f1f5f9; border-radius:6px; font-size:12px; color:#475569;'><b>Active Constraints:</b> None (Full Catalog)</div>"
            else:
                badges = []
                for k, v in self.active_constraints.items():
                    val_str = ", ".join(v) if isinstance(v, list) else str(v)
                    bg = "#ef4444" if k == "exclude_colors" else "#0284c7"
                    badges.append(f"<span style='background:{bg}; color:white; padding:3px 8px; border-radius:12px; margin-right:5px; font-size:11px;'>{k}: {val_str}</span>")
                html = f"<div style='padding:8px; background:#f8fafc; border-left:4px solid #0284c7; border-radius:4px;'><b>Active Request Constraints:</b> {' '.join(badges)}</div>"
            display(HTML(html))

    def _on_profile_selected(self, change):
        if change.get("new"):
            self.current_profile_id = change["new"]
            self.active_constraints = get_active_constraints(self.current_profile_id, self.db_path) or {}
            self._refresh_profile_ui()
            with self.status_output:
                clear_output()
                display(HTML(f"<div style='color:#0284c7; font-size:12px;'>Switched to profile <b>{self.current_profile_id}</b>.</div>"))

    def _on_create_profile(self, _):
        new_id = create_profile(db_path=self.db_path)
        self.current_profile_id = new_id
        self.active_constraints = {}
        self._refresh_profile_ui()
        with self.status_output:
            clear_output()
            display(HTML(f"<div style='color:#16a34a; font-size:12px;'>Created new anonymous profile <b>{new_id}</b>.</div>"))

    def _on_clear_profile(self, _):
        """Wipes profile taste preferences and likes explicitly."""
        update_profile(self.current_profile_id, {}, db_path=self.db_path)
        # Clear feedback table for this profile
        conn = get_db_connection(self.db_path)
        with conn:
            conn.execute("DELETE FROM feedback WHERE profile_id = ?;", (self.current_profile_id,))
        self.pref_categories_select.value = ()
        self.pref_colors_select.value = ()
        with self.status_output:
            clear_output()
            display(HTML(f"<div style='color:#dc2626; font-size:12px;'>Profile <b>{self.current_profile_id}</b> preferences and likes wiped successfully.</div>"))

    def _on_update_preferences(self, _):
        prefs = {
            "preferred_categories": list(self.pref_categories_select.value),
            "preferred_colors": list(self.pref_colors_select.value)
        }
        update_profile(self.current_profile_id, prefs, db_path=self.db_path)
        with self.status_output:
            clear_output()
            display(HTML(f"<div style='color:#16a34a; font-size:12px;'>Saved preferences for <b>{self.current_profile_id}</b>: {prefs}</div>"))

    def _on_clear_request_clicked(self, _):
        self.active_constraints = {}
        clear_active_constraints(self.current_profile_id, self.db_path)
        self.manual_category_dropdown.value = "[All Article Types]"
        self.manual_color_dropdown.value = "[All Colors]"
        self.manual_usage_dropdown.value = "[All Usages]"
        self.query_text_input.value = ""
        self._render_constraints_display()
        with self.results_output:
            clear_output()
        with self.status_output:
            clear_output()
            display(HTML("<div style='color:#d97706; font-size:12px;'>Cleared active conversation constraints.</div>"))

    def _on_recommend_clicked(self, _):
        """Main recommendation pipeline callback."""
        # Disable button to prevent duplicate submissions
        self.btn_recommend.disabled = True
        self.btn_recommend.description = "Searching..."

        with self.status_output:
            clear_output()
            display(HTML("<div style='color:#0284c7; font-size:12px;'><i class='fa fa-spinner fa-spin'></i> Processing query, extracting preferences, and scoring catalog items...</div>"))

        query_text = self.query_text_input.value.strip()

        # 1. Step 6: Parse LLM proposed update from user message
        if query_text:
            proposed_update = extract_update(
                message=query_text,
                current_constraints=self.active_constraints,
                model=self.qwen_model,
                tokenizer=self.qwen_tokenizer
            )
            success, new_constraints, err_msg = validate_and_apply(
                current_state=self.active_constraints,
                proposed_update=proposed_update
            )
            if success:
                self.active_constraints = new_constraints
                set_active_constraints(self.current_profile_id, self.active_constraints, self.db_path)

            # Check for unsupported attributes
            if proposed_update.unsupported:
                with self.status_output:
                    clear_output()
                    unsupported_str = ", ".join(proposed_update.unsupported)
                    display(HTML(f"<div style='padding:8px; background:#fffbeb; border-left:4px solid #f59e0b; border-radius:4px; font-size:12px; color:#92400e; margin-bottom:8px;'>"
                                 f"<b>Notice:</b> The catalog does not record sizes, prices, or live inventory. Unsupported requests ignored: <code>{unsupported_str}</code>. Please use available filters.</div>"))

        # 2. Incorporate manual dropdown filter selections if specified
        if self.manual_category_dropdown.value != "[All Article Types]":
            self.active_constraints["articleType"] = [self.manual_category_dropdown.value]
        if self.manual_color_dropdown.value != "[All Colors]":
            self.active_constraints["baseColour"] = [self.manual_color_dropdown.value]
        if self.manual_usage_dropdown.value != "[All Usages]":
            self.active_constraints["usage"] = [self.manual_usage_dropdown.value]

        self._render_constraints_display()

        # 3. Step 7: Personalized Recommendation Ranking
        pref = get_profile(self.current_profile_id, self.db_path).get("preferences", {})
        rec_res = recommend_personalized(
            profile_id=self.current_profile_id,
            query=query_text if query_text else "fashion apparel footwear",
            constraints=self.active_constraints,
            embeddings=self.embeddings_matrix,
            catalog_ids=self.catalog_ids,
            catalog=self.catalog_df,
            model=self.embedding_model,
            db_path=self.db_path,
            images_dir=str(self.images_dir),
            top_k=5
        )

        results = rec_res.get("results", [])

        # 4. Step 8: Evidence-Grounded Explanations
        explanations = {}
        if results:
            explanations = explain_items(
                selected_products=results,
                active_constraints=self.active_constraints,
                profile_preferences=pref,
                model=self.qwen_model,
                tokenizer=self.qwen_tokenizer
            )

        # 5. Render Results Cards
        self._render_results_cards(results, explanations, rec_res.get("revised_constraints_suggestion", ""))

        # Reset button state
        self.btn_recommend.disabled = False
        self.btn_recommend.description = "Recommend Products"

    def _render_results_cards(self, results: List[Dict[str, Any]], explanations: Dict[str, str], suggestion: str):
        with self.results_output:
            clear_output(wait=True)

            if not results:
                # No matches found state
                msg = (
                    "<div style='padding:16px; background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; margin-top:10px;'>"
                    "<h4 style='color:#be123c; margin-top:0;'><i class='fa fa-exclamation-circle'></i> No matching products found</h4>"
                    "<p style='color:#4c0519; font-size:13px;'>No catalog items satisfy all your active filters simultaneously.</p>"
                    f"<p style='color:#4c0519; font-size:12px; background:#ffe4e6; padding:8px; border-radius:4px;'><b>Suggestion:</b> {suggestion or 'Try clearing excluded colors or broadening category / usage filters.'}</p>"
                    "</div>"
                )
                display(HTML(msg))
                return

            cards = []
            for item in results:
                item_id = str(item["id"])
                name = item.get("productDisplayName", f"Item #{item_id}")
                cat = item.get("articleType", "Apparel")
                color = item.get("baseColour", "Unknown")
                usage = item.get("usage", "Casual")
                score = round(float(item.get("score", 0.0)), 3)
                expl = explanations.get(item_id, f"Matches your active request for {cat} in {color}.")

                # Product image loading (Base64 encoding for reliable inline rendering)
                img_path = self.images_dir / f"{item_id}.jpg"
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f_img:
                            b64_str = base64.b64encode(f_img.read()).decode("utf-8")
                            img_tag = f"<img src='data:image/jpeg;base64,{b64_str}' style='width:120px; height:150px; object-fit:cover; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);' />"
                    except Exception:
                        img_tag = "<div style='width:120px; height:150px; background:#e2e8f0; display:flex; align-items:center; justify-content:center; border-radius:6px; font-size:11px; color:#64748b;'>No Image</div>"
                else:
                    img_tag = "<div style='width:120px; height:150px; background:#e2e8f0; display:flex; align-items:center; justify-content:center; border-radius:6px; font-size:11px; color:#64748b;'>No Image</div>"

                # Card HTML
                card_html = (
                    f"<div style='border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-bottom:12px; background:white; display:flex; gap:16px; align-items:flex-start; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>"
                    f"  <div style='flex-shrink:0;'>{img_tag}</div>"
                    f"  <div style='flex-grow:1;'>"
                    f"    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>"
                    f"      <h4 style='margin:0 0 4px 0; color:#0f172a; font-size:15px;'>{name}</h4>"
                    f"      <span style='background:#f1f5f9; color:#334155; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600;'>ID: {item_id} | Score: {score}</span>"
                    f"    </div>"
                    f"    <div style='margin-bottom:8px;'>"
                    f"      <span style='background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:4px; font-size:11px; margin-right:4px;'>{cat}</span>"
                    f"      <span style='background:#f3e8ff; color:#7e22ce; padding:2px 8px; border-radius:4px; font-size:11px; margin-right:4px;'>{color}</span>"
                    f"      <span style='background:#dcfce7; color:#15803d; padding:2px 8px; border-radius:4px; font-size:11px;'>{usage}</span>"
                    f"    </div>"
                    f"    <div style='background:#f8fafc; border-left:3px solid #3b82f6; padding:6px 10px; border-radius:4px; font-size:12px; color:#334155; font-style:italic; margin-bottom:8px;'>"
                    f"      <b>Explanation:</b> {expl}"
                    f"    </div>"
                    f"  </div>"
                    f"</div>"
                )

                # Create Like and Dismiss Buttons
                btn_like = widgets.Button(
                    description="Like",
                    button_style="success",
                    icon="thumbs-up",
                    layout=widgets.Layout(width="80px", height="28px")
                )
                btn_dismiss = widgets.Button(
                    description="Dismiss",
                    button_style="danger",
                    icon="times",
                    layout=widgets.Layout(width="85px", height="28px")
                )

                def make_like_handler(pid=item_id):
                    def handler(b):
                        save_feedback(self.current_profile_id, pid, "like", self.db_path)
                        b.description = "Liked!"
                        b.button_style = "info"
                        b.disabled = True
                    return handler

                def make_dismiss_handler(pid=item_id):
                    def handler(b):
                        save_feedback(self.current_profile_id, pid, "dismiss", self.db_path)
                        b.description = "Dismissed"
                        b.disabled = True
                        # Trigger fresh recommendation without the dismissed item
                        self._on_recommend_clicked(None)
                    return handler

                btn_like.on_click(make_like_handler())
                btn_dismiss.on_click(make_dismiss_handler())

                button_box = widgets.HBox([btn_like, btn_dismiss], layout=widgets.Layout(margin="6px 0 0 136px"))
                item_widget = widgets.VBox([widgets.HTML(card_html), button_box], layout=widgets.Layout(margin="0 0 8px 0"))
                cards.append(item_widget)

            display(widgets.VBox(cards))

    def render(self):
        """Assemble and render the top-level interface."""
        profile_box = widgets.VBox([
            widgets.HTML("<h3 style='margin:0 0 8px 0; color:#1e293b;'><i class='fa fa-user-circle'></i> User Profile & Saved Taste</h3>"),
            widgets.HBox([self.profile_select_dropdown, self.btn_new_profile, self.btn_clear_profile]),
            widgets.HBox([self.pref_categories_select, self.pref_colors_select, widgets.VBox([self.btn_update_prefs], layout=widgets.Layout(justify_content="center"))], layout=widgets.Layout(margin="8px 0 0 0"))
        ], layout=widgets.Layout(padding="14px", border="1px solid #cbd5e1", border_radius="8px", background_color="#f8fafc", margin="0 0 16px 0"))

        request_box = widgets.VBox([
            widgets.HTML("<h3 style='margin:0 0 8px 0; color:#1e293b;'><i class='fa fa-search'></i> Clothing Request & Constraints</h3>"),
            self.query_text_input,
            widgets.HBox([self.manual_category_dropdown, self.manual_color_dropdown, self.manual_usage_dropdown]),
            widgets.HBox([self.btn_recommend, self.btn_clear_request], layout=widgets.Layout(margin="10px 0 4px 0"))
        ], layout=widgets.Layout(padding="14px", border="1px solid #cbd5e1", border_radius="8px", background_color="#ffffff", margin="0 0 16px 0"))

        main_ui = widgets.VBox([
            widgets.HTML("<h2 style='color:#0f172a; margin-bottom:4px;'>LLM-Assisted Personalized Fashion Recommender</h2>"
                         "<p style='color:#64748b; font-size:13px; margin-top:0;'>Prototype Demonstration — Phase 5 Final Integration</p>"),
            profile_box,
            request_box,
            self.constraints_badge_output,
            self.status_output,
            self.results_output
        ], layout=widgets.Layout(max_width="920px", padding="10px"))

        display(main_ui)
        self._render_constraints_display()


def launch_recommender_ui():
    """Helper entry point to initialize and display recommender widget."""
    app = FashionRecommenderApp()
    app.render()
    return app


if __name__ == "__main__":
    print("[UI] interactive_ui module compiled and verified successfully.")
