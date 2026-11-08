"""
test_streamlit_app.py
Unit tests for Streamlit application shell and UI components
LLM-Assisted Personalized Fashion Recommendation System
"""

import sys
import unittest
from pathlib import Path

# Ensure fashion_project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestStreamlitApp(unittest.TestCase):

    def test_01_app_and_ui_imports(self):
        """Verify all Streamlit UI modules import without fatal syntax or dependency errors."""
        import app
        import ui.styles as styles
        import ui.components as components
        import ui.sidebar as sidebar
        import ui.views.dashboard as dashboard
        import ui.views.discovery as discovery
        import ui.views.assistant as assistant
        import ui.views.collection as collection
        import ui.views.model_insights as model_insights

        self.assertIsNotNone(styles.CUSTOM_CSS)
        self.assertTrue(len(styles.CUSTOM_CSS) > 100)

    def test_02_component_html_generation(self):
        """Verify product card and status badge HTML renderers."""
        from ui.components import render_status_pill, render_product_card_html
        
        pill = render_status_pill("Operational", "ready")
        self.assertIn("status-pill-ready", pill)

        mock_product = {
            "id": "10000",
            "productDisplayName": "Puma Blue Shirt",
            "baseColour": "Blue",
            "articleType": "Shirts",
            "usage": "Casual"
        }
        card = render_product_card_html(mock_product, image_b64="data:image/jpeg;base64,mockdata", score=0.85)
        self.assertIn("Puma Blue Shirt", card)
        self.assertIn("data:image/jpeg;base64,mockdata", card)
        self.assertIn("0.850", card)


if __name__ == "__main__":
    unittest.main()
