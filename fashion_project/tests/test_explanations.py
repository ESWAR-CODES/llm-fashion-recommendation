"""
test_explanations.py
Test Suite for Phase 4: Grounded Explanations & Unsupported Claim Detection
Authoritative Specification: End_to_End PipeLine.pdf (Step 8)
"""

import os
import sys
import json
from pathlib import Path
import unittest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.explanation_engine import (
    validate_explanation,
    generate_template_explanation,
    explain_items
)


class TestExplanations(unittest.TestCase):

    def setUp(self):
        self.sample_product = {
            "id": "10051",
            "productDisplayName": "Puma Men Motorsport Navy Blue Shirts",
            "articleType": "Shirts",
            "baseColour": "Navy Blue",
            "usage": "Casual"
        }

    def test_01_factual_grounding_valid(self):
        valid_expl = "This is a Navy Blue Shirts marked Casual, matching your request."
        is_valid, reason = validate_explanation(
            valid_expl,
            self.sample_product,
            {"articleType": ["Shirts"]}
        )
        self.assertTrue(is_valid, f"Expected valid explanation, got: {reason}")

    def test_02_ungrounded_attribute_rejection(self):
        # Explanation claiming red color for a navy blue product
        invalid_expl = "This is a bright Red shirt marked Casual."
        is_valid, reason = validate_explanation(
            invalid_expl,
            self.sample_product,
            {"articleType": ["Shirts"]}
        )
        self.assertFalse(is_valid, "Expected rejection due to ungrounded color claim")

    def test_03_prohibited_claims_rejection(self):
        prohibited = [
            "This premium cotton shirt will fit you perfectly.",
            "This is a high quality cheap item in stock."
        ]
        for claim in prohibited:
            is_valid, reason = validate_explanation(claim, self.sample_product, {})
            self.assertFalse(is_valid, f"Expected rejection of prohibited claim: {claim}")

    def test_04_template_explanation_fallback(self):
        template = generate_template_explanation(
            self.sample_product,
            active_constraints={"articleType": ["Shirts"], "baseColour": ["Navy Blue"]},
            profile_preferences={"preferred_colors": ["Navy Blue"]}
        )
        self.assertIn("Navy Blue", template)
        self.assertIn("Shirts", template)
        self.assertIn("Casual", template)


if __name__ == "__main__":
    unittest.main()
