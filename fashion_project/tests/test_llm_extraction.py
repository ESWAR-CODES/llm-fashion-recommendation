"""
test_llm_extraction.py
Test Suite for Phase 4: Conversational Preference Extraction & Pydantic Validation
Authoritative Specification: End_to_End PipeLine.pdf (Step 6)
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

from engines.preference_extractor import (
    ProposedUpdate,
    StateUpdateItem,
    validate_and_apply,
    extract_update,
    rule_based_extract_update,
    get_allowed_catalog_values
)


class TestLLMExtraction(unittest.TestCase):

    def test_01_pydantic_schema_valid_and_invalid(self):
        # Valid schema
        valid_data = {
            "reset": False,
            "unsupported": [],
            "updates": [
                {"field": "articleType", "op": "set", "values": ["Shirts"]},
                {"field": "exclude_colors", "op": "add", "values": ["Red"]}
            ]
        }
        obj = ProposedUpdate(**valid_data)
        self.assertEqual(obj.reset, False)
        self.assertEqual(len(obj.updates), 2)

        # Invalid operation
        with self.assertRaises(Exception):
            ProposedUpdate(reset=False, unsupported=[], updates=[{"field": "articleType", "op": "invalid_op", "values": ["Shirts"]}])

    def test_02_set_add_remove_clear_operations(self):
        # Set operation
        curr = {"articleType": ["Kurtas"], "usage": ["Casual"]}
        upd = ProposedUpdate(reset=False, unsupported=[], updates=[
            StateUpdateItem(field="articleType", op="set", values=["Shirts"])
        ])
        ok, new_st, _ = validate_and_apply(curr, upd)
        self.assertTrue(ok)
        self.assertEqual(new_st["articleType"], ["Shirts"])
        self.assertEqual(new_st["usage"], ["Casual"])

        # Add operation
        upd_add = ProposedUpdate(reset=False, unsupported=[], updates=[
            StateUpdateItem(field="baseColour", op="add", values=["Navy Blue"])
        ])
        ok, new_st2, _ = validate_and_apply({"baseColour": ["Black"]}, upd_add)
        self.assertTrue(ok)
        self.assertEqual(sorted(new_st2["baseColour"]), ["Black", "Navy Blue"])

        # Remove operation
        upd_rem = ProposedUpdate(reset=False, unsupported=[], updates=[
            StateUpdateItem(field="baseColour", op="remove", values=["Black"])
        ])
        ok, new_st3, _ = validate_and_apply(new_st2, upd_rem)
        self.assertTrue(ok)
        self.assertEqual(new_st3["baseColour"], ["Navy Blue"])

        # Clear operation
        upd_clr = ProposedUpdate(reset=False, unsupported=[], updates=[
            StateUpdateItem(field="baseColour", op="clear", values=[])
        ])
        ok, new_st4, _ = validate_and_apply(new_st3, upd_clr)
        self.assertTrue(ok)
        self.assertNotIn("baseColour", new_st4)

    def test_03_reset_operation(self):
        curr = {"articleType": ["Shirts"], "baseColour": ["Black"]}
        upd_rst = ProposedUpdate(reset=True, unsupported=[], updates=[])
        ok, new_st, _ = validate_and_apply(curr, upd_rst)
        self.assertTrue(ok)
        self.assertEqual(new_st, {})

    def test_04_negation_and_exclusions(self):
        # "Avoid red"
        res = rule_based_extract_update("casual shirts but avoid red", {})
        self.assertFalse(res.reset)
        fields = {u.field: u.values for u in res.updates}
        self.assertIn("exclude_colors", fields)
        self.assertIn("Red", fields["exclude_colors"])

    def test_05_unsupported_attributes(self):
        # Size and price requests
        res = rule_based_extract_update("shoes in size 10 under $50", {})
        self.assertTrue(len(res.unsupported) > 0)
        unsupported_text = " ".join(res.unsupported).lower()
        self.assertTrue("size" in unsupported_text or "price" in unsupported_text)


if __name__ == "__main__":
    unittest.main()
