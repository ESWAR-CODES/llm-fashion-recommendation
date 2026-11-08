"""
services/explanation_service.py
Evidence-Based Recommendation Explanation Service
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Generate metadata-grounded explanations for recommended products.
- Strict factual verification against recorded catalog attributes.
- Immediate rejection of ungrounded attributes or subjective marketing claims.
- Deterministic template fallback when LLM is unavailable.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from engines.explanation_engine import (
    validate_explanation,
    generate_template_explanation,
    explain_items
)


class ExplanationService:
    """
    Service for generating and verifying evidence-based explanations.
    """

    _instance: Optional["ExplanationService"] = None

    @classmethod
    def get_instance(cls) -> "ExplanationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def explain(
        self,
        product: Dict[str, Any],
        active_constraints: Optional[Dict[str, Any]] = None,
        profile_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a strictly verified explanation for a single recommended product.
        """
        cons = active_constraints or {}
        prefs = profile_preferences or {}

        # Generate template explanation
        explanation = generate_template_explanation(product, cons, prefs)

        # Validate grounding
        is_valid, reason = validate_explanation(explanation, product, cons)
        if not is_valid:
            # Safe minimal fallback
            color = product.get("baseColour", "item")
            article = product.get("articleType", "clothing")
            return f"This is a {color} {article} selected from the catalog."

        return explanation

    def explain_batch(
        self,
        products: List[Dict[str, Any]],
        active_constraints: Optional[Dict[str, Any]] = None,
        profile_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate explanations for a batch of recommended products.
        Returns: Dict mapping product_id -> explanation_string
        """
        return {
            str(p.get("id")): self.explain(p, active_constraints, profile_preferences)
            for p in products
        }
