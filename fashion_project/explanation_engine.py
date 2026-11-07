"""
explanation_engine.py
Top-level export for engines.explanation_engine
"""

from engines.explanation_engine import (
    explain_items,
    validate_explanation,
    generate_template_explanation,
    build_explanation_prompt,
    PROHIBITED_UNSUPPORTED_PATTERNS
)

__all__ = [
    "explain_items",
    "validate_explanation",
    "generate_template_explanation",
    "build_explanation_prompt",
    "PROHIBITED_UNSUPPORTED_PATTERNS"
]
