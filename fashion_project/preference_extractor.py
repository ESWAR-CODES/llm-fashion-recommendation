"""
preference_extractor.py
Top-level export for engines.preference_extractor
"""

from engines.preference_extractor import (
    load_llm,
    get_hardware_info,
    get_model_status,
    StateUpdateItem,
    ProposedUpdate,
    validate_and_apply,
    validate_proposed_update_against_catalog,
    get_allowed_catalog_values,
    rule_based_extract_update,
    extract_update,
    extract_json_from_llm_text,
    SYNONYM_MAP,
    SUPPORTED_FIELDS,
    SUPPORTED_OPERATIONS
)

__all__ = [
    "load_llm",
    "get_hardware_info",
    "get_model_status",
    "StateUpdateItem",
    "ProposedUpdate",
    "validate_and_apply",
    "validate_proposed_update_against_catalog",
    "get_allowed_catalog_values",
    "rule_based_extract_update",
    "extract_update",
    "extract_json_from_llm_text",
    "SYNONYM_MAP",
    "SUPPORTED_FIELDS",
    "SUPPORTED_OPERATIONS"
]
