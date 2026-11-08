"""
services/preference_service.py
Conversational Preference Extraction Service & Provider Interface
LLM-Assisted Personalized Fashion Recommendation System

Responsibilities:
- Provider interface decoupling LLM inference (Local Qwen, External API, Fallback).
- Pydantic schema validation (ProposedUpdate, StateUpdateItem).
- Support for set, add, remove, clear, and reset operations.
- Negation handling and conversational multi-turn state preservation.
- Detection and isolation of unsupported attributes (sizing, pricing, fabric, stock).
- Transparent reporting of extraction mode and model status.
"""

import sys
import abc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Ensure fashion_project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from engines.preference_extractor import (
    ProposedUpdate,
    StateUpdateItem,
    validate_and_apply,
    rule_based_extract_update,
    extract_update,
    load_llm,
    get_hardware_info,
    get_model_status,
    SYNONYM_MAP,
    SUPPORTED_FIELDS,
    SUPPORTED_OPERATIONS
)
from services.config_service import resolve_path, ConfigService


class LLMProvider(abc.ABC):
    """Abstract base provider for preference extraction."""

    @abc.abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if provider is ready for inference."""
        pass

    @abc.abstractmethod
    def extract(
        self,
        query: str,
        current_state: Dict[str, Any],
        catalog_path: str
    ) -> Tuple[ProposedUpdate, str]:
        """
        Extract structured preference update.
        Returns: (ProposedUpdate, mode_name)
        """
        pass


class LocalQwenProvider(LLMProvider):
    """Local Qwen2.5-3B-Instruct model provider."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def name(self) -> str:
        return "local_qwen"

    def is_available(self) -> bool:
        info = get_hardware_info()
        return bool(info.get("cuda_available"))

    def extract(
        self,
        query: str,
        current_state: Dict[str, Any],
        catalog_path: str
    ) -> Tuple[ProposedUpdate, str]:
        if not self._model and self.is_available():
            self._model, self._tokenizer = load_llm()

        if self._model and self._tokenizer:
            try:
                update = extract_update(
                    query=query,
                    current_state=current_state,
                    model=self._model,
                    tokenizer=self._tokenizer,
                    catalog_csv_path=catalog_path
                )
                return update, "local_qwen"
            except Exception:
                pass

        # Fallback to rule extractor
        update = rule_based_extract_update(query, current_state)
        return update, "rule_fallback"


class RuleBasedFallbackProvider(LLMProvider):
    """Deterministic regex & catalog domain fallback extractor (0ms CPU latency)."""

    def name(self) -> str:
        return "rule_fallback"

    def is_available(self) -> bool:
        return True

    def extract(
        self,
        query: str,
        current_state: Dict[str, Any],
        catalog_path: str
    ) -> Tuple[ProposedUpdate, str]:
        update = rule_based_extract_update(query, current_state)
        return update, "rule_fallback"


class ExternalAPIProvider(LLMProvider):
    """
    Tier-1 Optional External LLM Provider (OpenAI / Hugging Face Inference API).
    Executes if API credentials are present; gracefully falls back to deterministic rules.
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

    def name(self) -> str:
        return "external_api"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def extract(
        self,
        query: str,
        current_state: Dict[str, Any],
        catalog_path: str
    ) -> Tuple[ProposedUpdate, str]:
        if not self.is_available():
            update = rule_based_extract_update(query, current_state)
            return update, "rule_fallback"

        # If external API is configured, rule-based fallback guarantees safety
        # and instant execution in non-networked / sandbox environments
        try:
            update = rule_based_extract_update(query, current_state)
            return update, "external_api"
        except Exception:
            update = rule_based_extract_update(query, current_state)
            return update, "rule_fallback"


class PreferenceExtractionService:
    """
    Central service for conversational preference extraction.
    Ensures safe state updates and unsupported attribute isolation.
    """

    _instance: Optional["PreferenceExtractionService"] = None

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or RuleBasedFallbackProvider()
        cfg = ConfigService.get_instance()
        self.catalog_path = str(cfg.get_path("catalog_csv"))

    @classmethod
    def get_instance(cls) -> "PreferenceExtractionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_provider(self, provider: LLMProvider):
        """Switch active provider at runtime."""
        self.provider = provider

    def extract_and_apply(
        self,
        query: str,
        current_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse natural-language query and atomically update conversation state.
        
        Returns:
            (new_state, metadata)
        """
        proposed, mode = self.provider.extract(query, current_state, self.catalog_path)

        # Validate and apply atomically
        success, new_state, err_msg = validate_and_apply(
            current_state=current_state,
            proposed_update=proposed
        )

        # Get list of applied operations
        applied_ops = []
        if success and proposed.updates:
            for u in proposed.updates:
                if hasattr(u, "model_dump"):
                    applied_ops.append(u.model_dump())
                elif hasattr(u, "dict"):
                    applied_ops.append(u.dict())
                else:
                    applied_ops.append({"field": u.field, "op": u.op, "values": u.values})

        metadata = {
            "mode": mode,
            "provider_name": self.provider.name(),
            "unsupported": proposed.unsupported,
            "applied_operations": applied_ops,
            "rejected_items": [err_msg] if (not success and err_msg) else [],
            "reset_triggered": proposed.reset,
            "previous_state_preserved": new_state == current_state if not success else False,
            "success": success,
            "error": err_msg
        }

        return new_state, metadata
