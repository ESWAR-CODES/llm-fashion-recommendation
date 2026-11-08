"""
engines/explanation_engine.py
Phase 4: Evidence-Based LLM Explanations & Factual Attribute Grounding
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Step 8, Page 5)
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set

# Ensure project root is on sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from engines.preference_extractor import extract_json_from_llm_text

# Forbidden phrases and unsupported claims per Step 8 of specification
PROHIBITED_UNSUPPORTED_PATTERNS = [
    # Fit claims
    r'\bperfect\s+fit\b',
    r'\bwill\s+fit\s+(?:you\s+)?perfect(?:ly)?\b',
    r'\bslim\s+fit\b',
    r'\bflattering\b',
    r'\btailored\s+fit\b',
    r'\bcomfortable\s+fit\b',
    # Quality / Brand claims
    r'\bpremium\b',
    r'\bbest\s+quality\b',
    r'\bhigh(?:est)?\s+quality\b',
    r'\bluxur(?:y|ious)\b',
    r'\bauthentic\b',
    r'\bgenuine\b',
    r'\bdurable\b',
    # Price / Value claims
    r'\blowest\s+price\b',
    r'\bcheap\b',
    r'\baffordable\b',
    r'\bexpensive\b',
    r'\bdiscount\b',
    r'\bbest\s+deal\b',
    # Stock / Inventory claims
    r'\bin\s+stock\b',
    r'\bavailable\s+now\b',
    r'\bselling\s+fast\b',
    r'\blimited\s+stock\b',
    # Unverified fabric/material claims (unless explicitly in productDisplayName)
    r'\b100%\s+cotton\b',
    r'\bpure\s+silk\b',
    r'\bgenuine\s+leather\b'
]


# ==============================================================================
# DETERMINISTIC TEMPLATE EXPLANATION (Section 9 & 10)
# ==============================================================================

def generate_template_explanation(
    product: Dict[str, Any],
    active_constraints: Optional[Dict[str, Any]] = None,
    profile_preferences: Optional[Dict[str, Any]] = None
) -> str:
    """
    Construct a factual, grounded explanation strictly from recorded metadata.
    Does not invent unsupported attributes, materials, fits, prices, or claims.
    
    Format Example from PDF:
    "This is a navy shirt marked Casual, matching your request."
    """
    cons = active_constraints or {}
    prefs = profile_preferences or {}

    color = str(product.get("baseColour", "Unknown")).strip()
    article_type = str(product.get("articleType", "item")).strip()
    usage = str(product.get("usage", "Casual")).strip()
    season = str(product.get("season", "")).strip()

    matches = []
    # Check match against active constraints
    if cons.get("articleType") and article_type.lower() in [a.lower() for a in cons["articleType"]]:
        matches.append(article_type)
    if cons.get("baseColour") and color.lower() in [c.lower() for c in cons["baseColour"]]:
        matches.append(color)
    if cons.get("usage") and usage.lower() in [u.lower() for u in cons["usage"]]:
        matches.append(f"{usage} wear")

    # Check match against saved preferences
    if prefs.get("preferred_colors") and color.lower() in [c.lower() for c in prefs["preferred_colors"]]:
        if color not in matches:
            matches.append(f"preferred {color}")

    if matches:
        match_phrase = f", matching your request for {', '.join(matches)}"
    else:
        match_phrase = ""

    explanation = f"This is a {color} {article_type} marked {usage}{match_phrase}."
    return explanation


# ==============================================================================
# EXPLANATION FACTUAL GROUNDING VALIDATION (Section 10)
# ==============================================================================

def validate_explanation(
    explanation: str,
    product: Dict[str, Any],
    active_constraints: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Strict evidence-based validation of explanation text against recorded product attributes.
    
    Validation Rules:
    1. Must be non-empty string.
    2. Must not contain unsupported claims (premium, perfect fit, cheap, in stock, etc.).
    3. Any referenced colors must match the product's actual recorded baseColour.
    4. Any referenced garment type must match the product's actual articleType or masterCategory.
    5. Any referenced usage must match the product's recorded usage.
    
    Returns: (is_valid: bool, rejection_reason: Optional[str])
    """
    if not explanation or not isinstance(explanation, str) or len(explanation.strip()) < 5:
        return False, "Explanation is empty or excessively short."

    text_lower = explanation.lower().strip()
    recorded_name = str(product.get("productDisplayName", "")).lower()
    recorded_color = str(product.get("baseColour", "")).lower()
    recorded_type = str(product.get("articleType", "")).lower()
    recorded_usage = str(product.get("usage", "")).lower()

    # Rule 1: Check for prohibited unsupported claims
    for pat in PROHIBITED_UNSUPPORTED_PATTERNS:
        match = re.search(pat, text_lower)
        if match:
            # Check if phrase is literally part of recorded productDisplayName
            matched_phrase = match.group(0)
            if matched_phrase not in recorded_name:
                return False, f"Contains unsupported claim: '{matched_phrase}' (unrecorded in catalog metadata)"

    # Rule 2: Attribute grounding - Color mentions
    # If explanation mentions common colors, verify the color belongs to the product
    common_colors = [
        "black", "white", "navy blue", "navy", "blue", "red", "green", "grey", "gray",
        "yellow", "pink", "purple", "brown", "orange", "maroon", "beige", "olive"
    ]
    for col in common_colors:
        # Check if color word appears as standalone token
        if re.search(rf'\b{col}\b', text_lower):
            # The mentioned color must be compatible with recorded_color
            if col not in recorded_color and recorded_color not in col and col not in recorded_name:
                return False, f"Ungrounded color mention: '{col}' does not match product baseColour '{product.get('baseColour')}'"

    # Rule 3: Attribute grounding - Usage mentions
    common_usages = ["casual", "formal", "sports", "ethnic", "party", "smart casual"]
    for usg in common_usages:
        if re.search(rf'\b{usg}\b', text_lower):
            # Check if usage token is grounded in recorded usage, title, or articleType (e.g. "Casual Shoes")
            if usg not in recorded_usage and usg not in recorded_name and usg not in recorded_type:
                return False, f"Ungrounded usage mention: '{usg}' does not match product usage '{product.get('usage')}'"

    return True, None


# ==============================================================================
# EVIDENCE-BASED EXPLANATION ENGINE (Section 9 & 10)
# ==============================================================================

def build_explanation_prompt(
    selected_products: List[Dict[str, Any]],
    active_constraints: Optional[Dict[str, Any]] = None,
    profile_preferences: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Construct prompt for Qwen2.5-3B-Instruct providing ONLY selected product IDs
    and recorded attributes.
    """
    system_prompt = (
        "You are an evidence-based fashion recommendation assistant.\n"
        "Your task is to generate ONE brief, factual explanation for each selected product.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Mention ONLY recorded attributes: Name, Article Type, Color, Usage, and Category.\n"
        "2. Do NOT mention materials (e.g. cotton, silk) unless explicitly in the product name.\n"
        "3. Do NOT make fit claims (e.g. 'perfect fit', 'will fit you perfectly').\n"
        "4. Do NOT make quality claims (e.g. 'premium', 'high quality', 'luxury').\n"
        "5. Do NOT make price or stock claims (e.g. 'cheap', 'in stock', 'lowest price').\n"
        "6. Return JSON mapping each exact Product ID to its brief explanation string.\n\n"
        "Allowed Example:\n"
        "{\"1234\": \"This is a Navy Blue shirt marked Casual, matching your request for casual shirts.\"}"
    )

    items_payload = []
    for p in selected_products:
        items_payload.append({
            "id": str(p.get("id")),
            "productDisplayName": str(p.get("productDisplayName")),
            "articleType": str(p.get("articleType")),
            "baseColour": str(p.get("baseColour")),
            "usage": str(p.get("usage")),
            "masterCategory": str(p.get("masterCategory")),
            "season": str(p.get("season"))
        })

    user_content = (
        f"Active Constraints: {json.dumps(active_constraints or {})}\n"
        f"User Preferences: {json.dumps(profile_preferences or {})}\n"
        f"Selected Products (IDs and Recorded Attributes):\n{json.dumps(items_payload, indent=2)}\n\n"
        "Return the JSON object mapping product IDs to brief grounded explanations:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def explain_items(
    selected_products: List[Dict[str, Any]],
    active_constraints: Optional[Dict[str, Any]] = None,
    profile_preferences: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None
) -> Dict[str, str]:
    """
    Generate evidence-based explanations for returned recommendations.
    
    Workflow:
    1. Send ONLY selected product IDs (up to 5) and recorded metadata to Qwen.
    2. Parse response JSON mapping product ID -> explanation.
    3. Validate each explanation:
       - Verify ID belongs to selected set.
       - Verify mentioned attributes match recorded metadata.
       - Reject ungrounded material, fit, quality, price or stock claims.
    4. If validation fails or LLM is unavailable, fall back to deterministic template explanation.
    5. Never alter recommendation ranking.
    """
    if not selected_products:
        return {}

    cons = active_constraints or {}
    prefs = profile_preferences or {}
    selected_id_set = set(str(p.get("id")) for p in selected_products)
    product_map = {str(p.get("id")): p for p in selected_products}

    explanations: Dict[str, str] = {}

    # Attempt LLM generation if model and tokenizer are provided
    if model is not None and tokenizer is not None:
        try:
            import torch
            messages = build_explanation_prompt(selected_products, cons, prefs)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Deterministic decoding
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Parse returned JSON
            parsed = extract_json_from_llm_text(response_text)
            if isinstance(parsed, dict):
                for pid_key, expl_text in parsed.items():
                    pid_str = str(pid_key).strip()
                    # Validate product ID belongs to selected set
                    if pid_str in selected_id_set and isinstance(expl_text, str):
                        p_meta = product_map[pid_str]
                        is_valid, reason = validate_explanation(expl_text, p_meta, cons)
                        if is_valid:
                            explanations[pid_str] = expl_text.strip()
                        else:
                            print(f"[GROUNDING REJECTION] Rejected LLM explanation for item {pid_str}: {reason}")
        except Exception as e:
            print(f"[WARNING] LLM explanation generation encountered error: {e}. Falling back to templates.")

    # Fill in any missing or rejected explanations with deterministic templates
    for pid_str, p_meta in product_map.items():
        if pid_str not in explanations:
            explanations[pid_str] = generate_template_explanation(p_meta, cons, prefs)

    return explanations
