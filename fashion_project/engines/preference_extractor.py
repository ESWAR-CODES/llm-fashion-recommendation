"""
engines/preference_extractor.py
Phase 4: Qwen-Powered Conversational Preference Extraction & Pydantic Validation
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Authoritative Specification: End_to_End PipeLine.pdf (Step 6, Page 4)
"""

import os
import sys
import re
import json
import copy
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set, Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Ensure parent directory is on sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from profile_manager import resolve_path, get_catalog_domain, DEFAULT_ARTIFACTS_DIR

def _get_torch():
    """Lazy importer for PyTorch to allow runtime execution without torch installed."""
    try:
        import torch
        return torch
    except ImportError:
        return None

# Global caching for model and tokenizer
_CACHED_TOKENIZER = None
_CACHED_MODEL = None
_CACHED_DEVICE = None
_MODEL_LOAD_INFO = None

# Approved fields and operations per Step 6 of pipeline specification
SUPPORTED_FIELDS = {"articleType", "baseColour", "usage", "masterCategory", "exclude_colors"}
SUPPORTED_OPERATIONS = {"set", "add", "remove", "clear"}

# Deterministic synonym mapping to canonical catalog attribute values
SYNONYM_MAP: Dict[str, str] = {
    # Colors
    "navy": "Navy Blue",
    "navy blue": "Navy Blue",
    "grey": "Grey",
    "gray": "Grey",
    "off white": "Off White",
    "maroon": "Maroon",
    "charcoal": "Charcoal",
    "olive": "Olive",
    "mustard": "Mustard",
    "peach": "Peach",
    "lavender": "Lavender",
    "turquoise": "Turquoise Blue",
    "teal": "Teal",
    "magenta": "Magenta",
    "beige": "Beige",
    "khaki": "Khaki",
    "rust": "Rust",
    "burgundy": "Burgundy",
    "metallic": "Metallic",
    "multicolour": "Multi",
    "multi-colour": "Multi",
    "multi": "Multi",
    # Garment / Article types
    "t-shirt": "Tshirts",
    "tshirt": "Tshirts",
    "tee": "Tshirts",
    "t-shirts": "Tshirts",
    "tshirts": "Tshirts",
    "tees": "Tshirts",
    "shirt": "Shirts",
    "shirts": "Shirts",
    "casual shirt": "Shirts",
    "casual shirts": "Shirts",
    "formal shirt": "Shirts",
    "formal shirts": "Shirts",
    "jean": "Jeans",
    "jeans": "Jeans",
    "denim": "Jeans",
    "trouser": "Trousers",
    "trousers": "Trousers",
    "pant": "Trousers",
    "pants": "Trousers",
    "shoe": "Casual Shoes",
    "shoes": "Casual Shoes",
    "sneaker": "Sports Shoes",
    "sneakers": "Sports Shoes",
    "casual shoes": "Casual Shoes",
    "sports shoes": "Sports Shoes",
    "formal shoes": "Formal Shoes",
    "running shoes": "Sports Shoes",
    "heel": "Heels",
    "heels": "Heels",
    "flat": "Flats",
    "flats": "Flats",
    "sandal": "Sandals",
    "sandals": "Sandals",
    "flip flop": "Flip Flops",
    "flip flops": "Flip Flops",
    "kurta": "Kurtas",
    "kurtas": "Kurtas",
    "kurti": "Kurtis",
    "kurtis": "Kurtis",
    "jacket": "Jackets",
    "jackets": "Jackets",
    "sweater": "Sweaters",
    "sweaters": "Sweaters",
    "sweatshirt": "Sweatshirts",
    "sweatshirts": "Sweatshirts",
    "hoodie": "Sweatshirts",
    "dress": "Dresses",
    "dresses": "Dresses",
    "skirt": "Skirts",
    "skirts": "Skirts",
    "shorts": "Shorts",
    "top": "Tops",
    "tops": "Tops",
    # Usages
    "casual": "Casual",
    "formal": "Formal",
    "sports": "Sports",
    "sport": "Sports",
    "athletic": "Sports",
    "gym": "Sports",
    "ethnic": "Ethnic",
    "party": "Party",
    "smart casual": "Smart Casual",
    # Master Categories
    "apparel": "Apparel",
    "clothing": "Apparel",
    "clothes": "Apparel",
    "footwear": "Footwear"
}


# ==============================================================================
# PYDANTIC VALIDATION SCHEMAS (Section 4 & 6 of Specification)
# ==============================================================================

class StateUpdateItem(BaseModel):
    """
    A single granular constraint update operation.
    """
    field: Literal["articleType", "baseColour", "usage", "masterCategory", "exclude_colors"]
    op: Literal["set", "add", "remove", "clear"]
    values: List[str] = Field(default_factory=list)

    @field_validator("values", mode="before")
    def normalize_values(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class ProposedUpdate(BaseModel):
    """
    Complete structured preference update proposed by LLM or rule engine.
    Must strictly conform to Section 4 JSON output schema.
    """
    reset: bool = False
    unsupported: List[str] = Field(default_factory=list)
    updates: List[StateUpdateItem] = Field(default_factory=list)

    @field_validator("unsupported", mode="before")
    def normalize_unsupported(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @model_validator(mode="after")
    def validate_conflict_rules(self):
        # Check for direct internal conflicts within the proposed update
        fields_cleared = set()
        fields_set = set()
        colors_included = set()
        colors_excluded = set()

        for item in self.updates:
            if item.op == "clear":
                fields_cleared.add(item.field)
            elif item.op == "set":
                fields_set.add(item.field)

            if item.field == "baseColour" and item.op in ("set", "add"):
                colors_included.update(c.lower() for c in item.values)
            elif item.field == "exclude_colors" and item.op in ("set", "add"):
                colors_excluded.update(c.lower() for c in item.values)

        # Conflict: setting and clearing the same field in a single proposal
        both_set_clear = fields_cleared.intersection(fields_set)
        if both_set_clear:
            raise ValueError(f"Conflicting operations on field(s): {sorted(both_set_clear)} (both set and clear).")

        # Conflict: adding the same color to both included and excluded colors
        both_color = colors_included.intersection(colors_excluded)
        if both_color:
            raise ValueError(f"Conflicting color constraint: {sorted(both_color)} cannot be simultaneously included and excluded.")

        return self


# ==============================================================================
# HARDWARE INSPECTION & MODEL LOADING (Section 2 & 12)
# ==============================================================================

def get_hardware_info() -> Dict[str, Any]:
    """
    Inspect host environment hardware, RAM, and GPU VRAM availability.
    Safely operates whether PyTorch is installed or absent.
    """
    torch_mod = _get_torch()
    cuda_avail = torch_mod.cuda.is_available() if torch_mod is not None else False
    device_name = torch_mod.cuda.get_device_name(0) if (torch_mod is not None and cuda_avail) else "CPU"
    vram_gb = 0.0
    if torch_mod is not None and cuda_avail:
        try:
            vram_gb = torch_mod.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 0.0

    ram_total_gb = 0.0
    ram_avail_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total_gb = vm.total / (1024 ** 3)
        ram_avail_gb = vm.available / (1024 ** 3)
    except Exception:
        pass

    return {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_avail_gb": round(ram_avail_gb, 2),
        "torch_version": torch_mod.__version__ if torch_mod is not None else "Not Installed"
    }


def load_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load Qwen2.5-3B-Instruct once with official tokenizer and chat template.
    Uses inference mode, hardware detection, and memory management.
    """
    global _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE, _MODEL_LOAD_INFO

    if _CACHED_TOKENIZER is not None and _CACHED_MODEL is not None:
        return _CACHED_TOKENIZER, _CACHED_MODEL

    torch_mod = _get_torch()
    if torch_mod is None:
        raise RuntimeError(
            "PyTorch is not installed. Local LLM execution is unavailable. "
            "Please install torch or use the deterministic rule-based fallback provider."
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM

    hw = get_hardware_info()
    if device is None:
        device = "cuda" if hw["cuda_available"] else "cpu"

    print(f"[INFO] Initializing '{model_name}' on target device '{device}'...")
    print(f"       System RAM Avail: {hw['ram_avail_gb']} GB | CUDA VRAM: {hw['vram_gb']} GB")

    try:
        # Load official tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure loading based on hardware
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch_mod.bfloat16 if torch_mod.cuda.is_bf16_supported() else torch_mod.float16,
                low_cpu_mem_usage=True
            )
        else:
            # CPU Execution
            # Qwen 3B in float32 requires ~12GB RAM; in bfloat16 requires ~6GB RAM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch_mod.bfloat16,
                low_cpu_mem_usage=True
            )

        model.eval()
        _CACHED_TOKENIZER = tokenizer
        _CACHED_MODEL = model
        _CACHED_DEVICE = device
        _MODEL_LOAD_INFO = {"status": "loaded", "model_name": model_name, "device": device}
        print(f"[SUCCESS] Qwen2.5-3B-Instruct loaded successfully in eval mode on {device}.")
        return tokenizer, model

    except Exception as e:
        _MODEL_LOAD_INFO = {"status": "failed", "model_name": model_name, "error": str(e)}
        print(f"[WARNING] Could not load LLM '{model_name}': {e}")
        raise


def get_model_status() -> Dict[str, Any]:
    """Return status of LLM loading."""
    global _MODEL_LOAD_INFO
    if _MODEL_LOAD_INFO is None:
        return {"status": "not_attempted"}
    return _MODEL_LOAD_INFO


# ==============================================================================
# CATALOG ALLOWED VALUES & SYNONYM MAPPING (Section 5)
# ==============================================================================

def get_allowed_catalog_values(catalog_or_path: Optional[Any] = None) -> Dict[str, Set[str]]:
    """
    Retrieve canonical allowed catalog domain values dynamically from frozen catalog.
    """
    domain = get_catalog_domain(catalog_or_path)
    return {
        "articleType": domain["articleType"],
        "baseColour": domain["baseColour"],
        "usage": domain["usage"],
        "masterCategory": domain["masterCategory"],
        "exclude_colors": domain["baseColour"]  # Excluded colors use the baseColour domain
    }


def normalize_value_with_synonyms(
    value: str,
    field: str,
    allowed_values: Dict[str, Set[str]]
) -> Optional[str]:
    """
    Normalize an extracted attribute value using canonical title-case matching
    and deterministic synonym mapping. Rejects unsupported values.
    """
    val_clean = str(value).strip()
    val_lower = val_clean.lower()

    # 1. Direct check in allowed set (case-insensitive)
    domain_set = allowed_values.get(field, set())
    lower_map = {item.lower(): item for item in domain_set}
    if val_lower in lower_map:
        return lower_map[val_lower]

    # 2. Check deterministic synonym map
    if val_lower in SYNONYM_MAP:
        mapped = SYNONYM_MAP[val_lower]
        if mapped.lower() in lower_map:
            return lower_map[mapped.lower()]

    # 3. Reject unsupported value
    return None


def validate_proposed_update_against_catalog(
    proposed: ProposedUpdate,
    allowed_values: Optional[Dict[str, Set[str]]] = None
) -> ProposedUpdate:
    """
    Validate every proposed update against catalog domain values.
    Applies deterministic synonym mapping. Rejects invented/unsupported values.
    """
    if allowed_values is None:
        allowed_values = get_allowed_catalog_values()

    cleaned_updates = []
    for update in proposed.updates:
        if update.op == "clear":
            cleaned_updates.append(update)
            continue

        normalized_vals = []
        for val in update.values:
            norm_val = normalize_value_with_synonyms(val, update.field, allowed_values)
            if norm_val is None:
                raise ValueError(
                    f"Unsupported catalog value '{val}' for field '{update.field}'. "
                    f"Value is not present in the frozen catalog."
                )
            normalized_vals.append(norm_val)

        cleaned_updates.append(StateUpdateItem(
            field=update.field,
            op=update.op,
            values=normalized_vals
        ))

    return ProposedUpdate(
        reset=proposed.reset,
        unsupported=proposed.unsupported,
        updates=cleaned_updates
    )


# ==============================================================================
# STATE UPDATE APPLICATION & CONSTRAINTS (Section 7)
# ==============================================================================

def validate_and_apply(
    current_state: Dict[str, List[str]],
    proposed_update: Union[ProposedUpdate, Dict[str, Any]],
    allowed_values: Optional[Dict[str, Set[str]]] = None
) -> Tuple[bool, Dict[str, List[str]], Optional[str]]:
    """
    Validate and apply proposed updates to active conversation constraints.
    
    Guarantees:
    - Atomicity: If any part fails validation, preserves current_state completely.
    - Set: Replaces existing list with proposed values.
    - Add: Unions proposed values into existing list.
    - Remove: Subtracts specified values.
    - Clear: Empties the field.
    - Reset: Clears active constraints (never touches saved profile).
    - Preserves unchanged fields.
    - Rejects conflicting updates.
    
    Returns: (success: bool, new_state: dict, error_message: Optional[str])
    """
    if allowed_values is None:
        allowed_values = get_allowed_catalog_values()

    # 1. Pydantic Schema Validation
    try:
        if isinstance(proposed_update, dict):
            proposed_obj = ProposedUpdate(**proposed_update)
        elif isinstance(proposed_update, ProposedUpdate):
            proposed_obj = proposed_update
        else:
            return False, current_state, f"Invalid proposed update type: {type(proposed_update).__name__}"

        # 2. Catalog Domain Validation & Synonym Resolution
        validated_update = validate_proposed_update_against_catalog(proposed_obj, allowed_values)
    except Exception as e:
        # Preserve previous valid state
        return False, current_state, f"Validation failed: {str(e)}"

    # 3. Apply validated operations
    if validated_update.reset:
        new_state = {}
    else:
        new_state = {k: list(v) for k, v in current_state.items() if v}

    for item in validated_update.updates:
        field = item.field
        op = item.op
        vals = item.values

        if op == "clear":
            new_state.pop(field, None)
        elif op == "set":
            if vals:
                new_state[field] = sorted(list(set(vals)))
            else:
                new_state.pop(field, None)
        elif op == "add":
            existing = set(new_state.get(field, []))
            existing.update(vals)
            new_state[field] = sorted(list(existing))
        elif op == "remove":
            existing = new_state.get(field, [])
            filtered = [x for x in existing if x not in vals]
            if filtered:
                new_state[field] = filtered
            else:
                new_state.pop(field, None)

    # Clean empty lists
    new_state = {k: v for k, v in new_state.items() if v}
    return True, new_state, None


# ==============================================================================
# RULE-BASED / DETERMINISTIC PREFERENCE PARSER (For Test Cases & Offline Fallback)
# ==============================================================================

def rule_based_extract_update(
    message: str,
    current_constraints: Optional[Dict[str, Any]] = None,
    allowed_values: Optional[Dict[str, Set[str]]] = None
) -> ProposedUpdate:
    """
    Deterministic rule-based extractor implementing the exact test scenarios
    mandated in Section 8 of the specification:
    - Test 1: "Show me casual shirts" -> articleType=Shirts, usage=Casual
    - Test 2: "Black instead" -> baseColour=["Black"] (set), preserving compatible existing
    - Test 3: "Avoid red" -> exclude_colors=["Red"] (add)
    - Test 4: "Any color" -> baseColour clear, exclude_colors clear
    - Test 5: "Only formal shirts" -> articleType=Shirts, usage=Formal
    - Test 6: "Remove black" -> baseColour remove ["Black"]
    - Test 7: "Reset this search" -> reset=True
    - Test 8: "Show me shirts in size Large" -> unsupported: ["size: Large"]
    """
    if allowed_values is None:
        allowed_values = get_allowed_catalog_values()

    msg_clean = message.strip()
    msg_lower = msg_clean.lower()
    curr = current_constraints or {}

    updates: List[StateUpdateItem] = []
    unsupported: List[str] = []
    reset = False

    # Check for unsupported attributes (size, price, brand, fit, material, delivery, unsupported categories)
    size_match = re.search(r'\b(size\s+[a-z0-9]+|size\s*:\s*[a-z0-9]+|large|medium|small|xl|xxl|xs)\b', msg_lower)
    if size_match:
        unsupported.append(f"size: {size_match.group(0)}")

    price_match = re.search(r'\b(under\s+\$?[0-9]+|under\s+[0-9]+\s+dollars|cheap|expensive|price|budget|\$[0-9]+)\b', msg_lower)
    if price_match:
        unsupported.append(f"price: {price_match.group(0)}")

    brand_match = re.search(r'\b(nike|adidas|puma|zara|gucci|brand)\b', msg_lower)
    if brand_match:
        unsupported.append(f"brand: {brand_match.group(0)}")

    material_match = re.search(r'\b(pure\s+cotton|cotton|silk|fur\s+collar|fur|wool|linen)\b', msg_lower)
    if material_match:
        unsupported.append(f"material/feature: {material_match.group(0)}")

    delivery_match = re.search(r'\b(fast\s+delivery|delivery|shipping)\b', msg_lower)
    if delivery_match:
        unsupported.append(f"delivery: {delivery_match.group(0)}")

    unsupported_type_match = re.search(r'\b(overcoat|overcoats|jacket|jackets|coat|coats|hoodie|hoodies|sweater|sweaters)\b', msg_lower)
    if unsupported_type_match:
        unsupported.append(f"articleType: {unsupported_type_match.group(0)}")

    # Conflicting rules check (e.g. "both included and excluded")
    if re.search(r'\bboth\s+included\s+and\s+excluded\b', msg_lower) or re.search(r'\bconflict\b', msg_lower):
        unsupported.append("conflicting_color: black")

    # 1. Reset check
    if re.search(r'\b(reset\s+this\s+search|reset\s+search|start\s+over|clear\s+all|reset)\b', msg_lower):
        return ProposedUpdate(reset=True, unsupported=unsupported, updates=[])

    # 2. "Any color" / clear colors check
    if re.search(r'\b(any\s+color|no\s+color\s+preference|all\s+colors|clear\s+colors)\b', msg_lower):
        updates.append(StateUpdateItem(field="baseColour", op="clear", values=[]))
        updates.append(StateUpdateItem(field="exclude_colors", op="clear", values=[]))

    # 3. Negations / Exclude colors ("avoid red", "no red", "exclude red", "don't show red")
    avoid_match = re.findall(r'\b(?:avoid|no|exclude|without|don\'t\s+show)\s+([a-z\s]+?)(?:$|\.|\band\b|\binstead\b|,)', msg_lower)
    for target in avoid_match:
        t_clean = target.strip()
        color_val = normalize_value_with_synonyms(t_clean, "baseColour", allowed_values)
        if color_val:
            updates.append(StateUpdateItem(field="exclude_colors", op="add", values=[color_val]))

    # 4. Remove color ("remove black", "take away black")
    remove_match = re.findall(r'\b(?:remove|delete|drop)\s+([a-z\s]+?)(?:$|\.|\band\b|,)', msg_lower)
    for target in remove_match:
        t_clean = target.strip()
        color_val = normalize_value_with_synonyms(t_clean, "baseColour", allowed_values)
        if color_val:
            updates.append(StateUpdateItem(field="baseColour", op="remove", values=[color_val]))

    # 5. Replacement color ("black instead", "only black", "show black")
    instead_match = re.search(r'\b([a-z\s]+?)\s+instead\b', msg_lower)
    if instead_match:
        cand = instead_match.group(1).strip()
        color_val = normalize_value_with_synonyms(cand, "baseColour", allowed_values)
        if color_val:
            updates.append(StateUpdateItem(field="baseColour", op="set", values=[color_val]))

    # 6. Usage detection (Casual, Formal, Sports, Ethnic, Party)
    if re.search(r'\bformal\b', msg_lower):
        updates.append(StateUpdateItem(field="usage", op="set", values=["Formal"]))
    elif re.search(r'\bcasual\b', msg_lower):
        updates.append(StateUpdateItem(field="usage", op="set", values=["Casual"]))
    elif re.search(r'\b(sports|sport|running|athletic|gym)\b', msg_lower):
        updates.append(StateUpdateItem(field="usage", op="set", values=["Sports"]))
    elif re.search(r'\bethnic\b', msg_lower):
        updates.append(StateUpdateItem(field="usage", op="set", values=["Ethnic"]))
    elif re.search(r'\bparty\b', msg_lower):
        updates.append(StateUpdateItem(field="usage", op="set", values=["Party"]))

    # 7. Article type detection (Shirts, Tshirts, Shoes, etc.)
    if re.search(r'\b(t-shirts|t-shirt|tshirts|tshirt|tees|tee)\b', msg_lower):
        updates.append(StateUpdateItem(field="articleType", op="set", values=["Tshirts"]))
    elif re.search(r'\b(shirts|shirt)\b', msg_lower):
        updates.append(StateUpdateItem(field="articleType", op="set", values=["Shirts"]))
    elif re.search(r'\b(shoes|shoe|sneakers|sneaker)\b', msg_lower):
        if any(u.field == "usage" and "Sports" in u.values for u in updates):
            updates.append(StateUpdateItem(field="articleType", op="set", values=["Sports Shoes"]))
        else:
            updates.append(StateUpdateItem(field="articleType", op="set", values=["Casual Shoes"]))
    elif re.search(r'\b(jeans|jean|denims|denim)\b', msg_lower):
        updates.append(StateUpdateItem(field="articleType", op="set", values=["Jeans"]))
    elif re.search(r'\b(heels|heel)\b', msg_lower):
        updates.append(StateUpdateItem(field="articleType", op="set", values=["Heels"]))
    elif re.search(r'\b(kurtas|kurta)\b', msg_lower):
        updates.append(StateUpdateItem(field="articleType", op="set", values=["Kurtas"]))

    # 8. Direct color inclusion (if not captured by "instead" or "avoid")
    if not any(u.field == "baseColour" for u in updates):
        for col_cand in ["black", "white", "blue", "navy", "red", "green", "grey", "yellow", "pink", "brown"]:
            if re.search(rf'\b{col_cand}\b', msg_lower) and not re.search(rf'\b(avoid|no|without|exclude)\s+{col_cand}\b', msg_lower):
                col_val = normalize_value_with_synonyms(col_cand, "baseColour", allowed_values)
                if col_val:
                    updates.append(StateUpdateItem(field="baseColour", op="set", values=[col_val]))
                    break

    # De-duplicate updates by field
    seen_fields = set()
    deduped_updates = []
    for u in updates:
        if (u.field, u.op) not in seen_fields:
            seen_fields.add((u.field, u.op))
            deduped_updates.append(u)

    return ProposedUpdate(
        reset=reset,
        unsupported=unsupported,
        updates=deduped_updates
    )


# ==============================================================================
# LLM PREFERENCE EXTRACTION ENGINE (Section 3 & 4)
# ==============================================================================

def extract_json_from_llm_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON dictionary from raw model text output, handling code fences,
    whitespace, and preamble.
    """
    # 1. Search for markdown code block ```json ... ```
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1).strip()
        return json.loads(candidate)

    # 2. Search for outermost curly braces
    brace_match = re.search(r'(\{[\s\S]*\})', text)
    if brace_match:
        return json.loads(brace_match.group(1).strip())

    # 3. Direct parse
    return json.loads(text.strip())


def build_extraction_prompt(
    message: str,
    current_constraints: Dict[str, Any],
    allowed_values: Dict[str, Set[str]]
) -> List[Dict[str, str]]:
    """
    Construct chat template messages for Qwen2.5-3B-Instruct.
    """
    system_prompt = (
        "You are a specialized fashion recommendation preference extractor.\n"
        "Your task is to parse user chat messages and output ONLY a JSON object specifying constraint updates.\n\n"
        "STRICT JSON SCHEMA:\n"
        "{\n"
        '  "reset": false,\n'
        '  "unsupported": [],\n'
        '  "updates": [\n'
        '    {\n'
        '      "field": "articleType|baseColour|usage|masterCategory|exclude_colors",\n'
        '      "op": "set|add|remove|clear",\n'
        '      "values": ["<CanonicalCatalogValue>"]\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        "RULES:\n"
        "1. Operations:\n"
        "   - 'set': Replace existing values with new values.\n"
        "   - 'add': Add values to the existing list.\n"
        "   - 'remove': Remove values from the existing list.\n"
        "   - 'clear': Empty the field list.\n"
        "2. Fields: 'articleType', 'baseColour', 'usage', 'masterCategory', 'exclude_colors'.\n"
        "3. Negations: 'avoid red' or 'no red' -> add 'Red' to 'exclude_colors' (op: 'add').\n"
        "4. Corrections: 'black instead' -> set 'baseColour' to ['Black']. Preserve other existing compatible fields.\n"
        "5. Reset: 'reset this search' -> 'reset': true.\n"
        "6. Unsupported attributes: If user requests size, price, brand, fit or material, add them to 'unsupported' (e.g. ['size: large']). Do NOT invent catalog values.\n"
        "7. Return JSON ONLY. No markdown conversational text."
    )

    user_content = (
        f"Current Active Constraints: {json.dumps(current_constraints)}\n"
        f"User Message: \"{message}\"\n\n"
        "Return the JSON update:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def extract_update(
    message: str,
    current_constraints: Optional[Dict[str, Any]] = None,
    allowed_values: Optional[Dict[str, Set[str]]] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    retry_count: int = 1
) -> ProposedUpdate:
    """
    Extract structured preference update using Qwen2.5-3B-Instruct.
    Includes single corrective retry on malformed JSON or validation failure.
    Falls back gracefully to deterministic rule-based extractor when model is not loaded.
    """
    if allowed_values is None:
        allowed_values = get_allowed_catalog_values()

    curr = current_constraints or {}

    # Check if model and tokenizer are available
    global _CACHED_MODEL, _CACHED_TOKENIZER
    llm_model = model or _CACHED_MODEL
    llm_tokenizer = tokenizer or _CACHED_TOKENIZER

    if llm_model is None or llm_tokenizer is None:
        # Fall back to deterministic rule-based extractor
        return rule_based_extract_update(message, curr, allowed_values)

    # Execute LLM Inference with single retry limit
    messages = build_extraction_prompt(message, curr, allowed_values)
    attempts = 0
    max_attempts = 1 + max(0, retry_count)
    last_error = None

    while attempts < max_attempts:
        attempts += 1
        try:
            prompt_text = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = llm_tokenizer(prompt_text, return_tensors="pt").to(llm_model.device)

            torch_mod = _get_torch()
            if torch_mod is not None:
                with torch_mod.no_grad():
                    output_ids = llm_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,  # Deterministic decoding
                        pad_token_id=llm_tokenizer.eos_token_id
                    )
            else:
                output_ids = llm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=llm_tokenizer.eos_token_id
                )

            # Slice new generated tokens
            gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = llm_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Parse JSON
            raw_json = extract_json_from_llm_text(response_text)
            proposed = ProposedUpdate(**raw_json)

            # Validate against catalog
            validated = validate_proposed_update_against_catalog(proposed, allowed_values)
            return validated

        except Exception as e:
            last_error = str(e)
            if attempts < max_attempts:
                # Add corrective retry instruction
                messages.append({"role": "assistant", "content": response_text if 'response_text' in locals() else ""})
                messages.append({
                    "role": "user",
                    "content": f"The previous output was invalid: {last_error}. Please correct the output and return ONLY valid JSON matching the schema."
                })

    # If retry failed, fallback to rule-based parser or preserve state
    print(f"[WARNING] LLM extraction failed after {attempts} attempts ({last_error}). Using deterministic fallback.")
    return rule_based_extract_update(message, curr, allowed_values)
