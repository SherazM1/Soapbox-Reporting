"""Deterministic image style-guide helpers for PDP audit metadata."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


_EMPTY_GUIDE: dict[str, Any] = {
    "category": "",
    "category_key": "",
    "guide_key": "",
    "global_guidance": {},
    "slot_definitions": {},
    "pages": {},
    "product_type_index": {},
}

_FOOD_BEVERAGE_ALIASES = {
    "food_beverage",
    "food beverage",
    "food and beverage",
    "food_beverageimg",
    "food beverageimg",
}

_BEAUTY_ALIASES = {
    "beauty",
    "beautyimg",
}

_HEALTH_PERSONAL_CARE_ALIASES = {
    "health_personal_care",
    "health personal care",
    "health and personal care",
    "healthpersonal",
    "healthpersonalimg",
}

_TOYS_ALIASES = {
    "toys",
    "toysimg",
}

_HOUSEHOLD_CLEAN_ALIASES = {
    "household_clean",
    "householdclean",
    "householdcleanimg",
    "household_industrial_cleaning_storage",
    "household industrial cleaning storage",
    "household_industrial_cleaning_and_storage",
    "household industrial cleaning and storage",
    "household cleaning",
}

_FURNITURE_ALIASES = {
    "furniture",
    "furnitureimg",
}

_BABY_ALIASES = {
    "baby",
    "babyimg",
}

_ARTS_CRAFTS_ALIASES = {
    "arts_crafts",
    "art_crafts",
    "arts_and_crafts",
    "art_and_crafts",
    "arts and crafts",
    "art and crafts",
    "artscrafts",
    "artscraftsimg",
}

_ANIMALS_ALIASES = {
    "animals",
    "animalsimg",
}

_ELECTRONICS_ALIASES = {
    "electronics",
    "photography",
    "electronics_photography",
    "electronics photography",
    "electronics and photography",
    "electronicsimg",
}

_MEDIA_ALIASES = {
    "media",
    "mediaimg",
}

_SEASONAL_ALIASES = {
    "seasonal",
    "seasonal_occasion",
    "seasonal occasion",
    "seasonal and occasion",
    "seasonalimg",
}

_SPORTS_OUTDOORS_ALIASES = {
    "sportsoutdoors",
    "sportsoutdoors json",
    "sports outdoors",
    "sports_outdoors",
    "sports and outdoors",
    "sports recreation outdoor",
    "sports recreation outdoors",
    "sports recreation and outdoor",
    "sports_recreation_outdoor",
    "sports_recreation_outdoors",
    "sportsoutdoorsimg",
}

_SLOT_RULES: list[tuple[str, float, list[str]]] = [
    (
        "silo_out_of_pack_whats_included",
        0.98,
        ["out of pack", "included pieces", "included accessories", "whats included", "what's included"],
    ),
    (
        "graphic_theme_collection",
        0.98,
        ["theme collection", "same theme", "party collection", "coordinated"],
    ),
    (
        "graphic_color_design_range",
        0.98,
        ["color range", "design range", "color schemes"],
    ),
    (
        "graphic_personalization",
        0.98,
        ["personalization", "personalized", "customizable", "monogram"],
    ),
    (
        "graphic_assembly_overview",
        0.98,
        ["assembly overview", "assembly diagram", "setup diagram"],
    ),
    (
        "silo_back",
        0.98,
        ["model back", "full visibility", "silo back", "rear"],
    ),
    (
        "silo_with_model",
        0.97,
        ["with model", "model", "full product", "no shadows"],
    ),
    (
        "silo_back",
        0.97,
        ["model back", "full visibility", "silo back", "rear"],
    ),
    (
        "dimensions_occasion_scale",
        0.97,
        ["human reference", "clear callouts"],
    ),
    (
        "graphic_size_guide",
        0.97,
        ["size guide", "size chart", "age range", "adult sizing"],
    ),
    (
        "graphic_count",
        0.97,
        ["piece count", "pack contents", "pack size", "bold graphics"],
    ),
    (
        "silo_detail",
        0.96,
        ["embroidery", "embellishment", "accessory detail", "quality detail"],
    ),
    (
        "silo_front_in_pack",
        0.96,
        ["front in pack", "package front", "branding", "boxed"],
    ),
    (
        "silo_back_in_pack",
        0.96,
        ["back in pack", "package back", "back label", "label details"],
    ),
    (
        "lifestyle_in_use",
        0.95,
        ["occasion", "party", "holiday", "festive", "celebration"],
    ),
    (
        "feature_graphic",
        0.95,
        ["pre-lit", "shatterproof", "fade-resistant", "light count", "timer", "power type"],
    ),
    (
        "silo_angle",
        0.95,
        ["angled", "profile", "thickness", "shape"],
    ),
    (
        "graphic_whats_included",
        0.97,
        ["what's included", "whats included", "included components", "flat-lay", "flat lay"],
    ),
    (
        "contents_view",
        0.96,
        ["box set", "collector edition", "multi-piece", "contents view"],
    ),
    (
        "interior_view",
        0.96,
        ["interior view", "interior pages", "open pages", "inside pages"],
    ),
    (
        "spine_view",
        0.96,
        ["book spine", "dvd spine", "case spine", "side view", "spine", "edge"],
    ),
    (
        "lifestyle_in_use",
        0.95,
        ["media in use", "interaction", "reading", "coloring", "watching", "listening"],
    ),
    (
        "silo_detail",
        0.95,
        ["print quality", "paper texture", "paper finish", "close-up print", "closeup print"],
    ),
    (
        "silo_front",
        0.94,
        ["front cover", "straight-on cover", "cover packaging"],
    ),
    (
        "graphic_folded_deployed",
        0.97,
        ["folded view", "deployed view", "changes form", "open state", "closed state", "folded", "deployed"],
    ),
    (
        "graphic_interior_organization",
        0.97,
        ["interior view", "interior organization", "compartments", "padding", "inside"],
    ),
    (
        "graphic_worn_fit",
        0.97,
        ["worn fit", "range of motion", "harness", "belt", "worn"],
    ),
    (
        "lifestyle_aerial_flight",
        0.96,
        ["active aerial", "flight environment", "aerial", "flight", "drone", "flying"],
    ),
    (
        "lifestyle_working_configuration",
        0.96,
        ["working configuration", "contact points", "support points", "configured"],
    ),
    (
        "function_graphic",
        0.96,
        ["primary function", "supports materials", "function", "holding", "protecting", "organizing", "processing"],
    ),
    (
        "graphic_installed_view",
        0.96,
        ["installed view", "mounted view", "full setup", "workflow", "placement"],
    ),
    (
        "graphic_interface",
        0.96,
        ["software interface", "app interface", "interface screen", "application"],
    ),
    (
        "graphic_warranty",
        0.96,
        ["warranty", "protection plan", "coverage"],
    ),
    (
        "graphic_whats_included",
        0.96,
        ["what's included", "whats included", "included items", "contents"],
    ),
    (
        "silo_front_software",
        0.95,
        ["software packaging", "software", "game media"],
    ),
    (
        "silo_front_gaming",
        0.95,
        ["gaming hardware", "game console", "console", "controller", "vr", "attached components"],
    ),
    (
        "silo_front_optics",
        0.95,
        ["optics", "binoculars", "microscope", "telescope", "lenses", "lens"],
    ),
    (
        "silo_front_radio",
        0.95,
        ["radio", "antenna", "transceiver", "two-way"],
    ),
    (
        "silo_front_display",
        0.95,
        ["display", "screen", "monitor", "television", "tv"],
    ),
    (
        "silo_front_kit",
        0.95,
        ["all primary components", "primary components", "kit"],
    ),
    (
        "silo_assortment",
        0.95,
        ["included pieces", "flat-lay", "flat lay", "assortment"],
    ),
    (
        "silo_front_system",
        0.95,
        ["complete system", "computer system", "desktop", "laptop", "server", "included accessories"],
    ),
    (
        "silo_front_case",
        0.95,
        ["protective case", "carrying case", "item inside", "case"],
    ),
    (
        "silo_front_skin",
        0.95,
        ["applied skin", "cosmetic wrap", "skin coverage", "wrap coverage", "skin", "wrap"],
    ),
    (
        "silo_alternate",
        0.94,
        ["three-quarter", "alternate angle", "different angle", "side angle", "right-facing", "configuration", "depth", "structure", "form"],
    ),
    (
        "feature_detail",
        0.94,
        ["feature detail", "hardware component", "control element"],
    ),
    (
        "silo_detail",
        0.94,
        ["close-up", "closeup", "hardware", "connector", "connectors", "ports", "controls", "component", "storage", "termination", "optical", "mechanical"],
    ),
    (
        "silo_back_detail",
        0.94,
        ["back detail", "network ports", "connections", "interfaces", "rear ports", "rear"],
    ),
    (
        "feature_detail",
        0.94,
        ["feature detail", "connection detail", "control detail", "hardware detail"],
    ),
    (
        "silo_front_in_pack",
        0.93,
        ["front in pack", "retail packaging", "package front", "boxed"],
    ),
    (
        "silo_assortment",
        0.92,
        ["included pieces", "included accessories", "flat-lay", "flat lay", "components"],
    ),
    (
        "graphic_instructions",
        0.92,
        ["setup instructions", "installation steps", "instructions", "setup", "how to", "special features"],
    ),
    (
        "lifestyle_active_play",
        0.94,
        ["active play", "dog playing", "cat playing", "motion blur", "pet active", "enrichment"],
    ),
    (
        "dimensions_toy_size",
        0.95,
        ["toy size", "size comparison", "familiar object", "dog size", "breed-sized"],
    ),
    (
        "graphic_breed_size_guide",
        0.94,
        ["breed guide", "weight guide", "breed size", "weight-based", "pet size", "dog size", "cat size"],
    ),
    (
        "dimensions_pet_size",
        0.94,
        ["interior dimensions", "exterior dimensions", "human scale", "breed size", "weight guide"],
    ),
    (
        "graphic_installation",
        0.94,
        ["wireless fence", "bark device", "installation", "setup configuration", "range coverage", "coverage distance"],
    ),
    (
        "graphic_compatibility",
        0.94,
        ["compatibility", "compatible", "pet types", "life stages", "small animal", "reptile"],
    ),
    (
        "graphic_safety_seal",
        0.94,
        ["safety seal", "quality certification", "brand seal", "natural rubber", "food-grade"],
    ),
    (
        "silo_components",
        0.94,
        ["included components", "kit contents"],
    ),
    (
        "graphic_color_material_range",
        0.93,
        ["color range", "material range", "colors", "variants", "lineup"],
    ),
    (
        "lifestyle_result",
        0.93,
        ["finished project", "completed project", "sewn", "knitted", "quilted", "sculpture", "diy", "wearable", "jewelry"],
    ),
    (
        "graphic_size_guide",
        0.92,
        ["stage guide", "age chart", "weight range", "feeding stage", "diaper size"],
    ),
    (
        "clinical_graphic",
        0.91,
        ["clinical", "accuracy", "range", "regulatory", "fda", "cleared", "status"],
    ),
    (
        "graphic_safety",
        0.91,
        ["skin-friendly", "hypoallergenic", "dermatologist", "age suitability", "rating"],
    ),
    (
        "graphic_certification",
        0.91,
        ["ap certified", "ap seal", "non-toxic", "nontoxic", "astm"],
    ),
    (
        "material_callout",
        0.91,
        ["bpa-free", "bpa free", "food-grade", "food grade", "safety certification", "jpma", "material claim"],
    ),
    (
        "graphic_safety",
        0.91,
        ["age suitability", "skin-friendly", "hypoallergenic", "dermatologist", "jpma", "bpa-free", "bpa free", "rating", "safety"],
    ),
    (
        "graphic_certification",
        0.9,
        [
            "certification",
            "certified",
            "badge",
            "badges",
            "nsf",
            "usp",
            "fda-cleared",
            "fda cleared",
            "official logo",
            "usda",
            "organic",
            "non-gmo",
            "kosher",
            "gluten free",
        ],
    ),
    (
        "silo_back_in_pack",
        0.9,
        ["back in pack", "side in pack", "package back", "pack back", "pack side", "back label", "side label", "label details", "no plunge"],
    ),
    (
        "graphic_ingredients",
        0.9,
        ["primary ingredients", "no artificial", "additives", "additive", "warning", "directions", "label"],
    ),
    (
        "graphic_supplement_fact",
        0.9,
        ["supplement facts", "drug facts", "usage instructions", "no glare", "active ingredient", "dosage", "serving size"],
    ),
    (
        "graphic_nutrition",
        0.9,
        ["nutrition facts", "nutrition", "calories", "serving size", "daily value"],
    ),
    (
        "graphic_ingredients",
        0.9,
        ["ingredient list", "ingredients", "ingredient", "contains", "allergen", "warning", "directions", "active ingredient", "label"],
    ),
    (
        "graphic_guarantee",
        0.88,
        ["satisfaction guaranteed", "private brand guarantee", "money back", "guarantee"],
    ),
    (
        "silo_detail",
        0.88,
        ["design detail", "material weave", "weave", "fill", "surface finish"],
    ),
    (
        "results_graphic",
        0.87,
        ["before and after", "visible results", "results", "before", "after", "finish", "effect", "applied", "disclaimer"],
    ),
    (
        "dimensions_assembled",
        0.87,
        ["assembled dimensions", "callout measurements", "configuration", "convertible"],
    ),
    (
        "graphic_size_guide",
        0.86,
        ["size guide", "size chart", "stage guide", "age chart", "weight range", "feeding stage", "diaper size", "fit guide"],
    ),
    (
        "in_use_dimensions",
        0.86,
        ["in-use dimensions", "in use dimensions", "usable size", "perfect size", "size for kids", "scale", "fit", "position"],
    ),
    (
        "dimensions",
        0.86,
        ["dimensions", "height", "width", "depth", "size chart", "scale", "inches", "in."],
    ),
    (
        "graphic_count",
        0.85,
        ["pack count", "bold typography", "count", "value", "packs", "rolls", "sheets", "pieces", "ct"],
    ),
    (
        "graphic_count_size",
        0.85,
        ["sheet size", "paper weight", "pack count", "gsm", "lb"],
    ),
    (
        "graphic_age_skill",
        0.85,
        ["skill level", "recommended age", "beginner", "intermediate", "advanced"],
    ),
    (
        "graphic_color_material_range",
        0.85,
        ["color range", "material range", "colors", "variants", "lineup"],
    ),
    (
        "silo_front_food_formula",
        0.85,
        ["age stage", "baby formula", "baby food", "formula", "food", "stage", "flavor", "claims", "quantity", "multipack"],
    ),
    (
        "dimensions_pet_size",
        0.85,
        ["interior dimensions", "exterior dimensions", "human scale", "breed", "weight"],
    ),
    (
        "dimensions_toy_size",
        0.85,
        ["toy size", "size comparison", "familiar object", "dog size", "breed-sized"],
    ),
    (
        "graphic_installation",
        0.85,
        ["wireless fence", "bark device", "installation", "setup", "configuration", "range", "coverage", "distance"],
    ),
    (
        "graphic_compatibility",
        0.85,
        ["compatibility", "compatible", "pet types", "breeds", "life stages", "small animal", "reptile"],
    ),
    (
        "graphic_use_occasion",
        0.85,
        ["use occasion", "ideal use", "iconographic", "travel", "home", "outdoor", "context"],
    ),
    (
        "graphic_safety_seal",
        0.85,
        ["safety seal", "quality certification", "brand seal", "natural rubber", "food-grade", "non-toxic", "bpa-free"],
    ),
    (
        "lifestyle_active_play",
        0.85,
        ["active play", "dog playing", "cat playing", "motion blur", "pet active", "enrichment"],
    ),
    (
        "lifestyle_alternate",
        0.85,
        ["alternate lifestyle", "different setting", "alternate use", "second room"],
    ),
    (
        "lifestyle_caregiver",
        0.85,
        ["caregiver", "changing", "diaper", "diverse"],
    ),
    (
        "lifestyle_sleep_nursery",
        0.85,
        ["nursery", "crib", "bassinet", "calm", "warm"],
    ),
    (
        "lifestyle_in_use_mattress",
        0.85,
        ["mattress", "sleep", "loft", "drape", "coverage"],
    ),
    (
        "lifestyle_result",
        0.85,
        ["finished project", "completed project", "sewn", "knitted", "quilted", "sculpture", "diy", "wearable", "jewelry"],
    ),
    (
        "swatch_graphic",
        0.84,
        ["swatch", "shades", "shade", "color range", "undertone", "skin tone", "pigment"],
    ),
    (
        "feature_graphic",
        0.82,
        [
            "feature",
            "callout",
            "gluten free",
            "organic",
            "protein",
            "no artificial",
            "benefits",
            "calories",
            "non-gmo",
            "whole grain",
        ],
    ),
    (
        "lifestyle_in_use",
        0.82,
        [
            "lifestyle",
            "in-use",
            "in use",
            "recipe",
            "serving suggestion",
            "prepared",
            "plated",
            "kitchen",
            "table",
            "meal",
        ],
    ),
    (
        "graphic_instructions",
        0.83,
        ["assembly", "setup", "gameplay instructions", "gameplay guide"],
    ),
    (
        "graphic_packaging",
        0.82,
        ["package directions", "how to use", "packaging", "directions", "usage", "step", "diagram", "instructions"],
    ),
    (
        "graphic_whats_included",
        0.82,
        ["what's included", "whats included", "included components", "accessories", "contents"],
    ),
    (
        "silo_front_in_pack",
        0.81,
        ["front in pack", "in-package", "package front", "front pack", "in package", "in pack", "packaging", "retail packaging", "retail-ready", "customer-facing", "box", "boxed"],
    ),
    (
        "silo_front_full_item",
        0.8,
        ["entire item", "full item", "whole item"],
    ),
    (
        "silo_front_profile",
        0.8,
        ["front profile", "mattress profile", "profile", "thickness", "shape"],
    ),
    (
        "silo_angle",
        0.8,
        ["silo angle", "angled"],
    ),
    (
        "silo_out_of_pack_whats_included",
        0.8,
        ["out of pack", "flat-lay", "flat lay", "kit components"],
    ),
    (
        "silo_components",
        0.8,
        ["included components", "kit contents", "what's included", "whats included"],
    ),
    (
        "silo_front_clinical",
        0.73,
        ["clinical front", "factual front", "front clinical", "front factual"],
    ),
    (
        "silo_back",
        0.8,
        ["silo back", "back panel", "back of product"],
    ),
    (
        "silo_alt_in_pack",
        0.8,
        ["alt in pack", "in pack", "in-pack", "side in pack", "back in pack", "alternate in pack", "pack side", "pack back"],
    ),
    (
        "graphic_texture",
        0.8,
        ["surface detail", "material", "texture graphic"],
    ),
    (
        "silo_detail",
        0.8,
        ["detail", "sliced", "halved", "cut open", "inside", "interior", "texture", "cross section", "material", "weave", "fill", "cover", "surface", "finish"],
    ),
    (
        "silo_assortment",
        0.8,
        ["assortment", "components", "pieces", "play pieces", "included", "small pieces"],
    ),
    (
        "silo_alternate",
        0.78,
        ["back", "side", "alternate", "supplement panel", "side panel", "rear", "nutrition side"],
    ),
    (
        "silo_single",
        0.76,
        ["single unit", "one piece", "single", "individual", "unit", "can", "bottle", "pack item", "one can", "one bottle"],
    ),
    (
        "silo_front",
        0.72,
        ["white background", "front", "main", "hero", "primary"],
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _empty_guide() -> dict[str, Any]:
    return json.loads(json.dumps(_EMPTY_GUIDE))


def _normalize_category_key(value: str) -> str:
    normalized = normalize_product_type(value)
    return normalized.replace(" ", "_")


def _guide_path_for_category(category_key: str) -> Path | None:
    normalized = _normalize_category_key(category_key)
    alias = normalized.replace("_", " ")
    if normalized in _FOOD_BEVERAGE_ALIASES or alias in _FOOD_BEVERAGE_ALIASES:
        return _repo_root() / "config" / "image_guides" / "food_beverageimg.json"
    if normalized in _BEAUTY_ALIASES or alias in _BEAUTY_ALIASES:
        return _repo_root() / "config" / "image_guides" / "beautyimg.json"
    if normalized in _HEALTH_PERSONAL_CARE_ALIASES or alias in _HEALTH_PERSONAL_CARE_ALIASES:
        return _repo_root() / "config" / "image_guides" / "healthpersonalimg.json"
    if normalized in _TOYS_ALIASES or alias in _TOYS_ALIASES:
        return _repo_root() / "config" / "image_guides" / "toysimg.json"
    if normalized in _HOUSEHOLD_CLEAN_ALIASES or alias in _HOUSEHOLD_CLEAN_ALIASES:
        return _repo_root() / "config" / "image_guides" / "householdcleanimg.json"
    if normalized in _FURNITURE_ALIASES or alias in _FURNITURE_ALIASES:
        return _repo_root() / "config" / "image_guides" / "furnitureimg.json"
    if normalized in _BABY_ALIASES or alias in _BABY_ALIASES:
        return _repo_root() / "config" / "image_guides" / "babyimg.json"
    if normalized in _ARTS_CRAFTS_ALIASES or alias in _ARTS_CRAFTS_ALIASES:
        return _repo_root() / "config" / "image_guides" / "artscraftsimg.json"
    if normalized in _ANIMALS_ALIASES or alias in _ANIMALS_ALIASES:
        return _repo_root() / "config" / "image_guides" / "animalsimg.json"
    if normalized in _ELECTRONICS_ALIASES or alias in _ELECTRONICS_ALIASES:
        return _repo_root() / "config" / "image_guides" / "electronicsimg.json"
    if normalized in _MEDIA_ALIASES or alias in _MEDIA_ALIASES:
        return _repo_root() / "config" / "image_guides" / "mediaimg.json"
    if normalized in _SEASONAL_ALIASES or alias in _SEASONAL_ALIASES:
        return _repo_root() / "config" / "image_guides" / "seasonalimg.json"
    if normalized in _SPORTS_OUTDOORS_ALIASES or alias in _SPORTS_OUTDOORS_ALIASES:
        return _repo_root() / "config" / "image_guides" / "sportsoutdoorsimg.json"
    return None


def load_image_guide(category_key: str) -> dict:
    """Load a category image guide, returning an empty guide if unavailable."""
    path = _guide_path_for_category(category_key or "")
    if path is None or not path.exists():
        return _empty_guide()
    try:
        with path.open("r", encoding="utf-8") as handle:
            guide = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _empty_guide()
    return guide if isinstance(guide, dict) else _empty_guide()


def normalize_product_type(value: str) -> str:
    """Normalize product type text enough for deterministic fallback matching."""
    if value is None:
        return ""
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s*-\s*", " ", text)
    punctuation = string.punctuation.replace("&", "").replace("-", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"\band\b", " and ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_image_guide_page(category_key: str, product_type: str) -> dict | None:
    """Return the matching image-guide page plus page_key for a product type."""
    guide = load_image_guide(category_key)
    pages = guide.get("pages") or {}
    product_type_index = guide.get("product_type_index") or {}
    if not isinstance(pages, dict) or not isinstance(product_type_index, dict):
        return None

    page_keys = product_type_index.get(product_type)
    if not page_keys:
        normalized_product_type = normalize_product_type(product_type)
        normalized_index = {
            normalize_product_type(index_product_type): index_page_key
            for index_product_type, index_page_key in product_type_index.items()
        }
        page_keys = normalized_index.get(normalized_product_type)

    if isinstance(page_keys, str):
        candidate_page_keys = [page_keys]
    elif isinstance(page_keys, list):
        candidate_page_keys = [page_key for page_key in page_keys if isinstance(page_key, str)]
    else:
        candidate_page_keys = []

    page_key = next((candidate for candidate in candidate_page_keys if isinstance(pages.get(candidate), dict)), None)
    page = pages.get(page_key) if page_key else None
    if not isinstance(page, dict):
        return None
    result = dict(page)
    result["page_key"] = page_key
    if len(candidate_page_keys) > 1:
        result["candidate_page_keys"] = candidate_page_keys
        result["candidate_pages"] = [
            {"page_key": candidate, **pages[candidate]}
            for candidate in candidate_page_keys
            if isinstance(pages.get(candidate), dict)
        ]
    return result


def _image_text(image: dict, position: int | None) -> tuple[str, int | None]:
    parts: list[str] = []
    for key in ("filename", "alt_text", "url", "text", "ocr_text", "title"):
        value = image.get(key) if isinstance(image, dict) else None
        if value:
            parts.append(str(value))
    image_position = position if position is not None else image.get("position") if isinstance(image, dict) else None
    try:
        parsed_position = int(image_position) if image_position is not None else None
    except (TypeError, ValueError):
        parsed_position = None
    return " ".join(parts).lower(), parsed_position


def classify_image_slot(image: dict, position: int | None = None) -> dict:
    """Classify one PDP image into a likely image-guide slot using metadata only."""
    text, image_position = _image_text(image or {}, position)
    reasons: list[str] = []

    for slot, confidence, hints in _SLOT_RULES:
        for hint in hints:
            if hint in text:
                reasons.append(f"matched {hint} text")
                return {"slot": slot, "confidence": confidence, "reasons": reasons}

    if image_position == 1:
        return {
            "slot": "silo_front",
            "confidence": 0.6,
            "reasons": ["used carousel position 1 fallback"],
        }

    return {"slot": None, "confidence": 0.0, "reasons": ["no slot metadata matched"]}


def _unique_in_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _recommendation_for_slot(guide: dict, slot: str) -> str:
    slot_definition = (guide.get("slot_definitions") or {}).get(slot) or {}
    recommendation = slot_definition.get("recommendation")
    if recommendation:
        return str(recommendation)
    label = slot_definition.get("label") or slot.replace("_", " ")
    return f"Add a {label} image to cover the expected PDP carousel slot."


def audit_pdp_images_against_guide(category_key: str, product_type: str, images: list[dict]) -> dict:
    """Compare PDP image metadata against the expected image-guide slots."""
    guide = load_image_guide(category_key)
    page = get_image_guide_page(category_key, product_type)
    if not page:
        return {
            "matched": False,
            "reason": "No image guide page found for product type",
            "product_type": product_type,
        }

    classifications = [
        classify_image_slot(image, index + 1)
        for index, image in enumerate(images or [])
        if isinstance(image, dict)
    ]
    detected_slots = _unique_in_order(
        [classification["slot"] for classification in classifications if classification.get("slot")]
    )
    expected_slots = page.get("required_slots") or []
    detected_counts: dict[str, int] = {}
    for classification in classifications:
        slot = classification.get("slot")
        if slot:
            detected_counts[slot] = detected_counts.get(slot, 0) + 1
    missing_required_slots: list[str] = []
    for slot in expected_slots:
        if detected_counts.get(slot, 0) > 0:
            detected_counts[slot] -= 1
        else:
            missing_required_slots.append(slot)
    additional_recommendations = list(page.get("additional_recommendations") or [])

    recommendation_bullets: list[str] = [
        _recommendation_for_slot(guide, slot) for slot in missing_required_slots
    ]
    for recommendation in additional_recommendations:
        if len(recommendation_bullets) >= 5:
            break
        recommendation_bullets.append(str(recommendation))

    return {
        "matched": True,
        "page_key": page.get("page_key"),
        "candidate_page_keys": page.get("candidate_page_keys"),
        "candidate_pages": page.get("candidate_pages"),
        "page_display_name": page.get("display_name"),
        "product_type": product_type,
        "expected_slots": expected_slots,
        "detected_slots": detected_slots,
        "missing_required_slots": missing_required_slots,
        "additional_recommendations": additional_recommendations,
        "optional_recommendations": additional_recommendations,
        "recommendation_bullets": recommendation_bullets[:5],
        "image_classifications": classifications,
    }
