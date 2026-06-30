"""Guide normalization and identity resolution for the new strategic PPTX path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.audit_helpers.image_guides import (
    get_image_guide_page,
    load_image_guide,
    normalize_product_type,
    resolve_image_guide_category,
)


STYLE_GUIDE_FILES: dict[str, str] = {
    "food_beverage": "food_beverage.json",
    "beauty": "beauty.json",
    "health_personal_care": "healthpersonal.json",
    "animals": "animals.json",
    "electronics": "electronics.json",
}

IMAGE_GUIDE_FILES: dict[str, str] = {
    "food_beverage": "food_beverageimg.json",
    "beauty": "beautyimg.json",
    "health_personal_care": "healthpersonalimg.json",
    "animals": "animalsimg.json",
    "electronics": "electronicsimg.json",
}

CATEGORY_VOCABULARY: dict[str, dict[str, str]] = {
    "food_beverage": {
        "benefit": "benefit communication",
        "ingredient": "ingredient communication",
        "usage": "breakfast and snack storytelling",
        "education": "nutrition and usage education",
        "visual": "appetite-led visual identity",
        "navigation": "pantry shelf navigation",
    },
    "beauty": {
        "benefit": "benefit communication",
        "ingredient": "formula communication",
        "usage": "routine storytelling",
        "education": "regimen education",
        "visual": "cohesive beauty presentation",
        "navigation": "concern-based navigation",
    },
    "health_personal_care": {
        "benefit": "wellness benefit communication",
        "ingredient": "active support communication",
        "usage": "symptom and routine guidance",
        "education": "wellness shopper education",
        "visual": "clinical visual clarity",
        "navigation": "wellness shelf navigation",
    },
    "animals": {
        "benefit": "pet benefit communication",
        "ingredient": "nutrition communication",
        "usage": "feeding and care guidance",
        "education": "life-stage shopper education",
        "visual": "care-led visual identity",
        "navigation": "life-stage navigation",
    },
    "electronics": {
        "benefit": "performance communication",
        "ingredient": "compatibility detail",
        "usage": "setup and use-case guidance",
        "education": "device shopper education",
        "visual": "technical visual clarity",
        "navigation": "device-use navigation",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return normalize_product_type(_safe_text(value))


def _slug(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


def _unique_text(values: list[Any], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _safe_text(value)
        key = _norm(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _style_path(category_key: str) -> Path | None:
    filename = STYLE_GUIDE_FILES.get(resolve_category_key(category_key))
    if not filename:
        return None
    path = _repo_root() / "config" / "style_guides" / filename
    return path if path.exists() else None


def _image_path(category_key: str) -> Path | None:
    filename = IMAGE_GUIDE_FILES.get(resolve_category_key(category_key))
    if not filename:
        return None
    path = _repo_root() / "config" / "image_guides" / filename
    return path if path.exists() else None


def resolve_category_key(value: str) -> str:
    """Resolve guide aliases and common evidence strings to supported keys."""
    blob = _norm(value)
    image_key = resolve_image_guide_category(value or "")
    if image_key in STYLE_GUIDE_FILES:
        return image_key
    if any(term in blob for term in ("food", "beverage", "pantry", "spread", "snack", "nutrition", "breakfast")):
        return "food_beverage"
    if any(term in blob for term in ("skin care", "hair care", "beauty", "makeup", "cosmetic")):
        return "beauty"
    if any(term in blob for term in ("pet", "dog", "cat", "animal", "feeding")):
        return "animals"
    if any(term in blob for term in ("electronics", "device", "compatibility", "laptop", "phone", "audio", "camera")):
        return "electronics"
    if any(term in blob for term in ("health", "wellness", "personal care", "vitamin", "supplement", "symptom")):
        return "health_personal_care"
    return image_key if image_key in STYLE_GUIDE_FILES else ""


def load_style_title_guide(category_key: str) -> dict[str, Any]:
    """Load one existing style/title guide JSON."""
    path = _style_path(category_key)
    return _load_json(path) if path else {}


def load_image_story_guide(category_key: str) -> dict[str, Any]:
    """Load one existing image guide JSON through the canonical image helper."""
    return load_image_guide(resolve_category_key(category_key))


def normalize_style_title_guide(category_key: str) -> dict[str, Any]:
    """Normalize style/title guide families and product types into one shape."""
    resolved_key = resolve_category_key(category_key)
    guide = load_style_title_guide(resolved_key)
    path = _style_path(resolved_key)
    notes: list[str] = []
    families: dict[str, Any] = {}
    for family_key, family in (guide.get("families") or {}).items():
        if not isinstance(family, dict):
            continue
        product_types: dict[str, Any] = {}
        for product_key, product in (family.get("product_types") or {}).items():
            if not isinstance(product, dict):
                continue
            display = _safe_text(product.get("display_name")) or str(product_key).replace("_", " ").title()
            formula = _unique_text(_as_list(product.get("formula")))
            attributes = _unique_text(_as_list(product.get("attributes")))
            title_keywords = _unique_text(_as_list(product.get("title_keywords")))
            context_keywords = _unique_text(_as_list(product.get("context_keywords")))
            product_types[str(product_key)] = {
                "product_type_key": str(product_key),
                "product_type_display": display,
                "aliases": _unique_text(_as_list(product.get("aliases"))),
                "attribute_cues": attributes,
                "benefit_cues": _unique_text([*attributes, *context_keywords]),
                "usage_occasion_cues": context_keywords,
                "education_cues": _unique_text([*formula, *attributes]),
                "comparison_cues": formula,
                "recommended_title_priorities": formula,
                "title_keywords": title_keywords,
                "context_keywords": context_keywords,
                "negative_keywords": _unique_text(_as_list(product.get("negative_keywords"))),
                "raw": product,
            }
        families[str(family_key)] = {
            "family_key": str(family_key),
            "family_display": _safe_text(family.get("display_name")) or str(family_key).replace("_", " ").title(),
            "product_types": product_types,
        }
    if not guide:
        notes.append("style guide was not available or could not be parsed")
    return {
        "category_key": resolved_key,
        "category_display": _safe_text(guide.get("category")) or resolved_key.replace("_", " ").title(),
        "families": families,
        "source_guides_used": [_relative_path(path)] if path else [],
        "normalization_notes": notes,
    }


def normalize_image_story_guide(category_key: str) -> dict[str, Any]:
    """Normalize image guide pages into visual/story priorities."""
    resolved_key = resolve_category_key(category_key)
    guide = load_image_story_guide(resolved_key)
    path = _image_path(resolved_key)
    notes: list[str] = []
    pages: dict[str, Any] = {}
    slot_definitions = guide.get("slot_definitions") or {}
    for page_key, page in (guide.get("pages") or {}).items():
        if not isinstance(page, dict):
            continue
        required_slots = _unique_text(_as_list(page.get("required_slots")), limit=20)
        visual_priorities: list[str] = []
        story_cues: list[str] = []
        for slot in required_slots:
            slot_data = slot_definitions.get(slot) if isinstance(slot_definitions, dict) else {}
            label = _safe_text((slot_data or {}).get("label")) or slot.replace("_", " ").title()
            guidance = _safe_text((slot_data or {}).get("guidance") or (slot_data or {}).get("description"))
            visual_priorities.append(label)
            if guidance:
                story_cues.append(guidance)
        pages[str(page_key)] = {
            "page_key": str(page_key),
            "page_display": _safe_text(page.get("display_name")) or str(page_key).replace("_", " ").title(),
            "product_types": _unique_text(_as_list(page.get("product_types")), limit=20),
            "required_slots": required_slots,
            "image_story_cues": _unique_text(story_cues, limit=10),
            "recommended_visual_priorities": _unique_text(
                [*visual_priorities, *_as_list(page.get("additional_recommendations"))],
                limit=12,
            ),
            "module_story_cues": _unique_text(_as_list(page.get("additional_recommendations")), limit=8),
        }
    if not guide or not pages:
        notes.append("image guide was not available or did not expose normalized pages")
    return {
        "category_key": resolved_key,
        "category_display": _safe_text(guide.get("category")) or resolved_key.replace("_", " ").title(),
        "guide_key": _safe_text(guide.get("guide_key")),
        "pages": pages,
        "product_type_index": guide.get("product_type_index") or {},
        "source_guides_used": [_relative_path(path)] if path else [],
        "normalization_notes": notes,
    }


def load_normalized_guides(category_key: str) -> dict[str, Any]:
    """Load and combine normalized style/title and image guide data."""
    resolved_key = resolve_category_key(category_key)
    style = normalize_style_title_guide(resolved_key)
    image = normalize_image_story_guide(resolved_key)
    return {
        "category_key": resolved_key,
        "category_display": style.get("category_display") or image.get("category_display") or resolved_key.replace("_", " ").title(),
        "style": style,
        "image": image,
        "source_guides_used": _unique_text(
            [*(style.get("source_guides_used") or []), *(image.get("source_guides_used") or [])],
            limit=8,
        ),
        "normalization_notes": [
            *(style.get("normalization_notes") or []),
            *(image.get("normalization_notes") or []),
        ],
    }


def _record_value(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        current: Any = record
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current not in (None, "", [], {}):
            return current
    return default


def _record_text(record: dict[str, Any]) -> str:
    parts = [
        _record_value(record, "product_title", "title", "productTitle"),
        _record_value(record, "description_body", "description", "shortDescription"),
        _record_value(record, "category", "categoryPathName"),
        _record_value(record, "product_type", "subcategory", "productType"),
        _record_value(record, "brand", "brandName"),
        *_as_list(_record_value(record, "description_bullets", "Description Bullets", default=[])),
        *_as_list(_record_value(record, "key_features", "Key Features", default=[])),
        *_as_list(record.get("categoryNavigation") if isinstance(record, dict) else []),
    ]
    return " ".join(_safe_text(part) for part in parts if _safe_text(part)).lower()


def _match_style_product_type(
    normalized_style: dict[str, Any],
    product_type: str,
    text_blob: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any], int]:
    best = ("", product_type, {}, {}, 0)
    wanted = _norm(product_type)
    for family_key, family in (normalized_style.get("families") or {}).items():
        for product_key, product in (family.get("product_types") or {}).items():
            score = 0
            display = _safe_text(product.get("product_type_display")) or str(product_key).replace("_", " ").title()
            terms = [
                display,
                *_as_list(product.get("aliases")),
                *_as_list(product.get("title_keywords")),
                *_as_list(product.get("context_keywords")),
            ]
            normalized_terms = [_norm(term) for term in terms if _safe_text(term)]
            if wanted and (wanted == _norm(display) or wanted in normalized_terms):
                score += 10
            for term in normalized_terms:
                if term and term in text_blob:
                    score += 2
            for term in [_norm(value) for value in _as_list(product.get("negative_keywords"))]:
                if term and term in text_blob:
                    score -= 6
            if score > best[4]:
                best = (str(family_key), display, family, product, score)
    return best


def _clean_phrase(value: str, fallback: str) -> str:
    text = _safe_text(value).replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or fallback


def build_identity_phrases(
    *,
    category_display: str,
    family_display: str,
    product_type_display: str,
) -> dict[str, str]:
    """Build display-safe phrase variants for later slide/cue prompts."""
    category = _clean_phrase(category_display, "category")
    family = _clean_phrase(family_display, "")
    product = _clean_phrase(product_type_display, family or category)
    if family and _norm(family) != _norm(product):
        combined = f"the {family} and {product} category"
        combined_alt = f"the {product} segment within {category}"
    else:
        combined = f"the {product} category"
        combined_alt = f"the {product} space within {category}"
    return {
        "combined_category_phrase": combined,
        "combined_category_phrase_alt": combined_alt,
        "shopping_context_phrase": f"{product} shopping journey",
        "product_type_focus_phrase": f"{product} positioning",
    }


def resolve_strategic_identity(
    records: list[dict[str, Any]],
    *,
    fallback_category: str = "",
    fallback_product_type: str = "",
) -> dict[str, Any]:
    """Resolve category/family/product type plus normalized guide context."""
    usable = [record for record in records or [] if isinstance(record, dict)]
    category = next(
        (
            _safe_text(_record_value(record, "category", "categoryPathName"))
            for record in usable
            if _safe_text(_record_value(record, "category", "categoryPathName"))
        ),
        fallback_category,
    )
    product_type = next(
        (
            _safe_text(_record_value(record, "product_type", "subcategory", "productType"))
            for record in usable
            if _safe_text(_record_value(record, "product_type", "subcategory", "productType"))
        ),
        fallback_product_type or category or "category",
    )
    text_blob = " ".join(_record_text(record) for record in usable)
    category_key = resolve_category_key(" ".join([category, product_type, text_blob]))
    guides = load_normalized_guides(category_key or category or product_type)
    family_key, product_display, family_data, product_data, score = _match_style_product_type(
        guides.get("style", {}),
        product_type,
        text_blob,
    )
    product_display = product_display or product_type or category or "category"
    family_display = _safe_text(family_data.get("family_display")) if family_data else ""
    category_display = guides.get("category_display") or _clean_phrase(category, category_key.replace("_", " ").title())

    image_page = get_image_guide_page(category_key, product_display) or get_image_guide_page(category_key, product_type) or {}
    normalized_image_page = {}
    if image_page:
        normalized_image_page = (guides.get("image", {}).get("pages") or {}).get(image_page.get("page_key"), {})

    phrases = build_identity_phrases(
        category_display=category_display,
        family_display=family_display,
        product_type_display=product_display,
    )
    vocab = CATEGORY_VOCABULARY.get(category_key, {})
    source_guides = guides.get("source_guides_used") or []
    notes = list(guides.get("normalization_notes") or [])
    if not family_key:
        notes.append("style product type was resolved from evidence fallback")

    return {
        "category_key": category_key,
        "category_display": category_display,
        "family_key": family_key,
        "family_display": family_display,
        "product_type_key": _slug(product_display),
        "product_type_display": product_display,
        **phrases,
        "attribute_cues": _unique_text(_as_list(product_data.get("attribute_cues")), limit=10),
        "benefit_cues": _unique_text([vocab.get("benefit", ""), *_as_list(product_data.get("benefit_cues"))], limit=10),
        "usage_occasion_cues": _unique_text([vocab.get("usage", ""), *_as_list(product_data.get("usage_occasion_cues"))], limit=10),
        "education_cues": _unique_text([vocab.get("education", ""), *_as_list(product_data.get("education_cues"))], limit=10),
        "comparison_cues": _unique_text(_as_list(product_data.get("comparison_cues")), limit=10),
        "image_story_cues": _unique_text(_as_list(normalized_image_page.get("image_story_cues")), limit=10),
        "module_story_cues": _unique_text(_as_list(normalized_image_page.get("module_story_cues")), limit=10),
        "recommended_visual_priorities": _unique_text(_as_list(normalized_image_page.get("recommended_visual_priorities")), limit=12),
        "recommended_title_priorities": _unique_text(_as_list(product_data.get("recommended_title_priorities")), limit=12),
        "source_guides_used": source_guides,
        "normalization_notes": notes,
        "style_guide_path": next((path for path in source_guides if "style_guides" in path), ""),
        "image_guide_path": next((path for path in source_guides if "image_guides" in path), ""),
        "style_product_type_score": score,
        "style_product_type": product_data.get("raw") or product_data,
        "image_page": image_page,
        "normalized_guides": guides,
    }


def identity_debug_payload(identity: dict[str, Any]) -> dict[str, Any]:
    """Return serializable resolver metadata for export debug structures."""
    hidden = {"style_product_type", "image_page", "normalized_guides"}
    return {key: value for key, value in identity.items() if key not in hidden}
