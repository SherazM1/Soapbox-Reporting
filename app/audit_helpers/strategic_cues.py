"""Cue-based strategic bullet helpers for the new audit PowerPoint path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.audit_helpers.image_guides import (
    audit_pdp_images_against_guide,
    get_image_guide_page,
    normalize_product_type,
    resolve_image_guide_category,
)


CUE_SLIDE_PRIORITY: dict[str, tuple[str, ...]] = {
    "slide2": (
        "product_positioning",
        "benefit_communication",
        "discoverability",
        "conversion_guidance",
        "review_or_trust_signals",
        "usage_storytelling",
    ),
    "slide3": (
        "discoverability",
        "keyword_alignment",
        "review_or_trust_signals",
        "assortment_segmentation",
    ),
    "slide4": (
        "product_positioning",
        "shopper_education",
        "usage_storytelling",
        "ingredient_or_formula_communication",
        "pack_or_spec_detail",
        "conversion_guidance",
        "visual_identity",
    ),
    "slide5": (
        "category_grouping",
        "discovery_pathways",
        "shopper_education",
        "usage_storytelling",
        "cross_category_navigation",
        "visual_identity",
        "conversion_guidance",
    ),
}

STYLE_GUIDE_FILES: dict[str, str] = {
    "food_beverage": "food_beverage.json",
    "beauty": "beauty.json",
    "health_personal_care": "healthpersonal.json",
    "animals": "animals.json",
    "electronics": "electronics.json",
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

CUE_NOUNS: dict[str, str] = {
    "product_positioning": "product positioning",
    "benefit_communication": "benefit communication",
    "ingredient_or_formula_communication": "ingredient communication",
    "shopper_education": "shopper education",
    "usage_storytelling": "usage storytelling",
    "visual_identity": "visual identity",
    "pack_or_spec_detail": "pack and nutrition detail",
    "keyword_alignment": "keyword alignment",
    "discoverability": "category discoverability",
    "assortment_segmentation": "assortment segmentation",
    "category_grouping": "category grouping",
    "discovery_pathways": "discovery pathways",
    "cross_category_navigation": "cross-category pathways",
    "review_or_trust_signals": "review and trust signals",
    "conversion_guidance": "conversion-focused guidance",
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


def _first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
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
        _first(record, "product_title", "title", "productTitle"),
        _first(record, "description_body", "description", "shortDescription"),
        _first(record, "category"),
        _first(record, "product_type", "subcategory"),
        _first(record, "brand", "brandName"),
        *_as_list(_first(record, "description_bullets", "key_features", default=[])),
        *_as_list(_first(record, "key_features", default=[])),
    ]
    return " ".join(_safe_text(part) for part in parts if _safe_text(part)).lower()


def _record_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _as_list(_first(record, "images", default=[])) if isinstance(item, dict)]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _style_guide_path(category_key: str) -> Path | None:
    filename = STYLE_GUIDE_FILES.get(category_key)
    if not filename:
        return None
    path = _repo_root() / "config" / "style_guides" / filename
    return path if path.exists() else None


def _load_style_guide(category_key: str) -> tuple[dict[str, Any], str]:
    path = _style_guide_path(category_key)
    if path is None:
        return {}, ""
    return _load_json(path), str(path.relative_to(_repo_root()))


def _infer_supported_category_key(text: str) -> str:
    blob = _norm(text)
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
    return ""


def _best_style_product_type(
    guide: dict[str, Any],
    product_type: str,
    text_blob: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any], int]:
    best = ("", product_type, {}, {}, 0)
    wanted = _norm(product_type)
    for family_key, family in (guide.get("families") or {}).items():
        if not isinstance(family, dict):
            continue
        for pt_key, pt_data in (family.get("product_types") or {}).items():
            if not isinstance(pt_data, dict):
                continue
            score = 0
            display = _safe_text(pt_data.get("display_name")) or pt_key.replace("_", " ").title()
            terms = [
                display,
                *_as_list(pt_data.get("aliases")),
                *_as_list(pt_data.get("title_keywords")),
                *_as_list(pt_data.get("context_keywords")),
            ]
            normalized_terms = [_norm(term) for term in terms if _safe_text(term)]
            if wanted and (wanted == _norm(display) or wanted in normalized_terms):
                score += 8
            for term in normalized_terms:
                if term and term in text_blob:
                    score += 2
            for term in [_norm(value) for value in _as_list(pt_data.get("negative_keywords"))]:
                if term and term in text_blob:
                    score -= 5
            if score > best[4]:
                best = (str(family_key), display, family, pt_data, score)
    return best


def resolve_strategic_identity(
    records: list[dict[str, Any]],
    *,
    fallback_category: str = "",
    fallback_product_type: str = "",
) -> dict[str, Any]:
    usable = [record for record in records if isinstance(record, dict)]
    category = next(
        (_safe_text(_first(record, "category")) for record in usable if _safe_text(_first(record, "category"))),
        fallback_category,
    )
    product_type = next(
        (
            _safe_text(_first(record, "product_type", "subcategory"))
            for record in usable
            if _safe_text(_first(record, "product_type", "subcategory"))
        ),
        fallback_product_type or category or "category",
    )
    text_blob = " ".join(_record_text(record) for record in usable)
    category_key = _infer_supported_category_key(" ".join([category, product_type, text_blob]))
    if not category_key:
        category_key = resolve_image_guide_category(category or product_type)
    style_guide, style_path = _load_style_guide(category_key)
    family_key, product_display, family_data, product_data, score = _best_style_product_type(
        style_guide,
        product_type,
        text_blob,
    )
    family_display = _safe_text(family_data.get("display_name")) if family_data else ""
    product_display = product_display or product_type or category or "category"
    category_display = _safe_text(style_guide.get("category")) or category or category_key.replace("_", " ").title()
    image_page = get_image_guide_page(category_key, product_display) or get_image_guide_page(category_key, product_type) or {}
    image_path = ""
    if category_key:
        image_path = f"config/image_guides/{category_key.replace('_care', '').replace('health_personal', 'healthpersonal')}img.json"
    category_clean = category_display
    product_clean = product_display
    combined = (
        f"the {family_display} and {product_clean} category"
        if family_display and _norm(family_display) != _norm(product_clean)
        else f"the {product_clean} segment"
    )
    return {
        "category_key": category_key,
        "category_display": category_display,
        "family_key": family_key,
        "family_display": family_display,
        "product_type_key": _slug(product_clean),
        "product_type_display": product_clean,
        "combined_category_phrase": combined,
        "combined_category_phrase_alt": f"the {product_clean} space",
        "shopping_context_phrase": f"{product_clean} shopping journey",
        "product_type_focus_phrase": f"{product_clean} positioning",
        "style_guide_path": style_path,
        "image_guide_path": image_path,
        "style_product_type_score": score,
        "style_product_type": product_data,
        "image_page": image_page,
    }


def _category_vocab(identity: dict[str, Any]) -> dict[str, str]:
    return CATEGORY_VOCABULARY.get(identity.get("category_key"), {})


def _cue_label(cue: str, identity: dict[str, Any]) -> str:
    vocab = _category_vocab(identity)
    if cue == "benefit_communication":
        return vocab.get("benefit", CUE_NOUNS[cue])
    if cue == "ingredient_or_formula_communication":
        return vocab.get("ingredient", CUE_NOUNS[cue])
    if cue == "usage_storytelling":
        return vocab.get("usage", CUE_NOUNS[cue])
    if cue == "shopper_education":
        return vocab.get("education", CUE_NOUNS[cue])
    if cue == "visual_identity":
        return vocab.get("visual", CUE_NOUNS[cue])
    if cue in {"category_grouping", "cross_category_navigation"}:
        return vocab.get("navigation", CUE_NOUNS[cue])
    return CUE_NOUNS.get(cue, cue.replace("_", " "))


def _text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _record_cues(record: dict[str, Any], identity: dict[str, Any]) -> dict[str, bool]:
    text = _record_text(record)
    images = _record_images(record)
    guide_audit = audit_pdp_images_against_guide(
        identity.get("category_key", ""),
        identity.get("product_type_display", ""),
        images,
    )
    missing_slots = set(guide_audit.get("missing_required_slots") or [])
    detected_slots = set(guide_audit.get("detected_slots") or [])
    reviews = _first(record, "reviews_summary", default={})
    rating_count = _first(record, "rating_count", "review_count", "reviews", default=None)
    if isinstance(reviews, dict):
        rating_count = rating_count or reviews.get("ratings_count") or reviews.get("review_count") or reviews.get("count")
    try:
        review_count = int(rating_count or 0)
    except (TypeError, ValueError):
        review_count = 0
    product_data = identity.get("style_product_type") or {}
    expected_terms = " ".join(
        _safe_text(value)
        for value in [
            *_as_list(product_data.get("attributes")),
            *_as_list(product_data.get("formula")),
            *_as_list(product_data.get("title_keywords")),
            *_as_list(product_data.get("context_keywords")),
        ]
    ).lower()
    return {
        "product_positioning": bool(identity.get("product_type_display")) and _norm(identity.get("product_type_display")) in _norm(text),
        "benefit_communication": _text_has_any(text, ("benefit", "protein", "hydrating", "moisture", "support", "performance", "organic", "gluten free", "compatible")),
        "ingredient_or_formula_communication": _text_has_any(text + " " + expected_terms, ("ingredient", "formula", "nutrition", "active", "protein", "compatibility", "spec")),
        "shopper_education": _text_has_any(text, ("how to", "use", "routine", "directions", "guide", "education", "instructions")) or bool({"feature_graphic", "graphic_instructions"} & detected_slots),
        "usage_storytelling": _text_has_any(text, ("breakfast", "snack", "recipe", "routine", "occasion", "setup", "feeding", "care")) or "lifestyle_in_use" in detected_slots,
        "visual_identity": len(images) >= 4 or bool(detected_slots),
        "pack_or_spec_detail": _text_has_any(text, ("nutrition", "ingredients", "size", "count", "compatibility", "dosage", "spec")) or bool({"graphic_nutrition", "graphic_ingredients", "dimensions"} & detected_slots),
        "review_or_trust_signals": review_count >= 25,
        "conversion_guidance": _text_has_any(text, ("compare", "choose", "benefit", "feature", "size", "count", "directions")) or len(missing_slots) <= 2,
        "discoverability": bool(identity.get("product_type_display")),
        "assortment_segmentation": _text_has_any(text, ("flavor", "variant", "size", "pack", "count", "type")),
        "category_grouping": bool(identity.get("family_display")),
        "discovery_pathways": len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 2,
        "cross_category_navigation": len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 3,
        "_guide_matched": bool(guide_audit.get("matched")),
        "_missing_slots": list(missing_slots),
        "_detected_slots": list(detected_slots),
    }


def aggregate_pdp_cues(
    records: list[dict[str, Any]],
    *,
    competitor_records: list[dict[str, Any]] | None = None,
    fallback_category: str = "",
    fallback_product_type: str = "",
) -> dict[str, Any]:
    usable = [record for record in records or [] if isinstance(record, dict)]
    competitors = [record for record in competitor_records or [] if isinstance(record, dict)]
    identity = resolve_strategic_identity(
        usable or competitors,
        fallback_category=fallback_category,
        fallback_product_type=fallback_product_type,
    )
    record_results = [_record_cues(record, identity) for record in usable]
    competitor_results = [_record_cues(record, identity) for record in competitors]
    total = max(1, len(record_results))
    comp_total = max(1, len(competitor_results))
    product_data = identity.get("style_product_type") or {}
    image_page = identity.get("image_page") or {}
    candidates: list[dict[str, Any]] = []
    for cue in sorted(set(CUE_NOUNS) | {"category_grouping", "discovery_pathways", "cross_category_navigation"}):
        coverage = sum(1 for result in record_results if result.get(cue)) / total
        comp_coverage = sum(1 for result in competitor_results if result.get(cue)) / comp_total if competitor_results else 0.0
        gap_ratio = 1.0 - coverage if record_results else 1.0
        classification = "context"
        if competitor_results and comp_coverage - coverage >= 0.35:
            classification = "pressure"
        elif gap_ratio >= 0.45:
            classification = "opportunity"
        elif coverage >= 0.55:
            classification = "strength"
        candidates.append(
            {
                "cue": cue,
                "classification": classification,
                "coverage_ratio": round(coverage, 2),
                "gap_ratio": round(gap_ratio, 2),
                "strength_ratio": round(coverage, 2),
                "competitive_delta": round(comp_coverage - coverage, 2),
                "consistency": "broad" if coverage >= 0.55 else "inconsistent" if record_results else "fallback",
                "confidence_tier": 2 if (product_data or image_page) else 1 if record_results else 4,
                "label": _cue_label(cue, identity),
            }
        )
    missing_slots = []
    detected_slots = []
    for result in record_results:
        missing_slots.extend(result.get("_missing_slots") or [])
        detected_slots.extend(result.get("_detected_slots") or [])
    return {
        "identity": identity,
        "candidate_cues": candidates,
        "debug": {
            "resolved_identity": {key: value for key, value in identity.items() if key not in {"style_product_type", "image_page"}},
            "guide_files_used": [value for value in [identity.get("style_guide_path"), identity.get("image_guide_path")] if value],
            "matched_guide_expectations": sorted(set(detected_slots)),
            "missing_guide_expectations": sorted(set(missing_slots)),
            "candidate_cues": candidates,
            "record_count": len(usable),
            "competitor_record_count": len(competitors),
        },
    }


def search_cue_context(
    search_term: str,
    products: list[dict[str, Any]],
    *,
    client_brand: str = "",
    side: str = "current",
) -> dict[str, Any]:
    records = [
        {
            "product_title": _safe_text(product.get("title") or product.get("productTitle") or product.get("name")),
            "brand": _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName")),
            "product_type": search_term,
            "category": search_term,
            "review_count": product.get("reviewCount") or product.get("reviews") or product.get("ratingCount"),
        }
        for product in products or []
        if isinstance(product, dict)
    ]
    context = aggregate_pdp_cues(records, fallback_product_type=search_term, fallback_category=search_term)
    candidates = context["candidate_cues"]
    brands = {
        _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName"))
        for product in products or []
        if isinstance(product, dict) and _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName"))
    }
    client_visible = any(client_brand and _norm(client_brand) == _norm(brand) for brand in brands)
    for candidate in candidates:
        if candidate["cue"] in {"discoverability", "keyword_alignment", "assortment_segmentation"}:
            if side == "benchmark" and len(brands) >= 2:
                candidate["classification"] = "pressure"
                candidate["competitive_delta"] = max(candidate.get("competitive_delta", 0), 0.5)
            elif side == "current" and not client_visible:
                candidate["classification"] = "opportunity"
    context["debug"]["search_term"] = search_term
    context["debug"]["brand_count"] = len(brands)
    context["debug"]["client_visible"] = client_visible
    return context


def brand_shop_cue_context(record: dict[str, Any]) -> dict[str, Any]:
    modules = _as_list(_first(record, "modules", "structuredModules", "moduleDetails", "data.modules", default=[]))
    module_text = " ".join(
        _safe_text(_first(module, "type", "moduleType", "name", "heading", "title", "description"))
        for module in modules
        if isinstance(module, dict)
    )
    categories = _as_list(_first(record, "categoryNavigation", "categories", "data.categoryNavigation", "data.categories", default=[]))
    pseudo_record = {
        "product_title": module_text,
        "description_body": " ".join(_safe_text(value) for value in categories),
        "category": " ".join(_safe_text(value) for value in categories[:1]),
        "product_type": _safe_text(categories[0]) if categories else "Brand Shop",
        "links": _first(record, "destinationLinks", "links", "navigationLinks", "data.destinationLinks", "data.links", "data.navigationLinks", default=[]),
        "images": [],
    }
    context = aggregate_pdp_cues([pseudo_record])
    for candidate in context["candidate_cues"]:
        if candidate["cue"] in {"category_grouping", "discovery_pathways", "cross_category_navigation"} and len(categories) >= 2:
            candidate["classification"] = "strength"
            candidate["coverage_ratio"] = 1.0
        if candidate["cue"] in {"visual_identity", "shopper_education"} and modules:
            candidate["classification"] = "strength"
            candidate["coverage_ratio"] = 1.0
    context["debug"]["module_count"] = len(modules)
    context["debug"]["category_navigation_count"] = len(categories)
    return context


def _ordered_candidates(context: dict[str, Any], slide_key: str, preferred: tuple[str, ...]) -> list[dict[str, Any]]:
    priority = CUE_SLIDE_PRIORITY.get(slide_key, tuple(CUE_NOUNS))
    preferred_rank = {kind: index for index, kind in enumerate(preferred)}
    cue_rank = {cue: index for index, cue in enumerate(priority)}
    candidates = [
        candidate
        for candidate in context.get("candidate_cues", [])
        if candidate.get("cue") in priority
    ]
    return sorted(
        candidates,
        key=lambda item: (
            preferred_rank.get(item.get("classification"), len(preferred_rank)),
            cue_rank.get(item.get("cue"), 99),
            -float(item.get("coverage_ratio", 0)),
        ),
    )


def _bullet_text(candidate: dict[str, Any], identity: dict[str, Any], *, slide_key: str, side: str = "") -> str:
    label = candidate.get("label") or _cue_label(candidate.get("cue", ""), identity)
    product = identity.get("product_type_display") or "category"
    classification = candidate.get("classification")
    cue = candidate.get("cue")
    family = identity.get("family_display") or identity.get("category_display") or product
    slide4_scope = ""
    if slide_key == "slide4":
        if side.startswith("competitor_2"):
            slide4_scope = "Secondary benchmark PDP"
        elif side.startswith("competitor"):
            slide4_scope = "Benchmark PDP"
        else:
            slide4_scope = "Current PDP"
    if classification == "pressure":
        if slide_key == "slide3":
            return f"Competitive pressure around {label}"
        if slide_key == "slide4" and slide4_scope:
            return f"Competitive pressure on {product} PDP content"
        return f"Competitive pressure on {label}"
    if classification == "opportunity":
        if slide_key == "slide4" and slide4_scope:
            if cue in {"shopper_education", "usage_storytelling"}:
                return f"Opportunity to deepen {product} shopper education"
            return f"Opportunity to strengthen {product} PDP content"
        if cue in {"discoverability", "keyword_alignment"}:
            return f"Opportunity to improve {label}"
        if cue in {"shopper_education", "usage_storytelling"}:
            return f"Opportunity to deepen {label}"
        return f"Opportunity to strengthen {label}"
    if classification == "strength":
        if slide_key == "slide5":
            if cue == "category_grouping":
                return f"Structured {family} navigation"
            if cue == "cross_category_navigation":
                return f"Effective {family} cross-category pathways"
            if cue == "discovery_pathways":
                return f"Enhanced {family} discovery pathways"
        if slide_key == "slide4":
            if cue == "product_positioning":
                return (
                    f"{slide4_scope.replace(' PDP', '')} {product} positioning"
                    if side.startswith("competitor")
                    else f"Strong {product} positioning"
                )
            if cue == "ingredient_or_formula_communication":
                return (
                    f"{slide4_scope.replace(' PDP', '')} {product} ingredient communication"
                    if side.startswith("competitor")
                    else f"Clear {product} ingredient communication"
                )
            if cue == "shopper_education":
                return (
                    f"{slide4_scope.replace(' PDP', '')} {product} shopper education"
                    if side.startswith("competitor")
                    else f"Structured {product} shopper education"
                )
            if cue == "usage_storytelling":
                return (
                    f"{slide4_scope.replace(' PDP', '')} {product} usage storytelling"
                    if side.startswith("competitor")
                    else f"Balanced {product} usage storytelling"
                )
            if cue == "visual_identity":
                return f"{slide4_scope} visual identity"
        prefixes = {
            "product_positioning": "Strong",
            "benefit_communication": "Clear",
            "ingredient_or_formula_communication": "Clear",
            "shopper_education": "Structured",
            "usage_storytelling": "Balanced",
            "visual_identity": "Cohesive",
            "pack_or_spec_detail": "Enhanced",
            "review_or_trust_signals": "Strong",
            "category_grouping": "Structured",
            "discovery_pathways": "Enhanced",
            "cross_category_navigation": "Effective",
        }
        return f"{prefixes.get(cue, 'Clear')} {label}"
    if slide_key == "slide2":
        return f"Broad {product} relevance"
    if slide_key == "slide3":
        return f"{product} search alignment"
    if slide_key == "slide5":
        return f"{product} shopping journey expansion"
    return f"{product} PDP content"


def translate_cues(
    context: dict[str, Any],
    *,
    slide_key: str,
    count: int,
    preferred_order: tuple[str, ...],
    side: str = "",
) -> tuple[list[str], list[dict[str, Any]]]:
    identity = context.get("identity", {})
    ordered = _ordered_candidates(context, slide_key, preferred_order)
    bullets: list[str] = []
    debug: list[dict[str, Any]] = []
    used: set[str] = set()
    for candidate in ordered:
        text = _bullet_text(candidate, identity, slide_key=slide_key, side=side)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if len(words) > 9:
            text = " ".join(words[:9]).rstrip(" ,;-")
        key = _norm(text)
        if not key or key in used:
            continue
        used.add(key)
        bullets.append(text)
        debug.append(
            {
                "text": text,
                "cue": candidate.get("cue"),
                "classification": candidate.get("classification"),
                "coverage_ratio": candidate.get("coverage_ratio"),
                "gap_ratio": candidate.get("gap_ratio"),
                "competitive_delta": candidate.get("competitive_delta"),
                "confidence_tier": candidate.get("confidence_tier"),
                "reason": "Translated from aggregated strategic cue evidence.",
            }
        )
        if len(bullets) >= count:
            break
    fallback_labels = [
        identity.get("product_type_focus_phrase") or "product positioning",
        _category_vocab(identity).get("education", "shopper education"),
        _category_vocab(identity).get("navigation", "category discoverability"),
        "conversion-focused guidance",
        "visual identity",
    ]
    for label in fallback_labels:
        if len(bullets) >= count:
            break
        text = f"Opportunity to strengthen {label}"
        key = _norm(text)
        if key in used:
            continue
        used.add(key)
        bullets.append(text)
        debug.append(
            {
                "text": text,
                "cue": "fallback",
                "classification": "opportunity",
                "coverage_ratio": 0,
                "gap_ratio": 1,
                "competitive_delta": 0,
                "confidence_tier": 4,
                "reason": "Soft fallback used to preserve slide bullet count.",
            }
        )
    return bullets[:count], debug[:count]
