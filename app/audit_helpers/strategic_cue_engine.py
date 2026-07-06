"""Backend cue aggregation for the new strategic PowerPoint export path."""

from __future__ import annotations

from typing import Any

from app.audit_helpers.image_guides import audit_pdp_images_against_guide, normalize_product_type
from app.audit_helpers.strategic_identity import identity_debug_payload, resolve_strategic_identity


APPROVED_CUE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "product_positioning": {
        "terms": ("positioning", "product type", "flavor", "variant", "premium", "organic", "compatible"),
        "guide_fields": ("recommended_title_priorities", "attribute_cues"),
        "slide_objective_tags": ("slide2_summary", "slide4_pdp_benchmark"),
    },
    "benefit_communication": {
        "terms": ("benefit", "protein", "hydrating", "moisture", "support", "performance", "organic", "gluten free"),
        "guide_fields": ("benefit_cues", "recommended_title_priorities"),
        "slide_objective_tags": ("slide2_summary", "slide4_pdp_benchmark"),
    },
    "ingredient_or_formula_communication": {
        "terms": ("ingredient", "formula", "nutrition", "active", "protein", "compatibility", "spec"),
        "guide_fields": ("attribute_cues", "education_cues"),
        "slide_objective_tags": ("slide4_pdp_benchmark",),
    },
    "shopper_education": {
        "terms": ("how to", "use", "routine", "directions", "guide", "education", "instructions"),
        "guide_fields": ("education_cues", "recommended_visual_priorities"),
        "slide_objective_tags": ("slide2_summary", "slide4_pdp_benchmark", "slide5_brand_shop"),
    },
    "usage_storytelling": {
        "terms": ("breakfast", "snack", "recipe", "routine", "occasion", "setup", "feeding", "care"),
        "guide_fields": ("usage_occasion_cues", "image_story_cues"),
        "slide_objective_tags": ("slide2_summary", "slide4_pdp_benchmark", "slide5_brand_shop"),
    },
    "visual_identity": {
        "terms": ("hero", "banner", "lifestyle", "image", "visual", "module", "pov"),
        "guide_fields": ("recommended_visual_priorities", "image_story_cues"),
        "slide_objective_tags": ("slide4_pdp_benchmark", "slide5_brand_shop"),
    },
    "pack_or_spec_detail": {
        "terms": ("nutrition", "ingredients", "size", "count", "compatibility", "dosage", "dimension", "spec"),
        "guide_fields": ("attribute_cues", "recommended_visual_priorities"),
        "slide_objective_tags": ("slide4_pdp_benchmark",),
    },
    "keyword_alignment": {
        "terms": ("search", "keyword", "query", "title", "intent"),
        "guide_fields": ("recommended_title_priorities",),
        "slide_objective_tags": ("slide3_search_discovery",),
    },
    "discoverability": {
        "terms": ("search", "shelf", "discover", "visible", "badge", "sponsored"),
        "guide_fields": ("recommended_title_priorities",),
        "slide_objective_tags": ("slide2_summary", "slide3_search_discovery", "slide5_brand_shop"),
    },
    "assortment_segmentation": {
        "terms": ("flavor", "variant", "size", "pack", "count", "type", "assortment"),
        "guide_fields": ("attribute_cues", "comparison_cues"),
        "slide_objective_tags": ("slide3_search_discovery", "slide5_brand_shop"),
    },
    "category_grouping": {
        "terms": ("category", "shop", "navigation", "department", "collection"),
        "guide_fields": ("comparison_cues",),
        "slide_objective_tags": ("slide2_summary", "slide5_brand_shop"),
    },
    "discovery_pathways": {
        "terms": ("link", "navigation", "collection", "carousel", "grid", "module"),
        "guide_fields": ("recommended_visual_priorities",),
        "slide_objective_tags": ("slide3_search_discovery", "slide5_brand_shop"),
    },
    "cross_category_navigation": {
        "terms": ("cross category", "category navigation", "shop all", "collection", "department"),
        "guide_fields": ("comparison_cues",),
        "slide_objective_tags": ("slide5_brand_shop",),
    },
    "review_or_trust_signals": {
        "terms": ("review", "rating", "walmart", "sold by", "shipped by", "trust"),
        "guide_fields": (),
        "slide_objective_tags": ("slide2_summary", "slide3_search_discovery", "slide4_pdp_benchmark"),
    },
    "conversion_guidance": {
        "terms": ("compare", "choose", "benefit", "feature", "size", "count", "directions", "buy"),
        "guide_fields": ("education_cues", "recommended_visual_priorities"),
        "slide_objective_tags": ("slide2_summary", "slide4_pdp_benchmark", "slide5_brand_shop"),
    },
}


CUE_CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
    "product_positioning": {"title": 1.2, "taxonomy": 1.1, "features": 0.8, "description": 0.5, "image_analysis": 0.4},
    "benefit_communication": {"features": 1.2, "bullets": 1.0, "title": 0.8, "description": 0.7, "image_analysis": 0.4},
    "ingredient_or_formula_communication": {"features": 1.1, "bullets": 1.0, "description": 0.8, "taxonomy": 0.5, "image_analysis": 0.5},
    "shopper_education": {"bullets": 1.1, "modules": 1.0, "description": 0.8, "image_analysis": 0.5},
    "usage_storytelling": {"modules": 1.1, "image_analysis": 0.9, "description": 0.6, "bullets": 0.5},
    "visual_identity": {"modules": 1.2, "image_analysis": 1.1, "brand_shop_media": 1.0},
    "pack_or_spec_detail": {"features": 1.0, "bullets": 0.9, "taxonomy": 0.7, "image_analysis": 0.6},
    "keyword_alignment": {"search_intent": 1.5, "title": 1.1, "taxonomy": 0.5},
    "discoverability": {"search_visibility": 1.5, "trust": 0.7, "navigation": 0.4},
    "assortment_segmentation": {"assortment": 1.5, "taxonomy": 0.9, "navigation": 0.7},
    "category_grouping": {"navigation": 1.5, "taxonomy": 0.8, "brand_shop_scale": 0.4},
    "discovery_pathways": {"brand_shop_pathways": 1.5, "modules": 0.9, "navigation": 0.5},
    "cross_category_navigation": {"cross_category": 1.6, "navigation": 0.7},
    "review_or_trust_signals": {"trust": 1.6, "search_visibility": 0.4},
    "conversion_guidance": {"conversion": 1.4, "modules": 0.8, "features": 0.5, "bullets": 0.5},
}


CUE_CHANNEL_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "keyword_alignment": {
        "search_intent": ("query match", "intent match", "keyword match", "exact search term", "aligned search term"),
        "title": ("search", "keyword", "intent"),
    },
    "discoverability": {
        "search_visibility": ("top result", "rank", "position", "sponsored", "badge", "visible", "shelf"),
        "trust": ("walmart", "seller", "rating", "review"),
    },
    "assortment_segmentation": {
        "assortment": ("variant spread", "product type spread", "assortment breadth", "size range", "pack range"),
        "taxonomy": ("flavor", "variant", "size", "pack", "count", "type"),
    },
    "category_grouping": {
        "navigation": ("category group", "category label", "collection label", "department label"),
        "taxonomy": ("category", "department"),
    },
    "discovery_pathways": {
        "brand_shop_pathways": ("link pathway", "destination link", "carousel", "grid", "shop module", "collection module"),
        "modules": ("carousel", "grid", "module", "collection"),
    },
    "cross_category_navigation": {
        "cross_category": ("cross category", "multiple departments", "shop all", "category expansion", "department expansion"),
        "navigation": ("department", "shop all"),
    },
    "visual_identity": {
        "modules": ("hero", "banner", "lifestyle", "video", "rich media"),
        "image_analysis": ("hero", "banner", "lifestyle", "image", "visual"),
        "brand_shop_media": ("hero", "banner", "video", "rich media", "lifestyle"),
    },
    "conversion_guidance": {
        "conversion": ("shop now", "buy now", "compare", "choose", "cta", "product tile", "how to choose"),
        "modules": ("compare", "choose", "shop now", "buy now"),
    },
}


CATEGORY_CUE_GUARDRAILS: dict[str, dict[str, Any]] = {
    "beauty": {
        "penalized_terms": ("nutrition", "protein", "calorie", "recipe", "breakfast", "snack", "feeding"),
        "cue_boosts": {
            "benefit_communication": ("hydrating", "moisture", "brightening", "blemish", "concern", "skin"),
            "ingredient_or_formula_communication": ("formula", "active", "hyaluronic", "niacinamide", "cleanser", "regimen"),
            "usage_storytelling": ("routine", "regimen", "morning", "night"),
        },
    },
    "electronics": {
        "penalized_terms": ("nutrition", "ingredient", "flavor", "recipe", "breakfast"),
        "cue_boosts": {
            "ingredient_or_formula_communication": ("compatibility", "spec", "dimension", "watt", "voltage"),
            "shopper_education": ("setup", "install", "instructions", "guide"),
            "usage_storytelling": ("use case", "gaming", "office", "travel", "device"),
            "pack_or_spec_detail": ("compatibility", "spec", "dimension", "model"),
        },
    },
    "food_beverage": {
        "penalized_terms": ("compatibility", "device", "install", "hyaluronic", "niacinamide"),
        "cue_boosts": {
            "ingredient_or_formula_communication": ("ingredient", "nutrition", "protein", "organic", "gluten free"),
            "usage_storytelling": ("recipe", "breakfast", "snack", "occasion"),
            "shopper_education": ("nutrition", "usage", "serving"),
        },
    },
}


REDUNDANT_CUE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("keyword_alignment", "discoverability", "assortment_segmentation"),
    ("category_grouping", "discovery_pathways", "cross_category_navigation"),
    ("visual_identity", "conversion_guidance"),
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return normalize_product_type(_safe_text(value))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


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
        _record_value(record, "product_title", "title", "productTitle", "name"),
        _record_value(record, "category", "categoryPathName"),
        _record_value(record, "product_type", "subcategory", "productType"),
        _record_value(record, "brand", "brandName"),
        _record_value(record, "description_body", "description", "shortDescription"),
        _record_value(record, "seller_name", "Seller Name"),
        *_as_list(_record_value(record, "description_bullets", "Description Bullets", default=[])),
        *_as_list(_record_value(record, "key_features", "Key Features", default=[])),
        *_as_list(record.get("badges") if isinstance(record, dict) else []),
        *_as_list(record.get("moduleTypes") if isinstance(record, dict) else []),
        *_as_list(record.get("categoryNavigation") if isinstance(record, dict) else []),
    ]
    modules = record.get("modules") if isinstance(record, dict) else []
    if isinstance(modules, list):
        for module in modules:
            if isinstance(module, dict):
                parts.extend([module.get("type"), module.get("heading"), module.get("title"), module.get("description")])
    return " ".join(_safe_text(part) for part in parts if _safe_text(part)).lower()


def _join_text(values: list[Any]) -> str:
    return " ".join(_safe_text(value) for value in values if _safe_text(value)).lower()


def _record_channels(record: dict[str, Any]) -> dict[str, str]:
    modules = record.get("modules") if isinstance(record, dict) else []
    module_parts: list[Any] = [record.get("moduleTypes") if isinstance(record, dict) else []]
    if isinstance(modules, list):
        for module in modules:
            if isinstance(module, dict):
                module_parts.extend([module.get("type"), module.get("heading"), module.get("title"), module.get("description")])
    navigation = _as_list(record.get("categoryNavigation") if isinstance(record, dict) else [])
    links = _as_list(record.get("links") or record.get("navigationLinks") or record.get("destinationLinks"))
    trust_parts = [
        _record_value(record, "seller_name", "Seller Name"),
        _record_value(record, "sold_by_walmart", "Sold by Walmart"),
        _record_value(record, "shipped_by_walmart", "Shipped by Walmart"),
        _record_value(record, "review_count", "Review Count", "reviews", "ratingCount", "reviews_summary.review_count"),
        _record_value(record, "rating", "averageRating", "reviews_summary.rating"),
        *_as_list(record.get("badges") if isinstance(record, dict) else []),
    ]
    return {
        "title": _join_text([_record_value(record, "product_title", "title", "productTitle", "name")]),
        "taxonomy": _join_text(
            [
                _record_value(record, "category", "categoryPathName"),
                _record_value(record, "product_type", "subcategory", "productType"),
                _record_value(record, "brand", "brandName"),
            ]
        ),
        "features": _join_text(_as_list(_record_value(record, "key_features", "Key Features", default=[]))),
        "bullets": _join_text(_as_list(_record_value(record, "description_bullets", "Description Bullets", default=[]))),
        "description": _join_text([_record_value(record, "description_body", "description", "shortDescription")]),
        "modules": _join_text(module_parts),
        "navigation": _join_text([*navigation, *links]),
        "image_analysis": _image_analysis_text(record),
        "trust": _join_text(trust_parts),
        "search_intent": _join_text(_as_list(record.get("search_intent_signals"))),
        "search_visibility": _join_text(_as_list(record.get("search_visibility_signals"))),
        "assortment": _join_text(_as_list(record.get("assortment_signals"))),
        "brand_shop_scale": _join_text(_as_list(record.get("brand_shop_scale_signals"))),
        "brand_shop_pathways": _join_text(_as_list(record.get("brand_shop_pathway_signals"))),
        "cross_category": _join_text(_as_list(record.get("cross_category_signals"))),
        "brand_shop_media": _join_text(_as_list(record.get("brand_shop_media_signals"))),
        "conversion": _join_text(_as_list(record.get("conversion_signals"))),
    }


def _numeric(record: dict[str, Any], *keys: str) -> float:
    value = _record_value(record, *keys, default=0)
    try:
        return float(str(value or 0).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _images(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _as_list(_record_value(record, "images", "ordered_images", default=[])) if isinstance(item, dict)]


def _image_analysis_text(record: dict[str, Any]) -> str:
    analysis = record.get("image_analysis") if isinstance(record, dict) else {}
    parts: list[str] = []
    if isinstance(analysis, dict):
        for image in _as_list(analysis.get("images")):
            if isinstance(image, dict):
                parts.extend(_safe_text(value) for value in _as_list(image.get("detected_signals")))
                parts.extend(_safe_text(value) for value in _as_list(image.get("ocr_tokens")))
                parts.append(_safe_text(image.get("probable_format")))
        parts.extend(_safe_text(value) for value in _as_list((analysis.get("stack_signals") or {}).values()))
    for image in _images(record):
        parts.extend([image.get("alt_text"), image.get("filename"), image.get("text"), image.get("ocr_text"), image.get("title")])
    return " ".join(_safe_text(part) for part in parts if _safe_text(part)).lower()


def _guide_rules(identity: dict[str, Any], cue_key: str) -> list[str]:
    definition = APPROVED_CUE_DEFINITIONS[cue_key]
    values: list[Any] = []
    for field in definition.get("guide_fields", ()):
        values.extend(_as_list(identity.get(field)))
    if cue_key == "product_positioning" and identity.get("product_type_display"):
        values.append(identity.get("product_type_display"))
    if cue_key == "category_grouping" and identity.get("family_display"):
        values.append(identity.get("family_display"))
    if cue_key in {"visual_identity", "pack_or_spec_detail"}:
        page = identity.get("image_page") or {}
        values.extend(_as_list(page.get("required_slots")))
    low_value_by_cue = {
        "benefit_communication": {"brand", "flavor", "size", "count", "pack", "product type", "spread", "spreads"},
        "shopper_education": {"brand", "flavor", "size", "count", "pack", "product type"},
        "conversion_guidance": {"brand", "flavor", "size", "count", "pack", "product type"},
    }
    seen: set[str] = set()
    rules: list[str] = []
    for value in values:
        text = _safe_text(value)
        key = _norm(text)
        if key in low_value_by_cue.get(cue_key, set()):
            continue
        if not text or not key or key in seen:
            continue
        seen.add(key)
        rules.append(text)
        if len(rules) >= 10:
            break
    return rules


def _match_rules(text: str, guide_text: str, rules: list[str]) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    text_norm = _norm(text)
    guide_norm = _norm(guide_text)
    for rule in rules:
        rule_norm = _norm(rule)
        if not rule_norm:
            continue
        if rule_norm in text_norm or rule_norm in guide_norm:
            matched.append(rule)
        else:
            missing.append(rule)
    return matched, missing


def _category_guardrail_multiplier(identity: dict[str, Any], cue_key: str, channels: dict[str, str]) -> float:
    category_key = _safe_text(identity.get("category_key"))
    product_type = _norm(identity.get("product_type_display"))
    guardrail = CATEGORY_CUE_GUARDRAILS.get(category_key, {})
    blob = " ".join(channels.values())
    multiplier = 1.0
    if any(_norm(term) in _norm(blob) for term in guardrail.get("penalized_terms", ())):
        if cue_key in {"ingredient_or_formula_communication", "pack_or_spec_detail", "usage_storytelling", "shopper_education"}:
            multiplier -= 0.35
    boost_terms = (guardrail.get("cue_boosts") or {}).get(cue_key, ())
    if any(_norm(term) in _norm(blob) for term in boost_terms):
        multiplier += 0.2
    if category_key == "beauty" and "cleanser" in product_type:
        if cue_key in {"benefit_communication", "ingredient_or_formula_communication", "shopper_education", "usage_storytelling"}:
            multiplier += 0.15
        if cue_key in {"pack_or_spec_detail", "usage_storytelling"} and any(
            term in _norm(blob) for term in ("nutrition", "recipe", "breakfast", "snack")
        ):
            multiplier -= 0.45
    return max(0.25, min(1.35, multiplier))


def _cue_evidence(record: dict[str, Any], cue_key: str, channels: dict[str, str], identity: dict[str, Any]) -> dict[str, Any]:
    definition = APPROVED_CUE_DEFINITIONS[cue_key]
    cue_terms = tuple(_norm(term) for term in definition["terms"])
    channel_terms = {
        channel: tuple(_norm(term) for term in terms)
        for channel, terms in CUE_CHANNEL_TERMS.get(cue_key, {}).items()
    }
    score = 0.0
    evidence_channels: list[str] = []
    for channel, weight in CUE_CHANNEL_WEIGHTS.get(cue_key, {}).items():
        channel_text = _norm(channels.get(channel, ""))
        if not channel_text:
            continue
        terms = channel_terms.get(channel, cue_terms)
        if any(term and term in channel_text for term in terms):
            score += weight
            evidence_channels.append(channel)
    if cue_key == "review_or_trust_signals":
        reviews = _numeric(record, "review_count", "Review Count", "reviews", "ratingCount", "reviews_summary.review_count", "reviews_summary.count")
        rating = _numeric(record, "rating", "averageRating", "reviews_summary.rating")
        if reviews >= 25:
            score += 1.2
            evidence_channels.append("trust")
        if rating >= 4.2 and reviews >= 10:
            score += 0.6
            evidence_channels.append("trust")
        if bool(_record_value(record, "sold_by_walmart", "Sold by Walmart")) or bool(_record_value(record, "shipped_by_walmart", "Shipped by Walmart")):
            score += 0.7
            evidence_channels.append("trust")
    if cue_key == "visual_identity":
        if len(_images(record)) >= 4 or _numeric(record, "image_count", "moduleCount") >= 4:
            score += 0.9
            evidence_channels.append("image_analysis")
    if cue_key == "discovery_pathways":
        if len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 2:
            score += 1.0
            evidence_channels.append("brand_shop_pathways")
        if _numeric(record, "productCount", "product_count") >= 4:
            score += 0.4
            evidence_channels.append("brand_shop_scale")
    if cue_key == "cross_category_navigation":
        if len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 3:
            score += 1.2
            evidence_channels.append("cross_category")
    if cue_key == "category_grouping":
        if len(_as_list(record.get("categoryNavigation"))) >= 2:
            score += 1.1
            evidence_channels.append("navigation")
        elif bool(_record_value(record, "category", "categoryPathName")):
            score += 0.5
            evidence_channels.append("taxonomy")
    if cue_key == "conversion_guidance":
        if bool(_record_value(record, "enhanced_brand_content", "enhanced_brand_content_present", "Enhanced Brand Content")):
            score += 0.9
            evidence_channels.append("conversion")
    if cue_key == "keyword_alignment":
        search_term = _norm(record.get("search_term"))
        title = _norm(channels.get("title"))
        if search_term and all(token in title for token in search_term.split() if len(token) > 2):
            score += 1.2
            evidence_channels.append("search_intent")
    if cue_key == "discoverability":
        rank = _numeric(record, "search_rank")
        if 0 < rank <= 4:
            score += 1.0
            evidence_channels.append("search_visibility")
        if bool(record.get("is_sponsored")):
            score += 0.5
            evidence_channels.append("search_visibility")
    if cue_key == "assortment_segmentation":
        if _numeric(record, "bucket_unique_product_type_count") >= 3:
            score += 1.0
            evidence_channels.append("assortment")
    multiplier = _category_guardrail_multiplier(identity, cue_key, channels)
    score = round(score * multiplier, 3)
    return {
        "hit": score >= 0.95,
        "score": score,
        "channels": sorted(set(evidence_channels)),
        "guardrail_multiplier": multiplier,
    }


def _record_profile(record: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    text = _record_text(record)
    channels = _record_channels(record)
    image_text = _image_analysis_text(record)
    guide_audit = audit_pdp_images_against_guide(
        identity.get("category_key", ""),
        identity.get("product_type_display", ""),
        _images(record),
    )
    guide_text = " ".join(
        [
            *[str(slot) for slot in guide_audit.get("detected_slots") or []],
            image_text,
        ]
    )
    cues: dict[str, dict[str, Any]] = {}
    for cue_key in APPROVED_CUE_DEFINITIONS:
        rules = _guide_rules(identity, cue_key)
        matched_rules, missing_rules = _match_rules(text, guide_text, rules)
        cue_evidence = _cue_evidence(record, cue_key, channels, identity)
        structured = bool(cue_evidence["hit"])
        image_support = bool(image_text and any(_norm(term) in _norm(image_text) for term in APPROVED_CUE_DEFINITIONS[cue_key]["terms"]))
        guide_aligned = bool(matched_rules)
        actual = structured or guide_aligned or image_support
        if structured:
            tier = 1
        elif guide_aligned:
            tier = 2
        elif image_support:
            tier = 3
        else:
            tier = 4
        sources: list[str] = []
        if structured:
            sources.append("structured")
            sources.extend(cue_evidence["channels"])
        if guide_aligned:
            sources.append("guide_expectation")
        if image_support:
            sources.append("image_analysis")
        if not sources:
            sources.append("fallback")
        cues[cue_key] = {
            "actual_present": actual,
            "confidence_tier": tier,
            "evidence_sources": sources,
            "matched_guide_rules": matched_rules,
            "missing_guide_rules": missing_rules,
            "guide_expected": bool(rules),
            "evidence_score": cue_evidence["score"],
            "evidence_channels": cue_evidence["channels"],
            "guardrail_multiplier": cue_evidence["guardrail_multiplier"],
        }
    return {
        "record_id": _safe_text(_record_value(record, "record_id", "id", "item_id", "productId")),
        "brand": _safe_text(_record_value(record, "brand", "brandName")),
        "cue_results": cues,
    }


def _search_records(search_evidence: dict[str, Any] | list[dict[str, Any]] | None, side: str) -> list[dict[str, Any]]:
    if isinstance(search_evidence, list):
        buckets = search_evidence
    elif isinstance(search_evidence, dict):
        buckets = _as_list(search_evidence.get(side))
        if not buckets and side == "competitor":
            buckets = _as_list(search_evidence.get("benchmark"))
    else:
        buckets = []
    records: list[dict[str, Any]] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        search_term = _safe_text(bucket.get("searchTerm") or bucket.get("search_term") or bucket.get("query"))
        products = bucket.get("products") or bucket.get("capturedProducts") or bucket.get("items") or []
        product_rows = [product for product in _as_list(products) if isinstance(product, dict)]
        product_types = [
            _safe_text(
                product.get("productType")
                or product.get("product_type")
                or product.get("subcategory")
                or product.get("category")
                or product.get("categoryPathName")
            )
            for product in product_rows
        ]
        unique_product_types = sorted({_norm(value) for value in product_types if _norm(value)})
        sponsored_count = sum(1 for product in product_rows if bool(product.get("sponsored") or product.get("isSponsored")))
        for index, product in enumerate(_as_list(products)):
            if not isinstance(product, dict):
                continue
            rank = _numeric(product, "rank", "position", "searchRank") or float(index + 1)
            title = product.get("title") or product.get("productTitle") or product.get("name")
            product_type = (
                product.get("productType")
                or product.get("product_type")
                or product.get("subcategory")
                or product.get("category")
                or search_term
            )
            sponsored = bool(product.get("sponsored") or product.get("isSponsored"))
            badges = product.get("badges") or product.get("flags") or []
            intent_signals = [search_term]
            if search_term and title and all(token in _norm(title) for token in _norm(search_term).split() if len(token) > 2):
                intent_signals.append("query match exact search term")
            visibility_signals = []
            if rank <= 4:
                visibility_signals.append("top result shelf visibility")
            if sponsored:
                visibility_signals.append("sponsored search pressure")
            if badges:
                visibility_signals.append("badge visible on shelf")
            assortment_signals = [f"product type spread {len(unique_product_types)}", *unique_product_types]
            records.append(
                {
                    "record_id": f"search-{side}-{index}",
                    "product_title": title,
                    "brand": product.get("brand") or product.get("brandName"),
                    "category": search_term,
                    "product_type": product_type,
                    "search_term": search_term,
                    "search_rank": rank,
                    "is_sponsored": sponsored,
                    "bucket_result_count": len(product_rows),
                    "bucket_sponsored_count": sponsored_count,
                    "bucket_unique_product_type_count": len(unique_product_types),
                    "review_count": product.get("reviewCount") or product.get("ratingCount"),
                    "rating": product.get("rating") or product.get("averageRating"),
                    "badges": badges,
                    "search_intent_signals": intent_signals,
                    "search_visibility_signals": visibility_signals,
                    "assortment_signals": assortment_signals,
                    "description_body": _safe_text(product.get("description") or product.get("shortDescription")),
                }
            )
    return records


def _brand_shop_records(brand_shop_evidence: dict[str, Any] | list[dict[str, Any]] | None, side: str) -> list[dict[str, Any]]:
    if isinstance(brand_shop_evidence, dict):
        rows = _as_list(brand_shop_evidence.get(side))
        if not rows and side == "competitor":
            rows = _as_list(brand_shop_evidence.get("benchmark"))
    elif isinstance(brand_shop_evidence, list):
        rows = brand_shop_evidence
    else:
        rows = []
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        categories = _as_list(row.get("categoryNavigation") or row.get("categories"))
        modules = row.get("modules") or []
        module_types = row.get("moduleTypes") or [module.get("type") for module in _as_list(modules) if isinstance(module, dict)]
        links = row.get("destinationLinks") or row.get("links") or []
        module_text = _join_text(
            [
                *module_types,
                *[
                    value
                    for module in _as_list(modules)
                    if isinstance(module, dict)
                    for value in (module.get("type"), module.get("heading"), module.get("title"), module.get("description"))
                ],
            ]
        )
        category_keys = sorted({_norm(category) for category in categories if _norm(category)})
        rich_media_terms = ("hero", "banner", "video", "lifestyle", "rich media")
        scale_signals = [
            f"assortment breadth {row.get('productCount') or 0}",
            f"module depth {row.get('moduleCount') or len(_as_list(modules))}",
        ]
        pathway_signals = []
        if links:
            pathway_signals.append(f"destination link pathway {len(_as_list(links))}")
        if any(term in module_text for term in ("carousel", "grid", "collection")):
            pathway_signals.append("carousel grid collection module")
        cross_category_signals = []
        if len(category_keys) >= 3:
            cross_category_signals.append("multiple departments cross category expansion")
        if any("shop all" in _norm(value) for value in [*categories, *links]):
            cross_category_signals.append("shop all cross category pathway")
        media_signals = [term for term in rich_media_terms if term in module_text]
        conversion_signals = []
        if any(term in module_text for term in ("shop now", "buy now", "compare", "choose", "cta")):
            conversion_signals.append("conversion cta guidance")
        if _numeric(row, "productCount") >= 4:
            conversion_signals.append("product tile shopping guidance")
        records.append(
            {
                "record_id": f"brand-shop-{side}-{index}",
                "brand": row.get("brandName") or row.get("brand_name"),
                "category": categories[0] if categories else row.get("category", ""),
                "product_type": categories[0] if categories else row.get("product_type", ""),
                "description_body": " ".join([_safe_text(row.get("screenshotAlt")), _safe_text(row.get("warnings"))]),
                "moduleTypes": module_types,
                "modules": modules,
                "categoryNavigation": categories,
                "links": links,
                "moduleCount": row.get("moduleCount"),
                "productCount": row.get("productCount"),
                "brand_shop_scale_signals": scale_signals,
                "brand_shop_pathway_signals": pathway_signals,
                "cross_category_signals": cross_category_signals,
                "brand_shop_media_signals": media_signals,
                "conversion_signals": conversion_signals,
            }
        )
    return records


def _consistency(coverage: float, total: int) -> str:
    if total <= 0:
        return "fallback"
    if coverage >= 0.7:
        return "broad"
    if coverage >= 0.35:
        return "mixed"
    return "limited"


def _classify_candidate(coverage: float, gap: float, competitive_delta: float, has_competitive_context: bool) -> str:
    if has_competitive_context and competitive_delta >= 0.35:
        return "pressure"
    if gap >= 0.55:
        return "opportunity"
    if coverage >= 0.55:
        return "strength"
    return "context"


def _candidate_score(candidate: dict[str, Any]) -> float:
    return (
        float(candidate.get("strength_ratio") or 0) * 3
        + float(candidate.get("coverage_ratio") or 0) * 2
        + float(candidate.get("average_evidence_score") or 0)
        + max(0.0, float(candidate.get("competitive_delta") or 0))
    )


def _channel_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def _suppress_redundant_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {candidate["cue_key"]: candidate for candidate in candidates}
    for group in REDUNDANT_CUE_GROUPS:
        ranked = sorted(
            [by_key[key] for key in group if key in by_key],
            key=_candidate_score,
            reverse=True,
        )
        winners: list[dict[str, Any]] = []
        for candidate in ranked:
            channels = candidate.get("evidence_channels") or []
            redundant_with = next(
                (
                    winner
                    for winner in winners
                    if _channel_overlap(channels, winner.get("evidence_channels") or []) >= 0.75
                    and abs(float(candidate.get("coverage_ratio") or 0) - float(winner.get("coverage_ratio") or 0)) <= 0.15
                ),
                None,
            )
            if redundant_with:
                candidate["redundancy_suppressed_by"] = redundant_with["cue_key"]
                candidate["strength_ratio"] = round(min(float(candidate.get("strength_ratio") or 0), 0.34), 2)
                if candidate["classification"] == "strength":
                    candidate["classification"] = "context"
                candidate["debug_reason"] += f" Redundant upstream cue suppressed behind {redundant_with['cue_key']}."
            else:
                winners.append(candidate)
    return candidates


def aggregate_strategic_cues(
    records: list[dict[str, Any]],
    *,
    competitor_records: list[dict[str, Any]] | None = None,
    search_evidence: dict[str, Any] | list[dict[str, Any]] | None = None,
    brand_shop_evidence: dict[str, Any] | list[dict[str, Any]] | None = None,
    identity: dict[str, Any] | None = None,
    fallback_category: str = "",
    fallback_product_type: str = "",
) -> dict[str, Any]:
    """Aggregate backend cue candidates across records and supporting evidence."""
    primary_records = [record for record in records or [] if isinstance(record, dict)]
    comparison_records = [record for record in competitor_records or [] if isinstance(record, dict)]
    primary_records = [
        *primary_records,
        *_search_records(search_evidence, "current"),
        *_brand_shop_records(brand_shop_evidence, "client"),
    ]
    comparison_records = [
        *comparison_records,
        *_search_records(search_evidence, "benchmark"),
        *_search_records(search_evidence, "competitor"),
        *_brand_shop_records(brand_shop_evidence, "competitor"),
    ]
    resolved_identity = identity or resolve_strategic_identity(
        primary_records or comparison_records,
        fallback_category=fallback_category,
        fallback_product_type=fallback_product_type,
    )
    primary_profiles = [_record_profile(record, resolved_identity) for record in primary_records]
    comparison_profiles = [_record_profile(record, resolved_identity) for record in comparison_records]
    total = max(1, len(primary_profiles))
    comp_total = max(1, len(comparison_profiles))
    candidates: list[dict[str, Any]] = []
    for cue_key, definition in APPROVED_CUE_DEFINITIONS.items():
        cue_rows = [profile["cue_results"][cue_key] for profile in primary_profiles]
        comp_rows = [profile["cue_results"][cue_key] for profile in comparison_profiles]
        actual_count = sum(1 for row in cue_rows if row["actual_present"])
        comp_count = sum(1 for row in comp_rows if row["actual_present"])
        guide_expected_count = sum(1 for row in cue_rows if row["guide_expected"])
        missing_count = sum(1 for row in cue_rows if row["guide_expected"] and not row["matched_guide_rules"])
        coverage = actual_count / total if primary_profiles else 0.0
        comp_coverage = comp_count / comp_total if comparison_profiles else 0.0
        gap = (missing_count / max(1, guide_expected_count)) if guide_expected_count else 1.0 - coverage
        evidence_scores = [float(row.get("evidence_score") or 0) for row in cue_rows]
        avg_evidence_score = sum(evidence_scores) / total if primary_profiles else 0.0
        strength = (
            sum(min(float(row.get("evidence_score") or 0), 2.0) / 2.0 for row in cue_rows if row["actual_present"])
            / total
            if primary_profiles
            else 0.0
        )
        competitive_delta = comp_coverage - coverage
        tiers = [row["confidence_tier"] for row in cue_rows if row["actual_present"]]
        confidence_tier = min(tiers) if tiers else (2 if guide_expected_count else 4)
        matched_rules = sorted({rule for row in cue_rows for rule in row["matched_guide_rules"]})
        missing_rules = sorted({rule for row in cue_rows for rule in row["missing_guide_rules"]})
        sources = sorted({source for row in cue_rows for source in row["evidence_sources"]})
        evidence_channels = sorted({channel for row in cue_rows for channel in row.get("evidence_channels", [])})
        guardrail_multiplier = (
            sum(float(row.get("guardrail_multiplier") or 1) for row in cue_rows) / total if primary_profiles else 1.0
        )
        classification = _classify_candidate(
            coverage,
            gap,
            competitive_delta,
            bool(comparison_profiles),
        )
        candidates.append(
            {
                "cue_key": cue_key,
                "cue": cue_key,
                "classification": classification,
                "confidence_tier": confidence_tier,
                "coverage_ratio": round(coverage, 2),
                "gap_ratio": round(gap, 2),
                "strength_ratio": round(strength, 2),
                "consistency": _consistency(coverage, len(primary_profiles)),
                "competitive_delta": round(competitive_delta, 2),
                "evidence_sources": sources,
                "evidence_channels": evidence_channels,
                "average_evidence_score": round(avg_evidence_score, 2),
                "guardrail_multiplier": round(guardrail_multiplier, 2),
                "matched_guide_rules": matched_rules[:8],
                "missing_guide_rules": missing_rules[:8],
                "slide_objective_tags": list(definition["slide_objective_tags"]),
                "debug_reason": (
                    f"{actual_count}/{len(primary_profiles)} primary records matched; "
                    f"{comp_count}/{len(comparison_profiles)} comparison records matched; "
                    f"{missing_count} guide expectation gaps."
                ),
            }
        )
    candidates = _suppress_redundant_candidates(candidates)
    return {
        "identity": resolved_identity,
        "candidate_cues": candidates,
        "debug": {
            "resolved_identity": identity_debug_payload(resolved_identity),
            "candidate_cues": candidates,
            "primary_record_count": len(primary_profiles),
            "comparison_record_count": len(comparison_profiles),
            "cue_definitions": {
                key: {
                    "guide_fields": list(value.get("guide_fields", ())),
                    "slide_objective_tags": list(value.get("slide_objective_tags", ())),
                }
                for key, value in APPROVED_CUE_DEFINITIONS.items()
            },
        },
    }


def cue_debug_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Return the serializable backend cue debug payload for export plans."""
    return context.get("debug", {})
