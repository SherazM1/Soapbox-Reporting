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


def _structured_hit(record: dict[str, Any], cue_key: str, text: str) -> bool:
    definition = APPROVED_CUE_DEFINITIONS[cue_key]
    terms = tuple(_norm(term) for term in definition["terms"])
    text_norm = _norm(text)
    if any(term and term in text_norm for term in terms):
        return True
    if cue_key == "review_or_trust_signals":
        reviews = _numeric(record, "review_count", "Review Count", "reviews", "ratingCount", "reviews_summary.review_count", "reviews_summary.count")
        return reviews >= 25 or bool(_record_value(record, "sold_by_walmart", "Sold by Walmart")) or bool(_record_value(record, "shipped_by_walmart", "Shipped by Walmart"))
    if cue_key == "visual_identity":
        return len(_images(record)) >= 4 or _numeric(record, "image_count", "moduleCount") >= 4
    if cue_key == "discovery_pathways":
        return len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 2 or _numeric(record, "productCount", "product_count") >= 4
    if cue_key == "cross_category_navigation":
        return len(_as_list(record.get("links") or record.get("navigationLinks") or record.get("categoryNavigation"))) >= 3
    if cue_key == "category_grouping":
        return bool(_record_value(record, "category", "categoryPathName")) or len(_as_list(record.get("categoryNavigation"))) >= 2
    if cue_key == "conversion_guidance":
        return bool(_record_value(record, "enhanced_brand_content", "enhanced_brand_content_present", "Enhanced Brand Content"))
    return False


def _record_profile(record: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    text = _record_text(record)
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
        structured = _structured_hit(record, cue_key, text)
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
        for index, product in enumerate(_as_list(products)):
            if not isinstance(product, dict):
                continue
            records.append(
                {
                    "record_id": f"search-{side}-{index}",
                    "product_title": product.get("title") or product.get("productTitle") or product.get("name"),
                    "brand": product.get("brand") or product.get("brandName"),
                    "category": search_term,
                    "product_type": search_term,
                    "review_count": product.get("reviewCount") or product.get("ratingCount"),
                    "badges": product.get("badges") or product.get("flags") or [],
                    "description_body": " ".join([search_term, _safe_text(product.get("sponsored") and "sponsored")]),
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
        records.append(
            {
                "record_id": f"brand-shop-{side}-{index}",
                "brand": row.get("brandName") or row.get("brand_name"),
                "category": categories[0] if categories else row.get("category", ""),
                "product_type": categories[0] if categories else row.get("product_type", ""),
                "description_body": " ".join([_safe_text(row.get("screenshotAlt")), _safe_text(row.get("warnings"))]),
                "moduleTypes": row.get("moduleTypes") or [module.get("type") for module in _as_list(row.get("modules")) if isinstance(module, dict)],
                "modules": row.get("modules") or [],
                "categoryNavigation": categories,
                "links": row.get("destinationLinks") or row.get("links") or [],
                "moduleCount": row.get("moduleCount"),
                "productCount": row.get("productCount"),
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
        strength = sum(1 for row in cue_rows if row["actual_present"] and row["confidence_tier"] <= 2) / total if primary_profiles else 0.0
        competitive_delta = comp_coverage - coverage
        tiers = [row["confidence_tier"] for row in cue_rows if row["actual_present"]]
        confidence_tier = min(tiers) if tiers else (2 if guide_expected_count else 4)
        matched_rules = sorted({rule for row in cue_rows for rule in row["matched_guide_rules"]})
        missing_rules = sorted({rule for row in cue_rows for rule in row["missing_guide_rules"]})
        sources = sorted({source for row in cue_rows for source in row["evidence_sources"]})
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
