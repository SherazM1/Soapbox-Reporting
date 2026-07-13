from __future__ import annotations

from collections import Counter
from typing import Any


SHOPPER_BUCKETS = (
    "opening_clarity",
    "benefit_proof",
    "usage_comparison",
    "trust_reassurance",
)

STRENGTH_SIGNALS = (
    "ingredient_or_flavor_storytelling",
    "nutrition_or_detail_support",
    "usage_or_recipe_storytelling",
    "lifestyle_or_contextual_storytelling",
    "benefit_forward_graphics",
    "routine_or_regimen_education",
    "trust_or_certification_support",
    "strong_opening_product_clarity",
    "clear_carousel_depth",
)

OPPORTUNITY_SIGNALS = (
    "weak_opening_sequence",
    "thin_carousel_depth",
    "missing_lifestyle_storytelling",
    "missing_usage_or_recipe_storytelling",
    "missing_benefit_graphics",
    "missing_nutrition_or_ingredient_detail",
    "missing_trust_or_certification_support",
    "text_heavy_without_clear_hierarchy",
    "duplicate_or_redundant_images",
    "limited_conversion_guidance",
)

SIGNAL_BUCKETS: dict[str, str] = {
    "strong_opening_product_clarity": "opening_clarity",
    "weak_opening_sequence": "opening_clarity",
    "ingredient_or_flavor_storytelling": "benefit_proof",
    "nutrition_or_detail_support": "benefit_proof",
    "benefit_forward_graphics": "benefit_proof",
    "missing_benefit_graphics": "benefit_proof",
    "missing_nutrition_or_ingredient_detail": "benefit_proof",
    "usage_or_recipe_storytelling": "usage_comparison",
    "lifestyle_or_contextual_storytelling": "usage_comparison",
    "routine_or_regimen_education": "usage_comparison",
    "clear_carousel_depth": "usage_comparison",
    "thin_carousel_depth": "usage_comparison",
    "missing_lifestyle_storytelling": "usage_comparison",
    "missing_usage_or_recipe_storytelling": "usage_comparison",
    "text_heavy_without_clear_hierarchy": "usage_comparison",
    "duplicate_or_redundant_images": "usage_comparison",
    "limited_conversion_guidance": "benefit_proof",
    "trust_or_certification_support": "trust_reassurance",
    "missing_trust_or_certification_support": "trust_reassurance",
}

CONFLICTING_CONTENT_OPPORTUNITIES: dict[str, tuple[str, ...]] = {
    "weak_opening_sequence": ("strong_opening_product_clarity",),
    "missing_benefit_graphics": (
        "benefit_forward_graphics",
        "ingredient_or_flavor_storytelling",
        "nutrition_or_detail_support",
    ),
    "missing_nutrition_or_ingredient_detail": (
        "benefit_forward_graphics",
        "ingredient_or_flavor_storytelling",
        "nutrition_or_detail_support",
    ),
    "missing_usage_or_recipe_storytelling": (
        "usage_or_recipe_storytelling",
        "routine_or_regimen_education",
    ),
    "missing_trust_or_certification_support": ("trust_or_certification_support",),
}

CATEGORY_WORDS: dict[str, dict[str, str]] = {
    "jam_preserves": {
        "product": "flavor and product cues",
        "benefit": "visible flavor and ingredient cues",
        "usage": "serving and use inspiration",
        "trust": "shopper confidence",
    },
    "nut_butter_spreads": {
        "product": "spread and pantry cues",
        "benefit": "protein and ingredient cues",
        "usage": "snack, breakfast, and comparison cues",
        "trust": "shopper confidence",
    },
    "food_generic": {
        "product": "flavor and product cues",
        "benefit": "visible reasons to buy",
        "usage": "serving and use inspiration",
        "trust": "shopper confidence",
    },
    "baby_care": {
        "product": "routine clarity",
        "benefit": "gentle-care benefits",
        "usage": "routine and application guidance",
        "trust": "parent reassurance",
    },
    "health_personal_care": {
        "product": "clearer product story",
        "benefit": "ingredient benefits and proof points",
        "usage": "regimen and application guidance",
        "trust": "efficacy reassurance",
    },
    "generic": {
        "product": "product understanding",
        "benefit": "visible benefits",
        "usage": "usage guidance",
        "trust": "reassurance",
    },
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _lower_blob(*values: Any) -> str:
    return " ".join(_safe_text(value).lower() for value in values if _safe_text(value))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, dict):
        return [_safe_text(item) for item in value.values() if _safe_text(item)]
    text = _safe_text(value)
    return [text] if text else []


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if _as_list(value):
            return value
    return None


def _content_blob(record: dict[str, Any], *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        parts.extend(_as_list(record.get(key)))
    return " ".join(parts).lower()


def _majority_threshold(count: int) -> int:
    return 1 if count == 1 else (count // 2) + 1


def _first_common(records: list[dict[str, Any]], *keys: str) -> str:
    counts: Counter[str] = Counter()
    for record in records:
        for key in keys:
            value = _safe_text(record.get(key))
            if value:
                counts[value] += 1
                break
    return counts.most_common(1)[0][0] if counts else ""


def _category_family(category: str, product_type: str, title: str) -> str:
    blob = _lower_blob(category, product_type, title)
    if any(term in blob for term in ("baby", "infant", "toddler")):
        return "baby_care"
    if any(term in blob for term in ("health", "personal care", "clinical", "dermatolog", "skin", "body", "bath")):
        return "health_personal_care"
    if any(term in blob for term in ("nut butter", "peanut butter", "almond butter")):
        return "nut_butter_spreads"
    if any(term in blob for term in ("jam", "jell", "preserve", "fruit spread")):
        return "jam_preserves"
    if any(term in blob for term in ("food", "beverage", "pantry", "snack", "sauce")):
        return "food_generic"
    return "generic"


def _image_signals(image: dict[str, Any]) -> set[str]:
    return {str(signal) for signal in (image.get("detected_signals") or []) if str(signal).strip()}


def _image_tokens(image: dict[str, Any]) -> set[str]:
    tokens = image.get("ocr_tokens") or []
    if not isinstance(tokens, list):
        return set()
    return {str(token).lower() for token in tokens if str(token).strip()}


def _content_count(record: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if _as_list(value):
            return len(_as_list(value))
    return 0


def _content_signal_evidence(record: dict[str, Any]) -> dict[str, list[str]]:
    title_blob = _content_blob(record, "titleNotes", "title_notes", "product_title", "title")
    benefit_blob = _content_blob(
        record,
        "benefitTerms",
        "benefit_terms",
        "ingredientTerms",
        "ingredient_terms",
        "claimStatements",
        "claim_statements",
        "descriptionNotes",
        "description_notes",
        "keyFeatureNotes",
        "key_feature_notes",
        "description_bullets",
        "descriptionBullets",
        "key_features",
        "keyFeaturesText",
    )
    usage_blob = _content_blob(
        record,
        "usageInstructions",
        "usage_instructions",
        "formFactorTerms",
        "form_factor_terms",
        "audienceTerms",
        "audience_terms",
        "descriptionBullets",
        "description_bullets",
        "keyFeaturesText",
        "key_features",
    )
    trust_blob = _content_blob(
        record,
        "claimStatements",
        "claim_statements",
        "warningStatements",
        "warning_statements",
        "reviews_summary",
        "average_rating",
        "review_count",
        "titleNotes",
        "descriptionNotes",
    )
    description_count = _content_count(record, "descriptionCount", "description_count", "descriptionBullets", "description_bullets")
    key_feature_count = _content_count(record, "keyFeatureCount", "key_feature_count", "keyFeaturesText", "key_features")

    evidence: dict[str, list[str]] = {}

    def add(signal: str, *items: str) -> None:
        evidence.setdefault(signal, [])
        evidence[signal].extend(item for item in items if item)

    if any(term in title_blob for term in ("clear", "specific", "size", "count", "flavor", "scent", "for ", "with ")):
        add("strong_opening_product_clarity", "title_quality")
    elif title_blob and any(term in title_blob for term in ("unclear", "generic", "missing", "thin", "weak")):
        add("weak_opening_sequence", "title_notes")

    if benefit_blob or description_count >= 2 or key_feature_count >= 3:
        if any(term in benefit_blob for term in ("benefit", "protein", "hydrating", "gentle", "clinical", "organic", "ingredient", "claim", "flavor", "nutrition", "proof")):
            add("benefit_forward_graphics", "benefit_terms", "claim_statements", "key_features")
        if key_feature_count >= 3 or description_count >= 2:
            add("nutrition_or_detail_support", "description_and_key_features")
        if any(term in benefit_blob for term in ("ingredient", "flavor", "nutrition", "protein", "organic", "cocoa", "peanut", "almond")):
            add("ingredient_or_flavor_storytelling", "ingredient_or_flavor_terms")
    else:
        add("missing_benefit_graphics", "limited_pdp_benefit_copy")

    if usage_blob:
        if any(term in usage_blob for term in ("use", "apply", "serving", "serve", "recipe", "routine", "regimen", "breakfast", "snack", "fit", "for ", "compare")):
            add("usage_or_recipe_storytelling", "usage_guidance")
        if any(term in usage_blob for term in ("routine", "regimen", "daily", "morning", "night", "step")):
            add("routine_or_regimen_education", "routine_guidance")
    elif description_count == 0 and key_feature_count <= 1:
        add("missing_usage_or_recipe_storytelling", "limited_usage_copy")

    if trust_blob:
        if any(term in trust_blob for term in ("dermatologist", "clinical", "safe", "warning", "caution", "guarantee", "certified", "organic", "rating", "review", "trusted", "reassurance")):
            add("trust_or_certification_support", "trust_or_reassurance_copy")
    else:
        add("missing_trust_or_certification_support", "limited_reassurance_copy")

    return {signal: list(dict.fromkeys(values)) for signal, values in evidence.items()}


def _record_signal_evidence(record: dict[str, Any]) -> tuple[set[str], dict[str, list[str]]]:
    analysis = record.get("image_analysis", {}) or {}
    images = [
        image
        for image in (analysis.get("images", []) or [])
        if isinstance(image, dict) and image.get("status") == "analyzed"
    ]
    analyzed_image_count = len(images)
    if analyzed_image_count <= 0:
        content_evidence = _content_signal_evidence(record)
        return set(content_evidence), content_evidence

    stack = analysis.get("stack_signals", {}) or {}
    all_detected: set[str] = set()
    all_tokens: set[str] = set()
    formats = Counter()
    for image in images:
        all_detected.update(_image_signals(image))
        all_tokens.update(_image_tokens(image))
        fmt = _safe_text(image.get("probable_format"))
        if fmt:
            formats[fmt] += 1

    image_count = int(analysis.get("image_count", 0) or record.get("image_count", 0) or analyzed_image_count)
    duplicate_count = int(stack.get("duplicate_image_count", 0) or 0)
    first_image = images[0] if images else {}
    first_format = _safe_text(first_image.get("probable_format"))
    first_white = float(first_image.get("white_background_ratio", 0.0) or 0.0)

    has_opening_clarity = first_format in {"product_silo", "mixed_product_graphic"} or first_white >= 0.72
    has_lifestyle = formats["lifestyle_or_scene"] > 0
    has_usage = bool({"usage_or_instructions", "recipe_or_serving"} & all_detected)
    has_recipe_tokens = bool({"recipe", "serving", "serve", "breakfast", "snack", "pairing", "toast"} & all_tokens)
    has_ingredient = bool({"ingredients", "nutrition", "flavor", "strawberry", "peanut", "almond"} & all_tokens)
    has_ingredient = has_ingredient or bool({"ingredients", "nutrition_or_ingredients"} & all_detected)
    has_detail = (
        "nutrition_or_ingredients" in all_detected
        or formats["nutrition_or_ingredients"] > 0
        or "dimensions_or_scale" in all_detected
        or formats["dimensions_or_instructions"] > 0
        or "size_or_count" in all_detected
    )
    has_benefit = bool({"feature_or_benefit_claim", "protein_or_nutrition_benefit"} & all_detected)
    has_routine = "routine_or_regimen" in all_detected
    has_trust = bool(
        {
            "organic_or_certification",
            "clinical_or_dermatologist",
            "sustainability_or_recycling",
            "guarantee",
        }
        & all_detected
    )
    text_heavy_count = formats["text_heavy_graphic"]

    signals: set[str] = set()
    evidence: dict[str, list[str]] = {}

    def add(signal: str, *items: str) -> None:
        signals.add(signal)
        evidence[signal] = [item for item in dict.fromkeys(items) if item]

    if has_opening_clarity:
        add("strong_opening_product_clarity", first_format or "hero_product_clarity")
    else:
        add("weak_opening_sequence", first_format or "unclear_opening_image")

    if image_count >= 6:
        add("clear_carousel_depth", f"{image_count}_images")
    elif image_count <= 3:
        add("thin_carousel_depth", f"{image_count}_images")

    if has_ingredient:
        add("ingredient_or_flavor_storytelling", "ingredients", "flavor")
    else:
        add("missing_nutrition_or_ingredient_detail", "missing_ingredient_detail")

    if has_detail:
        add("nutrition_or_detail_support", "nutrition_or_ingredients", "dimensions_or_detail")

    if has_usage or has_recipe_tokens:
        add("usage_or_recipe_storytelling", "usage_or_instructions", "recipe_or_serving")
    else:
        add("missing_usage_or_recipe_storytelling", "missing_recipe")

    if has_lifestyle:
        add("lifestyle_or_contextual_storytelling", "lifestyle_or_scene")
    else:
        add("missing_lifestyle_storytelling", "limited_lifestyle")

    if has_benefit:
        add("benefit_forward_graphics", "feature_or_benefit_claim")
    else:
        add("missing_benefit_graphics", "missing_benefit_graphics")

    if has_routine:
        add("routine_or_regimen_education", "routine_or_regimen")

    if has_trust:
        add("trust_or_certification_support", "trust_or_certification")
    else:
        add("missing_trust_or_certification_support", "missing_trust_support")

    if text_heavy_count >= max(2, round(analyzed_image_count * 0.5)) and not has_lifestyle:
        add("text_heavy_without_clear_hierarchy", "text_heavy_graphic")

    if duplicate_count > 0:
        add("duplicate_or_redundant_images", "duplicate_images")

    if not (has_benefit or has_usage or has_detail):
        add("limited_conversion_guidance", "limited_conversion_guidance")

    for signal, values in _content_signal_evidence(record).items():
        if any(positive in signals for positive in CONFLICTING_CONTENT_OPPORTUNITIES.get(signal, ())):
            continue
        signals.add(signal)
        evidence.setdefault(signal, [])
        evidence[signal].extend(values)
        evidence[signal] = list(dict.fromkeys(evidence[signal]))

    return signals, evidence


TEXT_BY_FAMILY: dict[str, dict[str, str]] = {
    "jam_preserves": {
        "ingredient_or_flavor_storytelling": "Flavor and ingredient cues make the product easier to understand",
        "nutrition_or_detail_support": "Product details give shoppers clearer reasons to buy",
        "usage_or_recipe_storytelling": "Serving guidance helps shoppers understand fit and use",
        "lifestyle_or_contextual_storytelling": "Use-case imagery makes serving occasions easier to picture",
        "benefit_forward_graphics": "Benefit communication is more visible across the PDP",
        "missing_lifestyle_storytelling": "More serving context could make product choice easier",
        "missing_usage_or_recipe_storytelling": "Clearer serving guidance could reduce shopper uncertainty",
    },
    "nut_butter_spreads": {
        "ingredient_or_flavor_storytelling": "Protein and ingredient cues make value easier to understand",
        "nutrition_or_detail_support": "Product details support easier spread comparison",
        "usage_or_recipe_storytelling": "Snack and breakfast guidance helps shoppers understand fit",
        "benefit_forward_graphics": "Benefit communication makes spread value more visible",
        "missing_usage_or_recipe_storytelling": "Clearer snack and breakfast guidance could make choice easier",
        "missing_lifestyle_storytelling": "More use-case imagery could make spread occasions clearer",
    },
    "baby_care": {
        "routine_or_regimen_education": "Routine guidance helps parents understand daily use",
        "usage_or_recipe_storytelling": "Application guidance makes the care routine clearer",
        "lifestyle_or_contextual_storytelling": "Family-care imagery adds parent reassurance",
        "benefit_forward_graphics": "Gentle-care benefits are easier to see across the PDP",
        "missing_lifestyle_storytelling": "More parent reassurance could help build confidence",
        "missing_usage_or_recipe_storytelling": "Clearer routine guidance could reduce uncertainty",
    },
    "health_personal_care": {
        "trust_or_certification_support": "Trust cues reinforce shopper confidence",
        "ingredient_or_flavor_storytelling": "Ingredient benefits make value easier to understand",
        "benefit_forward_graphics": "Proof points make product benefits more visible",
        "nutrition_or_detail_support": "Product details support clearer application guidance",
        "missing_trust_or_certification_support": "Stronger reassurance cues could help build confidence",
    },
    "generic": {
        "strong_opening_product_clarity": "Opening imagery makes the product easier to understand",
        "benefit_forward_graphics": "Benefit communication is more visible across the PDP",
        "nutrition_or_detail_support": "Product details give shoppers clearer reasons to buy",
        "missing_usage_or_recipe_storytelling": "Clearer usage guidance could reduce shopper uncertainty",
        "missing_lifestyle_storytelling": "More use-case context could make product choice easier",
        "limited_conversion_guidance": "The PDP needs more visible reasons to buy",
    },
}


def _finding_text(signal: str, family: str) -> str:
    family_map = TEXT_BY_FAMILY.get(family, {})
    if signal in family_map:
        return family_map[signal]
    if family == "food_generic":
        return TEXT_BY_FAMILY["jam_preserves"].get(signal) or TEXT_BY_FAMILY["generic"].get(signal) or _generic_text(signal)
    return TEXT_BY_FAMILY["generic"].get(signal) or _generic_text(signal)


def _generic_text(signal: str) -> str:
    return {
        "strong_opening_product_clarity": "Opening imagery makes the product easier to understand",
        "ingredient_or_flavor_storytelling": "Ingredient and product cues make value easier to understand",
        "nutrition_or_detail_support": "Product details give shoppers clearer reasons to buy",
        "usage_or_recipe_storytelling": "Usage guidance helps shoppers understand fit and application",
        "lifestyle_or_contextual_storytelling": "Use-case imagery makes product fit easier to picture",
        "benefit_forward_graphics": "Benefit communication is more visible across the PDP",
        "routine_or_regimen_education": "Routine guidance helps shoppers understand how to use the product",
        "trust_or_certification_support": "Trust and certification cues support shopper confidence",
        "clear_carousel_depth": "Carousel depth helps shoppers compare product details",
        "weak_opening_sequence": "The PDP would benefit from a clearer opening product story",
        "thin_carousel_depth": "More PDP detail could help shoppers compare options",
        "missing_lifestyle_storytelling": "More use-case context could make product choice easier",
        "missing_usage_or_recipe_storytelling": "Clearer usage guidance could reduce shopper uncertainty",
        "missing_benefit_graphics": "More visible proof points could strengthen reasons to buy",
        "missing_nutrition_or_ingredient_detail": "Clearer product detail could make value easier to understand",
        "missing_trust_or_certification_support": "Stronger reassurance cues could help build confidence",
        "text_heavy_without_clear_hierarchy": "Clearer visual hierarchy could make comparison easier",
        "duplicate_or_redundant_images": "More varied imagery could answer more shopper questions",
        "limited_conversion_guidance": "The PDP needs more visible reasons to buy",
    }.get(signal, signal.replace("_", " ").capitalize())


def _bucket_label(bucket: str) -> str:
    return {
        "opening_clarity": "Opening clarity",
        "benefit_proof": "Benefit / proof communication",
        "usage_comparison": "Usage / comparison guidance",
        "trust_reassurance": "Trust / reassurance",
    }.get(bucket, bucket.replace("_", " ").title())


def _category_words(family: str) -> dict[str, str]:
    return CATEGORY_WORDS.get(family) or CATEGORY_WORDS["generic"]


def _finding_priority(signal: str, evidence: list[str]) -> tuple[int, int]:
    bucket = SIGNAL_BUCKETS.get(signal, "")
    bucket_weight = {
        "opening_clarity": 4,
        "benefit_proof": 5,
        "usage_comparison": 4,
        "trust_reassurance": 3,
    }.get(bucket, 1)
    content_weight = sum(
        1
        for item in evidence
        if item
        in {
            "title_quality",
            "title_notes",
            "benefit_terms",
            "claim_statements",
            "key_features",
            "description_and_key_features",
            "usage_guidance",
            "routine_guidance",
            "trust_or_reassurance_copy",
        }
    )
    combined_weight = 3 if content_weight and any("graphic" in item or "image" in item or "format" in item for item in evidence) else 0
    return bucket_weight + min(content_weight, 3) + combined_weight, content_weight


def _build_finding(
    signal: str,
    *,
    family: str,
    supporting_pdps: int,
    analyzed_pdps: int,
    evidence: list[str],
) -> dict[str, Any]:
    bucket = SIGNAL_BUCKETS.get(signal, "usage_comparison")
    return {
        "text": _finding_text(signal, family),
        "signal": signal,
        "bucket": _bucket_label(bucket),
        "bucket_key": bucket,
        "supporting_pdps": supporting_pdps,
        "analyzed_pdps": analyzed_pdps,
        "evidence": evidence,
        "priority": _finding_priority(signal, evidence)[0],
    }


def build_slide4_group_findings(records: list[dict[str, Any]], group_label: str) -> dict[str, Any]:
    source_records = [record for record in (records or []) if isinstance(record, dict)]
    analyzed_records = [
        record
        for record in source_records
        if int((record.get("image_analysis", {}) or {}).get("analyzed_image_count", 0) or 0) > 0
    ]
    analyzed_count = len(analyzed_records)
    threshold = _majority_threshold(analyzed_count) if analyzed_count else 0
    category = _first_common(analyzed_records or source_records, "category")
    product_type = _first_common(analyzed_records or source_records, "product_type", "subcategory")
    title = _first_common(analyzed_records or source_records, "product_title", "title")
    family = _category_family(category, product_type, title)
    words = _category_words(family)

    signal_counts: Counter[str] = Counter()
    evidence_by_signal: dict[str, list[str]] = {}
    bucket_counts: Counter[str] = Counter()
    for record in analyzed_records:
        signals, evidence = _record_signal_evidence(record)
        signal_counts.update(signals)
        bucket_counts.update(SIGNAL_BUCKETS.get(signal, "usage_comparison") for signal in signals)
        for signal, values in evidence.items():
            evidence_by_signal.setdefault(signal, [])
            evidence_by_signal[signal].extend(values)

    strengths: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    if analyzed_count:
        for signal in STRENGTH_SIGNALS:
            count = signal_counts.get(signal, 0)
            if count >= threshold:
                strengths.append(
                    _build_finding(
                        signal,
                        family=family,
                        supporting_pdps=count,
                        analyzed_pdps=analyzed_count,
                        evidence=list(dict.fromkeys(evidence_by_signal.get(signal, [])))[:5],
                    )
                )
        for signal in OPPORTUNITY_SIGNALS:
            count = signal_counts.get(signal, 0)
            if count >= threshold:
                opportunities.append(
                    _build_finding(
                        signal,
                        family=family,
                        supporting_pdps=count,
                        analyzed_pdps=analyzed_count,
                        evidence=list(dict.fromkeys(evidence_by_signal.get(signal, [])))[:5],
                    )
                )

    strengths.sort(key=lambda finding: (finding["priority"], finding["supporting_pdps"]), reverse=True)
    opportunities.sort(key=lambda finding: (finding["priority"], finding["supporting_pdps"]), reverse=True)

    selected_findings = [*strengths, *opportunities]
    selected_findings = _dedupe_selected_findings(selected_findings)

    return {
        "group_label": group_label,
        "analyzed_pdp_count": analyzed_count,
        "majority_threshold": threshold,
        "category": category,
        "product_type": product_type,
        "strengths": strengths,
        "opportunities": opportunities,
        "slide4_bullets": [finding["text"] for finding in selected_findings],
        "debug": {
            "category_family": family,
            "category_words": words,
            "signal_counts": dict(signal_counts),
            "bucket_counts": dict(bucket_counts),
            "selected_signals": [finding["signal"] for finding in selected_findings],
            "selected_buckets": [finding.get("bucket_key") for finding in selected_findings],
        },
    }


def _dedupe_selected_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_buckets: set[str] = set()
    seen_text: set[str] = set()
    for finding in findings:
        text_key = _safe_text(finding.get("text")).lower()
        bucket_key = _safe_text(finding.get("bucket_key"))
        if text_key in seen_text or bucket_key in seen_buckets:
            continue
        selected.append(finding)
        seen_text.add(text_key)
        seen_buckets.add(bucket_key)
        if len(selected) >= 4:
            break
    if len(selected) < 4:
        for finding in findings:
            text_key = _safe_text(finding.get("text")).lower()
            if text_key not in seen_text:
                selected.append(finding)
                seen_text.add(text_key)
            if len(selected) >= 4:
                break
    return selected
