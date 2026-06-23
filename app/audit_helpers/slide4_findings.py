from __future__ import annotations

from collections import Counter
from typing import Any


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


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _lower_blob(*values: Any) -> str:
    return " ".join(_safe_text(value).lower() for value in values if _safe_text(value))


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


def _record_signal_evidence(record: dict[str, Any]) -> tuple[set[str], dict[str, list[str]]]:
    analysis = record.get("image_analysis", {}) or {}
    images = [
        image
        for image in (analysis.get("images", []) or [])
        if isinstance(image, dict) and image.get("status") == "analyzed"
    ]
    analyzed_image_count = len(images)
    if analyzed_image_count <= 0:
        return set(), {}

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

    return signals, evidence


TEXT_BY_FAMILY: dict[str, dict[str, str]] = {
    "jam_preserves": {
        "ingredient_or_flavor_storytelling": "Strong flavor-forward visual identity",
        "nutrition_or_detail_support": "Clear ingredient and nutrition support across carousel",
        "usage_or_recipe_storytelling": "Recipe-led serving inspiration supports shopper use cases",
        "lifestyle_or_contextual_storytelling": "Lifestyle imagery reinforces breakfast, snack, and pairing occasions",
        "benefit_forward_graphics": "Benefit and product-detail communication supports conversion confidence",
        "missing_lifestyle_storytelling": "Opportunity to strengthen breakfast, snack, and pairing use cases",
        "missing_usage_or_recipe_storytelling": "Opportunity to expand recipe-led serving inspiration",
    },
    "nut_butter_spreads": {
        "ingredient_or_flavor_storytelling": "Strong protein and ingredient-led benefit communication",
        "nutrition_or_detail_support": "Clear pack and nutrition detail support",
        "usage_or_recipe_storytelling": "Snack, breakfast, and recipe-based usage storytelling supports spread use cases",
        "benefit_forward_graphics": "Benefit-forward graphics help differentiate spread use cases",
        "missing_usage_or_recipe_storytelling": "Opportunity to expand snack, breakfast, and recipe-based usage storytelling",
        "missing_lifestyle_storytelling": "Opportunity to show more snack, breakfast, and recipe usage occasions",
    },
    "baby_care": {
        "routine_or_regimen_education": "Routine-based merchandising approach",
        "usage_or_recipe_storytelling": "Embedded usage and regimen education",
        "lifestyle_or_contextual_storytelling": "Soft lifestyle positioning aligned to family care",
        "benefit_forward_graphics": "Benefit communication integrated into use-case imagery",
        "missing_lifestyle_storytelling": "Opportunity to expand parent-focused reassurance and bath-time storytelling",
        "missing_usage_or_recipe_storytelling": "Opportunity to strengthen routine and usage education",
    },
    "health_personal_care": {
        "trust_or_certification_support": "Clinical trust and efficacy positioning",
        "ingredient_or_flavor_storytelling": "Ingredient-led educational storytelling",
        "benefit_forward_graphics": "Clear hierarchy of claims and product benefits",
        "nutrition_or_detail_support": "Structured shopper education throughout image stack",
        "missing_trust_or_certification_support": "Opportunity to strengthen trust and efficacy reassurance",
    },
    "generic": {
        "strong_opening_product_clarity": "Strong visual shelf presence",
        "benefit_forward_graphics": "Benefit-forward infographic integration",
        "nutrition_or_detail_support": "Clear product-detail and conversion support",
        "missing_usage_or_recipe_storytelling": "Opportunity to strengthen use-case and lifestyle storytelling",
        "missing_lifestyle_storytelling": "Opportunity to strengthen use-case and lifestyle storytelling",
        "limited_conversion_guidance": "Opportunity to expand educational merchandising depth",
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
        "strong_opening_product_clarity": "Strong visual shelf presence",
        "ingredient_or_flavor_storytelling": "Ingredient and flavor-forward merchandising is consistently present",
        "nutrition_or_detail_support": "Clear product-detail and conversion support",
        "usage_or_recipe_storytelling": "Use-case and usage storytelling is consistently present",
        "lifestyle_or_contextual_storytelling": "Lifestyle context supports shopper use-case understanding",
        "benefit_forward_graphics": "Benefit-forward infographic integration",
        "routine_or_regimen_education": "Routine-based education supports shopper understanding",
        "trust_or_certification_support": "Trust and certification cues support shopper confidence",
        "clear_carousel_depth": "Carousel depth supports shopper education",
        "weak_opening_sequence": "Opportunity to clarify the opening product sequence",
        "thin_carousel_depth": "Opportunity to expand educational merchandising depth",
        "missing_lifestyle_storytelling": "Opportunity to strengthen use-case and lifestyle storytelling",
        "missing_usage_or_recipe_storytelling": "Opportunity to expand usage and serving guidance",
        "missing_benefit_graphics": "Opportunity to strengthen benefit-forward graphic communication",
        "missing_nutrition_or_ingredient_detail": "Opportunity to add clearer product-detail support",
        "missing_trust_or_certification_support": "Opportunity to add stronger trust and reassurance cues",
        "text_heavy_without_clear_hierarchy": "Carousel appears to overuse text-heavy graphics without clear lifestyle hierarchy",
        "duplicate_or_redundant_images": "Opportunity to reduce redundant imagery and diversify shopper education",
        "limited_conversion_guidance": "Opportunity to expand educational merchandising depth",
    }.get(signal, signal.replace("_", " ").capitalize())


def _build_finding(
    signal: str,
    *,
    family: str,
    supporting_pdps: int,
    analyzed_pdps: int,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "text": _finding_text(signal, family),
        "signal": signal,
        "supporting_pdps": supporting_pdps,
        "analyzed_pdps": analyzed_pdps,
        "evidence": evidence,
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

    signal_counts: Counter[str] = Counter()
    evidence_by_signal: dict[str, list[str]] = {}
    for record in analyzed_records:
        signals, evidence = _record_signal_evidence(record)
        signal_counts.update(signals)
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

    selected_findings = [*strengths[:2], *opportunities[:2]]
    if len(selected_findings) < 4:
        selected_findings = [*strengths, *opportunities][:4]

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
            "signal_counts": dict(signal_counts),
            "selected_signals": [finding["signal"] for finding in selected_findings],
        },
    }
