"""Shared language helpers for strategic audit PowerPoint copy."""

from __future__ import annotations

import re
from typing import Any


LANGUAGE_PROFILES: dict[str, dict[str, Any]] = {
    "food_beverage": {
        "context": "pantry shelf",
        "benefit": "benefit and flavor communication",
        "formula": "ingredient and nutrition detail",
        "education": "nutrition and usage education",
        "story": "breakfast, snack, and recipe storytelling",
        "navigation": "pantry shelf navigation",
        "conversion": "serving and conversion guidance",
        "discovery": "category discovery and comparison",
        "search_intent": "pantry search intent alignment",
        "shelf_visibility": "pantry shelf visibility",
        "assortment": "flavor, size, and pack segmentation",
        "category_grouping": "pantry category grouping",
        "discovery_pathways": "pantry discovery pathways",
        "cross_category": "meal and pantry cross-shopping",
        "trust": "review and retailer trust signals",
    },
    "beauty": {
        "context": "beauty shelf",
        "benefit": "benefit and concern communication",
        "formula": "formula and ingredient detail",
        "education": "regimen education",
        "story": "routine and lifestyle storytelling",
        "navigation": "concern-based navigation",
        "conversion": "routine-building guidance",
        "discovery": "concern-led discovery and comparison",
        "search_intent": "concern-led search intent alignment",
        "shelf_visibility": "beauty shelf visibility",
        "assortment": "formula, concern, and format segmentation",
        "category_grouping": "concern-based category grouping",
        "discovery_pathways": "routine-led discovery pathways",
        "cross_category": "routine and regimen cross-shopping",
        "trust": "review and skin-confidence signals",
    },
    "health_personal_care": {
        "context": "wellness shelf",
        "benefit": "active support communication",
        "formula": "dosage and form detail",
        "education": "symptom and wellness guidance",
        "story": "routine and usage guidance",
        "navigation": "wellness shelf navigation",
        "conversion": "usage and confidence guidance",
        "discovery": "wellness discovery and comparison",
        "search_intent": "symptom and need-state search alignment",
        "shelf_visibility": "wellness shelf visibility",
        "assortment": "active, form, and dosage segmentation",
        "category_grouping": "wellness need-state grouping",
        "discovery_pathways": "condition-led discovery pathways",
        "cross_category": "wellness routine cross-shopping",
        "trust": "review and confidence signals",
    },
    "animals": {
        "context": "pet care shelf",
        "benefit": "nutrition and care communication",
        "formula": "ingredient and feeding detail",
        "education": "life-stage shopper education",
        "story": "feeding and care storytelling",
        "navigation": "life-stage navigation",
        "conversion": "feeding and care guidance",
        "discovery": "pet care discovery and comparison",
        "search_intent": "pet need-state search alignment",
        "shelf_visibility": "pet care shelf visibility",
        "assortment": "life-stage, size, and formula segmentation",
        "category_grouping": "life-stage category grouping",
        "discovery_pathways": "feeding and care discovery pathways",
        "cross_category": "pet routine cross-shopping",
        "trust": "review and care-confidence signals",
    },
    "electronics": {
        "context": "electronics shelf",
        "benefit": "performance communication",
        "formula": "compatibility and spec detail",
        "education": "setup and use-case guidance",
        "story": "use-case storytelling",
        "navigation": "device-use navigation",
        "conversion": "compatibility and purchase guidance",
        "discovery": "device discovery and comparison",
        "search_intent": "device search intent alignment",
        "shelf_visibility": "electronics shelf visibility",
        "assortment": "model, feature, and use-case segmentation",
        "category_grouping": "device-use category grouping",
        "discovery_pathways": "use-case discovery pathways",
        "cross_category": "device ecosystem cross-shopping",
        "trust": "review and spec-confidence signals",
    },
}

GENERIC_PROFILE = {
    "context": "category shelf",
    "benefit": "benefit communication",
    "formula": "pack and product detail",
    "education": "shopper education",
    "story": "usage storytelling",
    "navigation": "category navigation",
    "conversion": "conversion-focused guidance",
    "discovery": "category discovery and comparison",
    "search_intent": "search intent alignment",
    "shelf_visibility": "shelf visibility",
    "assortment": "assortment segmentation",
    "category_grouping": "category grouping",
    "discovery_pathways": "discovery pathways",
    "cross_category": "cross-category navigation",
    "trust": "review and trust signals",
}

CUE_LANGUAGE_KEYS = {
    "product_positioning": "benefit",
    "benefit_communication": "benefit",
    "ingredient_or_formula_communication": "formula",
    "pack_or_spec_detail": "formula",
    "shopper_education": "education",
    "usage_storytelling": "story",
    "visual_identity": "context",
    "keyword_alignment": "search_intent",
    "discoverability": "shelf_visibility",
    "assortment_segmentation": "assortment",
    "category_grouping": "category_grouping",
    "discovery_pathways": "discovery_pathways",
    "cross_category_navigation": "cross_category",
    "review_or_trust_signals": "trust",
    "conversion_guidance": "conversion",
}

PRODUCT_LANGUAGE_OVERLAYS: dict[str, tuple[dict[str, Any], ...]] = {
    "food_beverage": (
        {
            "terms": ("hazelnut", "chocolate spread"),
            "values": {
                "product_context": "hazelnut spread shelf",
                "benefit": "spreadability, flavor, and pantry relevance",
                "formula": "ingredient, nutrition, and flavor detail",
                "education": "serving, recipe, and breakfast education",
                "story": "breakfast, baking, and snack storytelling",
                "assortment": "flavor, size, and multipack segmentation",
                "conversion": "serving-size and basket-building guidance",
            },
        },
        {
            "terms": ("peanut butter", "nut butter", "almond butter"),
            "values": {
                "product_context": "nut butter shelf",
                "benefit": "protein, texture, and flavor communication",
                "formula": "ingredient, nutrition, and allergen detail",
                "education": "nutrition, recipe, and usage education",
                "story": "breakfast, snack, and recipe storytelling",
                "assortment": "texture, flavor, size, and pack segmentation",
                "conversion": "serving and pantry basket guidance",
            },
        },
    ),
    "beauty": (
        {
            "terms": ("facial cleanser", "face cleanser", "face wash", "cleanser"),
            "values": {
                "product_context": "facial cleanser shelf",
                "benefit": "clean-skin benefit and concern communication",
                "formula": "formula, active, and sensitive-skin detail",
                "education": "regimen and usage-step education",
                "story": "morning and night routine storytelling",
                "assortment": "skin concern, formula, and format segmentation",
                "conversion": "routine-building and skin-fit guidance",
                "discovery_pathways": "concern and regimen discovery pathways",
            },
        },
        {
            "terms": ("moisturizer", "moisturiser", "cream", "lotion"),
            "values": {
                "product_context": "moisturizer shelf",
                "benefit": "hydration, barrier, and concern communication",
                "formula": "formula, active, and skin-type detail",
                "education": "regimen education for layering and usage",
                "story": "daily hydration routine storytelling",
                "assortment": "skin type, benefit, and texture segmentation",
                "conversion": "skin-fit and routine-building guidance",
            },
        },
    ),
    "health_personal_care": (
        {
            "terms": ("vitamin", "supplement", "mineral", "gummy"),
            "values": {
                "product_context": "supplement shelf",
                "benefit": "active support and wellness benefit communication",
                "formula": "active, dosage, and format detail",
                "education": "dosage, routine, and wellness education",
                "story": "daily wellness routine guidance",
                "assortment": "active, form, count, and dosage segmentation",
                "conversion": "need-state and confidence guidance",
            },
        },
    ),
    "animals": (
        {
            "terms": ("dog", "cat", "pet food", "treat", "chew"),
            "values": {
                "product_context": "pet nutrition shelf",
                "benefit": "nutrition, life-stage, and care communication",
                "formula": "ingredient, feeding, and support detail",
                "education": "feeding, life-stage, and care education",
                "story": "feeding and care routine storytelling",
                "assortment": "life-stage, breed size, and formula segmentation",
                "conversion": "feeding-fit and care guidance",
            },
        },
    ),
    "electronics": (
        {
            "terms": ("speaker", "headphone", "earbud", "audio", "bluetooth"),
            "values": {
                "product_context": "audio device shelf",
                "benefit": "sound, portability, and performance communication",
                "formula": "compatibility, battery, and spec detail",
                "education": "setup, pairing, and use-case guidance",
                "story": "home, travel, and entertainment use-case storytelling",
                "assortment": "model, feature, and use-case segmentation",
                "conversion": "compatibility and purchase-confidence guidance",
            },
        },
        {
            "terms": ("charger", "cable", "adapter", "case"),
            "values": {
                "product_context": "device accessory shelf",
                "benefit": "compatibility and utility communication",
                "formula": "device-fit, spec, and setup detail",
                "education": "compatibility and setup guidance",
                "story": "everyday device-use storytelling",
                "assortment": "device, model, and spec segmentation",
                "conversion": "fit-check and purchase-confidence guidance",
            },
        },
    ),
}

INVALID_LANGUAGE_GUARDS: dict[str, tuple[tuple[str, str], ...]] = {
    "food_beverage": (
        ("regimen education", "nutrition and usage education"),
        ("skin-fit", "shopper-fit"),
        ("sensitive-skin", "ingredient"),
        ("device", "pantry"),
        ("compatibility and spec", "ingredient and nutrition"),
    ),
    "beauty": (
        ("nutrition detail", "formula detail"),
        ("nutrition and usage", "regimen and usage"),
        ("breakfast, snack, and recipe", "routine and lifestyle"),
        ("serving", "usage"),
        ("pantry", "beauty"),
        ("feeding", "routine"),
    ),
    "electronics": (
        ("ingredient and nutrition", "compatibility and spec"),
        ("ingredient detail", "spec detail"),
        ("formula and ingredient", "compatibility and spec"),
        ("regimen", "setup"),
        ("nutrition", "performance"),
        ("feeding", "setup"),
    ),
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _safe_text(value).lower()).strip()


def _first_nonempty(values: list[Any], fallback: str = "") -> str:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return fallback


def _identity_blob(identity: dict[str, Any]) -> str:
    values = [
        identity.get("product_type_display"),
        identity.get("family_display"),
        identity.get("category_display"),
        identity.get("product_type_key"),
        identity.get("family_key"),
        identity.get("combined_category_phrase"),
        identity.get("shopping_context_phrase"),
    ]
    return _norm(" ".join(_safe_text(value) for value in values if _safe_text(value)))


def _overlay_values(identity: dict[str, Any]) -> dict[str, str]:
    category_key = _safe_text(identity.get("category_key"))
    blob = _identity_blob(identity)
    values: dict[str, str] = {}
    for overlay in PRODUCT_LANGUAGE_OVERLAYS.get(category_key, ()):
        if any(_norm(term) and _norm(term) in blob for term in overlay.get("terms", ())):
            values.update(overlay.get("values", {}))
    return values


def _guard_language(text: str, profile: dict[str, str]) -> str:
    category_key = profile.get("category_key", "")
    guarded = _safe_text(text)
    for bad, replacement in INVALID_LANGUAGE_GUARDS.get(category_key, ()):
        guarded = re.sub(re.escape(bad), replacement, guarded, flags=re.I)
    return re.sub(r"\s+", " ", guarded).strip()


def _phrase_variant(options: tuple[str, ...], candidate: dict[str, Any], side: str = "") -> str:
    if not options:
        return ""
    seed = " ".join(
        [
            _safe_text(candidate.get("cue_key") or candidate.get("cue")),
            _safe_text(candidate.get("classification")),
            _safe_text(side),
        ]
    )
    return options[sum(ord(char) for char in seed) % len(options)]


def language_profile(identity: dict[str, Any] | None) -> dict[str, str]:
    identity = identity or {}
    profile = dict(GENERIC_PROFILE)
    profile.update(LANGUAGE_PROFILES.get(identity.get("category_key"), {}))
    profile.update(_overlay_values(identity))
    product = _first_nonempty(
        [
            identity.get("product_type_display"),
            identity.get("family_display"),
            identity.get("category_display"),
        ],
        "category",
    )
    family = _first_nonempty([identity.get("family_display"), identity.get("category_display")], product)
    profile["product"] = product
    profile["family"] = family
    profile["category"] = _first_nonempty([identity.get("category_display")], family)
    profile["shopping_context"] = _first_nonempty(
        [identity.get("shopping_context_phrase")],
        f"{product} shopping journey",
    )
    profile["product_focus"] = _first_nonempty(
        [identity.get("product_type_focus_phrase")],
        f"{product} positioning",
    )
    profile["category_key"] = _safe_text(identity.get("category_key"))
    profile["product_context"] = profile.get("product_context") or profile["context"]
    for key, value in list(profile.items()):
        if isinstance(value, str):
            profile[key] = _guard_language(value, profile)
    return profile


def cue_language_label(identity: dict[str, Any], cue_key: str) -> str:
    profile = language_profile(identity)
    label = profile.get(CUE_LANGUAGE_KEYS.get(cue_key, ""), cue_key.replace("_", " "))
    return _guard_language(label, profile)


def strategic_bullet_text(
    candidate: dict[str, Any],
    identity: dict[str, Any],
    *,
    slide_key: str,
    side: str = "",
    evidence_terms: dict[str, Any] | None = None,
) -> str:
    """Translate one cue candidate into compact slide-style language."""
    profile = language_profile(identity)
    cue = candidate.get("cue_key") or candidate.get("cue") or ""
    classification = candidate.get("classification") or "context"
    product = profile["product"]
    evidence_terms = evidence_terms or {}
    context = profile["product_context"]
    action_tone = "Opportunity to" if classification == "opportunity" else "Clearer" if classification == "pressure" else "Stronger"

    if slide_key == "slide2":
        if cue in {"product_positioning", "benefit_communication"}:
            text = _phrase_variant(
                (
                    f"Sharper {profile['benefit']} across Walmart",
                    f"{action_tone} {product} value communication",
                    f"More ownable {context} positioning",
                ),
                candidate,
                side,
            )
            return _guard_language(text, profile)
        if cue == "keyword_alignment":
            return _guard_language(f"Sharper {profile['search_intent']} opportunity", profile)
        if cue == "discoverability":
            return _guard_language(f"Clearer {profile['shelf_visibility']} opportunity", profile)
        if cue == "assortment_segmentation":
            return _guard_language(f"More useful {profile['assortment']} story", profile)
        if cue in {"shopper_education", "usage_storytelling"}:
            text = (
                f"Structured {profile['education']} opportunity"
                if cue == "shopper_education"
                else f"Richer {profile['story']} opportunity"
            )
            return _guard_language(text, profile)
        if cue == "review_or_trust_signals":
            return _guard_language(f"{profile['trust'].title()} support conversion confidence", profile)
        if cue == "conversion_guidance":
            return _guard_language(f"Focused {profile['conversion']} opportunity", profile)
        return _guard_language(f"Focused {context} opportunity", profile)

    if slide_key == "slide3":
        if cue == "keyword_alignment":
            return _guard_language(f"Sharper {profile['search_intent']}", profile)
        if cue == "discoverability":
            return _guard_language(f"Clearer {profile['shelf_visibility']} and discovery", profile)
        if cue == "assortment_segmentation":
            return _guard_language(f"Sharper {profile['assortment']}", profile)
        if cue == "review_or_trust_signals":
            return _guard_language(f"{profile['trust'].title()} shape shelf confidence", profile)
        if cue == "conversion_guidance":
            return _guard_language(f"Clearer {profile['conversion']} from search", profile)
        return _guard_language(f"Stronger {context} discovery signals", profile)

    if slide_key == "slide4":
        theme = _safe_text(evidence_terms.get("theme"))
        if cue == "product_positioning":
            text = _first_nonempty(
                [evidence_terms.get("positioning")],
                _phrase_variant(
                    (
                        f"Benefit-forward {product} PDP positioning",
                        f"Sharper {profile['product_focus']} on the PDP",
                        f"More ownable {context} PDP positioning",
                    ),
                    candidate,
                    side,
                ),
            )
            return _guard_language(text, profile)
        if cue == "benefit_communication":
            return _guard_language(_first_nonempty([evidence_terms.get("benefit")], f"Clear {profile['benefit']}"), profile)
        if cue in {"ingredient_or_formula_communication", "pack_or_spec_detail"}:
            prefix = "Complete" if cue == "pack_or_spec_detail" else "Clear"
            return _guard_language(_first_nonempty([evidence_terms.get("detail")], f"{prefix} {profile['formula']}"), profile)
        if cue == "shopper_education":
            return _guard_language(f"Structured {profile['education']}", profile)
        if cue == "usage_storytelling":
            return _guard_language(_first_nonempty([evidence_terms.get("story")], f"Balanced {profile['story']}"), profile)
        if cue == "visual_identity":
            return _guard_language(_first_nonempty([evidence_terms.get("visual")], f"Cohesive {context} visual education"), profile)
        if cue == "review_or_trust_signals":
            return _guard_language(f"{profile['trust'].title()} strengthen purchase confidence", profile)
        if cue == "conversion_guidance":
            return _guard_language(f"Clear {profile['conversion']}", profile)
        if classification == "opportunity":
            return _guard_language(f"Opportunity to deepen {profile['conversion']}", profile)
        return _guard_language(f"Focused {product} PDP guidance" if not theme else f"Focused {theme} PDP guidance", profile)

    if slide_key == "slide5":
        if cue == "category_grouping":
            return _guard_language(_first_nonempty([evidence_terms.get("category_grouping")], f"Structured {profile['category_grouping']}"), profile)
        if cue == "discovery_pathways":
            return _guard_language(_first_nonempty([evidence_terms.get("discovery")], f"Expanded {profile['discovery_pathways']}"), profile)
        if cue == "cross_category_navigation":
            return _guard_language(_first_nonempty([evidence_terms.get("cross_category")], f"Stronger {profile['cross_category']}"), profile)
        if cue == "shopper_education":
            return _guard_language(_first_nonempty([evidence_terms.get("education")], f"Structured {profile['education']}"), profile)
        if cue == "usage_storytelling":
            return _guard_language(_first_nonempty([evidence_terms.get("story")], f"Immersive {profile['story']}"), profile)
        if cue == "visual_identity":
            return _guard_language(f"Cohesive {context} identity anchors the Brand Shop", profile)
        if cue == "assortment_segmentation":
            return _guard_language(f"{profile['assortment'].title()} clarifies product choice", profile)
        if cue == "conversion_guidance":
            return _guard_language(f"Benefit-led {profile['conversion']}", profile)
        if cue == "review_or_trust_signals":
            return _guard_language(f"{profile['trust'].title()} reinforce Brand Shop confidence", profile)
        return _guard_language(f"Focused {profile['navigation']}", profile)

    return _guard_language(f"Focused {product} content", profile)


def recommendation_phrase(identity: dict[str, Any], recommendation_key: str) -> str:
    profile = language_profile(identity)
    phrases = {
        "seo": f"Strengthen {profile['product']} SEO around {profile['search_intent']}",
        "search_intent": f"Align titles and PDP language to {profile['search_intent']}",
        "assortment": f"Clarify {profile['assortment']} across shelf and Brand Shop",
        "brand_shop": f"Expand Brand Shop modules around {profile['discovery_pathways']}",
        "education": f"Build richer {profile['education']} across PDP and Brand Shop",
        "storytelling": f"Add commercial {profile['story']} to support shopper context",
        "taxonomy": f"Tighten taxonomy and filters around {profile['category_grouping']}",
        "attributes": f"Complete priority attributes for {profile['formula']}",
        "discovery": f"Expand {profile['discovery']} through {profile['discovery_pathways']}",
        "trust": f"Strengthen {profile['trust']} near key conversion points",
        "reviews": f"Increase review depth and visibility for {profile['product']}",
        "conversion": f"Add clearer {profile['conversion']} at decision points",
        "category_grouping": f"Structure Brand Shop shelves around {profile['category_grouping']}",
        "cross_category": f"Build {profile['cross_category']} for adjacent shopper missions",
        "visual_identity": f"Create a more cohesive {profile['product_context']} visual system",
        "pathways": f"Add guided entry points for {profile['discovery_pathways']}",
    }
    return _guard_language(
        phrases.get(recommendation_key, f"Strengthen {profile['product']} shopper guidance"),
        profile,
    )
