"""Shared language helpers for strategic audit PowerPoint copy."""

from __future__ import annotations

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
}

CUE_LANGUAGE_KEYS = {
    "product_positioning": "benefit",
    "benefit_communication": "benefit",
    "ingredient_or_formula_communication": "formula",
    "pack_or_spec_detail": "formula",
    "shopper_education": "education",
    "usage_storytelling": "story",
    "visual_identity": "context",
    "keyword_alignment": "discovery",
    "discoverability": "discovery",
    "assortment_segmentation": "discovery",
    "category_grouping": "navigation",
    "discovery_pathways": "navigation",
    "cross_category_navigation": "navigation",
    "review_or_trust_signals": "conversion",
    "conversion_guidance": "conversion",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _first_nonempty(values: list[Any], fallback: str = "") -> str:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return fallback


def language_profile(identity: dict[str, Any] | None) -> dict[str, str]:
    identity = identity or {}
    profile = dict(GENERIC_PROFILE)
    profile.update(LANGUAGE_PROFILES.get(identity.get("category_key"), {}))
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
    return profile


def cue_language_label(identity: dict[str, Any], cue_key: str) -> str:
    profile = language_profile(identity)
    return profile.get(CUE_LANGUAGE_KEYS.get(cue_key, ""), cue_key.replace("_", " "))


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

    if slide_key == "slide2":
        if cue in {"product_positioning", "benefit_communication"}:
            return f"Stronger {profile['benefit']} across Walmart"
        if cue in {"discoverability", "keyword_alignment"}:
            return f"Clearer {profile['discovery']} opportunity"
        if cue in {"shopper_education", "usage_storytelling"}:
            return f"Structured {profile['education']} opportunity"
        if cue == "review_or_trust_signals":
            return "Review depth supports conversion confidence"
        return f"Focused {profile['conversion']} opportunity"

    if slide_key == "slide3":
        if cue == "keyword_alignment":
            return f"Sharper {product} search intent alignment"
        if cue == "discoverability":
            return f"Clearer {product} search and discovery visibility"
        if cue == "assortment_segmentation":
            return f"Sharper {profile['discovery']} cues"
        if cue == "review_or_trust_signals":
            return "Review depth shapes shelf confidence"
        return f"Stronger {profile['context']} discovery signals"

    if slide_key == "slide4":
        theme = _safe_text(evidence_terms.get("theme"))
        if cue == "product_positioning":
            return _first_nonempty([evidence_terms.get("positioning")], f"Benefit-forward {product} PDP positioning")
        if cue == "benefit_communication":
            return _first_nonempty([evidence_terms.get("benefit")], f"Clear {profile['benefit']}")
        if cue in {"ingredient_or_formula_communication", "pack_or_spec_detail"}:
            return _first_nonempty([evidence_terms.get("detail")], f"Clear {profile['formula']}")
        if cue == "shopper_education":
            return f"Structured {profile['education']}"
        if cue == "usage_storytelling":
            return _first_nonempty([evidence_terms.get("story")], f"Balanced {profile['story']}")
        if cue == "visual_identity":
            return _first_nonempty([evidence_terms.get("visual")], "Cohesive PDP visual education")
        if cue == "review_or_trust_signals":
            return "Review depth strengthens purchase confidence"
        if classification == "opportunity":
            return f"Opportunity to deepen {profile['conversion']}"
        return f"Focused {product} PDP guidance" if not theme else f"Focused {theme} PDP guidance"

    if slide_key == "slide5":
        if cue == "category_grouping":
            return _first_nonempty([evidence_terms.get("category_grouping")], f"Structured {profile['navigation']}")
        if cue == "discovery_pathways":
            return _first_nonempty([evidence_terms.get("discovery")], "Broad assortment depth supports discovery")
        if cue == "cross_category_navigation":
            return _first_nonempty([evidence_terms.get("cross_category")], "Cross-category pathways support exploration")
        if cue == "shopper_education":
            return _first_nonempty([evidence_terms.get("education")], f"Structured {profile['education']}")
        if cue == "usage_storytelling":
            return _first_nonempty([evidence_terms.get("story")], f"Immersive {profile['story']}")
        if cue == "visual_identity":
            return "Cohesive visual identity anchors the Brand Shop"
        if cue == "assortment_segmentation":
            return "Assortment segmentation clarifies product choice"
        if cue == "conversion_guidance":
            return f"Benefit-led {profile['conversion']}"
        return f"Focused {profile['navigation']}"

    return f"Focused {product} content"


def recommendation_phrase(identity: dict[str, Any], recommendation_key: str) -> str:
    profile = language_profile(identity)
    phrases = {
        "seo": f"Strengthen {profile['product']} SEO and search intent alignment",
        "assortment": f"Clarify {profile['product']} assortment segmentation",
        "brand_shop": f"Expand {profile['navigation']} in the Brand Shop",
        "education": f"Deepen {profile['education']}",
        "taxonomy": f"Tighten {profile['navigation']} taxonomy",
        "attributes": f"Complete {profile['formula']}",
        "discovery": f"Expand {profile['discovery']} pathways",
        "trust": "Strengthen review and trust signals",
        "conversion": f"Strengthen {profile['conversion']}",
        "cross_category": f"Build cross-category {profile['navigation']}",
    }
    return phrases.get(recommendation_key, f"Strengthen {profile['product']} shopper guidance")
