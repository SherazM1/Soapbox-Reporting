from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


CONFIDENCE_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "exact": 4,
}


def load_style_guides(base_dir: str = "config/style_guides") -> list[dict[str, Any]]:
    """Load JSON style guide files from a directory."""
    guide_dir = Path(base_dir)
    if not guide_dir.exists():
        return []

    guides: list[dict[str, Any]] = []
    for path in sorted(guide_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            guides.append(json.load(handle))
    return guides


def match_style_guide_rule(
    product: dict[str, Any],
    guides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    guides = guides if guides is not None else load_style_guides()
    if not guides:
        return _no_match()

    title = _first_text(product, ("title", "current_title", "product_title"))
    category = _first_text(product, ("category", "product_category"))
    type_text = " ".join(
        value
        for value in (
            _first_text(product, ("product_type",)),
            _first_text(product, ("subcategory",)),
        )
        if value
    )
    context = _context_text(product)

    candidates: list[dict[str, Any]] = []
    for guide in guides:
        for family_id, family in guide.get("families", {}).items():
            for product_type_id, rule in family.get("product_types", {}).items():
                candidate = _score_rule(
                    guide=guide,
                    family_id=family_id,
                    family=family,
                    product_type_id=product_type_id,
                    rule=rule,
                    title=title,
                    category=category,
                    type_text=type_text,
                    context=context,
                )
                if candidate["matched"]:
                    candidates.append(candidate)

    if not candidates:
        return _no_match()

    candidates.sort(
        key=lambda item: (
            CONFIDENCE_RANK[item["confidence"]],
            item.get("_phrase_length", 0),
            item.get("_priority", 0),
        ),
        reverse=True,
    )
    best = candidates[0]
    best.pop("_phrase_length", None)
    best.pop("_priority", None)
    return best


def _score_rule(
    *,
    guide: dict[str, Any],
    family_id: str,
    family: dict[str, Any],
    product_type_id: str,
    rule: dict[str, Any],
    title: str,
    category: str,
    type_text: str,
    context: str,
) -> dict[str, Any]:
    normalized_title = _normalize(title)
    normalized_context = _normalize(context)
    normalized_category = _normalize(category)
    normalized_type = _normalize(type_text)
    guide_category = _normalize(str(guide.get("category", "")))
    family_name = str(family.get("display_name", ""))
    product_type_name = str(rule.get("display_name", ""))

    negative_keywords = rule.get("negative_keywords", [])
    has_negative_title = any(_contains_phrase(normalized_title, keyword) for keyword in negative_keywords)

    if normalized_category == guide_category and normalized_type:
        names = [product_type_name, family_name, product_type_id.replace("_", " ")]
        for name in names:
            if _normalize(name) == normalized_type:
                return _match(
                    guide,
                    family_id,
                    family_name,
                    product_type_id,
                    rule,
                    "exact:product_type",
                    "exact",
                    len(normalized_type),
                    100,
                )

    aliases = rule.get("aliases", [])
    for phrase in _sorted_phrases(aliases):
        if _contains_phrase(normalized_type, phrase):
            return _match(
                guide,
                family_id,
                family_name,
                product_type_id,
                rule,
                f"alias:product_type:{_clean_reason_phrase(phrase)}",
                "high",
                len(_normalize(phrase)),
                90,
            )

    if not has_negative_title:
        for phrase in _sorted_phrases(rule.get("title_keywords", [])):
            if _contains_phrase(normalized_title, phrase):
                return _match(
                    guide,
                    family_id,
                    family_name,
                    product_type_id,
                    rule,
                    f"keyword:title:{_clean_reason_phrase(phrase)}",
                    "high",
                    len(_normalize(phrase)),
                    80,
                )

    for phrase in _sorted_phrases(rule.get("title_keywords", [])):
        if _contains_phrase(normalized_context, phrase):
            return _match(
                guide,
                family_id,
                family_name,
                product_type_id,
                rule,
                f"keyword:context:{_clean_reason_phrase(phrase)}",
                "medium",
                len(_normalize(phrase)),
                70,
            )

    if not has_negative_title:
        for phrase in _sorted_phrases(rule.get("context_keywords", [])):
            if _contains_phrase(normalized_context, phrase):
                return _match(
                    guide,
                    family_id,
                    family_name,
                    product_type_id,
                    rule,
                    f"context:{_clean_reason_phrase(phrase)}",
                    "low",
                    len(_normalize(phrase)),
                    60,
                )

    return _no_match()


def _match(
    guide: dict[str, Any],
    family_id: str,
    family_name: str,
    product_type_id: str,
    rule: dict[str, Any],
    match_reason: str,
    confidence: str,
    phrase_length: int,
    priority: int,
) -> dict[str, Any]:
    return {
        "matched": True,
        "category": str(guide.get("category", "")),
        "family_id": family_id,
        "family_name": family_name,
        "product_type_id": product_type_id,
        "product_type_name": str(rule.get("display_name", "")),
        "formula": list(rule.get("formula", [])),
        "attributes": list(rule.get("attributes", [])),
        "match_reason": match_reason,
        "confidence": confidence,
        "_phrase_length": phrase_length,
        "_priority": priority,
    }


def _no_match() -> dict[str, Any]:
    return {
        "matched": False,
        "category": "",
        "family_id": "",
        "family_name": "",
        "product_type_id": "",
        "product_type_name": "",
        "formula": [],
        "attributes": [],
        "match_reason": "no_match",
        "confidence": "none",
    }


def _first_text(product: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = product.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _context_text(product: dict[str, Any]) -> str:
    parts: list[str] = []
    description = product.get("description")
    if isinstance(description, str):
        parts.append(description)

    key_features = product.get("key_features")
    if isinstance(key_features, list):
        parts.extend(str(item) for item in key_features if item)
    elif isinstance(key_features, str):
        parts.append(key_features)

    return " ".join(parts)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9&]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize(phrase)
    if not text or not normalized_phrase:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}s?(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _sorted_phrases(phrases: list[str]) -> list[str]:
    return sorted((str(phrase) for phrase in phrases), key=lambda item: len(_normalize(item)), reverse=True)


def _clean_reason_phrase(phrase: str) -> str:
    return _normalize(phrase)


def _self_test() -> None:
    guides = load_style_guides()
    cases = [
        ("Jif Creamy Peanut Butter", "nut_butters_spreads"),
        ("Nutella Hazelnut Spread", "nut_butters_spreads"),
        ("Good Good Strawberry Jam", "jams_jellies_preserves"),
        ("Good Good Four Fruit Preserves", "jams_jellies_preserves"),
        ("Marketside Guacamole", "guacamole"),
        ("Queso Cheese Dip", "cheese_dips_spreads"),
    ]

    for title, expected_product_type_id in cases:
        result = match_style_guide_rule({"title": title}, guides)
        actual = result["product_type_id"]
        assert actual == expected_product_type_id, (
            f"{title!r}: expected {expected_product_type_id!r}, got {actual!r} "
            f"({result['match_reason']})"
        )
        print(f"PASS {title} -> {result['product_type_name']} ({result['confidence']})")


if __name__ == "__main__":
    _self_test()
