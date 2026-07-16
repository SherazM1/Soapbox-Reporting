from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Callable


_MAX_TERM_LENGTH = 28
_MAX_TERM_WORDS = 4

_FALLBACK_TERMS: dict[str, tuple[str, ...]] = {
    "beauty": (
        "face cleanser",
        "face wash",
        "hydrating cleanser",
        "gentle cleanser",
        "sensitive skin cleanser",
        "oily skin cleanser",
    ),
    "health": (
        "antacid",
        "stomach relief",
        "heartburn relief",
        "acid reducer",
        "upset stomach relief",
        "indigestion relief",
    ),
    "food": (
        "peanut butter",
        "almond butter",
        "jelly",
        "fruit spread",
        "protein shake",
        "electrolyte drink",
    ),
    "pet": (
        "dog treats",
        "cat treats",
        "dog shampoo",
        "flea treatment",
        "dog supplement",
        "cat litter",
    ),
    "electronics": (
        "wireless earbuds",
        "bluetooth speaker",
        "phone charger",
        "usb c cable",
        "laptop stand",
        "phone case",
    ),
}

_FAMILY_LABELS: dict[str, str] = {
    "beauty": "skin care essentials",
    "health": "over-the-counter medicine",
    "food": "food and beverage essentials",
    "pet": "pet care essentials",
    "electronics": "electronics essentials",
}

_SLIDE4_PRODUCT_PHRASES: dict[str, tuple[str, ...]] = {
    "beauty": ("skin care product", "cleanser", "face cleanser", "face wash"),
    "health": ("over-the-counter medicine", "antacid", "stomach relief product"),
    "food": ("food or beverage product", "pantry staple", "beverage item"),
    "pet": ("pet care product", "pet item", "pet treatment"),
    "electronics": ("electronics product", "device accessory", "tech accessory"),
}

_SLIDE4_GENERIC_SAFE_PHRASES = {
    "the product",
    "the PDP",
    "shopper confidence",
    "usage guidance",
    "comparison",
    "product role",
}

_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "beauty": (
        "beauty",
        "skin care",
        "skincare",
        "cleanser",
        "face wash",
        "moisturizer",
        "lotion",
        "serum",
        "oily skin",
        "sensitive skin",
    ),
    "health": (
        "health",
        "otc",
        "medicine",
        "antacid",
        "heartburn",
        "acid reducer",
        "stomach",
        "indigestion",
        "pain relief",
    ),
    "food": (
        "food",
        "beverage",
        "peanut butter",
        "almond butter",
        "jelly",
        "fruit spread",
        "protein shake",
        "electrolyte",
        "jam",
    ),
    "pet": (
        "pet",
        "dog",
        "cat",
        "flea",
        "litter",
        "treats",
        "shampoo",
        "supplement",
    ),
    "electronics": (
        "electronics",
        "earbuds",
        "bluetooth",
        "speaker",
        "charger",
        "usb",
        "laptop",
        "phone case",
        "cable",
    ),
}

_OFF_CATEGORY_MARKERS: dict[str, tuple[str, ...]] = {
    "beauty": ("antacid", "heartburn", "peanut butter", "dog ", "cat ", "earbuds", "charger"),
    "health": (
        "vaccaria",
        "acupressure",
        "face wash",
        "cleanser",
        "peanut butter",
        "mixed spices",
        "seasoning",
        "pantry",
        "beverage",
        "dog ",
        "cat ",
        "earbuds",
    ),
    "food": ("antacid", "heartburn", "cleanser", "dog ", "cat ", "earbuds", "charger"),
    "pet": ("antacid", "heartburn", "cleanser", "peanut butter", "earbuds", "charger"),
    "electronics": ("antacid", "heartburn", "cleanser", "peanut butter", "dog ", "cat ", "flea"),
}

_BUCKET_PHRASES = {
    "options",
    "solutions",
    "needs",
    "audience",
    "ingredient",
    "recommended",
    "use case",
    "use-case",
    "benefit led",
    "benefit-led",
    "assortment",
    "coverage",
    "audience and ingredient needs",
    "benefit led solutions",
    "ingredient needs",
}

_BAD_EXACT_TERMS = {
    "ear vaccaria seed",
    "ear acupressure seeds",
    "dermatologist recommended",
    "health and medicine",
    "multicultural otc",
    "international antacids",
    "international antacids solutions",
    "mixed spices and seasoning",
    "medicine dosing container",
}

_BAD_TEXT_MARKERS = (
    "ear vaccaria seed",
    "ear acupressure",
    "health and medicine/",
    "multicultural otc/",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    "300x250",
    "audience and ingredient needs",
    "benefit-led solutions",
    "benefit led solutions",
    "ingredient needs",
)

_SLIDE4_CONTAMINATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bear\s+vaccaria\s+seeds?\b", re.I),
    re.compile(r"\bear\s+acupressure\s+seeds?\b", re.I),
    re.compile(r"\bhealth\s+and\s+medicine\s*/\s*[^,.;|]+", re.I),
    re.compile(r"\bmulticultural\s+otc\s*/\s*[^,.;|]+", re.I),
    re.compile(r"\binternational\s+antacids(?:\s+solutions)?\b", re.I),
    re.compile(r"\b[\w.-]+\.(?:jpg|jpeg|png|webp)\b", re.I),
    re.compile(r"\b\d{2,5}\s*x\s*\d{2,5}\b", re.I),
    re.compile(r"\baudience\s+and\s+ingredient\s+needs\b", re.I),
    re.compile(r"\bbenefit[-\s]+led\s+solutions\b", re.I),
    re.compile(r"\bingredient\s+needs\b", re.I),
    re.compile(r"\bmixed\s+spices?\s+(?:and\s+)?seasonings?\b", re.I),
    re.compile(r"\bmedicine\s+dosing\s+container\b", re.I),
)

SLIDE_CLEANUP_SEQUENCE: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("slide6_visibility", "Slide 6", lambda payload: cleanup_slide6(payload)),
    ("slide4_findings", "Slide 4", lambda payload: cleanup_slide4(payload)),
    ("slide3_search_benchmark", "Slide 3", lambda payload: cleanup_slide3(payload)),
    ("slide5_brand_shop", "Slide 5", lambda payload: cleanup_slide5(payload)),
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{_text(key)} {_text(item)}" for key, item in value.items())
    return "" if value is None else str(value)


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).strip())


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


def _contains_any(text: str, markers: tuple[str, ...] | set[str]) -> bool:
    padded = f" {text} "
    return any(marker in padded or marker in text for marker in markers)


def _detect_slide6_family(payload: dict[str, Any], segments: list[Any]) -> str:
    haystack_parts: list[str] = [
        _text(payload.get("category_phrase")),
        _text(payload.get("pack_id")),
        _text(payload.get("intro")),
        _text(payload.get("takeaway")),
        _text(payload.get("debug")),
    ]
    for segment in segments:
        if isinstance(segment, dict):
            haystack_parts.append(_text(segment.get("segment")))
            haystack_parts.append(_text(segment.get("debug")))
        else:
            haystack_parts.append(_text(segment))
    haystack = _norm(" ".join(haystack_parts))
    scores = {
        family: sum(1 for marker in markers if marker in haystack)
        for family, markers in _FAMILY_MARKERS.items()
    }
    best_family, best_score = max(scores.items(), key=lambda item: item[1])
    return best_family if best_score > 0 else "food"


def _detect_cleanup_family(payload: dict[str, Any]) -> str:
    columns = payload.get("columns", []) if isinstance(payload, dict) else []
    pseudo_segments: list[Any] = []
    if isinstance(columns, list):
        for column in columns:
            if isinstance(column, dict):
                pseudo_segments.append(
                    {
                        "segment": " ".join(str(item) for item in (column.get("bullets", []) or [])),
                        "debug": {
                            "category": column.get("category", ""),
                            "product_type": column.get("product_type", ""),
                            "product_title": column.get("product_title", ""),
                            "findings": column.get("findings", {}),
                        },
                    }
                )
    if not pseudo_segments and isinstance(payload, dict):
        pseudo_segments.append({"segment": _text(payload), "debug": {}})
    return _detect_slide6_family(payload, pseudo_segments)


def _is_bad_term(term: str, family: str) -> bool:
    normalized = _norm(term)
    if not normalized:
        return True
    if normalized in _BAD_EXACT_TERMS:
        return True
    raw = term.lower()
    if "/" in raw:
        return True
    if re.search(r"\.(?:jpe?g|png|webp)\b", raw):
        return True
    if re.search(r"\b\d{2,5}\s*x\s*\d{2,5}\b", raw):
        return True
    if len(term) > _MAX_TERM_LENGTH or len(normalized.split()) > _MAX_TERM_WORDS:
        return True
    if any(phrase == normalized for phrase in _BUCKET_PHRASES):
        return True
    if any(f" {phrase} " in f" {normalized} " for phrase in _BUCKET_PHRASES):
        return True
    if _contains_any(normalized, (" taxonomy ", " path ", " metadata ", " filename ", " asset ")):
        return True
    if _contains_any(raw, _OFF_CATEGORY_MARKERS.get(family, ())):
        return True
    if re.search(r"[,;|+&]", raw):
        return True
    return False


def _first_safe_slash_segment(term: str, family: str) -> str:
    candidate = _normalize_space(term.split("/", 1)[0])
    if candidate and not _is_bad_term(candidate, family):
        return candidate
    return ""


def _next_fallback(family: str, used: set[str]) -> str:
    for fallback in _FALLBACK_TERMS.get(family, _FALLBACK_TERMS["food"]):
        normalized = _norm(fallback)
        if normalized not in used:
            used.add(normalized)
            return fallback
    for fallback in _FALLBACK_TERMS["food"]:
        normalized = _norm(fallback)
        if normalized not in used:
            used.add(normalized)
            return fallback
    return "shopper search"


def _clean_term(value: Any, family: str, used: set[str], rejected: list[str], fallback_terms_used: list[str]) -> str:
    original = _normalize_space(value)
    candidate = original
    if "/" in candidate:
        candidate = _first_safe_slash_segment(candidate, family)
    candidate = _normalize_space(candidate).lower()
    normalized = _norm(candidate)
    if not candidate or _is_bad_term(candidate, family) or normalized in used:
        if original:
            rejected.append(original)
        fallback = _next_fallback(family, used)
        fallback_terms_used.append(fallback)
        return fallback
    used.add(normalized)
    return candidate


def _clean_segments(segments: list[Any], family: str) -> tuple[list[Any], dict[str, Any]]:
    cleaned_segments: list[Any] = []
    used: set[str] = set()
    rejected: list[str] = []
    fallback_terms_used: list[str] = []

    for segment in segments[:6]:
        if isinstance(segment, dict):
            cleaned_segment = deepcopy(segment)
            cleaned_segment["segment"] = _clean_term(
                cleaned_segment.get("segment"),
                family,
                used,
                rejected,
                fallback_terms_used,
            )
            cleaned_segments.append(cleaned_segment)
        else:
            cleaned_segments.append(_clean_term(segment, family, used, rejected, fallback_terms_used))

    template_segment = next((item for item in cleaned_segments if isinstance(item, dict)), None)
    while len(cleaned_segments) < 6:
        fallback = _next_fallback(family, used)
        fallback_terms_used.append(fallback)
        if template_segment is not None:
            added = deepcopy(template_segment)
            added["segment"] = fallback
            added["warnings"] = []
            cleaned_segments.append(added)
        else:
            cleaned_segments.append({"segment": fallback})

    return cleaned_segments, {
        "detected_category_family": family,
        "rejected_row_terms": rejected,
        "fallback_terms_used": fallback_terms_used,
    }


def _text_is_contaminated(value: Any) -> bool:
    raw = _text(value).lower()
    normalized = _norm(raw)
    return any(marker in raw for marker in _BAD_TEXT_MARKERS) or any(
        phrase == normalized or f" {phrase} " in f" {normalized} " for phrase in _BUCKET_PHRASES
    )


def _clean_slide6_text(value: Any, family: str, field_name: str, client_label: str) -> tuple[Any, bool]:
    if not isinstance(value, str) or not _text_is_contaminated(value):
        return value, False
    family_label = _FAMILY_LABELS.get(family, "category search terms")
    if field_name == "intro":
        return (
            f"Competitor and {client_label} PDP content is compared for alignment with "
            f"realistic {family_label} shopper search paths.",
            True,
        )
    return (
        f"Focused {family_label} language helps protect search visibility without relying on "
        "taxonomy or internal planning terms.",
        True,
    )


def _slide4_safe_product_phrase(family: str, bullet: str) -> str:
    phrases = _SLIDE4_PRODUCT_PHRASES.get(family, _SLIDE4_PRODUCT_PHRASES["food"])
    lower = bullet.lower()
    if "comparison" in lower:
        return "comparison"
    if "usage" in lower or "education" in lower or "guidance" in lower:
        return "usage guidance"
    if "confidence" in lower or "review" in lower or "trust" in lower:
        return "shopper confidence"
    if "title" in lower or "role" in lower:
        return "product role"
    return phrases[0]


def _slide4_family_safe_phrase(family: str, phrase: str) -> str:
    if phrase in _SLIDE4_GENERIC_SAFE_PHRASES:
        return phrase
    allowed = _SLIDE4_PRODUCT_PHRASES.get(family, _SLIDE4_PRODUCT_PHRASES["food"])
    return phrase if phrase in allowed else allowed[0]


def _slide4_direct_rewrite(bullet: str, family: str) -> str:
    lower = bullet.lower()
    product_phrase = _slide4_family_safe_phrase(family, _SLIDE4_PRODUCT_PHRASES.get(family, _SLIDE4_PRODUCT_PHRASES["food"])[0])
    if "title" in lower and ("can name" in lower or "more directly" in lower):
        if family == "health":
            return "Title can name the over-the-counter medicine more directly"
        return "Title can name the product more directly"
    if "title" in lower and ("clarifies" in lower or "clarity" in lower or "role" in lower):
        return "Title clarity makes the product role easier to understand"
    if "feature" in lower and "comparison" in lower:
        if family == "health":
            return "Feature detail supports over-the-counter medicine comparison"
        return "Feature detail makes comparison easier"
    if ("image" in lower or "carousel" in lower) and ("usage" in lower or "education" in lower):
        return "Image stack adds usage education"
    if "review" in lower and "confidence" in lower:
        return "Review depth helps reinforce shopper confidence"
    if "benefit" in lower and ("communication" in lower or "positioning" in lower):
        return "Benefit communication is clearer across the PDP"
    if ("pack" in lower or "spec" in lower) and ("detail" in lower or "understand" in lower):
        return "Pack and spec detail are easier to understand"
    return product_phrase


def _slide4_has_contamination(text: str, family: str) -> bool:
    normalized = _norm(text)
    raw = text.lower()
    if not normalized:
        return False
    if any(pattern.search(text) for pattern in _SLIDE4_CONTAMINATED_PATTERNS):
        return True
    if "/" in raw:
        return True
    if any(phrase == normalized for phrase in _BUCKET_PHRASES):
        return True
    if any((" " in phrase or "-" in phrase) and f" {phrase} " in f" {normalized} " for phrase in _BUCKET_PHRASES):
        return True
    if _contains_any(raw, _OFF_CATEGORY_MARKERS.get(family, ())):
        return True
    return False


def _rewrite_contaminated_slide4_bullet(text: Any, family: str) -> tuple[Any, bool]:
    if not isinstance(text, str):
        return text, False
    bullet = _normalize_space(text)
    if not _slide4_has_contamination(bullet, family):
        return text, False

    direct = _slide4_direct_rewrite(bullet, family)
    if direct != _SLIDE4_PRODUCT_PHRASES.get(family, _SLIDE4_PRODUCT_PHRASES["food"])[0]:
        return direct, True

    replacement = _slide4_safe_product_phrase(family, bullet)
    cleaned = bullet
    for pattern in _SLIDE4_CONTAMINATED_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    if "/" in cleaned:
        cleaned = re.sub(r"\b[a-z0-9][a-z0-9\s&-]{1,40}/[a-z0-9][a-z0-9\s/&-]{1,80}", replacement, cleaned, flags=re.I)
    for phrase in sorted(_BUCKET_PHRASES, key=len, reverse=True):
        if " " in phrase or "-" in phrase:
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", replacement, cleaned, flags=re.I)
    cleaned = _normalize_space(cleaned)
    if not cleaned or _slide4_has_contamination(cleaned, family):
        cleaned = _slide4_direct_rewrite(bullet, family)
        if not cleaned or _slide4_has_contamination(cleaned, family):
            return text, False
    return cleaned, cleaned != bullet


def _clean_slide4_bullet_list(values: Any, family: str, changed_terms: list[str]) -> Any:
    if not isinstance(values, list):
        return values
    cleaned_values: list[Any] = []
    for value in values:
        cleaned, changed = _rewrite_contaminated_slide4_bullet(value, family)
        if changed:
            changed_terms.append(_normalize_space(value))
        cleaned_values.append(cleaned)
    return cleaned_values


def _clean_slide4_findings_payload(findings: Any, family: str, changed_terms: list[str]) -> None:
    if not isinstance(findings, dict):
        return
    if isinstance(findings.get("slide4_bullets"), list):
        findings["slide4_bullets"] = _clean_slide4_bullet_list(findings.get("slide4_bullets"), family, changed_terms)
    for list_key in ("strengths", "opportunities"):
        items = findings.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "text" not in item:
                continue
            cleaned, changed = _rewrite_contaminated_slide4_bullet(item.get("text"), family)
            if changed:
                changed_terms.append(_normalize_space(item.get("text")))
                item["text"] = cleaned


def cleanup_slide6(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    try:
        cleaned_payload = deepcopy(payload)
        original_segments = list(cleaned_payload.get("segments", []) or [])
        family = _detect_slide6_family(cleaned_payload, original_segments)
        cleaned_segments, cleanup_debug = _clean_segments(original_segments, family)
        cleaned_payload["segments"] = cleaned_segments

        client_label = _normalize_space(cleaned_payload.get("client_label")) or "Client"
        adjusted_text_fields: list[str] = []
        for field_name in ("intro", "takeaway", "intro_text", "takeaway_text"):
            if field_name in cleaned_payload:
                cleaned_text, adjusted = _clean_slide6_text(
                    cleaned_payload.get(field_name),
                    family,
                    "intro" if "intro" in field_name else "takeaway",
                    client_label,
                )
                cleaned_payload[field_name] = cleaned_text
                if adjusted:
                    adjusted_text_fields.append(field_name)

        for list_key in ("surfaced_terms", "terms", "search_terms", "row_terms"):
            values = cleaned_payload.get(list_key)
            if isinstance(values, list):
                used: set[str] = set()
                cleaned_payload[list_key] = [
                    _clean_term(item, family, used, cleanup_debug["rejected_row_terms"], cleanup_debug["fallback_terms_used"])
                    for item in values[:6]
                ]

        cleanup_debug["adjusted_text_fields"] = adjusted_text_fields
        debug = cleaned_payload.get("debug")
        if isinstance(debug, dict):
            debug["slide_cleanup"] = cleanup_debug
        warnings = cleaned_payload.get("warnings")
        if isinstance(warnings, list) and cleanup_debug["rejected_row_terms"]:
            warnings.append(
                "Slide 6 cleanup replaced invalid or table-unfriendly shopper terms after generation."
            )
        return cleaned_payload
    except Exception:
        return payload


def cleanup_slide4(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    try:
        cleaned_payload = deepcopy(payload)
        family = _detect_cleanup_family(cleaned_payload)
        changed_terms: list[str] = []

        columns = cleaned_payload.get("columns")
        if isinstance(columns, list):
            for column in columns:
                if not isinstance(column, dict):
                    continue
                if isinstance(column.get("bullets"), list):
                    column["bullets"] = _clean_slide4_bullet_list(column.get("bullets"), family, changed_terms)
                _clean_slide4_findings_payload(column.get("findings"), family, changed_terms)

        slide4_findings = cleaned_payload.get("slide4_findings")
        if isinstance(slide4_findings, dict):
            for findings in slide4_findings.values():
                _clean_slide4_findings_payload(findings, family, changed_terms)

        debug = cleaned_payload.get("debug")
        if isinstance(debug, dict):
            final_bullets = debug.get("final_bullets")
            if isinstance(final_bullets, dict):
                for key, bullets in list(final_bullets.items()):
                    final_bullets[key] = _clean_slide4_bullet_list(bullets, family, changed_terms)
            debug["slide4_cleanup"] = {
                "detected_category_family": family,
                "rewritten_bullet_count": len(changed_terms),
                "rewritten_bullets": changed_terms,
            }
        warnings = cleaned_payload.get("warnings")
        if isinstance(warnings, list) and changed_terms:
            warnings.append("Slide 4 cleanup replaced off-scope product terms after generation.")
        return cleaned_payload
    except Exception:
        return payload


def cleanup_slide3(payload: Any) -> Any:
    return payload


def cleanup_slide5(payload: Any) -> Any:
    return payload


def cleanup_generated_audit_plan(export_plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "has_run": True,
        "succeeded": False,
        "active": False,
        "slides_cleaned": [],
        "slides_skipped": [],
        "warnings": [],
    }

    try:
        cleaned_plan = deepcopy(export_plan or {})
    except Exception as exc:
        metadata["warnings"].append(f"Unable to clone generated audit plan: {exc}")
        return export_plan or {}, metadata

    for payload_key, slide_label, cleanup_hook in SLIDE_CLEANUP_SEQUENCE:
        original_payload = cleaned_plan.get(payload_key)
        try:
            cleaned_plan[payload_key] = cleanup_hook(deepcopy(original_payload))
            metadata["slides_cleaned"].append(slide_label)
        except Exception as exc:
            cleaned_plan[payload_key] = original_payload
            metadata["slides_skipped"].append(slide_label)
            metadata["warnings"].append(f"{slide_label} cleanup skipped: {exc}")

    metadata["succeeded"] = True
    metadata["active"] = True
    return cleaned_plan, metadata
