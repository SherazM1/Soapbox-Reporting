from __future__ import annotations

import re
from collections import Counter
from typing import Any
from uuid import uuid4


PROMO_TERMS = ("best selling", "free shipping")
RETAILER_TERMS = ("walmart", "amazon", "target", "instacart", "costco", "kroger")
USE_CASE_TERMS = (
    "for",
    "ideal",
    "perfect",
    "use",
    "usage",
    "designed",
    "great for",
)
OUTCOME_TERMS = (
    "helps",
    "support",
    "improve",
    "benefit",
    "reduce",
    "increase",
    "save",
    "protect",
    "comfort",
)
FORBIDDEN_SPECIAL_CHARS = ("$", "#", "*", "!")
UNIVERSAL_USE_CASE_TERMS = (
    "use",
    "great for",
    "ideal for",
    "perfect for",
    "designed for",
    "helps",
    "supports",
    "enjoy",
    "serve",
    "apply",
    "clean",
    "protect",
    "organize",
    "store",
    "wear",
    "build",
    "play",
    "pair with",
    "top with",
    "spread on",
)

WALMART_TOP_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "Animals": ("animals", "pet", "pets", "pet supplies"),
    "Arts & Crafts": ("arts & crafts", "arts and crafts", "craft", "crafts"),
    "Baby": ("baby", "infant", "toddler"),
    "Beauty": ("beauty", "cosmetic", "cosmetics", "skin care", "hair care"),
    "Business & Industrial": ("business & industrial", "business and industrial", "industrial"),
    "Electronics & Photography": ("electronics & photography", "electronics and photography", "electronics", "photography"),
    "Everything Else": ("everything else",),
    "Fashion": ("fashion", "apparel", "clothing", "accessories"),
    "Food & Beverage": ("food & beverage", "food and beverage", "food", "beverage", "grocery"),
    "Furniture": ("furniture",),
    "Garden & Patio": ("garden & patio", "garden and patio", "garden", "patio", "outdoor living"),
    "Health & Personal Care": ("health & personal care", "health and personal care", "health", "personal care"),
    "Home": ("home", "kitchen", "housewares"),
    "Home Improvement": ("home improvement", "tools", "hardware"),
    "Household, Industrial Cleaning & Storage": (
        "household, industrial cleaning & storage",
        "household industrial cleaning storage",
        "household cleaning",
        "cleaning",
        "storage",
    ),
    "Media": ("media", "books", "movies", "music"),
    "Musical Instruments": ("musical instruments", "instrument", "instruments"),
    "Office & Stationery": ("office & stationery", "office and stationery", "office", "stationery"),
    "Safety & Emergency": ("safety & emergency", "safety and emergency", "safety", "emergency"),
    "Seasonal & Occasion": ("seasonal & occasion", "seasonal and occasion", "seasonal", "occasion"),
    "Sports, Recreation & Outdoor": (
        "sports, recreation & outdoor",
        "sports recreation outdoor",
        "sports",
        "recreation",
        "outdoor",
    ),
    "Toys": ("toys", "toy"),
    "Vehicle": ("vehicle", "automotive", "auto", "car", "truck"),
}

PRIORITY_CATEGORY_USE_CASE_TERMS: dict[str, tuple[str, ...]] = {
    "Food & Beverage": (
        "serve", "spread on", "pair with", "top with", "enjoy", "snack", "breakfast",
        "dessert", "meal", "recipe", "glaze", "toast", "waffles",
    ),
    "Electronics & Photography": (
        "charge", "connect", "stream", "listen", "work", "gaming", "travel", "protect", "power", "compatible with",
    ),
    "Beauty": (
        "apply", "routine", "daily use", "for skin", "for hair", "massage", "cleanse", "moisturize", "hydrate", "refresh",
    ),
    "Furniture": (
        "sit", "store", "organize", "display", "workspace", "bedroom", "living room", "patio", "comfort", "space-saving",
    ),
    "Home": (
        "organize", "store", "cook", "prep", "serve", "display", "kitchen", "pantry", "countertop", "space-saving",
    ),
    "Toys": (
        "build", "create", "play", "gift", "screen-free", "activity", "learn", "collect", "imagination", "pretend play",
    ),
    "Vehicle": (
        "install", "protect", "repair", "replace", "maintain", "drive", "towing", "storage", "travel", "performance",
    ),
}

LIGHT_CATEGORY_USE_CASE_HINTS: dict[str, tuple[str, ...]] = {
    "Animals": ("feed", "walk", "train"),
    "Arts & Crafts": ("create", "paint", "craft"),
    "Baby": ("daily use", "feeding", "comfort"),
    "Business & Industrial": ("worksite", "operate", "maintain"),
    "Everything Else": ("everyday use", "practical", "convenient"),
    "Fashion": ("wear", "style", "daily wear"),
    "Garden & Patio": ("outdoor use", "plant", "maintain"),
    "Health & Personal Care": ("daily use", "apply", "wellness"),
    "Home Improvement": ("install", "repair", "upgrade"),
    "Household, Industrial Cleaning & Storage": ("clean", "organize", "store"),
    "Media": ("read", "watch", "listen"),
    "Musical Instruments": ("practice", "play", "perform"),
    "Office & Stationery": ("write", "organize", "workspace"),
    "Safety & Emergency": ("protect", "prepare", "emergency use"),
    "Seasonal & Occasion": ("celebrate", "decorate", "gift"),
    "Sports, Recreation & Outdoor": ("train", "outdoor use", "play"),
}
FEATURE_HEAVY_TERMS = (
    "ingredients",
    "formula",
    "pack",
    "jar",
    "ounce",
    "oz",
    "spec",
    "specification",
    "manufactured",
    "process",
    "compliant",
    "certified",
    "dimensions",
    "size",
)
MANUFACTURING_DETAIL_TERMS = (
    "manufactured",
    "manufacturing",
    "facility",
    "batch",
    "compliance",
    "specification",
    "pack of",
    "ounces",
    "oz",
    "jar",
    "bottle",
    "count",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _lower(text: str) -> str:
    return _norm(text).lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _lower(text))


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    t = _lower(text)
    return any(term in t for term in terms)


def _get_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    source_meta = record.get("source_metadata")
    ingest_meta = record.get("ingest_metadata")
    merged: dict[str, Any] = {}
    if isinstance(source_meta, dict):
        merged.update(source_meta)
    if isinstance(ingest_meta, dict):
        merged.update(ingest_meta)
    return merged


def _get_meta_value(meta: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in meta:
            return meta.get(key)
    return None


def _meta_text(meta: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _get_meta_value(meta, keys)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else text


def _meta_int(meta: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    text = _meta_text(meta, keys)
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except ValueError:
        return None


def _meta_float(meta: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    text = _meta_text(meta, keys)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _has_issue(findings: list[dict[str, Any]], issue_type: str) -> bool:
    return any(str(f.get("issue_type", "")) == issue_type for f in findings)


def _add_universal_finding(
    findings: list[dict[str, Any]],
    *,
    section: str,
    issue_type: str,
    message: str,
    severity: str,
    evidence: dict[str, Any],
    recommendation_theme: str,
) -> None:
    if _has_issue(findings, issue_type):
        return
    findings.append(
        _section_find(
            section=section,
            issue_type=issue_type,
            severity=severity,
            message=message,
            evidence=evidence,
            recommendation_theme=recommendation_theme,
            source="universal_rule",
        )
    )


def _resolve_walmart_top_level_category(record: dict[str, Any]) -> str | None:
    category = _lower(str(record.get("category", "")))
    subcategory = _lower(str(record.get("subcategory", "")))
    product_type = _lower(str(record.get("product_type", "")))
    title = _lower(str(record.get("current_title") or record.get("product_title") or ""))

    weighted_sources = (
        (category, 3),
        (subcategory, 2),
        (product_type, 2),
        (title, 1),
    )

    best_category: str | None = None
    best_score = 0
    for top_category, aliases in WALMART_TOP_CATEGORY_ALIASES.items():
        score = 0
        for source_text, weight in weighted_sources:
            if not source_text:
                continue
            for alias in aliases:
                alias_norm = _lower(alias)
                if alias_norm and alias_norm in source_text:
                    score += weight
                    break
        if score > best_score:
            best_score = score
            best_category = top_category

    return best_category if best_score > 0 else None


def _combined_use_case_term_bank(record: dict[str, Any]) -> tuple[str, ...]:
    top_category = _resolve_walmart_top_level_category(record)
    terms = list(UNIVERSAL_USE_CASE_TERMS)
    if top_category in PRIORITY_CATEGORY_USE_CASE_TERMS:
        terms.extend(PRIORITY_CATEGORY_USE_CASE_TERMS[top_category])
    elif top_category in LIGHT_CATEGORY_USE_CASE_HINTS:
        terms.extend(LIGHT_CATEGORY_USE_CASE_HINTS[top_category])

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        t = _lower(term)
        if not t or t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return tuple(deduped)


def _section_find(
    *,
    section: str,
    issue_type: str,
    severity: str,
    message: str,
    evidence: dict[str, Any],
    recommendation_theme: str,
    source: str = "rule_check",
    content_source: str | None = None,
) -> dict[str, Any]:
    finding = {
        "finding_id": f"f-{uuid4().hex[:10]}",
        "section": section,
        "issue_type": issue_type,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "recommendation_theme": recommendation_theme,
        "source": source,
    }
    if content_source:
        finding["content_source"] = content_source
    return finding


def analyze_title(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    title = _norm(record.get("current_title") or record.get("product_title") or "")
    title_lower = title.lower()
    meta = _get_record_metadata(record)
    title_count = _meta_int(meta, ("title_count", "Title Count", "titleCount"))
    title_notes = _meta_text(meta, ("title_notes", "Title Notes", "titleNotes")).lower()

    if not title:
        findings.append(
            _section_find(
                section="title",
                issue_type="missing_title",
                severity="high",
                message="No usable product title was extracted.",
                evidence={"title": title},
                recommendation_theme="clarity",
            )
        )
        return findings

    if title_count is not None and title_count > 90 and not _has_issue(findings, "title_too_long"):
        findings.append(
            _section_find(
                section="title",
                issue_type="title_too_long",
                severity="high",
                message="Title exceeds the 90-character baseline.",
                evidence={"title_length": title_count, "max_length": 90},
                recommendation_theme="seo",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    if len(title) > 90:
        findings.append(
            _section_find(
                section="title",
                issue_type="title_too_long",
                severity="high",
                message="Title exceeds the 90-character baseline.",
                evidence={"title_length": len(title), "max_length": 90},
                recommendation_theme="seo",
            )
        )

    if any(term in title_lower for term in PROMO_TERMS):
        findings.append(
            _section_find(
                section="title",
                issue_type="promo_language",
                severity="medium",
                message="Title contains promo language that should be avoided.",
                evidence={"title": title, "promo_terms": [t for t in PROMO_TERMS if t in title_lower]},
                recommendation_theme="formatting",
            )
        )

    letters = [ch for ch in title if ch.isalpha()]
    if len(letters) >= 8 and all(ch.isupper() for ch in letters):
        findings.append(
            _section_find(
                section="title",
                issue_type="all_caps",
                severity="medium",
                message="Title appears to be in all caps.",
                evidence={"title": title},
                recommendation_theme="clarity",
            )
        )

    title_tokens = [t for t in _tokens(title) if len(t) >= 3]
    if title_tokens:
        counts = Counter(title_tokens)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3:
            findings.append(
                _section_find(
                    section="title",
                    issue_type="keyword_stuffing",
                    severity="medium",
                    message="Title shows repeated keyword usage.",
                    evidence={
                        "word": most_common_word,
                        "repetitions": most_common_count,
                        "title": title,
                    },
                    recommendation_theme="seo",
                source="heuristic",
            )
        )

    if title_count is not None and title_count < 20 and not _has_issue(findings, "too_short"):
        findings.append(
            _section_find(
                section="title",
                issue_type="too_short",
                severity="medium",
                message="Title may be too short to be descriptive.",
                evidence={"title_length": title_count, "title": title},
                recommendation_theme="clarity",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    if len(title) < 20:
        findings.append(
            _section_find(
                section="title",
                issue_type="too_short",
                severity="low",
                message="Title may be too short to be descriptive.",
                evidence={"title_length": len(title), "title": title},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    if title_notes:
        if ("too long" in title_notes or "length" in title_notes) and not _has_issue(findings, "title_too_long"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="title_too_long",
                    severity="medium",
                    message="Extension notes indicate the title should be tightened.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="seo",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if (
            "too short" in title_notes
            or "not descriptive" in title_notes
            or "missing keyword" in title_notes
        ) and not _has_issue(findings, "too_short"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="too_short",
                    severity="medium",
                    message="Extension notes indicate the title lacks sufficient detail.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if "promo" in title_notes and not _has_issue(findings, "promo_language"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="promo_language",
                    severity="medium",
                    message="Extension notes flag promotional wording in the title.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("all caps" in title_notes or "capitalization" in title_notes) and not _has_issue(findings, "all_caps"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="all_caps",
                    severity="medium",
                    message="Extension notes indicate capitalization formatting issues in the title.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    return findings


def analyze_description(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    body = _norm(record.get("current_description_body", ""))
    bullets = [str(b).strip() for b in record.get("current_description_bullets", []) if str(b).strip()]
    combined = _norm(record.get("current_description_combined", "") or " ".join([body, *bullets]))
    combined_lower = combined.lower()
    wc = _word_count(combined)
    resolved_top_category = _resolve_walmart_top_level_category(record)
    use_case_term_bank = _combined_use_case_term_bank(record)
    meta = _get_record_metadata(record)
    description_count = _meta_int(meta, ("description_count", "Description Count", "descriptionCount"))
    description_notes = _meta_text(meta, ("description_notes", "Description Notes", "descriptionNotes")).lower()
    content_score = _meta_float(meta, ("content_score", "Content Score", "contentScore"))

    if wc < 60:
        findings.append(
            _section_find(
                section="description",
                issue_type="too_short",
                severity="high",
                message="Description content is below the 60-word minimum baseline.",
                evidence={"word_count": wc, "minimum_words": 60},
                recommendation_theme="seo",
            )
        )
    elif wc < 100:
        findings.append(
            _section_find(
                section="description",
                issue_type="below_recommended_length",
                severity="low",
                message="Description is below the 100-word recommended length.",
                evidence={"word_count": wc, "recommended_words": 100},
                recommendation_theme="clarity",
            )
        )

    if description_count is not None:
        if description_count < 60 and not _has_issue(findings, "too_short"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="too_short",
                    severity="high",
                    message="Description content is below the 60-word minimum baseline.",
                    evidence={"word_count": description_count, "minimum_words": 60},
                    recommendation_theme="seo",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        elif description_count < 100 and not _has_issue(findings, "below_recommended_length"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="below_recommended_length",
                    severity="low",
                    message="Description is below the 100-word recommended length.",
                    evidence={"word_count": description_count, "recommended_words": 100},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    title = _norm(record.get("current_title") or record.get("product_title") or "")
    title_words = [t for t in _tokens(title) if len(t) > 2]
    title_anchor = " ".join(title_words[:2]) if len(title_words) >= 2 else (title_words[0] if title_words else "")
    if title_anchor and title_anchor not in combined_lower:
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_product_name",
                severity="medium",
                message="Description does not appear to repeat the product name.",
                evidence={"product_name_anchor": title_anchor},
                recommendation_theme="seo",
                source="heuristic",
            )
        )

    if _has_any_term(combined, RETAILER_TERMS):
        findings.append(
            _section_find(
                section="description",
                issue_type="retailer_mention",
                severity="high",
                message="Description appears to mention another retailer.",
                evidence={"retailer_terms_detected": [t for t in RETAILER_TERMS if t in combined_lower]},
                recommendation_theme="formatting",
            )
        )

    if not body and bullets:
        findings.append(
            _section_find(
                section="description",
                issue_type="generic_content",
                severity="medium",
                message="Description appears to be bullet-only with no clear paragraph body.",
                evidence={"body_present": False, "bullet_count": len(bullets)},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    has_use_case_language = _has_any_term(combined, use_case_term_bank)
    if combined and not has_use_case_language:
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_use_case",
                severity="medium",
                message="Description lacks clear use-case language.",
                evidence={"word_count": wc, "resolved_top_category": resolved_top_category or ""},
                recommendation_theme="use_case",
                source="heuristic",
            )
        )

    if combined and not _has_any_term(combined, OUTCOME_TERMS):
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_outcome_focus",
                severity="low",
                message="Description does not clearly communicate outcomes or shopper benefit.",
                evidence={"word_count": wc},
                recommendation_theme="conversion",
                source="heuristic",
            )
        )

    html_like = bool(re.search(r"<[^>]+>", body))
    if body and not html_like:
        findings.append(
            _section_find(
                section="description",
                issue_type="html_structure_missing",
                severity="low",
                message="Description appears plain-text only; consider adding clean HTML structure if channel supports it.",
                evidence={"html_like_detected": html_like},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    if description_notes:
        if ("too short" in description_notes or "thin" in description_notes) and not _has_issue(findings, "too_short"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="too_short",
                    severity="high",
                    message="Extension notes indicate description depth is insufficient.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="seo",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("too long" in description_notes or "verbose" in description_notes) and not _has_issue(findings, "too_long"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="too_long",
                    severity="low",
                    message="Description may be overly long and should be tightened for scanability.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("generic" in description_notes or "vague" in description_notes) and not _has_issue(findings, "generic_content"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="generic_content",
                    severity="medium",
                    message="Extension notes indicate description copy is generic.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="differentiation",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("use case" in description_notes or "usage" in description_notes) and not _has_issue(findings, "missing_use_case"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="missing_use_case",
                    severity="medium",
                    message="Extension notes indicate missing use-case language.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="use_case",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("benefit" in description_notes or "outcome" in description_notes) and not _has_issue(findings, "missing_outcome_focus"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="missing_outcome_focus",
                    severity="low",
                    message="Extension notes indicate weak benefit/outcome language.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="conversion",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("html" in description_notes or "format" in description_notes) and not _has_issue(findings, "html_structure_missing"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="html_structure_missing",
                    severity="low",
                    message="Extension notes indicate description formatting/HTML structure needs cleanup.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    if content_score is not None and content_score < 75:
        findings.append(
            _section_find(
                section="description",
                issue_type="content_score_low",
                severity="medium" if content_score < 60 else "low",
                message="Content score indicates broader copy quality gaps.",
                evidence={"content_score": content_score, "target_minimum": 75},
                recommendation_theme="conversion",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    unique_ratio = (len(set(_tokens(combined))) / max(1, len(_tokens(combined)))) if combined else 0.0
    if wc > 0 and wc < 80 and unique_ratio < 0.55:
        findings.append(
            _section_find(
                section="description",
                issue_type="generic_content",
                severity="low",
                message="Description appears thin or generic.",
                evidence={"word_count": wc, "unique_word_ratio": round(unique_ratio, 2)},
                recommendation_theme="differentiation",
                source="heuristic",
            )
        )

    # Additive approved universal statements layer for description recommendations.
    body_lower = body.lower()
    has_html_structure = bool(re.search(r"(?is)<\s*(p|ul|ol|li)\b", body))
    if body and not has_html_structure:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_not_html_format",
            message="Not in HTML format (<p>...</p>)",
            severity="low",
            evidence={"html_structure_detected": has_html_structure},
            recommendation_theme="formatting",
        )

    effective_wc = description_count if description_count is not None else wc
    if effective_wc < 60:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_below_60_word_minimum",
            message="Below 60-word minimum",
            severity="high",
            evidence={"word_count": effective_wc},
            recommendation_theme="seo",
        )

    if title_anchor and title_anchor not in body_lower and title_anchor not in combined_lower:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_product_name_not_repeated_for_seo",
            message="Product name not repeated for SEO",
            severity="medium",
            evidence={"product_name_anchor": title_anchor},
            recommendation_theme="seo",
        )

    has_use_case = has_use_case_language
    has_outcome = _has_any_term(combined, OUTCOME_TERMS)
    if combined and not has_use_case:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_missing_clear_use_cases",
            message="Missing clear use cases",
            severity="medium",
            evidence={"word_count": wc, "resolved_top_category": resolved_top_category or ""},
            recommendation_theme="use_case",
        )

    feature_term_hits = sum(1 for term in FEATURE_HEAVY_TERMS if term in combined_lower)
    if wc >= 35 and feature_term_hits >= 2 and not has_use_case and not has_outcome:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_too_feature_focused_not_benefit_use_case_driven",
            message="Too feature-focused, not benefit/use-case driven",
            severity="medium",
            evidence={"feature_term_hits": feature_term_hits, "word_count": wc},
            recommendation_theme="conversion",
        )

    token_counts = Counter([t for t in _tokens(combined) if len(t) >= 4])
    repeated_pattern = token_counts.most_common(1)[0][1] >= 4 if token_counts else False
    thin_generic = wc > 0 and wc < 90 and unique_ratio < 0.6
    if combined and (thin_generic or repeated_pattern):
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_likely_duplicated_content",
            message="Likely duplicated content",
            severity="low",
            evidence={
                "word_count": wc,
                "unique_word_ratio": round(unique_ratio, 2),
                "repeated_pattern": repeated_pattern,
            },
            recommendation_theme="differentiation",
        )

    return findings


def analyze_key_features(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    features = record.get("current_key_features", []) or []
    bullets = [_norm(str(f.get("text", ""))) for f in features if _norm(str(f.get("text", "")))]
    bullet_count = len(bullets)
    full_desc = _lower(record.get("current_description_combined", ""))
    meta = _get_record_metadata(record)
    meta_bullet_count = _meta_int(
        meta,
        ("description_bullet_count", "Description Bullet Count", "descriptionBulletCount"),
    )
    bullet_notes = _meta_text(
        meta,
        ("description_bullet_notes", "Description Bullet Notes", "descriptionBulletNotes"),
    ).lower()

    if bullet_count < 3:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="insufficient_bullets",
                severity="high",
                message="Key features contain fewer than 3 bullets.",
                evidence={"bullet_count": bullet_count, "minimum_bullets": 3},
                recommendation_theme="clarity",
            )
        )
    elif bullet_count < 5:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="recommended_bullets_missing",
                severity="low",
                message="Key features are below the 5-bullet recommendation.",
                evidence={"bullet_count": bullet_count, "recommended_bullets": 5},
                recommendation_theme="conversion",
            )
        )

    if meta_bullet_count is not None:
        if meta_bullet_count < 3 and not _has_issue(findings, "insufficient_bullets"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="insufficient_bullets",
                    severity="high",
                    message="Key features contain fewer than 3 bullets.",
                    evidence={"bullet_count": meta_bullet_count, "minimum_bullets": 3},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        elif meta_bullet_count < 5 and not _has_issue(findings, "recommended_bullets_missing"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="recommended_bullets_missing",
                    severity="low",
                    message="Key features are below the 5-bullet recommendation.",
                    evidence={"bullet_count": meta_bullet_count, "recommended_bullets": 5},
                    recommendation_theme="conversion",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    punctuated = [b for b in bullets if re.search(r"[.!?;:]\s*$", b)]
    if punctuated:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="ending_punctuation",
                severity="medium",
                message="One or more bullets end with punctuation.",
                evidence={"examples": punctuated[:3], "count": len(punctuated)},
                recommendation_theme="formatting",
            )
        )

    sentence_like = []
    for b in bullets:
        words = _tokens(b)
        if len(words) >= 8 and (re.search(r"[.!?]\s*$", b) or re.search(r"\b(this|it|you|we|they|is|are|will|can)\b", b.lower())):
            sentence_like.append(b)
    if sentence_like:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="full_sentences",
                severity="medium",
                message="One or more bullets read like full sentences rather than fragments.",
                evidence={"examples": sentence_like[:3], "count": len(sentence_like)},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    starts_upper = [bool(re.match(r"^[A-Z]", b)) for b in bullets if b]
    if starts_upper and not all(starts_upper):
        findings.append(
            _section_find(
                section="key_features",
                issue_type="inconsistent_formatting",
                severity="low",
                message="Bullet capitalization appears inconsistent.",
                evidence={"starts_with_capital_count": sum(starts_upper), "bullet_count": len(starts_upper)},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    joined = " ".join(bullets).lower()
    if _has_any_term(joined, RETAILER_TERMS):
        findings.append(
            _section_find(
                section="key_features",
                issue_type="retailer_mention",
                severity="high",
                message="Key features appear to mention another retailer.",
                evidence={"retailer_terms_detected": [t for t in RETAILER_TERMS if t in joined]},
                recommendation_theme="formatting",
            )
        )

    forbidden_hits = [c for c in FORBIDDEN_SPECIAL_CHARS if c in " ".join(bullets)]
    if forbidden_hits:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="forbidden_special_characters",
                severity="medium",
                message="Key features include forbidden special characters.",
                evidence={"forbidden_characters": forbidden_hits},
                recommendation_theme="formatting",
            )
        )

    overlap_count = 0
    for b in bullets:
        b_low = b.lower()
        if b_low and b_low in full_desc:
            overlap_count += 1
    if bullet_count > 0 and overlap_count / bullet_count >= 0.5:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="overlap_with_other_section",
                severity="medium",
                message="Key features heavily overlap with the description content.",
                evidence={"overlapping_bullets": overlap_count, "bullet_count": bullet_count},
                recommendation_theme="differentiation",
                source="heuristic",
            )
        )

    if bullets and not _has_any_term(joined, USE_CASE_TERMS + OUTCOME_TERMS):
        findings.append(
            _section_find(
                section="key_features",
                issue_type="missing_use_case",
                severity="low",
                message="Key features have weak use-case or benefit language.",
                evidence={"bullet_count": bullet_count},
                recommendation_theme="use_case",
                source="heuristic",
            )
        )

    if bullet_notes:
        if ("full sentence" in bullet_notes or "sentence" in bullet_notes) and not _has_issue(findings, "full_sentences"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="full_sentences",
                    severity="medium",
                    message="Extension notes indicate bullets read as full sentences.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("punctuation" in bullet_notes or "period" in bullet_notes) and not _has_issue(findings, "ending_punctuation"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="ending_punctuation",
                    severity="medium",
                    message="Extension notes indicate bullet punctuation cleanup is needed.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("overlap" in bullet_notes or "duplicate" in bullet_notes) and not _has_issue(findings, "overlap_with_other_section"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="overlap_with_other_section",
                    severity="medium",
                    message="Extension notes indicate overlap between bullets and description.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="differentiation",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("benefit" in bullet_notes or "use case" in bullet_notes) and not _has_issue(findings, "missing_use_case"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="missing_use_case",
                    severity="low",
                    message="Extension notes indicate bullets need stronger benefit/use-case language.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="use_case",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    # Additive approved universal statements layer for key features recommendations.
    sentence_like_trigger = bool(sentence_like) or bool(punctuated)
    if sentence_like_trigger:
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
            message="Bullets are full sentences (should be fragments, no punctuation)",
            severity="medium",
            evidence={"sentence_like_count": len(sentence_like), "punctuated_count": len(punctuated)},
            recommendation_theme="formatting",
        )

    repeated_start_count = 0
    if bullets:
        starts = [" ".join(_tokens(b)[:3]) for b in bullets if _tokens(b)]
        start_counts = Counter([s for s in starts if s])
        repeated_start_count = max(start_counts.values()) if start_counts else 0
    duplicate_ratio = (len(set([b.lower() for b in bullets])) / max(1, len(bullets))) if bullets else 1.0
    if bullet_count >= 3 and (repeated_start_count >= 2 or duplicate_ratio < 0.75):
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_duplicate_repetitive_content",
            message="Duplicate/repetitive content",
            severity="medium",
            evidence={
                "bullet_count": bullet_count,
                "repeated_start_count": repeated_start_count,
                "unique_ratio": round(duplicate_ratio, 2),
            },
            recommendation_theme="differentiation",
        )

    sentence_style_flags = [bool(re.search(r"[.!?]\s*$", b)) for b in bullets]
    if bullets and ((starts_upper and not all(starts_upper)) or (sentence_style_flags and any(sentence_style_flags) and not all(sentence_style_flags))):
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_inconsistent_formatting_capitalization_structure",
            message="Inconsistent formatting (capitalization/structure)",
            severity="low",
            evidence={
                "starts_upper_mixed": bool(starts_upper and not all(starts_upper)),
                "sentence_style_mixed": bool(sentence_style_flags and any(sentence_style_flags) and not all(sentence_style_flags)),
            },
            recommendation_theme="formatting",
        )

    manufacturing_hits = sum(1 for term in MANUFACTURING_DETAIL_TERMS if term in joined)
    benefit_hits = sum(1 for term in OUTCOME_TERMS + UNIVERSAL_USE_CASE_TERMS if term in joined)
    if bullet_count >= 2 and manufacturing_hits >= 2 and benefit_hits <= 1:
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_too_much_manufacturing_detail",
            message="Too much manufacturing detail",
            severity="low",
            evidence={"manufacturing_hits": manufacturing_hits, "benefit_hits": benefit_hits},
            recommendation_theme="conversion",
        )

    if bullet_count > 0 and overlap_count / bullet_count >= 0.4:
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_repeats_description_content",
            message="Repeats description content",
            severity="medium",
            evidence={"overlapping_bullets": overlap_count, "bullet_count": bullet_count},
            recommendation_theme="differentiation",
        )

    return findings


def analyze_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    images = record.get("images", []) or []
    image_count = len(images)
    has_hero = any(bool(img.get("is_hero")) for img in images)

    if image_count == 0:
        findings.append(
            _section_find(
                section="images",
                issue_type="missing_images",
                severity="high",
                message="No usable product images were extracted.",
                evidence={"image_count": image_count},
                recommendation_theme="imagery",
            )
        )
        return findings

    if image_count < 3:
        findings.append(
            _section_find(
                section="images",
                issue_type="low_image_count",
                severity="medium",
                message="Image stack appears limited.",
                evidence={"image_count": image_count, "recommended_minimum": 3},
                recommendation_theme="imagery",
            )
        )

    if not has_hero:
        findings.append(
            _section_find(
                section="images",
                issue_type="missing_clear_hero",
                severity="low",
                message="No explicit hero flag was detected; first image fallback is being used.",
                evidence={"image_count": image_count, "hero_flag_detected": has_hero},
                recommendation_theme="imagery",
                source="heuristic",
            )
        )

    return findings


def analyze_primary_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if (record.get("source_type") or "").lower() != "primary":
        return []

    findings: list[dict[str, Any]] = []
    findings.extend(analyze_title(record))
    findings.extend(analyze_description(record))
    findings.extend(analyze_key_features(record))
    findings.extend(analyze_images(record))
    return findings
