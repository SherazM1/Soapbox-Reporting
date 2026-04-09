from __future__ import annotations

import re
from typing import Any


SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}
STATUS_WEIGHT = {"fail": 4, "warning": 3, "acceptable": 2, "recommended": 1}

PROMO_TERMS = ("best selling", "free shipping")
RETAILER_TERMS = ("walmart", "amazon", "target", "instacart", "costco", "kroger")
TITLE_OFFER_SHIPPING_TERMS = (
    "free shipping",
    "ships free",
    "in stock",
    "limited stock",
    "buy now",
    "save",
    "deal",
    "offer",
    "clearance",
    "discount",
    "sale",
)
TITLE_SUSPICIOUS_CHARS = ("|", "{", "}", "[", "]", "<", ">", "^", "~", "`")

APPROVED_DESCRIPTION_LINES = [
    ("universal_desc_not_html_format", "Not in HTML format (<p>...</p>)"),
    ("universal_desc_below_60_word_minimum", "Below 60-word minimum"),
    ("universal_desc_product_name_not_repeated_for_seo", "Product name not repeated for SEO"),
    ("universal_desc_missing_clear_use_cases", "Missing clear use cases"),
    (
        "universal_desc_too_feature_focused_not_benefit_use_case_driven",
        "Too feature-focused, not benefit/use-case driven",
    ),
    ("universal_desc_likely_duplicated_content", "Likely duplicated content"),
]

APPROVED_KEY_FEATURE_LINES = [
    (
        "universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
        "Bullets are full sentences (should be fragments, no punctuation)",
    ),
    ("universal_kf_duplicate_repetitive_content", "Duplicate/repetitive content"),
    (
        "universal_kf_inconsistent_formatting_capitalization_structure",
        "Inconsistent formatting (capitalization/structure)",
    ),
    ("universal_kf_too_much_manufacturing_detail", "Too much manufacturing detail"),
    ("universal_kf_repeats_description_content", "Repeats description content"),
]

DESCRIPTION_RECOMMENDATION_ISSUES = {
    "too_short",
    "below_recommended_length",
    "too_long",
    "missing_product_name",
    "retailer_mention",
    "missing_use_case",
    "missing_outcome_focus",
    "generic_content",
    "html_structure_missing",
    "content_score_low",
    "universal_desc_not_html_format",
    "universal_desc_below_60_word_minimum",
    "universal_desc_product_name_not_repeated_for_seo",
    "universal_desc_missing_clear_use_cases",
    "universal_desc_too_feature_focused_not_benefit_use_case_driven",
    "universal_desc_likely_duplicated_content",
    "bullet_style_description",
    "promo_language",
    "missing_shopper_usefulness",
}

KEY_FEATURE_RECOMMENDATION_ISSUES = {
    "insufficient_bullets",
    "recommended_bullets_missing",
    "ending_punctuation",
    "full_sentences",
    "inconsistent_formatting",
    "retailer_mention",
    "forbidden_special_characters",
    "overlap_with_other_section",
    "missing_use_case",
    "universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
    "universal_kf_duplicate_repetitive_content",
    "universal_kf_inconsistent_formatting_capitalization_structure",
    "universal_kf_too_much_manufacturing_detail",
    "universal_kf_repeats_description_content",
    "over_recommended_bullets",
    "promo_language",
}

TITLE_TERMS = ("title", "headline")
DESCRIPTION_TERMS = (
    "description",
    "html",
    "word",
    "seo",
    "paragraph",
    "readability",
    "benefit",
    "outcome",
    "use-case",
    "use case",
    "generic",
    "duplicate content",
    "thin content",
    "product name",
)
KEY_FEATURE_TERMS = (
    "key feature",
    "key-feature",
    "bullet",
    "fragment",
    "punctuation",
    "capitalization",
    "formatting",
    "overlap",
    "sentence",
    "hierarchy",
    "order",
)
RECOMMENDATION_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "into",
    "from",
    "your",
    "when",
    "where",
    "how",
    "more",
    "clear",
    "clean",
    "stronger",
}
GENERIC_PHRASES = (
    "can be strengthened",
    "quality signals are low",
    "appears",
    "should be",
    "consider adding",
    "prioritize clarity",
)
SECTION_TIER_ISSUE_TYPES: dict[str, dict[int, set[str]]] = {
    "description": {
        1: {
            "universal_desc_not_html_format",
            "html_structure_missing",
            "universal_desc_below_60_word_minimum",
            "too_short",
            "universal_desc_likely_duplicated_content",
            "generic_content",
        },
        2: {
            "universal_desc_missing_clear_use_cases",
            "missing_use_case",
            "universal_desc_too_feature_focused_not_benefit_use_case_driven",
            "missing_outcome_focus",
            "universal_desc_product_name_not_repeated_for_seo",
            "missing_product_name",
            "content_score_low",
            "missing_shopper_usefulness",
        },
        3: {
            "below_recommended_length",
            "too_long",
            "retailer_mention",
            "promo_language",
            "bullet_style_description",
        },
    },
    "key_features": {
        1: {
            "insufficient_bullets",
            "recommended_bullets_missing",
            "over_recommended_bullets",
            "full_sentences",
            "universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
            "ending_punctuation",
            "universal_kf_duplicate_repetitive_content",
        },
        2: {
            "universal_kf_repeats_description_content",
            "overlap_with_other_section",
            "universal_kf_too_much_manufacturing_detail",
            "missing_use_case",
        },
        3: {
            "inconsistent_formatting",
            "universal_kf_inconsistent_formatting_capitalization_structure",
            "forbidden_special_characters",
            "retailer_mention",
            "promo_language",
        },
    },
}
APPROVED_DESCRIPTION_ISSUES = {issue_type for issue_type, _ in APPROVED_DESCRIPTION_LINES}
APPROVED_KEY_FEATURE_ISSUES = {issue_type for issue_type, _ in APPROVED_KEY_FEATURE_LINES}
FAIL_ENFORCED_ISSUE_TYPES: dict[str, set[str]] = {
    "description": {
        "too_short",
        "html_structure_missing",
        "universal_desc_not_html_format",
    },
    "key_features": {
        "insufficient_bullets",
        "full_sentences",
        "universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
    },
}
WARNING_ISSUE_TYPES: dict[str, set[str]] = {
    "title": {
        "keyword_stuffing",
        "suspicious_special_characters",
    },
    "description": {
        "below_recommended_length",
        "too_long",
        "promo_language",
        "retailer_mention",
        "bullet_style_description",
        "missing_shopper_usefulness",
    },
    "key_features": {
        "over_recommended_bullets",
        "promo_language",
        "retailer_mention",
        "forbidden_special_characters",
        "inconsistent_formatting",
    },
}
VALIDATION_GOVERNED_ISSUE_TYPES: dict[str, set[str]] = {
    "title": {
        "title_too_long",
        "promo_language",
        "offer_shipping_availability_language",
        "disallowed_or_construction",
        "suspicious_special_characters",
        "keyword_stuffing",
    },
    "description": DESCRIPTION_RECOMMENDATION_ISSUES,
    "key_features": KEY_FEATURE_RECOMMENDATION_ISSUES,
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = _norm(item)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _semantic_signature(text: str, *, section: str) -> str:
    lowered = _norm(text).lower()
    if section == "description":
        if ("html format" in lowered) or ("html structure" in lowered) or ("plain-text only" in lowered):
            return "description:html_format"
        if ("below 60-word minimum" in lowered) or ("depth is light" in lowered and "expanded" in lowered):
            return "description:below_60_word_minimum"
        if ("product name not repeated" in lowered) or ("reintroduce the product name" in lowered):
            return "description:product_name_seo"
        if ("missing clear use cases" in lowered) or ("missing use-case" in lowered) or ("add clear use-case language" in lowered):
            return "description:missing_use_cases"
        if ("feature-focused" in lowered and "benefit" in lowered):
            return "description:feature_focused"
        if ("duplicated content" in lowered) or ("duplicate content" in lowered):
            return "description:duplicated_content"
        if ("promotional/offer language" in lowered) or ("promotional language" in lowered):
            return "description:promo_language"
        if ("paragraph form" in lowered) or ("bullet formatting" in lowered):
            return "description:bullet_style_formatting"
        if ("shopper-useful context" in lowered) or ("shopper-facing context" in lowered):
            return "description:shopper_usefulness"

    if section == "key_features":
        if ("full sentences" in lowered) or ("sentence-style bullets" in lowered):
            return "key_features:full_sentences"
        if ("duplicate/repetitive content" in lowered) or ("repetitive content" in lowered):
            return "key_features:duplicate_repetitive"
        if ("inconsistent formatting" in lowered) or ("standardize capitalization" in lowered):
            return "key_features:inconsistent_formatting"
        if "manufacturing detail" in lowered:
            return "key_features:manufacturing_detail"
        if ("repeats description content" in lowered) or ("description overlap" in lowered):
            return "key_features:repeats_description"
        if ("promotional/offer language" in lowered) or ("promotional language" in lowered):
            return "key_features:promo_language"
        if ("most important six bullets" in lowered) or ("6-bullet recommended limit" in lowered):
            return "key_features:over_recommended_bullets"
    return f"{section}:raw:{lowered}"


def _dedupe_semantic_keep_order(items: list[str], *, section: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = _norm(item)
        if not clean:
            continue
        key = _semantic_signature(clean, section=section)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _text_tokens_for_similarity(text: str) -> set[str]:
    lowered = _norm(text).lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return {t for t in tokens if len(t) >= 4 and t not in RECOMMENDATION_STOPWORDS}


def _recommendation_tier(*, section: str, issue_type: str) -> int:
    rules = SECTION_TIER_ISSUE_TYPES.get(section, {})
    for tier, issue_types in rules.items():
        if issue_type in issue_types:
            return tier
    return 3


def _recommendation_specificity(text: str) -> int:
    tokens = _text_tokens_for_similarity(text)
    return len(tokens)


def _is_generic_recommendation(text: str) -> bool:
    lowered = _norm(text).lower()
    return any(phrase in lowered for phrase in GENERIC_PHRASES)


def _is_near_duplicate(a: str, b: str) -> bool:
    a_tokens = _text_tokens_for_similarity(a)
    b_tokens = _text_tokens_for_similarity(b)
    if not a_tokens or not b_tokens:
        return False
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    if union == 0:
        return False
    jaccard = inter / union
    if jaccard >= 0.85:
        return True
    a_norm = _norm(a).lower()
    b_norm = _norm(b).lower()
    return len(a_norm) >= 30 and len(b_norm) >= 30 and (a_norm in b_norm or b_norm in a_norm)


def _recommendation_strength(item: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    # Stronger means: fail-enforced support, validation-backed support, validation status, tier impact, approved universal phrasing, and specificity.
    must_keep_score = 1 if bool(item.get("must_keep")) else 0
    validation_backed_score = 1 if str(item.get("support_source_type", "")) == "validation_backed" else 0
    status_score = STATUS_WEIGHT.get(str(item.get("validation_status", "warning")), 3)
    tier_score = 4 - int(item.get("tier", 3))
    universal_score = 1 if bool(item.get("is_approved_universal")) else 0
    specificity = _recommendation_specificity(str(item.get("text", "")))
    return (must_keep_score, validation_backed_score, status_score, tier_score, universal_score, specificity)


def _findings_by_section(findings: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("section") == section]


def _is_fail_enforced_issue(*, section: str, issue_type: str, severity: str) -> bool:
    if issue_type in FAIL_ENFORCED_ISSUE_TYPES.get(section, set()):
        return True
    return severity == "high" and issue_type in {"title_too_long", "offer_shipping_availability_language"}


def _issue_validation_status(*, section: str, issue_type: str, severity: str) -> str:
    if _is_fail_enforced_issue(section=section, issue_type=issue_type, severity=severity):
        return "fail"
    if issue_type in WARNING_ISSUE_TYPES.get(section, set()) or severity in {"medium", "low"}:
        return "warning"
    return "warning"


def _best_evidence_int(findings: list[dict[str, Any]], issue_types: set[str], key: str) -> int | None:
    vals: list[int] = []
    for f in findings:
        if str(f.get("issue_type", "")) not in issue_types:
            continue
        evidence = f.get("evidence", {}) or {}
        try:
            if key in evidence:
                vals.append(int(float(str(evidence.get(key)))))
        except (TypeError, ValueError):
            continue
    return vals[0] if vals else None


def _section_validation_applicability(findings: list[dict[str, Any]], *, section: str) -> dict[str, str]:
    section_findings = _findings_by_section(findings, section)
    issue_types = {str(f.get("issue_type", "")) for f in section_findings}
    out: dict[str, str] = {}

    if section == "description":
        wc = _best_evidence_int(
            section_findings,
            {"too_short", "below_recommended_length", "universal_desc_below_60_word_minimum"},
            "word_count",
        )
        if "too_short" in issue_types:
            out["length"] = "fail"
        elif "below_recommended_length" in issue_types:
            out["length"] = "warning"
        elif wc is not None:
            out["length"] = "recommended" if wc >= 100 else "acceptable"

        out["html"] = "fail" if ("html_structure_missing" in issue_types or "universal_desc_not_html_format" in issue_types) else "acceptable"
        out["repetition"] = "warning" if ("generic_content" in issue_types or "universal_desc_likely_duplicated_content" in issue_types) else "acceptable"

    if section == "key_features":
        bullet_count = _best_evidence_int(
            section_findings,
            {"insufficient_bullets", "recommended_bullets_missing", "over_recommended_bullets"},
            "bullet_count",
        )
        if "insufficient_bullets" in issue_types:
            out["count"] = "fail"
        elif "over_recommended_bullets" in issue_types:
            out["count"] = "warning"
        elif bullet_count is not None:
            if bullet_count >= 3:
                out["count"] = "recommended" if bullet_count == 6 else "acceptable"

        out["sentence_style"] = (
            "fail"
            if (
                "full_sentences" in issue_types
                or "ending_punctuation" in issue_types
                or "universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation" in issue_types
            )
            else "acceptable"
        )
        out["repetition"] = (
            "warning"
            if (
                "overlap_with_other_section" in issue_types
                or "universal_kf_repeats_description_content" in issue_types
                or "universal_kf_duplicate_repetitive_content" in issue_types
            )
            else "acceptable"
        )

    if section == "title":
        out["length"] = "fail" if "title_too_long" in issue_types else "acceptable"
        out["promo_offer"] = (
            "fail"
            if ("promo_language" in issue_types or "offer_shipping_availability_language" in issue_types)
            else "acceptable"
        )
    return out


def _suppress_if_non_applicable(item: dict[str, Any], *, section: str, applicability: dict[str, str]) -> bool:
    issue_type = str(item.get("issue_type", ""))

    if section == "description":
        if issue_type in {"too_short", "universal_desc_below_60_word_minimum"}:
            return applicability.get("length") in {"acceptable", "recommended"}
        if issue_type in {"html_structure_missing", "universal_desc_not_html_format"}:
            return applicability.get("html") in {"acceptable", "recommended"}

    if section == "key_features":
        if issue_type in {"insufficient_bullets", "recommended_bullets_missing"}:
            return applicability.get("count") in {"acceptable", "recommended"}
        if issue_type in {"over_recommended_bullets"}:
            return applicability.get("count") in {"acceptable", "recommended"}

    return False


def _support_source_type(*, section: str, issue_type: str, source: str) -> str:
    validation_governed = issue_type in VALIDATION_GOVERNED_ISSUE_TYPES.get(section, set())
    if validation_governed and source in {"heuristic", "sheet_metadata", "rule_check"}:
        return "validation_backed"
    if source == "universal_rule":
        return "universal_rule_backed"
    return "rule_backed"


def _approved_lines_for_section(
    findings: list[dict[str, Any]],
    *,
    section: str,
    ordered_rules: list[tuple[str, str]],
) -> list[str]:
    section_findings = _findings_by_section(findings, section)
    issue_types = {str(f.get("issue_type", "")) for f in section_findings}
    return [line for issue_type, line in ordered_rules if issue_type in issue_types]


def _issue_templates(section: str) -> dict[str, str]:
    templates = {
        "image_recommendations": {
            "missing_images": "Add a clear front-of-pack hero image on a clean background",
            "low_image_count": "Expand the image stack with lifestyle and in-use visuals",
            "missing_clear_hero": "Promote the strongest product shot as the hero image",
        },
        "description_recommendations": {
            "too_short": "Description depth is light and should be expanded with concrete product details",
            "below_recommended_length": "Description can be strengthened with additional shopper-relevant context",
            "too_long": "Description is dense and should be tightened for faster scanability",
            "missing_product_name": "Reintroduce the product name naturally in the body for SEO continuity",
            "retailer_mention": "Remove retailer references and keep copy channel-neutral",
            "missing_use_case": "Add clear use-case language for when and how the product is used",
            "missing_outcome_focus": "Add stronger benefit/outcome language tied to shopper value",
            "generic_content": "Replace generic statements with specific differentiators and proof points",
            "html_structure_missing": "Improve structure with short sections/bullets for better readability",
            "content_score_low": "Content quality signals are low; prioritize clarity, specificity, and benefits",
            "bullet_style_description": "Rewrite description content into paragraph form instead of bullet formatting",
            "promo_language": "Remove promotional/offer language and keep description copy product-focused",
            "missing_shopper_usefulness": "Add clearer shopper-useful context, outcomes, and practical usage guidance",
        },
        "key_features_recommendations": {
            "insufficient_bullets": "Add more key feature bullets to meet the minimum coverage baseline",
            "recommended_bullets_missing": "Build toward a 5-bullet structure to improve scanability",
            "over_recommended_bullets": "Trim key features to the most important six bullets to improve scanability",
            "ending_punctuation": "Remove ending punctuation so bullets read as clean fragments",
            "full_sentences": "Convert sentence-style bullets into concise feature fragments",
            "inconsistent_formatting": "Standardize capitalization and structure across all bullets",
            "retailer_mention": "Remove retailer references from key features",
            "forbidden_special_characters": "Remove non-compliant special characters from bullet text",
            "overlap_with_other_section": "Reduce description overlap and keep bullets focused on distinct points",
            "missing_use_case": "Strengthen benefit and use-case framing across bullets",
            "promo_language": "Remove promotional/offer language from key features",
        },
    }
    return templates.get(section, {})


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = _norm(text).lower()
    return any(term in lowered for term in terms)


def _route_recommendation_section(*, source_section: str, issue_type: str, recommendation: str) -> str | None:
    rec = _norm(recommendation)
    if not rec:
        return None

    if issue_type in DESCRIPTION_RECOMMENDATION_ISSUES:
        return "description"
    if issue_type in KEY_FEATURE_RECOMMENDATION_ISSUES:
        return "key_features"

    if _contains_any_term(rec, TITLE_TERMS):
        return None

    has_description_intent = _contains_any_term(rec, DESCRIPTION_TERMS)
    has_key_feature_intent = _contains_any_term(rec, KEY_FEATURE_TERMS)

    if source_section == "description":
        return "description"
    if source_section == "key_features":
        if has_key_feature_intent:
            return "key_features"
        if has_description_intent:
            return "description"
        return None
    return None


def _collect_routed_recommendations(findings: list[dict[str, Any]], *, target_section: str) -> list[str]:
    out: list[str] = []
    section_to_output_field = {
        "description": "description_recommendations",
        "key_features": "key_features_recommendations",
    }
    for finding in findings:
        source_section = str(finding.get("section", ""))
        if source_section not in section_to_output_field:
            continue

        issue_type = str(finding.get("issue_type", ""))
        templates = _issue_templates(section_to_output_field[source_section])
        recommendation = templates.get(issue_type) or _norm(str(finding.get("message", "")))
        routed_section = _route_recommendation_section(
            source_section=source_section,
            issue_type=issue_type,
            recommendation=recommendation,
        )
        if routed_section == target_section and recommendation:
            out.append(recommendation)
    return _dedupe_keep_order(out)


def _collect_routed_recommendation_items(
    findings: list[dict[str, Any]], *, target_section: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    section_to_output_field = {
        "description": "description_recommendations",
        "key_features": "key_features_recommendations",
    }
    approved_issue_types = (
        APPROVED_DESCRIPTION_ISSUES if target_section == "description" else APPROVED_KEY_FEATURE_ISSUES
    )
    for idx, finding in enumerate(findings):
        source_section = str(finding.get("section", ""))
        if source_section not in section_to_output_field:
            continue

        issue_type = str(finding.get("issue_type", ""))
        templates = _issue_templates(section_to_output_field[source_section])
        recommendation = templates.get(issue_type) or _norm(str(finding.get("message", "")))
        routed_section = _route_recommendation_section(
            source_section=source_section,
            issue_type=issue_type,
            recommendation=recommendation,
        )
        if routed_section != target_section or not recommendation:
            continue
        out.append(
            {
                "text": recommendation,
                "issue_type": issue_type,
                "source_section": source_section,
                "severity": str(finding.get("severity", "low")),
                "source": str(finding.get("source", "")),
                "evidence": finding.get("evidence", {}) or {},
                "normalized_issue_key": _semantic_signature(recommendation, section=target_section),
                "validation_governed": issue_type in VALIDATION_GOVERNED_ISSUE_TYPES.get(target_section, set()),
                "support_source_type": _support_source_type(
                    section=target_section,
                    issue_type=issue_type,
                    source=str(finding.get("source", "")),
                ),
                "is_approved_universal": issue_type in approved_issue_types,
                "tier": _recommendation_tier(section=target_section, issue_type=issue_type),
                "validation_status": _issue_validation_status(
                    section=target_section,
                    issue_type=issue_type,
                    severity=str(finding.get("severity", "low")),
                ),
                "must_keep": _is_fail_enforced_issue(
                    section=target_section,
                    issue_type=issue_type,
                    severity=str(finding.get("severity", "low")),
                ),
                "order": idx,
            }
        )
    return out


def _finalize_section_recommendations(items: list[dict[str, Any]], *, section: str) -> list[str]:
    if not items:
        return []
    applicability = _section_validation_applicability(
        [
            {
                "section": section,
                "issue_type": i.get("issue_type", ""),
                "severity": i.get("severity", "low"),
                "evidence": i.get("evidence", {}) or {},
            }
            for i in items
        ],
        section=section,
    )

    # Stage 1: semantic de-duplication by keeping the strongest item per semantic signature.
    best_by_signature: dict[str, dict[str, Any]] = {}
    for item in items:
        if _suppress_if_non_applicable(item, section=section, applicability=applicability):
            item["applicability_state"] = "suppressed_non_applicable"
            continue
        item["applicability_state"] = "applicable"
        text = _norm(str(item.get("text", "")))
        if not text:
            continue
        item["text"] = text
        sig = _semantic_signature(text, section=section)
        existing = best_by_signature.get(sig)
        if existing is None:
            best_by_signature[sig] = item
            continue
        if _recommendation_strength(item) > _recommendation_strength(existing):
            best_by_signature[sig] = item
        elif _recommendation_strength(item) == _recommendation_strength(existing):
            if int(item.get("order", 0)) < int(existing.get("order", 0)):
                best_by_signature[sig] = item

    candidates = list(best_by_signature.values())
    candidates.sort(
        key=lambda i: (
            int(i.get("tier", 3)),
            -_recommendation_strength(i)[1],
            -_recommendation_strength(i)[2],
            -SEVERITY_WEIGHT.get(str(i.get("severity", "low")), 1),
            -_recommendation_strength(i)[4],
            -_recommendation_strength(i)[5],
            int(i.get("order", 0)),
        )
    )

    # Stage 2: near-duplicate suppression with stronger-line preference.
    kept: list[dict[str, Any]] = []
    for item in candidates:
        text = str(item.get("text", ""))
        dropped = False
        for strong in kept:
            if not _is_near_duplicate(text, str(strong.get("text", ""))):
                continue
            # Keep distinct actions if they are in different semantic groups.
            if _semantic_signature(text, section=section) != _semantic_signature(str(strong.get("text", "")), section=section):
                continue
            if _recommendation_strength(item) <= _recommendation_strength(strong):
                dropped = True
                break
        if dropped:
            continue
        kept.append(item)

    # Stage 3: weak generic suppression when a stronger retained line exists.
    final_items: list[dict[str, Any]] = []
    for item in kept:
        text = str(item.get("text", ""))
        if bool(item.get("must_keep")):
            final_items.append(item)
            continue
        if not _is_generic_recommendation(text):
            final_items.append(item)
            continue
        has_stronger_overlap = any(
            _is_near_duplicate(text, str(other.get("text", "")))
            and _recommendation_strength(other) >= _recommendation_strength(item)
            and not _is_generic_recommendation(str(other.get("text", "")))
            for other in kept
            if other is not item
        )
        if not has_stronger_overlap:
            final_items.append(item)

    return [str(i.get("text", "")) for i in final_items if _norm(str(i.get("text", "")))]


def _clean_title_candidate(title: str) -> str:
    candidate = _norm(title)
    lowered = candidate.lower()
    for term in PROMO_TERMS:
        lowered = lowered.replace(term, "")
    for term in RETAILER_TERMS:
        lowered = lowered.replace(term, "")
    candidate = _norm(lowered)

    words = candidate.split(" ")
    compact: list[str] = []
    last = ""
    for w in words:
        if not w:
            continue
        wl = w.lower()
        if wl == last:
            continue
        compact.append(w)
        last = wl
    candidate = " ".join(compact)

    letters = [ch for ch in candidate if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        candidate = candidate.title()
    if len(candidate) > 90:
        candidate = candidate[:90].rstrip(" ,;-")
    return candidate


def _title_case_phrase(text: str) -> str:
    words = [w for w in _norm(text).split(" ") if w]
    return " ".join(w.capitalize() if w.isalpha() else w for w in words)


def _clean_title_from_validation_flags(title: str, issue_types: set[str]) -> str:
    candidate = _norm(title)
    lowered = candidate.lower()

    if {"promo_language", "offer_shipping_availability_language"} & issue_types:
        for term in (*PROMO_TERMS, *TITLE_OFFER_SHIPPING_TERMS):
            lowered = lowered.replace(term, " ")
        candidate = _norm(lowered)

    if "disallowed_or_construction" in issue_types:
        candidate = re.split(r"\s+or\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip() or candidate

    if "suspicious_special_characters" in issue_types:
        for ch in TITLE_SUSPICIOUS_CHARS:
            candidate = candidate.replace(ch, " ")
        candidate = re.sub(r"\s{2,}", " ", candidate).strip()

    if "keyword_stuffing" in issue_types:
        words = _norm(candidate).split(" ")
        compressed: list[str] = []
        seen_recent: list[str] = []
        for w in words:
            wl = w.lower()
            if wl in seen_recent[-2:]:
                continue
            compressed.append(w)
            seen_recent.append(wl)
        candidate = _norm(" ".join(compressed))

    return candidate


def generate_recommended_title(record: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    base = _norm(record.get("current_title") or record.get("product_title") or "")
    if not base:
        return "Add clear product title with brand, product type, and key differentiator"

    issue_types = {f.get("issue_type") for f in _findings_by_section(findings, "title")}
    if not issue_types:
        return base

    candidate = _clean_title_from_validation_flags(base, issue_types)
    candidate = _clean_title_candidate(candidate)
    if not candidate:
        candidate = base

    if "too_short" in issue_types and len(candidate) < 20:
        brand = _norm(str(record.get("brand", "")))
        product_type = _norm(str(record.get("subcategory") or record.get("category") or ""))
        tail_tokens = [t for t in re.findall(r"[A-Za-z0-9]+", base) if len(t) > 2]
        tail = " ".join(tail_tokens[:4]).strip()
        parts = [p for p in [brand, product_type, tail] if p]
        if parts:
            candidate = _norm(" - ".join(parts[:2]) + (f", {parts[2]}" if len(parts) > 2 else ""))

    if "all_caps" in issue_types:
        candidate = _title_case_phrase(candidate)
    if "title_too_long" in issue_types and len(candidate) > 90:
        candidate = candidate[:90].rstrip(" ,;-")
    return candidate


def _build_section_recommendations(
    *,
    findings: list[dict[str, Any]],
    section: str,
    output_field: str,
) -> list[str]:
    templates = _issue_templates(output_field)
    out: list[str] = []
    for f in _findings_by_section(findings, section):
        issue_type = str(f.get("issue_type", ""))
        if issue_type in templates:
            out.append(templates[issue_type])
        else:
            msg = _norm(str(f.get("message", "")))
            if msg:
                out.append(msg)
    return _dedupe_keep_order(out)


def generate_image_recommendations(record: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    recs = _build_section_recommendations(
        findings=findings,
        section="images",
        output_field="image_recommendations",
    )
    if any(f.get("issue_type") == "low_image_count" for f in _findings_by_section(findings, "images")):
        recs.extend(
            [
                "Add a product-in-context lifestyle image",
                "Add an infographic image highlighting key benefits",
                "Add a what's-included or size-comparison visual",
            ]
        )
    if not recs:
        image_count = int(record.get("image_count") or 0)
        if image_count <= 2:
            recs = [
                "Add a product-in-context lifestyle image",
                "Add a feature/benefit infographic visual",
                "Add a what's-included or step-by-step usage visual",
            ]
        else:
            recs = ["Maintain a strong hero image and diversify supporting visual types"]
    return _dedupe_keep_order(recs)[:5]


def generate_description_recommendations(findings: list[dict[str, Any]]) -> list[str]:
    items = _collect_routed_recommendation_items(findings, target_section="description")
    return _finalize_section_recommendations(items, section="description")


def generate_key_features_recommendations(findings: list[dict[str, Any]]) -> list[str]:
    items = _collect_routed_recommendation_items(findings, target_section="key_features")
    return _finalize_section_recommendations(items, section="key_features")


def generate_top_priority_fixes(findings: list[dict[str, Any]]) -> list[str]:
    title_issue_types = {str(f.get("issue_type", "")) for f in _findings_by_section(findings, "title")}
    title_specific_fix = "Tighten the title to a clear, structured format with key differentiators"
    if "title_too_long" in title_issue_types:
        title_specific_fix = "Shorten the title to 90 characters or less while keeping key product identifiers"
    elif {"promo_language", "offer_shipping_availability_language"} & title_issue_types:
        title_specific_fix = "Remove promotional, shipping, and offer language from the title"
    elif "disallowed_or_construction" in title_issue_types:
        title_specific_fix = 'Replace ambiguous "or" title construction with one clear product naming path'
    elif "suspicious_special_characters" in title_issue_types:
        title_specific_fix = "Remove non-compliant special characters from the title"
    elif "keyword_stuffing" in title_issue_types:
        title_specific_fix = "Reduce repetitive keywords in the title and keep phrasing concise"

    theme_map = {
        "title": title_specific_fix,
        "description": "Strengthen description depth with clearer use-case and benefit language",
        "key_features": "Rebuild key features into clean, non-overlapping shopper-focused bullets",
        "images": "Upgrade the visual stack with stronger hero and support imagery",
    }
    ranked = sorted(
        findings,
        key=lambda f: (-SEVERITY_WEIGHT.get(str(f.get("severity", "low")), 1), str(f.get("section", ""))),
    )
    fixes: list[str] = []
    seen_sections: set[str] = set()
    for f in ranked:
        sec = str(f.get("section", ""))
        if sec in seen_sections:
            continue
        if sec in theme_map:
            fixes.append(theme_map[sec])
            seen_sections.add(sec)
        if len(fixes) >= 5:
            break

    if not fixes:
        fixes = ["Maintain current content strengths and address minor consistency issues"]

    fallback_themes = [
        theme_map["title"],
        theme_map["description"],
        theme_map["key_features"],
        theme_map["images"],
    ]
    for item in fallback_themes:
        if len(fixes) >= 3:
            break
        if item not in fixes:
            fixes.append(item)
    return fixes[:5]


def generate_mvp_outputs_for_primary_entry(entry: dict[str, Any]) -> dict[str, Any]:
    record = entry.get("cached_record", {})
    findings = entry.get("rule_findings", []) or []
    return {
        "image_recommendations": generate_image_recommendations(record, findings),
        "recommended_title": generate_recommended_title(record, findings),
        "description_recommendations": generate_description_recommendations(findings),
        "key_features_recommendations": generate_key_features_recommendations(findings),
        "top_priority_fixes": generate_top_priority_fixes(findings),
    }


def is_output_shell_empty(outputs: dict[str, Any] | None) -> bool:
    if not outputs:
        return True
    if _norm(str(outputs.get("recommended_title", ""))):
        return False
    for k in (
        "image_recommendations",
        "description_recommendations",
        "key_features_recommendations",
        "top_priority_fixes",
    ):
        vals = outputs.get(k, [])
        if isinstance(vals, list) and any(_norm(str(v)) for v in vals):
            return False
    return True
