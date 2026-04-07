from __future__ import annotations

import re
from typing import Any


SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

PROMO_TERMS = ("best selling", "free shipping")
RETAILER_TERMS = ("walmart", "amazon", "target", "instacart", "costco", "kroger")


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


def _findings_by_section(findings: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("section") == section]


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
        },
        "key_features_recommendations": {
            "insufficient_bullets": "Add more key feature bullets to meet the minimum coverage baseline",
            "recommended_bullets_missing": "Build toward a 5-bullet structure to improve scanability",
            "ending_punctuation": "Remove ending punctuation so bullets read as clean fragments",
            "full_sentences": "Convert sentence-style bullets into concise feature fragments",
            "inconsistent_formatting": "Standardize capitalization and structure across all bullets",
            "retailer_mention": "Remove retailer references from key features",
            "forbidden_special_characters": "Remove non-compliant special characters from bullet text",
            "overlap_with_other_section": "Reduce description overlap and keep bullets focused on distinct points",
            "missing_use_case": "Strengthen benefit and use-case framing across bullets",
        },
    }
    return templates.get(section, {})


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


def generate_recommended_title(record: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    base = _norm(record.get("current_title") or record.get("product_title") or "")
    if not base:
        return "Add clear product title with brand, product type, and key differentiator"

    issue_types = {f.get("issue_type") for f in _findings_by_section(findings, "title")}
    if not issue_types:
        return base

    candidate = _clean_title_candidate(base)
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
    recs = _build_section_recommendations(
        findings=findings,
        section="description",
        output_field="description_recommendations",
    )
    return recs[:6]


def generate_key_features_recommendations(findings: list[dict[str, Any]]) -> list[str]:
    recs = _build_section_recommendations(
        findings=findings,
        section="key_features",
        output_field="key_features_recommendations",
    )
    return recs[:6]


def generate_top_priority_fixes(findings: list[dict[str, Any]]) -> list[str]:
    theme_map = {
        "title": "Tighten the title to a clear, structured format with key differentiators",
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
