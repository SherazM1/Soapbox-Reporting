from __future__ import annotations

import re
from typing import Any


SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

PROMO_TERMS = ("best selling", "free shipping")


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
            "missing_images": "Add finished product hero image",
            "low_image_count": "Add lifestyle image showing product in use",
            "missing_clear_hero": "Clarify hero image selection with strongest front-of-pack visual",
        },
        "description_recommendations": {
            "too_short": "Description is below minimum content depth and should be expanded",
            "below_recommended_length": "Description is below recommended depth and can be strengthened",
            "missing_product_name": "Product name should be repeated naturally in the body for SEO",
            "retailer_mention": "Remove retailer mentions and keep description channel-neutral",
            "missing_use_case": "Add concrete use-case language for when and how the product is used",
            "missing_outcome_focus": "Add outcome-focused language that explains customer benefit",
            "generic_content": "Replace generic copy with specific product details and value",
        },
        "key_features_recommendations": {
            "insufficient_bullets": "Fewer than 3 key feature bullets are present",
            "recommended_bullets_missing": "Expand key features toward a 5-bullet structure",
            "ending_punctuation": "Remove ending punctuation so bullets read as clean fragments",
            "full_sentences": "Convert full-sentence bullets into concise feature fragments",
            "inconsistent_formatting": "Align bullet capitalization and formatting consistently",
            "retailer_mention": "Remove retailer mentions from key features",
            "forbidden_special_characters": "Remove forbidden special characters in bullet text",
            "overlap_with_other_section": "Reduce overlap with description and emphasize distinct feature points",
            "missing_use_case": "Add clearer benefit and use-case cues in key features",
        },
    }
    return templates.get(section, {})


def _clean_title_candidate(title: str) -> str:
    candidate = _norm(title)
    lowered = candidate.lower()
    for term in PROMO_TERMS:
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
    if not recs:
        image_count = int(record.get("image_count") or 0)
        if image_count <= 2:
            recs = [
                "Add lifestyle image showing product in use",
                "Add feature/benefit infographic image",
                "Add what-is-included or key-attributes visual",
            ]
        else:
            recs = ["Maintain clear hero image and diversify secondary supporting visuals"]
    return recs[:5]


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
        "title": "Rewrite title for stronger SEO clarity and cleaner structure",
        "description": "Strengthen description depth with clearer use-case and benefit language",
        "key_features": "Improve key feature structure and formatting for scanability",
        "images": "Upgrade image stack with stronger hero, lifestyle, and supporting visuals",
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
