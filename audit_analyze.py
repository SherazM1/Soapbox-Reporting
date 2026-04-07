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

    if combined and not _has_any_term(combined, USE_CASE_TERMS):
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_use_case",
                severity="medium",
                message="Description lacks clear use-case language.",
                evidence={"word_count": wc},
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
