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

    return findings


def analyze_description(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    body = _norm(record.get("current_description_body", ""))
    bullets = [str(b).strip() for b in record.get("current_description_bullets", []) if str(b).strip()]
    combined = _norm(record.get("current_description_combined", "") or " ".join([body, *bullets]))
    combined_lower = combined.lower()
    wc = _word_count(combined)

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
