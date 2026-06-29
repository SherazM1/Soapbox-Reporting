from __future__ import annotations

from statistics import mean
from typing import Any

from app.audit_helpers.bullet_uniqueness import normalize_bullet_text


# TODO: If this wording needs non-code tuning by strategists, move these banks to
# a versioned JSON config. Keep them centralized here for this hardening pass.
RATING_SCALES: dict[str, dict[str, Any]] = {
    "consumer_demand": {
        "allowed": ("Strong", "Emerging", "Limited"),
        "default": "Emerging",
    },
    "walmart_opportunity": {
        "allowed": ("Significant", "Meaningful", "Selective"),
        "default": "Meaningful",
    },
    "competitive_benchmark": {
        "allowed": ("Competitive", "Evolving", "Limited"),
        "default": "Evolving",
    },
}

CONSUMER_DEMAND_LABELS = set(RATING_SCALES["consumer_demand"]["allowed"])
WALMART_OPPORTUNITY_LABELS = set(RATING_SCALES["walmart_opportunity"]["allowed"])
COMPETITIVE_BENCHMARK_LABELS = set(RATING_SCALES["competitive_benchmark"]["allowed"])

BULLET_BANK: dict[str, dict[str, str]] = {
    "consumer_demand": {
        "established_trust": "Established shopper trust across {category_phrase}",
        "strong_benefit_positioning": "Strong {benefit_phrase} positioning",
        "broad_relevance": "Broad {category_phrase} relevance across Walmart PDPs",
        "positive_review_foundation": "Positive review foundation supports consumer confidence",
        "clear_shopping_journey_fit": "Clear fit within Walmart's {category_phrase} shopping journey",
        "growing_relevance": "Growing shopper relevance across {category_phrase}",
        "review_confidence_opportunity": "Opportunity to strengthen review-backed consumer confidence",
        "limited_review_evidence": "Limited review evidence creates room to build shopper confidence",
        "benefit_clarity_opportunity": "Opportunity to clarify {benefit_phrase} positioning",
    },
    "walmart_opportunity": {
        "shelf_ownership": "{category_phrase} shelf ownership can be strengthened through PDP execution",
        "conversion_optimization": "Content cleanup creates room for clearer Walmart shopper guidance",
        "visual_storytelling_gap": "Opportunity to expand {visual_phrase}",
        "assortment_segmentation": "Sharper assortment cues can help shoppers compare variants faster",
        "shopper_guidance": "PDP content can guide shoppers more directly from need state to product choice",
    },
    "competitive_benchmark": {
        "broader_discoverability": "Competitor evidence shows broader {category_phrase} discoverability",
        "stronger_visual_storytelling": "Competitive PDPs show stronger {visual_phrase}",
        "educational_merchandising": "Educational merchandising supports clearer {category_phrase} comparison",
        "benefit_education": "Category leaders use clearer {benefit_phrase} education",
        "search_visibility": "Search visibility is increasingly shaped by benefit and use-case language",
        "limited_competitor_evidence": "Limited competitor evidence available for benchmarking",
    },
}

FALLBACK_BULLET_IDS: dict[str, tuple[str, str, str]] = {
    "consumer_demand": (
        "growing_relevance",
        "clear_shopping_journey_fit",
        "review_confidence_opportunity",
    ),
    "walmart_opportunity": (
        "shelf_ownership",
        "conversion_optimization",
        "shopper_guidance",
    ),
    "competitive_benchmark": (
        "limited_competitor_evidence",
        "search_visibility",
        "educational_merchandising",
    ),
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _blob(records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for record in records:
        parts.extend(
            [
                _safe_text(record.get("category")),
                _safe_text(record.get("product_type") or record.get("subcategory")),
                _safe_text(record.get("product_title") or record.get("title")),
                _safe_text(record.get("brand")),
            ]
        )
    return " ".join(part for part in parts if part).lower()


def _first_value(records: list[dict[str, Any]], *keys: str) -> str:
    for record in records:
        for key in keys:
            value = _safe_text(record.get(key))
            if value:
                return value
    return ""


def resolve_slide2_phrases(primary_records: list[dict[str, Any]]) -> dict[str, str]:
    records = [record for record in (primary_records or []) if isinstance(record, dict)]
    blob = _blob(records)
    category = _first_value(records, "category")
    product_type = _first_value(records, "product_type", "subcategory")

    if any(term in blob for term in ("baby wash", "baby care", "infant", "toddler", "baby")):
        return {
            "category_phrase": "baby care",
            "benefit_phrase": "gentle family-care",
            "visual_phrase": "routine-based bath-time storytelling",
            "shopper_phrase": "parents and family-care shoppers",
        }
    if any(term in blob for term in ("skin care", "skincare", "dermatolog", "body wash", "lotion", "serum")):
        return {
            "category_phrase": "skin care",
            "benefit_phrase": "ingredient-led",
            "visual_phrase": "regimen and usage education",
            "shopper_phrase": "care-focused shoppers",
        }
    if any(term in blob for term in ("nut butter", "peanut butter", "almond butter")):
        return {
            "category_phrase": "nut butter and spreads",
            "benefit_phrase": "protein and ingredient-led",
            "visual_phrase": "snack, breakfast, and recipe-based usage storytelling",
            "shopper_phrase": "family pantry shoppers",
        }
    if any(term in blob for term in ("jam", "jell", "preserve", "fruit spread")):
        return {
            "category_phrase": "jams and fruit spreads",
            "benefit_phrase": "flavor-forward",
            "visual_phrase": "recipe-led serving inspiration",
            "shopper_phrase": "breakfast and snacking shoppers",
        }
    if any(term in blob for term in ("household cleaning", "cleaner", "surface", "disinfect")):
        return {
            "category_phrase": "household cleaning",
            "benefit_phrase": "efficacy-led",
            "visual_phrase": "usage and surface-specific education",
            "shopper_phrase": "solution-seeking household shoppers",
        }

    return {
        "category_phrase": product_type or category or "the category",
        "benefit_phrase": "benefit-led",
        "visual_phrase": "use-case and educational storytelling",
        "shopper_phrase": "Walmart shoppers",
    }


def _ratings(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        reviews = record.get("reviews_summary", {}) or {}
        rating = reviews.get("average_rating")
        try:
            if rating is not None:
                values.append(float(rating))
        except (TypeError, ValueError):
            continue
    return values


def _rating_counts(records: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for record in records:
        reviews = record.get("reviews_summary", {}) or {}
        count = reviews.get("ratings_count")
        try:
            if count is not None:
                values.append(int(count))
        except (TypeError, ValueError):
            continue
    return values


def _validate_rating(section_key: str, rating: str, warnings: list[str]) -> str:
    scale = RATING_SCALES[section_key]
    allowed = set(scale["allowed"])
    if rating in allowed:
        return rating
    fallback = str(scale["default"])
    warnings.append(
        f"{section_key} produced invalid rating '{rating or '<empty>'}'; "
        f"replaced with default '{fallback}'."
    )
    return fallback


def _consumer_demand_rating(records: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    ratings = _ratings(records)
    rating_counts = _rating_counts(records)
    warnings: list[str] = []
    signals: list[str] = []
    if ratings:
        avg_rating = mean(ratings)
        signals.append(f"avg_rating={avg_rating:.2f}")
    else:
        avg_rating = 0.0
        signals.append("no_rating_evidence")
    total_reviews = sum(rating_counts)
    signals.append(f"rating_count={total_reviews}")

    if not records:
        warnings.append("No primary records available; Consumer Demand used default fallback logic.")
        return "Emerging", signals, warnings
    if not ratings:
        warnings.append("No review/rating evidence available; Consumer Demand used limited-evidence fallback.")
        return "Limited", signals, warnings
    if avg_rating >= 4.2 and total_reviews >= 50:
        return "Strong", signals, warnings
    if avg_rating >= 3.7 and total_reviews >= 10:
        return "Emerging", signals, warnings
    return "Limited", signals, warnings


def _gap_count(record: dict[str, Any]) -> int:
    if "_slide2_gap_count" in record:
        try:
            return int(record.get("_slide2_gap_count", 0) or 0)
        except (TypeError, ValueError):
            return 0
    count = 0
    if record.get("recommended_title"):
        count += 1
    for key in ("image_recommendations", "description_recommendations", "key_features_recommendations"):
        values = record.get(key, [])
        if isinstance(values, list):
            count += len([value for value in values if _safe_text(value)])
    return count


def _walmart_opportunity_rating(
    records: list[dict[str, Any]], slide4_findings: dict[str, Any] | None
) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    gap_counts = [_gap_count(record) for record in records]
    avg_gaps = mean(gap_counts) if gap_counts else 0.0
    client_findings = ((slide4_findings or {}).get("client", {}) or {})
    opportunity_count = len(client_findings.get("opportunities", []) or [])
    signals = [f"avg_gap_count={avg_gaps:.2f}", f"slide4_opportunities={opportunity_count}"]
    if not records:
        warnings.append("No primary records available; Walmart Opportunity used default fallback logic.")
        return "Meaningful", signals, warnings
    if avg_gaps >= 3 or opportunity_count >= 3:
        return "Significant", signals, warnings
    if avg_gaps >= 1 or opportunity_count >= 1:
        return "Meaningful", signals, warnings
    warnings.append("Limited gap evidence available; Walmart Opportunity used selective/low-gap fallback.")
    return "Selective", signals, warnings


def _competitive_rating(
    primary_records: list[dict[str, Any]],
    competitor_records: list[dict[str, Any]],
    slide4_findings: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    if not competitor_records:
        warnings.append("No competitor records available; Competitive Benchmark used default fallback logic.")
        return "Evolving", ["no_competitor_records"], warnings

    client_findings = ((slide4_findings or {}).get("client", {}) or {})
    competitor_findings = [
        (slide4_findings or {}).get("competitor_1", {}) or {},
        (slide4_findings or {}).get("competitor_2", {}) or {},
    ]
    client_opportunities = len(client_findings.get("opportunities", []) or [])
    competitor_strengths = sum(len(finding.get("strengths", []) or []) for finding in competitor_findings)
    analyzed_competitors = sum(
        1
        for record in competitor_records
        if int((record.get("image_analysis", {}) or {}).get("analyzed_image_count", 0) or 0) > 0
    )
    signals = [
        f"client_slide4_opportunities={client_opportunities}",
        f"competitor_slide4_strengths={competitor_strengths}",
        f"analyzed_competitors={analyzed_competitors}",
    ]
    if competitor_strengths >= 4 and client_opportunities >= 2:
        return "Competitive", signals, warnings
    if analyzed_competitors or len(competitor_records) >= 2:
        return "Evolving", signals, warnings
    warnings.append("Competitor records exist but lack analysis depth; Competitive Benchmark used default fallback logic.")
    return "Evolving", signals, warnings


def _slide4_support(
    slide4_findings: dict[str, Any] | None,
    *,
    group_key: str,
    signals: tuple[str, ...],
) -> tuple[list[str], int, int]:
    findings = ((slide4_findings or {}).get(group_key, {}) or {})
    matched_signals: list[str] = []
    supporting_count = 0
    analyzed_count = int(findings.get("analyzed_pdp_count", 0) or 0)
    for collection_key in ("strengths", "opportunities"):
        for finding in findings.get(collection_key, []) or []:
            signal = _safe_text(finding.get("signal"))
            if signal in signals:
                matched_signals.append(signal)
                supporting_count = max(supporting_count, int(finding.get("supporting_pdps", 0) or 0))
                analyzed_count = max(analyzed_count, int(finding.get("analyzed_pdps", 0) or 0))
    return list(dict.fromkeys(matched_signals)), supporting_count, analyzed_count


def _bullet(
    section: str,
    template_id: str,
    phrases: dict[str, str],
    *,
    signals: list[str],
    reason: str,
    supporting_count: int = 0,
    analyzed_count: int = 0,
) -> dict[str, Any]:
    text = BULLET_BANK[section][template_id].format(**phrases)
    return {
        "text": text,
        "template_id": template_id,
        "section": section,
        "signals": signals,
        "supporting_count": supporting_count,
        "analyzed_count": analyzed_count,
        "reason": reason,
    }


def _append_unique_bullet(target: list[dict[str, Any]], bullet: dict[str, Any]) -> None:
    if normalize_bullet_text(bullet["text"]) not in {normalize_bullet_text(item["text"]) for item in target}:
        target.append(bullet)


def _ensure_minimum_bullets(
    section: str,
    bullets: list[dict[str, Any]],
    phrases: dict[str, str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    for template_id in FALLBACK_BULLET_IDS[section]:
        if len(bullets) >= 3:
            break
        _append_unique_bullet(
            bullets,
            _bullet(
                section,
                template_id,
                phrases,
                signals=["fallback_controlled_bullet"],
                reason="Selected as a safe controlled fallback because fewer than three evidence-backed bullets were available.",
            ),
        )
    if len(bullets) < 3:
        warnings.append(f"{section} had fewer than three unique controlled fallback bullets.")
    return bullets[:4]


def _consumer_bullets(
    rating: str,
    phrases: dict[str, str],
    signals: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    rating_signal = f"consumer_rating={rating}"
    if rating == "Strong":
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "established_trust",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because rating and review volume indicate strong shopper trust.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "strong_benefit_positioning",
                phrases,
                signals=[rating_signal, "category_phrase_match"],
                reason="Selected to connect demand strength to the resolved category benefit positioning.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "positive_review_foundation",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because review evidence supports shopper confidence.",
            ),
        )
    elif rating == "Emerging":
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "growing_relevance",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because available demand signals are moderate or mixed.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "clear_shopping_journey_fit",
                phrases,
                signals=[rating_signal, "category_phrase_match"],
                reason="Selected to preserve category relevance when demand evidence is not fully strong.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "review_confidence_opportunity",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because the review foundation can be strengthened.",
            ),
        )
    else:
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "limited_review_evidence",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because review/rating evidence is sparse or weak.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "clear_shopping_journey_fit",
                phrases,
                signals=[rating_signal, "category_phrase_match"],
                reason="Selected as a controlled category-fit fallback despite limited demand evidence.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "consumer_demand",
                "benefit_clarity_opportunity",
                phrases,
                signals=[rating_signal, "benefit_phrase_match"],
                reason="Selected to avoid overclaiming demand while keeping benefit positioning actionable.",
            ),
        )
    _append_unique_bullet(
        bullets,
        _bullet(
            "consumer_demand",
            "broad_relevance",
            phrases,
            signals=[rating_signal, "category_phrase_match"],
            reason="Selected as controlled category relevance support for the demand section.",
        ),
    )
    return _ensure_minimum_bullets("consumer_demand", bullets, phrases, warnings)


def _opportunity_bullets(
    rating: str,
    phrases: dict[str, str],
    slide4_findings: dict[str, Any] | None,
    signals: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    rating_signal = f"walmart_opportunity_rating={rating}"
    _append_unique_bullet(
        bullets,
        _bullet(
            "walmart_opportunity",
            "shelf_ownership",
            phrases,
            signals=[rating_signal, *signals],
            reason="Selected because PDP/content or visual findings indicate fixable Walmart shelf-ownership opportunity.",
        ),
    )
    _append_unique_bullet(
        bullets,
        _bullet(
            "walmart_opportunity",
            "conversion_optimization",
            phrases,
            signals=[rating_signal, *signals],
            reason="Selected because the section rating is driven by conversion-relevant content or image gaps.",
        ),
    )
    visual_signals, supporting_count, analyzed_count = _slide4_support(
        slide4_findings,
        group_key="client",
        signals=("missing_usage_or_recipe_storytelling", "missing_lifestyle_storytelling", "limited_conversion_guidance"),
    )
    if visual_signals:
        _append_unique_bullet(
            bullets,
            _bullet(
                "walmart_opportunity",
                "visual_storytelling_gap",
                phrases,
                signals=visual_signals,
                supporting_count=supporting_count,
                analyzed_count=analyzed_count,
                reason="Selected because Slide 4 findings showed a majority gap in usage, recipe, lifestyle, or conversion storytelling.",
            ),
        )
    if rating == "Significant":
        _append_unique_bullet(
            bullets,
            _bullet(
                "walmart_opportunity",
                "shopper_guidance",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because broad gap evidence supports a shopper-guidance opportunity.",
            ),
        )
    else:
        _append_unique_bullet(
            bullets,
            _bullet(
                "walmart_opportunity",
                "assortment_segmentation",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected as a focused opportunity when the gap evidence is more selective.",
            ),
        )
    return _ensure_minimum_bullets("walmart_opportunity", bullets, phrases, warnings)


def _competitive_bullets(
    rating: str,
    phrases: dict[str, str],
    slide4_findings: dict[str, Any] | None,
    signals: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    rating_signal = f"competitive_benchmark_rating={rating}"
    if rating == "Limited" or "no_competitor_records" in signals:
        _append_unique_bullet(
            bullets,
            _bullet(
                "competitive_benchmark",
                "limited_competitor_evidence",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because competitor records or competitor analysis evidence are limited.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "competitive_benchmark",
                "search_visibility",
                phrases,
                signals=[rating_signal, "safe_category_benchmark"],
                reason="Selected as a generic controlled benchmark when competitor evidence is weak.",
            ),
        )
    else:
        _append_unique_bullet(
            bullets,
            _bullet(
                "competitive_benchmark",
                "broader_discoverability",
                phrases,
                signals=[rating_signal, *signals],
                reason="Selected because competitor data creates benchmark pressure in category discoverability.",
            ),
        )
        comp_signals, supporting_count, analyzed_count = _slide4_support(
            slide4_findings,
            group_key="competitor_1",
            signals=("usage_or_recipe_storytelling", "lifestyle_or_contextual_storytelling", "benefit_forward_graphics"),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "competitive_benchmark",
                "stronger_visual_storytelling",
                phrases,
                signals=comp_signals or [rating_signal, "competitor_benchmark_available"],
                supporting_count=supporting_count,
                analyzed_count=analyzed_count,
                reason="Selected because competitor evidence supports comparison against visual or educational storytelling.",
            ),
        )
        _append_unique_bullet(
            bullets,
            _bullet(
                "competitive_benchmark",
                "benefit_education",
                phrases,
                signals=[rating_signal, "benefit_phrase_match"],
                reason="Selected to frame competitor benchmarking around controlled benefit education language.",
            ),
        )
    _append_unique_bullet(
        bullets,
        _bullet(
            "competitive_benchmark",
            "educational_merchandising",
            phrases,
            signals=[rating_signal, "category_phrase_match"],
            reason="Selected as controlled category benchmark language for educational merchandising.",
        ),
    )
    return _ensure_minimum_bullets("competitive_benchmark", bullets, phrases, warnings)


def _section_payload(
    *,
    section_key: str,
    label: str,
    rating: str,
    signals: list[str],
    bullet_debug: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    validated_rating = _validate_rating(section_key, rating, warnings)
    return {
        "label": label,
        "rating": validated_rating,
        "allowed_ratings": list(RATING_SCALES[section_key]["allowed"]),
        "default_rating": RATING_SCALES[section_key]["default"],
        "signals": signals,
        "bullets": [item["text"] for item in bullet_debug],
        "bullet_debug": bullet_debug,
        "debug_warnings": warnings,
    }


def build_slide2_summary_payload(
    primary_records: list[dict[str, Any]],
    competitor_records: list[dict[str, Any]] | None = None,
    slide4_findings: dict[str, Any] | None = None,
    audit_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_records = [record for record in (primary_records or []) if isinstance(record, dict)]
    competitor_records = [record for record in (competitor_records or []) if isinstance(record, dict)]
    audit_metadata = audit_metadata or {}
    phrases = resolve_slide2_phrases(primary_records)
    client_name = _safe_text(
        audit_metadata.get("client_company_name") or audit_metadata.get("client_name") or "The brand"
    )

    consumer_rating, consumer_signals, consumer_warnings = _consumer_demand_rating(primary_records)
    opportunity_rating, opportunity_signals, opportunity_warnings = _walmart_opportunity_rating(
        primary_records, slide4_findings
    )
    competitive_rating, competitive_signals, competitive_warnings = _competitive_rating(
        primary_records, competitor_records, slide4_findings
    )

    consumer_rating = _validate_rating("consumer_demand", consumer_rating, consumer_warnings)
    opportunity_rating = _validate_rating("walmart_opportunity", opportunity_rating, opportunity_warnings)
    competitive_rating = _validate_rating("competitive_benchmark", competitive_rating, competitive_warnings)

    intro_copy = (
        f"{client_name} has built a foundation of shopper relevance across "
        f"{phrases['category_phrase']}, with the next opportunity centered on translating "
        "that equity into stronger Walmart digital shelf ownership, category discoverability, "
        "and conversion-focused PDP execution."
    )
    sections = {
        "consumer_demand": _section_payload(
            section_key="consumer_demand",
            label="Consumer Demand",
            rating=consumer_rating,
            signals=consumer_signals,
            bullet_debug=_consumer_bullets(consumer_rating, phrases, consumer_signals, consumer_warnings),
            warnings=consumer_warnings,
        ),
        "walmart_opportunity": _section_payload(
            section_key="walmart_opportunity",
            label="Walmart Opportunity",
            rating=opportunity_rating,
            signals=opportunity_signals,
            bullet_debug=_opportunity_bullets(
                opportunity_rating,
                phrases,
                slide4_findings,
                opportunity_signals,
                opportunity_warnings,
            ),
            warnings=opportunity_warnings,
        ),
        "competitive_benchmark": _section_payload(
            section_key="competitive_benchmark",
            label="Competitive Benchmark",
            rating=competitive_rating,
            signals=competitive_signals,
            bullet_debug=_competitive_bullets(
                competitive_rating,
                phrases,
                slide4_findings,
                competitive_signals,
                competitive_warnings,
            ),
            warnings=competitive_warnings,
        ),
    }
    debug_warnings = []
    for section in sections.values():
        debug_warnings.extend(section.get("debug_warnings", []) or [])
    return {
        "intro_copy": intro_copy,
        "sections": sections,
        "phrases": phrases,
        "debug": {
            "primary_record_count": len(primary_records),
            "competitor_record_count": len(competitor_records),
            "warnings": debug_warnings,
        },
    }
