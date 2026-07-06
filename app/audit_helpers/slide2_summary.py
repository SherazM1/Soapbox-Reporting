from __future__ import annotations

from statistics import mean
from typing import Any

from app.audit_helpers.bullet_uniqueness import normalize_bullet_text
from app.audit_helpers.strategic_cues import aggregate_pdp_cues, translate_cues


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
        "established_trust": "Established shopper trust in {category_context_phrase}",
        "strong_benefit_positioning": "Strong {product_positioning_phrase}",
        "broad_relevance": "Fits high-frequency Walmart pantry routines",
        "positive_review_foundation": "Reviews provide a trust base for conversion",
        "clear_shopping_journey_fit": "Clear fit for the Walmart {shopping_journey_phrase}",
        "growing_relevance": "Growing relevance for {shopper_phrase}",
        "review_confidence_opportunity": "Review depth can further reduce purchase friction",
        "limited_review_evidence": "Sparse reviews make trust-building more important",
        "benefit_clarity_opportunity": "Clarify {benefit_phrase} benefits earlier",
    },
    "walmart_opportunity": {
        "shelf_ownership": "Stronger Walmart PDP content can deepen shelf ownership",
        "conversion_optimization": "Sharper PDP guidance can reduce purchase friction",
        "visual_storytelling_gap": "Expand {visual_phrase}",
        "assortment_segmentation": "Sharper assortment cues can help shoppers compare variants faster",
        "shopper_guidance": "Connect shopper needs to clearer product choice",
    },
    "competitive_benchmark": {
        "broader_discoverability": "Competitors broaden {discovery_phrase}",
        "stronger_visual_storytelling": "Benchmark PDPs use stronger {visual_phrase}",
        "educational_merchandising": "Education helps shoppers compare formats faster",
        "benefit_education": "Category leaders use clearer {benefit_phrase} education",
        "search_visibility": "Benefit and use-case language shape {shelf_navigation_phrase}",
        "limited_competitor_evidence": "Limited competitor evidence available for benchmarking",
    },
}

FALLBACK_BULLET_IDS: dict[str, tuple[str, ...]] = {
    "consumer_demand": (
        "growing_relevance",
        "clear_shopping_journey_fit",
        "review_confidence_opportunity",
        "benefit_clarity_opportunity",
    ),
    "walmart_opportunity": (
        "shelf_ownership",
        "conversion_optimization",
        "shopper_guidance",
        "assortment_segmentation",
    ),
    "competitive_benchmark": (
        "limited_competitor_evidence",
        "search_visibility",
        "educational_merchandising",
        "benefit_education",
    ),
}

SLIDE2_MAX_BULLET_CHARS = 64
MIN_SECTION_BANK_BULLETS = 2
MAX_CUE_SWAPS_PER_SECTION = 2

SECTION_CUE_RULES: dict[str, dict[str, Any]] = {
    "consumer_demand": {
        "allowed_cues": {"product_positioning", "benefit_communication", "review_or_trust_signals"},
        "allowed_classifications": {"strength", "context"},
        "required_terms": {
            "trust",
            "review",
            "relevance",
            "confidence",
            "fit",
            "foundation",
            "positioning",
            "demand",
        },
        "blocked_terms": {"opportunity", "competitor", "benchmark", "pressure", "gap"},
    },
    "walmart_opportunity": {
        "allowed_cues": {
            "keyword_alignment",
            "discoverability",
            "assortment_segmentation",
            "shopper_education",
            "conversion_guidance",
            "visual_identity",
            "product_positioning",
        },
        "allowed_classifications": {"opportunity", "pressure", "context", "strength"},
        "required_terms": {
            "opportunity",
            "walmart",
            "pdp",
            "content",
            "shelf",
            "conversion",
            "guidance",
            "discoverability",
            "visibility",
            "assortment",
        },
        "blocked_terms": {"competitor", "benchmark", "category leaders"},
    },
    "competitive_benchmark": {
        "allowed_cues": {
            "keyword_alignment",
            "discoverability",
            "shopper_education",
            "usage_storytelling",
            "visual_identity",
            "benefit_communication",
            "review_or_trust_signals",
        },
        "allowed_classifications": {"pressure", "context", "strength", "opportunity"},
        "required_terms": {
            "competitor",
            "benchmark",
            "competitive",
            "category leaders",
            "comparison",
            "pressure",
            "limited competitor",
        },
        "blocked_terms": {"client-side", "walmart opportunity", "ownable"},
    },
}

SYNTHETIC_FILLER_PATTERNS = (
    "more ownable",
    "enhanced ownable",
    "value communication",
    "focused ",
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _shorten_slide2_bullet(text: str, *, hard_limit: int = SLIDE2_MAX_BULLET_CHARS) -> str:
    clean = " ".join(_safe_text(text).split())
    replacements = (
        ("Opportunity to strengthen", "Strengthens"),
        ("Opportunity to expand", "Expands"),
        ("Opportunity to clarify", "Clarifies"),
        ("Opportunity to", "Can"),
        ("creates room for", "supports"),
        ("conversion-focused", "conversion"),
        ("Walmart shopper", "shopper"),
        ("shopper confidence", "confidence"),
        ("category discoverability", "discoverability"),
        ("educational merchandising", "education"),
    )
    for old, new in replacements:
        clean = clean.replace(old, new)
    clean = clean.replace(" And ", " and ")
    if len(clean) <= hard_limit:
        return clean
    words = clean.split()
    while words and len(" ".join(words)) > hard_limit:
        words.pop()
    while words and words[-1].lower() in {"and", "or", "for", "with", "to"}:
        words.pop()
    shortened = " ".join(words).rstrip(" ,;-")
    return shortened or clean[:hard_limit].rsplit(" ", 1)[0].rstrip(" ,;-")


def _dedupe_and_fit_slide2_sections(
    sections: dict[str, dict[str, Any]],
    phrases: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    original: dict[str, list[str]] = {
        key: list(section.get("bullets", []) or [])
        for key, section in sections.items()
        if isinstance(section, dict)
    }
    deduped: dict[str, list[str]] = {}
    shortened: dict[str, list[str]] = {}
    final: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    used: set[str] = set()

    for section_key, section in sections.items():
        if not isinstance(section, dict):
            continue
        deduped[section_key] = []
        shortened[section_key] = []
        final[section_key] = []
        dropped[section_key] = []
        for bullet in section.get("bullets", []) or []:
            clean = _safe_text(bullet)
            if not clean:
                continue
            normalized = normalize_bullet_text(clean)
            if normalized in used or any(
                normalized and (normalized in existing or existing in normalized)
                for existing in used
            ):
                dropped[section_key].append(clean)
                continue
            used.add(normalized)
            deduped[section_key].append(clean)
            shortened_text = _shorten_slide2_bullet(clean)
            shortened[section_key].append(shortened_text)
            if len(final[section_key]) < 4:
                final[section_key].append(shortened_text)
            else:
                dropped[section_key].append(clean)

        for template_id in FALLBACK_BULLET_IDS.get(section_key, ()):
            if len(final[section_key]) >= 4:
                break
            fallback = _shorten_slide2_bullet(
                BULLET_BANK[section_key][template_id].format(**phrases)
            )
            normalized = normalize_bullet_text(fallback)
            if normalized and normalized not in used:
                used.add(normalized)
                final[section_key].append(fallback)
                shortened[section_key].append(fallback)
                section.setdefault("bullet_debug", []).append(
                    {
                        "text": fallback,
                        "template_id": template_id,
                        "section": section_key,
                        "source_tag": "section_bank_fallback",
                        "signals": ["dedupe_refill_controlled_bullet"],
                        "supporting_count": 0,
                        "analyzed_count": 0,
                        "reason": "Controlled refill added after cross-section dedupe to preserve four bullets.",
                    }
                )

        section["bullets"] = final[section_key]
        selected_debug = []
        final_norms = [normalize_bullet_text(value) for value in final[section_key]]
        for normalized in final_norms:
            match = next(
                (
                    item
                    for item in section.get("bullet_debug", []) or []
                    if normalize_bullet_text(_shorten_slide2_bullet(item.get("text", ""))) == normalized
                ),
                None,
            )
            if match:
                copied = dict(match)
                copied["text"] = _shorten_slide2_bullet(copied.get("text", ""))
                selected_debug.append(copied)
        section["bullet_debug"] = selected_debug
        section["bullet_fit_debug"] = {
            "original_candidates": original.get(section_key, []),
            "deduped_bullets": deduped[section_key],
            "shortened_bullets": shortened[section_key],
            "dropped_bullets": dropped[section_key],
            "final_rendered_bullets": final[section_key],
            "final_bullet_count": len(final[section_key]),
            "dedupe_result": "passed" if len(final[section_key]) == len(set(final_norms)) else "duplicates_remaining",
            "fit_result": "passed" if all(len(value) <= SLIDE2_MAX_BULLET_CHARS for value in final[section_key]) else "over_limit",
        }
        section["final_validation"] = {
            "final_bullet_count": len(final[section_key]),
            "dedupe_result": section["bullet_fit_debug"]["dedupe_result"],
            "fit_result": section["bullet_fit_debug"]["fit_result"],
            "final_rating": section.get("rating"),
            "final_rating_valid": section.get("rating") in RATING_SCALES[section_key]["allowed"],
            "required_count_met": len(final[section_key]) == 4,
        }

    return sections, {
        "original_candidate_bullets": original,
        "deduped_bullets": deduped,
        "shortened_bullets": shortened,
        "dropped_bullets": dropped,
        "final_rendered_bullets": final,
    }


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


def _clean_phrase(value: str) -> str:
    cleaned = " ".join(_safe_text(value).replace("&", "and").split())
    return cleaned.strip(" /-").lower()


def _last_category_segment(category: str) -> str:
    parts = [_clean_phrase(part) for part in _safe_text(category).split("/") if _clean_phrase(part)]
    return parts[-1] if parts else ""


def _phrase_set(
    *,
    category_phrase: str,
    category: str,
    product_type: str,
    benefit_phrase: str,
    visual_phrase: str,
    shopper_phrase: str,
) -> dict[str, str]:
    category_label = _clean_phrase(_last_category_segment(category) or category_phrase or "the category")
    product_label = _clean_phrase(product_type or category_phrase or category_label or "the product type")
    if category_label.startswith("the "):
        category_context = category_label
    elif category_label == product_label:
        category_context = f"the {category_label} category"
    else:
        category_context = f"the {category_label} and {product_label} category"
    if category_label == product_label:
        segment_phrase = f"the {product_label} segment"
    else:
        segment_phrase = f"the {product_label} segment within {category_label}"
    return {
        "category_phrase": category_phrase,
        "category_context_phrase": category_context,
        "product_type_phrase": product_label,
        "product_positioning_phrase": f"{product_label} positioning",
        "shopping_journey_phrase": f"{product_label} shopping journey",
        "discovery_phrase": f"{category_label} discovery and comparison",
        "shelf_navigation_phrase": f"{product_label} shelf navigation",
        "segment_phrase": segment_phrase,
        "benefit_phrase": benefit_phrase,
        "visual_phrase": visual_phrase,
        "shopper_phrase": shopper_phrase,
    }


def resolve_slide2_phrases(primary_records: list[dict[str, Any]]) -> dict[str, str]:
    records = [record for record in (primary_records or []) if isinstance(record, dict)]
    blob = _blob(records)
    category = _first_value(records, "category")
    product_type = _first_value(records, "product_type", "subcategory")

    if any(term in blob for term in ("baby wash", "baby care", "infant", "toddler", "baby")):
        return _phrase_set(
            category_phrase="baby care",
            category=category or "baby care",
            product_type=product_type or "baby care",
            benefit_phrase="gentle family-care",
            visual_phrase="routine-based bath-time storytelling",
            shopper_phrase="parents and family-care shoppers",
        )
    if any(term in blob for term in ("skin care", "skincare", "dermatolog", "body wash", "lotion", "serum")):
        return _phrase_set(
            category_phrase="skin care",
            category=category or "skin care",
            product_type=product_type or "skin care",
            benefit_phrase="ingredient-led",
            visual_phrase="regimen and usage education",
            shopper_phrase="care-focused shoppers",
        )
    if any(term in blob for term in ("nut butter", "peanut butter", "almond butter")):
        return _phrase_set(
            category_phrase="nut butter and spreads",
            category=category or "pantry spreads",
            product_type=product_type or "nut butter spreads",
            benefit_phrase="protein and ingredient-led",
            visual_phrase="snack, breakfast, and recipe-based usage storytelling",
            shopper_phrase="family pantry shoppers",
        )
    if any(term in blob for term in ("jam", "jell", "preserve", "fruit spread")):
        return _phrase_set(
            category_phrase="jams and fruit spreads",
            category=category or "fruit spreads",
            product_type=product_type or "jams and preserves",
            benefit_phrase="flavor-forward",
            visual_phrase="recipe-led serving inspiration",
            shopper_phrase="breakfast and snacking shoppers",
        )
    if any(term in blob for term in ("household cleaning", "cleaner", "surface", "disinfect")):
        return _phrase_set(
            category_phrase="household cleaning",
            category=category or "household cleaning",
            product_type=product_type or "cleaning products",
            benefit_phrase="efficacy-led",
            visual_phrase="usage and surface-specific education",
            shopper_phrase="solution-seeking household shoppers",
        )

    fallback = product_type or category or "the category"
    return _phrase_set(
        category_phrase=fallback,
        category=category or fallback,
        product_type=product_type or fallback,
        benefit_phrase="benefit-led",
        visual_phrase="use-case and educational storytelling",
        shopper_phrase="Walmart shoppers",
    )


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
    source_tag: str = "section_bank_original",
) -> dict[str, Any]:
    text = BULLET_BANK[section][template_id].format(**phrases)
    return {
        "text": text,
        "template_id": template_id,
        "section": section,
        "source_tag": source_tag,
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
        if len(bullets) >= 4:
            break
        _append_unique_bullet(
            bullets,
            _bullet(
                section,
                template_id,
                phrases,
                signals=["fallback_controlled_bullet"],
                reason="Selected as a safe controlled fallback because fewer than four evidence-backed bullets were available.",
                source_tag="section_bank_fallback",
            ),
        )
    if len(bullets) < 4:
        warnings.append(f"{section} had fewer than four unique controlled fallback bullets.")
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
    rating_reason = _rating_reason(section_key, validated_rating, signals, warnings)
    return {
        "label": label,
        "rating": validated_rating,
        "allowed_ratings": list(RATING_SCALES[section_key]["allowed"]),
        "default_rating": RATING_SCALES[section_key]["default"],
        "signals": signals,
        "rating_reason": rating_reason,
        "rating_signals": signals,
        "rating_inputs": {
            "section_key": section_key,
            "raw_rating": rating,
            "validated_rating": validated_rating,
            "warnings": list(warnings),
        },
        "bullets": [item["text"] for item in bullet_debug],
        "bullet_debug": bullet_debug,
        "debug_warnings": warnings,
    }


def _rating_reason(section_key: str, rating: str, signals: list[str], warnings: list[str]) -> str:
    if warnings:
        warning_text = warnings[-1]
        if "No" in warning_text or "Limited" in warning_text:
            return warning_text
    if section_key == "consumer_demand":
        if rating == "Strong":
            return "Strong was selected because review volume and rating quality indicate credible shopper trust."
        if rating == "Emerging":
            return "Emerging was selected because demand evidence is present but not yet broad enough for Strong."
        return "Limited was selected because review or rating evidence is sparse, weak, or unavailable."
    if section_key == "walmart_opportunity":
        if rating == "Significant":
            return "Significant was selected because PDP/content gap evidence indicates a material Walmart opportunity."
        if rating == "Meaningful":
            return "Meaningful was selected because gap evidence indicates real but not maximal improvement potential."
        return "Selective was selected because available gap evidence points to narrower targeted improvements."
    if section_key == "competitive_benchmark":
        if rating == "Competitive":
            return "Competitive was selected because competitor strengths and client opportunities create benchmark pressure."
        if rating == "Evolving":
            return "Evolving was selected because competitor evidence provides moderate benchmark context."
        return "Limited was selected because competitor benchmark evidence is constrained."
    return f"{rating} was selected from signals: {', '.join(signals)}"


def _apply_cue_bullets_to_slide2_sections(
    sections: dict[str, dict[str, Any]],
    cue_context: dict[str, Any],
) -> None:
    section_orders = {
        "consumer_demand": ("context", "strength", "opportunity"),
        "walmart_opportunity": ("opportunity", "pressure", "context", "strength"),
        "competitive_benchmark": ("pressure", "opportunity", "context", "strength"),
    }
    for section_key, preferred_order in section_orders.items():
        section = sections.get(section_key)
        if not isinstance(section, dict):
            continue
        original_bullets = list(section.get("bullets", []) or [])
        original_debug = [dict(item) for item in section.get("bullet_debug", []) or []]
        bullets, debug = translate_cues(
            cue_context,
            slide_key="slide2",
            count=4,
            preferred_order=preferred_order,
            side=section_key,
        )
        refinement_debug = {
            "mode": "controlled_swap",
            "max_swaps": MAX_CUE_SWAPS_PER_SECTION,
            "minimum_section_bank_bullets": MIN_SECTION_BANK_BULLETS,
            "original_bullets": original_bullets,
            "accepted_swaps": [],
            "rejected_candidates": [],
        }
        section["cue_translation_debug"] = []
        swap_count = 0
        current_debug = [dict(item) for item in original_debug]
        for index, translated in enumerate(bullets):
            cue_item = debug[index] if index < len(debug) else {}
            candidate_text = _shorten_slide2_bullet(translated)
            decision = _cue_swap_decision(
                section_key=section_key,
                candidate_text=candidate_text,
                cue_item=cue_item,
                current_debug=current_debug,
                swap_count=swap_count,
            )
            traced_cue = {
                **cue_item,
                "text": candidate_text,
                "section": section_key,
                "template_id": f"cue_{cue_item.get('cue')}",
                "signals": [cue_item.get("classification", ""), cue_item.get("cue", "")],
                "supporting_count": 0,
                "analyzed_count": 0,
                "source_tag": "cue_candidate_reviewed",
                "accepted": decision["accepted"],
                "decision_reason": decision["reason"],
            }
            section["cue_translation_debug"].append(traced_cue)
            if not decision["accepted"]:
                refinement_debug["rejected_candidates"].append(
                    {
                        "replacement_bullet": candidate_text,
                        "cue": cue_item.get("cue"),
                        "classification": cue_item.get("classification"),
                        "reason": decision["reason"],
                    }
                )
                continue
            replaced = current_debug[decision["replace_index"]]
            replacement_debug = {
                **traced_cue,
                "source_tag": "cue_refined_swap",
                "original_bullet_text": replaced.get("text"),
                "replacement_bullet_text": candidate_text,
                "swap_reason": decision["reason"],
                "reason": "Controlled cue refinement accepted because it was section-specific and more concrete than the replaced bullet.",
            }
            current_debug[decision["replace_index"]] = replacement_debug
            swap_count += 1
            refinement_debug["accepted_swaps"].append(
                {
                    "original_bullet_text": replaced.get("text"),
                    "replacement_bullet_text": candidate_text,
                    "cue": cue_item.get("cue"),
                    "classification": cue_item.get("classification"),
                    "reason": decision["reason"],
                }
            )
        section["bullet_debug"] = current_debug[:4]
        section["bullets"] = [item.get("text", "") for item in section["bullet_debug"]]
        refinement_debug["final_bullets_before_fit"] = list(section["bullets"])
        refinement_debug["swap_count"] = swap_count
        section["cue_refinement_debug"] = refinement_debug


def _cue_swap_decision(
    *,
    section_key: str,
    candidate_text: str,
    cue_item: dict[str, Any],
    current_debug: list[dict[str, Any]],
    swap_count: int,
) -> dict[str, Any]:
    if swap_count >= MAX_CUE_SWAPS_PER_SECTION:
        return {"accepted": False, "reason": "Rejected because the section already reached the cue swap limit."}
    if not candidate_text or len(candidate_text) > SLIDE2_MAX_BULLET_CHARS:
        return {"accepted": False, "reason": "Rejected because the cue bullet was empty or exceeded the Slide 2 fit limit."}
    normalized_candidate = normalize_bullet_text(candidate_text)
    if any(pattern in candidate_text.lower() for pattern in SYNTHETIC_FILLER_PATTERNS):
        return {"accepted": False, "reason": "Rejected because the cue bullet used synthetic filler language."}
    if any(normalized_candidate == normalize_bullet_text(item.get("text", "")) for item in current_debug):
        return {"accepted": False, "reason": "Rejected because the cue bullet duplicated an existing section bullet."}
    if any(
        normalized_candidate and (
            normalized_candidate in normalize_bullet_text(item.get("text", ""))
            or normalize_bullet_text(item.get("text", "")) in normalized_candidate
        )
        for item in current_debug
    ):
        return {"accepted": False, "reason": "Rejected because the cue bullet was a near-duplicate within the section."}

    rules = SECTION_CUE_RULES[section_key]
    cue = _safe_text(cue_item.get("cue"))
    classification = _safe_text(cue_item.get("classification"))
    lowered = candidate_text.lower()
    if cue not in rules["allowed_cues"]:
        return {"accepted": False, "reason": f"Rejected because cue '{cue}' does not belong to {section_key}."}
    if classification and classification not in rules["allowed_classifications"]:
        return {
            "accepted": False,
            "reason": f"Rejected because classification '{classification}' does not align with {section_key}.",
        }
    if any(term in lowered for term in rules["blocked_terms"]):
        return {"accepted": False, "reason": "Rejected because the cue bullet crossed into another Slide 2 section meaning."}
    if not any(term in lowered for term in rules["required_terms"]):
        return {"accepted": False, "reason": "Rejected because the cue bullet lacked section-specific language."}
    section_bank_count = sum(1 for item in current_debug if item.get("source_tag") != "cue_refined_swap")
    if section_bank_count <= MIN_SECTION_BANK_BULLETS:
        return {"accepted": False, "reason": "Rejected to preserve at least two original section-built bullets."}

    candidate_score = _section_specificity_score(section_key, candidate_text, cue_item)
    replace_index = -1
    replace_score = 999
    for index, item in enumerate(current_debug):
        if item.get("source_tag") == "cue_refined_swap":
            continue
        score = _section_specificity_score(section_key, _safe_text(item.get("text")), item)
        if score < replace_score:
            replace_index = index
            replace_score = score
    if replace_index < 0:
        return {"accepted": False, "reason": "Rejected because no replaceable section-bank bullet was available."}
    if candidate_score <= replace_score:
        return {
            "accepted": False,
            "reason": "Rejected because the cue bullet was not more specific than the section-built bullet.",
        }
    return {
        "accepted": True,
        "replace_index": replace_index,
        "reason": (
            "Accepted because it matched the section meaning, improved specificity, "
            "and preserved the section-bank bullet majority."
        ),
    }


def _section_specificity_score(section_key: str, text: str, cue_item: dict[str, Any] | None = None) -> int:
    lowered = _safe_text(text).lower()
    rules = SECTION_CUE_RULES[section_key]
    score = 0
    score += sum(2 for term in rules["required_terms"] if term in lowered)
    score += 1 if any(char.isdigit() for char in lowered) else 0
    score += 1 if cue_item and cue_item.get("cue") in rules["allowed_cues"] else 0
    score -= sum(3 for term in rules["blocked_terms"] if term in lowered)
    score -= sum(2 for pattern in SYNTHETIC_FILLER_PATTERNS if pattern in lowered)
    if len(lowered.split()) <= 8:
        score += 1
    return score


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
        f"{phrases['category_context_phrase']}, with the next opportunity centered on translating "
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
    cue_context = aggregate_pdp_cues(
        primary_records,
        competitor_records=competitor_records,
    )
    _apply_cue_bullets_to_slide2_sections(sections, cue_context)
    sections, fit_debug = _dedupe_and_fit_slide2_sections(sections, phrases)
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
            "bullet_fit": fit_debug,
            "strategic_cues": cue_context.get("debug", {}),
        },
    }
