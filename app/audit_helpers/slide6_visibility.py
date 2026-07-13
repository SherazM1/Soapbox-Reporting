from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.audit_helpers.strategic_identity import resolve_strategic_identity


VISIBILITY_LABELS = ("Strong", "Moderate", "Partial", "Limited")
_VISIBILITY_SET = set(VISIBILITY_LABELS)
_STRUCTURED_FIELDS = (
    "category",
    "resolved_category",
    "family",
    "resolved_family",
    "product_type",
    "resolved_product_type",
    "subcategory",
)
_TEXT_FIELDS = (
    "product_title",
    "title",
    "description",
    "key_features",
    "features",
    "content_signals",
)
QUERY_ROW_TYPES = ("core", "attribute", "variant", "adjacent")
QUERY_BLOCKLIST = {
    "recommended",
    "routine",
    "shopper education",
    "category discoverability",
    "benchmark visibility",
    "merchandising",
    "communication",
    "alignment",
    "need state",
    "content coverage",
    "product detail language",
    "audience ingredient needs",
    "food and beverage",
    "dips and spread",
    "animal health",
    "organic option",
    "product option",
    "flavor jam",
    "wash cleanser",
    "wash sensitive",
    "hydrating face",
    "cleansing cleanser",
}
ATTRIBUTE_STOPWORDS = {
    "brand",
    "size",
    "count",
    "pack",
    "retail packaging",
    "net content",
    "multipack quantity",
    "product type",
    "product",
    "type",
}
PRODUCT_ANCHOR_WORDS = {
    "balm",
    "butter",
    "cleaner",
    "cleanser",
    "cream",
    "diaper",
    "gel",
    "jam",
    "jelly",
    "lotion",
    "moisturizer",
    "preserve",
    "serum",
    "spread",
    "wash",
    "wipes",
}
MODIFIER_ANCHOR_WORDS = {
    "fragrance",
    "free",
    "gentle",
    "hydrating",
    "hypoallergenic",
    "low",
    "natural",
    "normal",
    "oily",
    "organic",
    "protein",
    "sensitive",
    "sugar",
}
FORM_ANCHOR_WORDS = {
    "bar",
    "cleansing",
    "cream",
    "foam",
    "foaming",
    "gel",
    "spray",
    "stick",
    "wash",
}


def _segment(
    segment_id: str,
    display_name: str,
    positive_terms: tuple[str, ...],
    *,
    supporting_terms: tuple[str, ...] = (),
    negative_terms: tuple[str, ...] = (),
    fields: tuple[str, ...] = (*_STRUCTURED_FIELDS, *_TEXT_FIELDS),
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "display_name": display_name,
        "positive_terms": positive_terms,
        "supporting_terms": supporting_terms,
        "negative_terms": negative_terms,
        "fields": fields,
    }


SEGMENT_PACKS: dict[str, dict[str, Any]] = {
    "baby_care": {
        "category_phrase": "baby care",
        "match_terms": ("baby", "infant", "toddler", "diaper", "baby wipe"),
        "segments": (
            _segment("baby_care", "Baby Care", ("baby care", "baby", "infant")),
            _segment("diapers", "Diapers", ("diaper", "nappy")),
            _segment("baby_wipes", "Baby Wipes", ("baby wipe", "wipes")),
            _segment("sensitive_baby_care", "Sensitive Baby Care", ("sensitive", "hypoallergenic"), supporting_terms=("baby",)),
            _segment("clean_lifestyle", "Clean Lifestyle", ("clean", "free from", "non toxic"), supporting_terms=("baby", "gentle")),
            _segment("plant_based_baby_care", "Plant-Based Baby Care", ("plant based", "plant derived", "botanical"), supporting_terms=("baby",)),
        ),
    },
    "jam_fruit_spreads": {
        "category_phrase": "jams and fruit spreads",
        "match_terms": ("jam", "jelly", "preserve", "fruit spread", "marmalade"),
        "segments": (
            _segment("jam_jelly", "Jam & Jelly", ("jam", "jelly")),
            _segment("fruit_spread", "Fruit Spread", ("fruit spread", "fruit preserves")),
            _segment("strawberry_preserves", "Strawberry Preserves", ("strawberry preserve", "strawberry jam", "strawberry")),
            _segment("breakfast_spreads", "Breakfast Spreads", ("breakfast", "toast"), supporting_terms=("spread", "jam", "jelly", "preserve")),
            _segment("low_sugar_jam", "Low Sugar Jam", ("low sugar", "reduced sugar", "no sugar added"), supporting_terms=("jam", "spread")),
            _segment("organic_fruit_spread", "Organic Fruit Spread", ("organic",), supporting_terms=("fruit", "spread", "jam", "preserve")),
        ),
    },
    "nut_butter_spreads": {
        "category_phrase": "nut butters and spreads",
        "match_terms": ("peanut butter", "nut butter", "almond butter", "hazelnut spread"),
        "segments": (
            _segment("peanut_butter", "Peanut Butter", ("peanut butter", "peanut")),
            _segment("nut_butter", "Nut Butter", ("nut butter", "almond butter", "cashew butter")),
            _segment("chocolate_hazelnut_spread", "Chocolate Hazelnut Spread", ("chocolate hazelnut", "hazelnut spread")),
            _segment("protein_spread", "Protein Spread", ("protein",), supporting_terms=("spread", "butter")),
            _segment("snack_spread", "Snack Spread", ("snack", "snacking"), supporting_terms=("spread", "butter")),
            _segment("natural_peanut_butter", "Natural Peanut Butter", ("natural peanut butter", "natural"), supporting_terms=("peanut butter",)),
        ),
    },
    "skin_care": {
        "category_phrase": "skin care",
        "match_terms": ("skin care", "skincare", "moisturizer", "lotion", "serum", "dermatolog"),
        "segments": (
            _segment("skin_care", "Skin Care", ("skin care", "skincare", "skin")),
            _segment("face_moisturizer", "Face Moisturizer", ("face moisturizer", "facial moisturizer", "face cream")),
            _segment("sensitive_skin", "Sensitive Skin", ("sensitive skin", "sensitive", "hypoallergenic")),
            _segment("clean_skin_care", "Clean Skin Care", ("clean skin", "clean beauty", "free from"), supporting_terms=("skin",)),
            _segment("dermatologist_recommended", "Dermatologist Recommended", ("dermatologist recommended", "dermatologist tested", "dermatolog")),
            _segment("daily_skin_routine", "Daily Skin Routine", ("daily", "routine", "regimen"), supporting_terms=("skin", "moisturizer", "lotion")),
        ),
    },
    "household_cleaning": {
        "category_phrase": "household cleaning",
        "match_terms": ("household cleaning", "cleaner", "cleaning", "disinfect", "surface"),
        "segments": (
            _segment("household_cleaning", "Household Cleaning", ("household cleaning", "cleaner", "cleaning")),
            _segment("multi_surface_cleaner", "Multi-Surface Cleaner", ("multi surface", "all purpose cleaner")),
            _segment("disinfecting_cleaner", "Disinfecting Cleaner", ("disinfect", "kills germs", "antibacterial")),
            _segment("kitchen_cleaner", "Kitchen Cleaner", ("kitchen", "grease", "degreaser"), supporting_terms=("cleaner",)),
            _segment("bathroom_cleaner", "Bathroom Cleaner", ("bathroom", "soap scum", "shower"), supporting_terms=("cleaner",)),
            _segment("plant_based_cleaner", "Plant-Based Cleaner", ("plant based", "plant derived", "non toxic"), supporting_terms=("cleaner", "cleaning")),
        ),
    },
}


def _safe_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{_safe_text(key)} {_safe_text(item)}" for key, item in value.items())
    return str(value or "").strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _safe_text(value).lower()).strip()


def _record_field_text(record: dict[str, Any], field: str) -> str:
    return _normalize(record.get(field))


def _ocr_text(record: dict[str, Any]) -> str:
    analysis = record.get("image_analysis", {}) or {}
    parts: list[str] = []
    for image in analysis.get("images", []) or []:
        if not isinstance(image, dict) or image.get("status") not in (None, "", "analyzed"):
            continue
        parts.append(_safe_text(image.get("ocr_text")))
        parts.append(_safe_text(image.get("ocr_tokens")))
    return _normalize(parts)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


def _first_value(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        current: Any = record
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current not in (None, "", [], {}):
            return current
    return default


def _record_value(record: dict[str, Any], key: str, default: Any = "") -> Any:
    return _first_value(record, key, f"data.{key}", default=default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return _normalize(value) in {"1", "true", "yes", "y", "present", "detected"}


def _record_int(record: dict[str, Any], key: str, default: int = 0) -> int:
    return _to_int(_record_value(record, key, default), default)


def _record_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _to_float(_record_value(record, key, default), default)


def _record_bool(record: dict[str, Any], key: str) -> bool:
    return _to_bool(_record_value(record, key, False))


def _display_query(value: Any) -> str:
    text = re.sub(r"\s+", " ", _safe_text(value).replace("&", " and ")).strip(" -")
    return text.lower()


def _singular_query(value: str) -> str:
    text = _display_query(value)
    replacements = (
        ("facial cleansers", "face cleanser"),
        ("facial cleanser", "face cleanser"),
        ("jams, jellies and preserves", "jam"),
        ("jams jellies and preserves", "jam"),
        ("jams, jellies and preserves", "jam"),
        ("jams jellies preserves", "jam"),
        ("nut butters and spreads", "peanut butter"),
        ("nut butters spreads", "peanut butter"),
    )
    for old, new in replacements:
        if _normalize(text) == _normalize(old):
            return new
    words = text.split()
    if len(words) <= 3 and words and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _query_like(value: str, negative_keywords: tuple[str, ...] = ()) -> tuple[bool, str]:
    query = _display_query(value)
    normalized = _normalize(query)
    if not normalized:
        return False, "blank"
    if any(blocked in normalized for blocked in QUERY_BLOCKLIST):
        return False, "strategist_or_internal_label"
    if any(_normalize(term) and _normalize(term) in normalized for term in negative_keywords):
        return False, "negative_keyword"
    words = normalized.split()
    if len(words) > 5:
        return False, "too_long_for_search_query"
    if normalized.endswith(" type") or " type " in normalized:
        return False, "attribute_taxonomy_label"
    if len(words) == 1 and len(words[0]) < 4 and words[0] not in {"jam", "bbq", "tea"}:
        return False, "too_short"
    if normalized in {"beauty shelf", "category core", "product type"}:
        return False, "too_broad"
    return True, "query_like"


def _record_blob(records: list[dict[str, Any]], fields: tuple[str, ...] = (*_STRUCTURED_FIELDS, *_TEXT_FIELDS)) -> str:
    return " ".join(
        _record_field_text(record, field)
        for record in records
        for field in fields
        if isinstance(record, dict)
    )


def _candidate(
    query: str,
    row_type: str,
    source: str,
    *,
    positive_terms: tuple[str, ...] = (),
    supporting_terms: tuple[str, ...] = (),
    negative_terms: tuple[str, ...] = (),
    base_score: float = 0.0,
    guide_support: tuple[str, ...] = (),
    layer: str = "",
) -> dict[str, Any]:
    display = _display_query(query)
    terms = tuple(dict.fromkeys(_normalize(term) for term in (positive_terms or (display,)) if _normalize(term)))
    return {
        "segment_id": f"query_{_normalize(display).replace(' ', '_')}",
        "display_name": display,
        "positive_terms": terms or (_normalize(display),),
        "supporting_terms": tuple(dict.fromkeys(_normalize(term) for term in supporting_terms if _normalize(term))),
        "negative_terms": tuple(dict.fromkeys(_normalize(term) for term in negative_terms if _normalize(term))),
        "fields": (*_STRUCTURED_FIELDS, *_TEXT_FIELDS),
        "row_type": row_type if row_type in QUERY_ROW_TYPES else "adjacent",
        "candidate_source": source,
        "base_score": base_score,
        "guide_support": list(guide_support),
        "query_layer": layer or _query_layer(display, row_type, source),
    }


def _query_layer(query: str, row_type: str, source: str) -> str:
    normalized = _normalize(query)
    words = set(normalized.split())
    modifier_terms = {
        "sensitive",
        "hydrating",
        "fragrance",
        "free",
        "natural",
        "organic",
        "low",
        "sugar",
        "protein",
        "creamy",
        "gentle",
    }
    form_terms = {"wash", "foam", "foaming", "gel", "cream", "stick", "bar", "spray"}
    if words & modifier_terms:
        return "modifier_attribute"
    if source in {"actual_search_evidence", "resolved_identity", "resolved_category_anchor", "resolved_family_anchor"}:
        return "core_search_family"
    if words & form_terms:
        return "form_variant"
    if row_type == "core":
        return "product_type_anchor"
    if row_type == "adjacent":
        return "adjacent_discovery"
    return "product_type_anchor"


def _extract_search_evidence(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        values.extend(
            [
                record.get("searchTerm"),
                record.get("search_term"),
                record.get("query"),
                record.get("keyword"),
            ]
        )
        values.extend(_as_list(record.get("search_terms") or record.get("queries") or record.get("keywords")))
    return [_display_query(value) for value in values if _safe_text(value)]


def _record_search_terms(record: dict[str, Any]) -> list[str]:
    values: list[Any] = [
        _record_value(record, "searchTerm"),
        _record_value(record, "search_term"),
        _record_value(record, "query"),
        _record_value(record, "keyword"),
    ]
    values.extend(_as_list(_record_value(record, "search_terms", [])))
    values.extend(_as_list(_record_value(record, "queries", [])))
    values.extend(_as_list(_record_value(record, "keywords", [])))
    return _unique_queries(values, 12)


def _record_int_list(record: dict[str, Any], key: str) -> list[int]:
    values = _as_list(_record_value(record, key, []))
    output: list[int] = []
    for value in values:
        if isinstance(value, dict):
            value = _first_value(value, "rank", "position", "value", default="")
        parsed = _to_int(value, -1)
        if parsed > 0:
            output.append(parsed)
    return output


def _record_text_list(record: dict[str, Any], key: str) -> list[str]:
    return _unique_queries(_as_list(_record_value(record, key, [])), 20)


def _query_matches_record_search(record: dict[str, Any], definition: dict[str, Any]) -> bool:
    query_terms = {
        _normalize(definition.get("display_name")),
        *(_normalize(term) for term in definition.get("positive_terms", ()) if _normalize(term)),
    }
    search_terms = {_normalize(term) for term in _record_search_terms(record) if _normalize(term)}
    if not query_terms or not search_terms:
        return False
    return any(
        query and term and (query == term or query in term or term in query)
        for query in query_terms
        for term in search_terms
    )


def _search_shelf_summary(records: list[dict[str, Any]], definition: dict[str, Any]) -> dict[str, Any]:
    direct_rows = 0
    top3 = top5 = top10 = 0
    brand_matches = 0
    best_rank = 10**9
    share_values: list[float] = []
    dominant_names: list[str] = []
    badge_count = 0
    sponsored_count = 0
    review_values: list[float] = []
    rating_values: list[float] = []
    form_terms: list[str] = []
    solution_terms: list[str] = []
    use_case_terms: list[str] = []
    diversity_values: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        direct = _query_matches_record_search(record, definition)
        if direct:
            direct_rows += 1
        ranks = _record_int_list(record, "visible_brand_ranks")
        first_rank = _record_int(record, "first_brand_rank", 0)
        if first_rank > 0:
            ranks.append(first_rank)
        if _record_bool(record, "brand_in_top_3") or any(rank <= 3 for rank in ranks):
            top3 += 1
        if _record_bool(record, "brand_in_top_5") or any(rank <= 5 for rank in ranks):
            top5 += 1
        if _record_bool(record, "brand_in_top_10") or any(rank <= 10 for rank in ranks):
            top10 += 1
        if ranks:
            best_rank = min(best_rank, min(ranks))
        brand_matches += _record_int(record, "brand_match_count_visible")
        share = _record_float(record, "brand_share_top_10", 0.0)
        if share > 1:
            share = share / 100
        if share > 0:
            share_values.append(share)
        dominant_names.extend(_record_text_list(record, "dominant_brand_names"))
        badge_count += _record_int(record, "top_10_badge_count")
        badge_count += _record_int(record, "top_10_best_seller_count")
        badge_count += _record_int(record, "top_10_overall_pick_count")
        sponsored_count += _record_int(record, "top_10_sponsored_count")
        review = _record_float(record, "top_5_avg_review_count", 0.0)
        rating = _record_float(record, "top_5_avg_rating", 0.0)
        if review > 0:
            review_values.append(review)
        if rating > 0:
            rating_values.append(rating)
        form_terms.extend(_record_text_list(record, "top_10_form_factors"))
        solution_terms.extend(_record_text_list(record, "top_10_solution_types"))
        use_case_terms.extend(_record_text_list(record, "visible_use_case_terms"))
        diversity = _record_float(record, "result_form_diversity", 0.0)
        if diversity > 0:
            diversity_values.append(diversity)
    max_share = max(share_values, default=0.0)
    avg_review = sum(review_values) / len(review_values) if review_values else 0.0
    avg_rating = sum(rating_values) / len(rating_values) if rating_values else 0.0
    diversity = max(diversity_values, default=0.0)
    shelf_score = (
        direct_rows * 2.5
        + top3 * 4.0
        + max(0, top5 - top3) * 2.75
        + max(0, top10 - top5) * 1.5
        + min(brand_matches, 6) * 0.6
        + (2.0 if max_share >= 0.30 else 1.0 if max_share >= 0.15 else 0.0)
        + min(badge_count, 6) * 0.3
        + min(sponsored_count, 6) * 0.2
        + (1.0 if avg_review >= 500 else 0.5 if avg_review >= 100 else 0.0)
        + (0.5 if avg_rating >= 4.5 else 0.0)
        + min(len(set(form_terms + solution_terms + use_case_terms)), 6) * 0.15
    )
    return {
        "direct_search_rows": direct_rows,
        "top3_presence": top3,
        "top5_presence": top5,
        "top10_presence": top10,
        "best_rank": best_rank if best_rank != 10**9 else None,
        "visible_brand_match_count": brand_matches,
        "brand_share_top_10": round(max_share, 3),
        "dominant_brand_names": list(dict.fromkeys(dominant_names))[:6],
        "dominant_brand_count": max((_record_int(record, "dominant_brand_count") for record in records if isinstance(record, dict)), default=0),
        "badge_pressure_count": badge_count,
        "sponsored_pressure_count": sponsored_count,
        "top_5_avg_review_count": round(avg_review, 1),
        "top_5_avg_rating": round(avg_rating, 2),
        "top_10_form_factors": list(dict.fromkeys(form_terms))[:8],
        "top_10_solution_types": list(dict.fromkeys(solution_terms))[:8],
        "visible_use_case_terms": list(dict.fromkeys(use_case_terms))[:8],
        "result_form_diversity": round(diversity, 2),
        "shelf_signal_score": round(shelf_score, 2),
    }


def _shelf_visibility_tier(summary: dict[str, Any]) -> str:
    best_rank = summary.get("best_rank")
    brand_matches = int(summary.get("visible_brand_match_count", 0) or 0)
    share = float(summary.get("brand_share_top_10", 0) or 0)
    if summary.get("top3_presence") or (best_rank and best_rank <= 3):
        return "Strong"
    if summary.get("top5_presence") or (best_rank and best_rank <= 5) or brand_matches >= 3 or share >= 0.25:
        return "Moderate"
    if summary.get("top10_presence") or (best_rank and best_rank <= 10) or brand_matches >= 1 or summary.get("direct_search_rows"):
        return "Partial"
    return "Limited"


def _best_title_phrases(records: list[dict[str, Any]], negative_keywords: tuple[str, ...]) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        title = _record_field_text(record, "product_title") or _record_field_text(record, "title")
        words = [word for word in title.split() if len(word) >= 3 and word not in {"with", "and", "for", "the", "pack", "count"}]
        for size in (2, 3):
            for index in range(0, max(0, len(words) - size + 1)):
                phrase = " ".join(words[index : index + size])
                ok, _ = _query_like(phrase, negative_keywords)
                if ok:
                    counts[phrase] += 1
    return [phrase for phrase, _ in counts.most_common(8)]


def _guide_product_data(identity: dict[str, Any]) -> dict[str, Any]:
    data = identity.get("style_product_type") or {}
    return data if isinstance(data, dict) else {}


def _unique_queries(values: list[Any], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for value in values:
        query = _display_query(value)
        key = _normalize(query)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if limit is not None and len(queries) >= limit:
            break
    return queries


def _brand_terms(records: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    values: list[Any] = [
        metadata.get("client_company_name"),
        metadata.get("client_name"),
        metadata.get("brand"),
        metadata.get("client_brand"),
    ]
    for record in records:
        if not isinstance(record, dict):
            continue
        values.extend(
            [
                record.get("brand"),
                record.get("brand_name"),
                record.get("manufacturer"),
                record.get("seller_name"),
            ]
        )
    return _unique_queries(values, 8)


def _support_terms(*groups: list[str]) -> dict[str, int]:
    support: Counter[str] = Counter()
    for weight, terms in enumerate(groups, start=1):
        for term in terms:
            normalized = _normalize(term)
            if normalized:
                support[normalized] += weight
    return dict(support)


def _build_search_identity(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    pack: dict[str, Any],
    audit_metadata: dict[str, Any] | None = None,
    competitor_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    product_data = _guide_product_data(identity)
    negative_keywords = _unique_queries(_as_list(product_data.get("negative_keywords")), 20)
    product = _singular_query(identity.get("product_type_display") or _common_value(records, "product_type", "subcategory"))
    family = _singular_query(identity.get("family_display") or "")
    category = _singular_query(identity.get("category_display") or _common_value(records, "category") or pack.get("category_phrase"))
    aliases = _unique_queries(_as_list(product_data.get("aliases")), 12)
    title_keywords = _unique_queries(_as_list(product_data.get("title_keywords")), 16)
    context_keywords = _unique_queries(_as_list(product_data.get("context_keywords")), 16)
    attributes = _unique_queries(
        [
            value
            for value in _as_list(product_data.get("attributes"))
            if _normalize(value) and _normalize(value) not in ATTRIBUTE_STOPWORDS
        ],
        12,
    )
    search_terms = _unique_queries(_extract_search_evidence(records), 16)
    shelf_form_terms = _unique_queries(
        [
            term
            for record in records
            if isinstance(record, dict)
            for term in [
                *_record_text_list(record, "top_10_form_factors"),
                *_record_text_list(record, "top_10_solution_types"),
                *_record_text_list(record, "visible_use_case_terms"),
            ]
        ],
        16,
    )
    title_phrases = _unique_queries(_best_title_phrases(records, tuple(negative_keywords)), 12)
    modifier_markers = {
        "sensitive",
        "hydrating",
        "fragrance",
        "free",
        "natural",
        "organic",
        "low",
        "sugar",
        "protein",
        "creamy",
        "gentle",
    }
    form_markers = {"wash", "foam", "foaming", "gel", "cream", "stick", "bar", "spray"}
    modifier_terms = _unique_queries(
        [
            term
            for term in [*context_keywords, *attributes, *title_phrases, *search_terms]
            if set(_normalize(term).split()) & modifier_markers
        ],
        16,
    )
    form_terms = _unique_queries(
        [
            term
            for term in [*title_keywords, *context_keywords, *attributes, *title_phrases, *search_terms, *shelf_form_terms]
            if set(_normalize(term).split()) & form_markers
        ],
        12,
    )
    category_anchors = _unique_queries([category, pack.get("category_phrase")], 6)
    family_anchors = _unique_queries([family], 6)
    product_type_anchors = _unique_queries([product, *aliases[:6], *title_keywords[:8]], 12)
    adjacent_terms = _unique_queries(
        [
            segment.get("display_name")
            for segment in pack.get("segments", ())
            if isinstance(segment, dict)
        ],
        10,
    )
    support_by_family = {
        "category_anchors": _support_terms(category_anchors, search_terms),
        "family_anchors": _support_terms(family_anchors, search_terms),
        "product_type_anchors": _support_terms(product_type_anchors, title_phrases, search_terms),
        "modifier_terms": _support_terms(modifier_terms, title_phrases, context_keywords),
        "attribute_form_terms": _support_terms(form_terms, attributes, title_phrases),
        "adjacent_discovery_terms": _support_terms(adjacent_terms),
    }
    return {
        "category_key": identity.get("category_key"),
        "category_anchor_terms": category_anchors,
        "family_anchor_terms": family_anchors,
        "product_type_anchor_terms": product_type_anchors,
        "modifier_terms": modifier_terms,
        "attribute_form_terms": form_terms,
        "adjacent_discovery_terms": adjacent_terms,
        "banned_negative_terms": negative_keywords,
        "client_brand_context": _brand_terms(records, audit_metadata),
        "competitor_brand_context": _brand_terms(competitor_records or []),
        "source_terms": {
            "aliases": aliases,
            "title_keywords": title_keywords,
            "context_keywords": context_keywords,
            "attributes": attributes,
            "search_evidence": search_terms,
            "shelf_form_terms": shelf_form_terms,
            "title_phrases": title_phrases,
        },
        "support_by_term_family": support_by_family,
        "confidence": {
            "style_product_type_score": identity.get("style_product_type_score"),
            "has_search_evidence": bool(search_terms),
            "has_guide_product_type": bool(product_data),
            "has_title_language": bool(title_phrases),
        },
    }


def _candidate_bucket(query: str, row_type: str, source: str, layer: str, search_identity: dict[str, Any]) -> str:
    normalized = _normalize(query)
    if source == "resolved_category_anchor":
        return "category_anchor"
    if source == "resolved_family_anchor":
        return "family_anchor"
    if layer == "product_type_anchor" or source == "resolved_identity":
        return "product_type_anchor"
    if layer == "modifier_attribute":
        return "modifier"
    if layer == "form_variant" or row_type == "variant":
        return "attribute_form"
    if layer == "adjacent_discovery" or row_type == "adjacent":
        return "adjacent_discovery"
    if normalized in {_normalize(term) for term in search_identity.get("product_type_anchor_terms", [])}:
        return "product_type_anchor"
    return "modifier" if row_type == "attribute" else "adjacent_discovery"


def _support_family_for_bucket(bucket: str) -> str:
    return {
        "category_anchor": "category_anchors",
        "family_anchor": "family_anchors",
        "product_type_anchor": "product_type_anchors",
        "modifier": "modifier_terms",
        "attribute_form": "attribute_form_terms",
        "adjacent_discovery": "adjacent_discovery_terms",
    }.get(bucket, bucket)


def _build_query_candidates(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    pack: dict[str, Any],
    audit_metadata: dict[str, Any] | None = None,
    competitor_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search_identity = _build_search_identity(records, identity, pack, audit_metadata, competitor_records)
    source_terms = search_identity["source_terms"]
    negative_keywords = tuple(search_identity["banned_negative_terms"])
    product = (search_identity["product_type_anchor_terms"] or [""])[0]
    family = (search_identity["family_anchor_terms"] or [""])[0]
    category = (search_identity["category_anchor_terms"] or [""])[0]
    aliases = source_terms["aliases"]
    title_keywords = source_terms["title_keywords"]
    context_keywords = source_terms["context_keywords"]
    attributes = source_terms["attributes"]
    search_terms = source_terms["search_evidence"]
    shelf_form_terms = source_terms.get("shelf_form_terms", [])
    title_phrases = source_terms["title_phrases"]
    pool: list[dict[str, Any]] = []

    def add(
        query: str,
        row_type: str,
        source: str,
        score: float,
        guide_support: tuple[str, ...] = (),
        positive_terms: tuple[str, ...] = (),
        layer: str = "",
    ) -> None:
        ok, reason = _query_like(query, negative_keywords)
        if not ok:
            rejected.append({"query": _display_query(query), "source": source, "reason": reason})
            return
        support = tuple(term for term in (product, family, category) if term and _normalize(term) != _normalize(query))
        resolved_layer = layer or _query_layer(query, row_type, source)
        bucket = _candidate_bucket(query, row_type, source, resolved_layer, search_identity)
        term_support = search_identity["support_by_term_family"].get(_support_family_for_bucket(bucket), {})
        support_key = _normalize(query)
        pool.append(
            _candidate(
                query,
                row_type,
                source,
                positive_terms=positive_terms or (query,),
                supporting_terms=support,
                negative_terms=negative_keywords,
                base_score=score,
                guide_support=guide_support,
                layer=resolved_layer,
            )
        )
        pool[-1]["candidate_bucket"] = bucket
        pool[-1]["identity_support"] = {
            "bucket": bucket,
            "term_family_support": term_support.get(support_key, 0),
            "brand_context_available": bool(search_identity["client_brand_context"]),
        }

    rejected: list[dict[str, str]] = []
    for query in (product,):
        add(
            query,
            "core",
            "resolved_identity",
            10.0,
            positive_terms=(query, _safe_text(identity.get("product_type_display"))),
            layer="product_type_anchor",
        )
    for query, source, score in (
        (category, "resolved_category_anchor", 8.5),
        (family, "resolved_family_anchor", 8.0),
    ):
        add(
            query,
            "core",
            source,
            score,
            positive_terms=(query,),
            layer="core_search_family",
        )
    for query in [*aliases[:5], *title_keywords[:6]]:
        layer = _query_layer(query, "core", "style_guide_title_or_alias")
        add(
            query,
            "variant" if layer == "form_variant" else "core",
            "style_guide_title_or_alias",
            9.0,
            ("aliases/title_keywords",),
            layer=layer,
        )
    for query in search_terms:
        layer = _query_layer(query, "core", "actual_search_evidence")
        add(
            query,
            "attribute" if layer == "modifier_attribute" else "core",
            "actual_search_evidence",
            11.0,
            layer=layer,
        )
    for query in shelf_form_terms:
        layer = _query_layer(query, "variant", "search_shelf_structure")
        add(
            query,
            "variant" if layer == "form_variant" else "attribute",
            "search_shelf_structure",
            8.0,
            ("search_shelf_structure",),
            layer=layer,
        )
    for query in title_phrases:
        add(query, "attribute", "actual_pdp_title_language", 7.5)

    title_blob = _record_blob(records, ("product_title", "title"))
    product_anchor = product.split()[-1] if product else ""
    if product_anchor:
        modifier_phrases = (
            "sensitive skin",
            "normal to oily skin",
            "oily skin",
            "fragrance free",
            "hydrating",
            "foaming",
            "gentle",
            "natural",
            "creamy",
            "low sugar",
            "protein",
        )
        for modifier in modifier_phrases:
            if _normalize(modifier) not in title_blob:
                continue
            query = f"{modifier} {product_anchor}"
            layer = _query_layer(query, "attribute", "actual_pdp_title_language")
            add(
                query,
                "variant" if layer == "form_variant" else "attribute",
                "actual_pdp_title_language",
                8.2,
                ("title_modifier_product_anchor",),
                positive_terms=(modifier, query),
                layer=layer,
            )

    product_head = product.split()[-1] if product else ""
    for term in [*context_keywords[:10], *attributes[:8]]:
        term_norm = _normalize(term)
        if not term_norm:
            continue
        if product_head and product_head not in term_norm and len(term_norm.split()) <= 2:
            query = f"{term} {product_head}"
        else:
            query = term
        layer = _query_layer(query, "attribute", "style_guide_context_or_attribute")
        row_type = "variant" if layer == "form_variant" else "attribute"
        add(query, row_type, "style_guide_context_or_attribute", 7.0, ("context_keywords/attributes",), layer=layer)

    for segment in pack.get("segments", ())[:8]:
        if not isinstance(segment, dict):
            continue
        ok, reason = _query_like(segment.get("display_name", ""), negative_keywords)
        if not ok:
            rejected.append({"query": _display_query(segment.get("display_name", "")), "source": "segment_pack_fallback_seed", "reason": reason})
            continue
        segment_name = _display_query(segment.get("display_name", ""))
        segment_terms = tuple(segment.get("positive_terms") or ())
        segment_supporting = tuple(segment.get("supporting_terms") or ())
        source_type = "attribute" if any(
            term in _normalize(segment_name)
            for term in ("organic", "low sugar", "sensitive", "hydrating", "natural", "chocolate")
        ) else "adjacent"
        layer = _query_layer(segment_name, source_type, "segment_pack_fallback_seed")
        pool.append(
            _candidate(
                segment_name,
                "variant" if layer == "form_variant" else source_type,
                "segment_pack_fallback_seed",
                positive_terms=(*segment_terms, segment_name),
                supporting_terms=segment_supporting,
                negative_terms=negative_keywords,
                base_score=5.5,
                layer=layer,
            )
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in pool:
        key = _query_cluster_key(item["display_name"])
        existing = deduped.get(key)
        if not existing or float(item.get("base_score", 0)) > float(existing.get("base_score", 0)):
            if existing:
                item["positive_terms"] = tuple(dict.fromkeys((*item.get("positive_terms", ()), *existing.get("positive_terms", ()))))
                item["supporting_terms"] = tuple(dict.fromkeys((*item.get("supporting_terms", ()), *existing.get("supporting_terms", ()))))
                item["guide_support"] = list(dict.fromkeys([*item.get("guide_support", []), *existing.get("guide_support", [])]))
                item["merged_sources"] = list(dict.fromkeys([*existing.get("merged_sources", []), existing.get("candidate_source")]))
            deduped[key] = item
        elif existing:
            existing.setdefault("merged_sources", []).append(item.get("candidate_source"))
            existing["positive_terms"] = tuple(dict.fromkeys((*existing.get("positive_terms", ()), *item.get("positive_terms", ()))))
            existing["supporting_terms"] = tuple(dict.fromkeys((*existing.get("supporting_terms", ()), *item.get("supporting_terms", ()))))
            existing["guide_support"] = list(dict.fromkeys([*existing.get("guide_support", []), *item.get("guide_support", [])]))
    return list(deduped.values()), {
        "identity": {
            "category_key": identity.get("category_key"),
            "product_type_display": identity.get("product_type_display"),
            "family_display": identity.get("family_display"),
            "style_guide_path": identity.get("style_guide_path"),
            "style_product_type_score": identity.get("style_product_type_score"),
        },
        "shared_search_identity": search_identity,
        "guide_terms_used": {
            "aliases": aliases[:8],
            "title_keywords": title_keywords[:8],
            "context_keywords": context_keywords[:10],
            "attributes": attributes[:8],
            "shelf_form_terms": shelf_form_terms[:8],
            "negative_keywords": list(negative_keywords)[:10],
        },
        "rejected_candidates": rejected[:20],
        "raw_candidate_count": len(pool),
        "deduped_candidate_count": len(deduped),
    }


def _query_cluster_key(query: str) -> str:
    normalized = _normalize(query)
    replacements = {
        "facial cleanser": "face cleanser",
        "facial cleansers": "face cleanser",
        "face wash": "face cleanser",
        "jams jellies preserves": "jam",
        "jam and jelly": "jam",
        "jellies": "jam",
        "jelly": "jam",
        "marmalade": "jam",
        "preserve": "jam",
        "preserves": "jam",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    words = [word[:-1] if word.endswith("s") and not word.endswith("ss") else word for word in normalized.split()]
    return " ".join(words)


def _row_similarity_reason(candidate_key: str, selected_keys: set[str]) -> str:
    key_words = set(candidate_key.split())
    for existing in selected_keys:
        existing_words = set(existing.split())
        if not key_words or not existing_words:
            continue
        shared = key_words & existing_words
        if len(shared) >= min(len(key_words), len(existing_words)):
            if abs(len(key_words) - len(existing_words)) <= 1:
                return "near_duplicate"
        if len(shared) >= 2 and len(shared) >= min(len(key_words), len(existing_words)) * 0.75:
            return "redundant_query_theme"
    return ""


def _candidate_evidence_summary(records: list[dict[str, Any]], definition: dict[str, Any]) -> dict[str, Any]:
    matches = [
        _score_record(record, definition, include_concept_support=False)
        for record in records
        if isinstance(record, dict)
    ]
    return {
        "matched_terms": list(dict.fromkeys(term for match in matches for term in match.get("matched_terms", []))),
        "matched_fields": list(dict.fromkeys(field for match in matches for field in match.get("matched_fields", []))),
        "supporting_records": sum(1 for match in matches if match.get("weight", 0) > 0),
    }


def _rank_candidate(
    definition: dict[str, Any],
    records: list[dict[str, Any]],
    competitors: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    client_summary = _candidate_evidence_summary(records, definition)
    competitor_summary = _candidate_evidence_summary(competitors, definition)
    source_bonus = {
        "actual_search_evidence": 5.0,
        "search_shelf_structure": 4.0,
        "resolved_identity": 4.0,
        "resolved_category_anchor": 3.5,
        "resolved_family_anchor": 3.0,
        "style_guide_title_or_alias": 3.5,
        "actual_pdp_title_language": 4.0,
        "style_guide_context_or_attribute": 2.0,
        "segment_pack_fallback_seed": 0.5,
    }.get(str(definition.get("candidate_source")), 1.0)
    evidence_support = client_summary["supporting_records"] + competitor_summary["supporting_records"]
    client_shelf = _search_shelf_summary(records, definition)
    competitor_shelf = _search_shelf_summary(competitors, definition)
    combined_shelf_score = float(client_shelf.get("shelf_signal_score", 0) or 0) + float(competitor_shelf.get("shelf_signal_score", 0) or 0)
    direct_search_rows = int(client_shelf.get("direct_search_rows", 0) or 0) + int(competitor_shelf.get("direct_search_rows", 0) or 0)
    top10_presence = int(client_shelf.get("top10_presence", 0) or 0) + int(competitor_shelf.get("top10_presence", 0) or 0)
    benchmark_usefulness = 1.0 if client_summary["supporting_records"] != competitor_summary["supporting_records"] else 0.0
    if _shelf_visibility_tier(client_shelf) != _shelf_visibility_tier(competitor_shelf):
        benchmark_usefulness += 1.0
    row_type_bonus = {"core": 4.0, "attribute": 3.0, "variant": 2.0, "adjacent": 0.5}.get(str(definition.get("row_type")), 1.0)
    layer = str(definition.get("query_layer") or "")
    layer_bonus = {
        "core_search_family": 3.0,
        "product_type_anchor": 2.5,
        "modifier_attribute": 2.0,
        "form_variant": 0.75,
        "adjacent_discovery": -0.5,
    }.get(layer, 0.0)
    realism = _query_realism_score(str(definition.get("display_name")))
    product_type_fit = _product_type_fit_score(definition)
    guide_support_score = min(len(definition.get("guide_support") or []) * 1.5, 3.0)
    identity_support_score = min(float(((definition.get("identity_support") or {}).get("term_family_support", 0)) or 0), 4.0)
    distinctiveness = _query_distinctiveness_score(str(definition.get("display_name")), str(definition.get("candidate_bucket") or ""))
    quality_penalty, quality_reasons = _query_quality_penalty(definition, evidence_support)
    score = (
        float(definition.get("base_score", 0))
        + source_bonus
        + row_type_bonus
        + layer_bonus
        + evidence_support * 3
        + min(combined_shelf_score, 9.0)
        + direct_search_rows * 2.0
        + top10_presence * 1.0
        + benchmark_usefulness
        + realism
        + product_type_fit
        + guide_support_score
        + identity_support_score
        + distinctiveness
        - quality_penalty
    )
    factors = {
        "base_score": definition.get("base_score", 0),
        "source_bonus": source_bonus,
        "row_type_bonus": row_type_bonus,
        "layer_bonus": layer_bonus,
        "evidence_support": evidence_support,
        "direct_search_rows": direct_search_rows,
        "top10_presence": top10_presence,
        "search_shelf_signal_score": round(combined_shelf_score, 2),
        "search_shelf_tier": {
            "client": _shelf_visibility_tier(client_shelf),
            "competitor": _shelf_visibility_tier(competitor_shelf),
        },
        "search_shelf_signals": {
            "client": client_shelf,
            "competitor": competitor_shelf,
        },
        "benchmark_usefulness": benchmark_usefulness,
        "shopper_query_realism": realism,
        "product_type_fit": product_type_fit,
        "guide_support": guide_support_score,
        "identity_support": identity_support_score,
        "distinctiveness": distinctiveness,
        "quality_penalty": quality_penalty,
        "quality_penalty_reasons": quality_reasons,
        "query_layer": layer,
        "candidate_bucket": definition.get("candidate_bucket"),
        "client_supporting_records": client_summary["supporting_records"],
        "competitor_supporting_records": competitor_summary["supporting_records"],
    }
    return score, factors


def _product_type_fit_score(definition: dict[str, Any]) -> float:
    bucket = str(definition.get("candidate_bucket") or "")
    source = str(definition.get("candidate_source") or "")
    layer = str(definition.get("query_layer") or "")
    if bucket == "product_type_anchor" or source == "resolved_identity":
        return 4.0
    if bucket == "modifier" and layer == "modifier_attribute":
        return 3.0
    if bucket == "attribute_form":
        return 2.0
    if bucket == "family_anchor":
        return 1.5
    if bucket == "category_anchor":
        return 0.5
    if bucket == "adjacent_discovery":
        return -0.5
    return 0.0


def _query_distinctiveness_score(query: str, bucket: str) -> float:
    normalized = _normalize(query)
    words = normalized.split()
    word_set = set(words)
    if bucket == "category_anchor" and len(words) <= 2:
        return -1.0
    if bucket == "adjacent_discovery":
        return -0.5
    score = 0.0
    if 2 <= len(words) <= 4:
        score += 1.5
    if any(term in normalized for term in ("sensitive", "fragrance free", "low sugar", "natural", "organic", "hydrating", "foaming")):
        score += 1.0
    if normalized in {"product", "category", "option", "routine"}:
        score -= 4.0
    return score


def _query_quality_penalty(definition: dict[str, Any], evidence_support: int) -> tuple[float, list[str]]:
    query = str(definition.get("display_name") or "")
    normalized = _normalize(query)
    words = normalized.split()
    word_set = set(words)
    source = str(definition.get("candidate_source") or "")
    row_type = str(definition.get("row_type") or "")
    reasons: list[str] = []
    penalty = 0.0
    weak_pairs = {
        "cleansing cleanser",
        "cleanser face",
        "facial cleanser face",
        "gentle facial",
        "free foaming",
        "free foaming face",
        "hydrating fragrance",
        "hydrating fragrance free",
        "fragrance free foaming",
        "foaming face",
        "daily cleanser",
        "wash cleanser",
        "wash sensitive",
        "hydrating face",
        "organic option",
        "flavor jam",
        "option cleanser",
        "product cleanser",
        "normal oily",
        "normal to oily",
        "oily skin",
        "fragrance free face",
        "free face",
        "sensitive face",
        "skin care concern",
        "beauty skin care",
    }
    if normalized in weak_pairs:
        penalty += 8.0
        reasons.append("weak_synthetic_fragment")
    if len(words) == 1 and source not in {"resolved_identity", "actual_search_evidence"}:
        penalty += 4.0
        reasons.append("single_word_non_core_candidate")
    if row_type == "adjacent" and evidence_support <= 0:
        penalty += 5.0
        reasons.append("unsupported_adjacent_row")
    if source == "segment_pack_fallback_seed" and evidence_support <= 0:
        penalty += 2.0
        reasons.append("fallback_without_visible_evidence")
    if row_type == "variant" and evidence_support <= 0:
        penalty += 2.5
        reasons.append("unsupported_form_variant")
    if normalized in {"daily cleanser", "foam cleanser", "gel cleanser", "cream cleanser"} and evidence_support <= 0:
        penalty += 3.0
        reasons.append("weak_generic_form_variant")
    if source == "actual_pdp_title_language" and evidence_support <= 0:
        benefit_or_concern_terms = {
            "hydrating",
            "sensitive",
            "gentle",
            "natural",
            "organic",
            "low",
            "sugar",
            "protein",
            "fragrance",
            "free",
        }
        if "option" in words or "product" in words:
            penalty += 6.0
            reasons.append("synthetic_title_fragment")
        product_or_concern_anchors = {
            "cleanser",
            "wash",
            "butter",
            "jam",
            "spread",
            "moisturizer",
            "cleaner",
        }
        if benefit_or_concern_terms & word_set and not (product_or_concern_anchors & word_set):
            penalty += 4.0
            reasons.append("benefit_fragment_without_product_anchor")
        elif not (benefit_or_concern_terms & set(words)):
            penalty += 5.0
            reasons.append("unsupported_title_fragment")
    if source == "actual_pdp_title_language" and row_type == "attribute" and len(words) <= 2:
        product_or_concern_anchors = {
            "cleanser",
            "wash",
            "butter",
            "jam",
            "spread",
            "moisturizer",
            "cleaner",
        }
        benefit_or_concern_terms = {
            "hydrating",
            "sensitive",
            "gentle",
            "natural",
            "organic",
            "low",
            "sugar",
            "protein",
            "fragrance",
            "free",
        }
        if benefit_or_concern_terms & set(words) and not (product_or_concern_anchors & set(words)):
            penalty += 2.0
            reasons.append("prefer_product_anchored_modifier")
    bucket = str(definition.get("candidate_bucket") or "")
    if "concern" in words and not (PRODUCT_ANCHOR_WORDS & word_set):
        penalty += 8.0
        reasons.append("incomplete_concern_fragment")
    if bucket in {"category_anchor", "family_anchor"} and len(words) <= 2:
        penalty += 8.0
        reasons.append("broad_category_or_family_row")
    if bucket == "product_type_anchor" and not (PRODUCT_ANCHOR_WORDS & word_set) and word_set & {"beauty", "skin", "care"}:
        penalty += 8.0
        reasons.append("broad_product_type_anchor")
    if bucket == "modifier":
        if not (PRODUCT_ANCHOR_WORDS & word_set):
            penalty += 8.5
            reasons.append("incomplete_modifier_without_product_anchor")
        elif len(words) <= 2 and MODIFIER_ANCHOR_WORDS & word_set:
            penalty += 2.5
            reasons.append("modifier_phrase_too_thin")
    if source == "segment_pack_fallback_seed" and evidence_support <= 0 and row_type != "adjacent":
        penalty += 3.0
        reasons.append("unsupported_fallback_attribute")
    if words and len(set(words)) < len(words):
        penalty += 3.0
        reasons.append("repeated_query_token")
    if len(words) == 2 and words[0] in {"organic", "classic", "natural", "gentle", "clean"} and evidence_support <= 0:
        penalty += 3.0
        reasons.append("modifier_without_product_evidence")
    return penalty, reasons


def _query_realism_score(query: str) -> float:
    normalized = _normalize(query)
    words = normalized.split()
    score = 3.0
    if 2 <= len(words) <= 4:
        score += 2.0
    if len(words) == 1:
        score -= 1.5
    if any(blocked in normalized for blocked in QUERY_BLOCKLIST):
        score -= 8.0
    return score


def _select_final_query_rows(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    competitors: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        score, factors = _rank_candidate(candidate, records, competitors)
        ranked.append((score, candidate, factors))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    type_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    rejected_similar: list[dict[str, str]] = []

    def can_add(candidate: dict[str, Any]) -> tuple[bool, str]:
        key = _query_cluster_key(candidate["display_name"])
        if key in selected_keys:
            return False, "near_duplicate"
        similarity_reason = _row_similarity_reason(key, selected_keys)
        if similarity_reason:
            return False, similarity_reason
        row_type = str(candidate.get("row_type"))
        layer = str(candidate.get("query_layer") or "")
        bucket = str(candidate.get("candidate_bucket") or "")
        bucket_caps = {
            "category_anchor": 1,
            "family_anchor": 1,
            "product_type_anchor": 2,
            "modifier": 4,
            "attribute_form": 3,
            "adjacent_discovery": 1,
        }
        if bucket in bucket_caps and bucket_counts[bucket] >= bucket_caps[bucket]:
            return False, f"{bucket}_quota_met"
        if row_type == "core" and type_counts[row_type] >= 2:
            return False, "core_row_quota_met"
        if row_type == "attribute" and type_counts[row_type] >= 4:
            return False, "attribute_row_quota_met"
        if row_type == "variant" and type_counts[row_type] >= 3:
            return False, "variant_row_quota_met"
        if row_type == "adjacent" and type_counts[row_type] >= 1:
            return False, "adjacent_row_quota_met"
        if layer == "adjacent_discovery" and layer_counts[layer] >= 1:
            return False, "adjacent_layer_quota_met"
        if layer == "form_variant" and layer_counts[layer] >= 3:
            return False, "form_variant_quota_met"
        penalty = float(((candidate.get("ranking_factors") or {}).get("quality_penalty", 0)) or 0)
        if penalty >= 8:
            return False, "weak_query_quality"
        factors = candidate.get("ranking_factors") or {}
        evidence_support = int(factors.get("evidence_support", 0) or 0)
        direct_search_rows = int(factors.get("direct_search_rows", 0) or 0)
        shelf_score = float(factors.get("search_shelf_signal_score", 0) or 0)
        source = str(candidate.get("candidate_source") or "")
        bucket = str(candidate.get("candidate_bucket") or "")
        if not evidence_support and not direct_search_rows and shelf_score <= 0:
            if source in {"segment_pack_fallback_seed", "style_guide_context_or_attribute"}:
                return False, "no_real_row_evidence"
            if bucket in {"adjacent_discovery", "attribute_form"}:
                return False, "no_real_row_evidence"
        if bucket == "adjacent_discovery" and direct_search_rows <= 0 and evidence_support <= 1 and shelf_score < 3:
            return False, "weak_adjacent_shelf_support"
        if len(selected) >= 4:
            score = float(candidate.get("selection_score", 0) or 0)
            if score < 10:
                return False, "weak_trailing_score"
            if source == "segment_pack_fallback_seed" and evidence_support <= 0 and penalty >= 5:
                return False, "weak_trailing_fallback_row"
            if layer == "adjacent_discovery" and evidence_support <= 0:
                return False, "weak_trailing_layer_row"
            if layer == "form_variant" and evidence_support <= 0 and score < 20:
                return False, "weak_trailing_layer_row"
        return True, "selected"

    def ranked_in_bucket(bucket: str) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
        return [item for item in ranked if str(item[1].get("candidate_bucket") or "") == bucket]

    selection_passes: list[tuple[float, dict[str, Any], dict[str, Any]]] = [
        *ranked_in_bucket("product_type_anchor")[:1],
        *ranked_in_bucket("family_anchor")[:1],
    ]
    if len(selection_passes) < 2:
        selection_passes.extend(ranked_in_bucket("category_anchor")[: 2 - len(selection_passes)])
    seen_pass_ids = {id(item[1]) for item in selection_passes}
    selection_passes.extend(item for item in ranked if id(item[1]) not in seen_pass_ids)

    for score, candidate, factors in selection_passes:
        candidate = dict(candidate)
        candidate["ranking_factors"] = factors
        candidate["selection_score"] = round(score, 2)
        ok, reason = can_add(candidate)
        if not ok:
            rejected_similar.append({"query": candidate["display_name"], "reason": reason})
            continue
        selected.append(candidate)
        selected_keys.add(_query_cluster_key(candidate["display_name"]))
        type_counts[str(candidate.get("row_type"))] += 1
        layer_counts[str(candidate.get("query_layer") or "")] += 1
        bucket_counts[str(candidate.get("candidate_bucket") or "")] += 1
        if len(selected) == 6:
            break
    if len(selected) < 6:
        warnings.append("Query candidate pool was thin; static segment pack fallback filled remaining rows.")
        for score, candidate, factors in ranked:
            if len(selected) == 6:
                break
            key = _query_cluster_key(candidate["display_name"])
            if key in selected_keys:
                continue
            similarity_reason = _row_similarity_reason(key, selected_keys)
            if similarity_reason:
                rejected_similar.append({"query": candidate["display_name"], "reason": similarity_reason})
                continue
            candidate = dict(candidate)
            candidate["ranking_factors"] = factors
            candidate["selection_score"] = round(score, 2)
            ok, reason = can_add(candidate)
            if not ok:
                rejected_similar.append({"query": candidate["display_name"], "reason": reason})
                continue
            selected.append(candidate)
            selected_keys.add(key)
            type_counts[str(candidate.get("row_type"))] += 1
            layer_counts[str(candidate.get("query_layer") or "")] += 1
            bucket_counts[str(candidate.get("candidate_bucket") or "")] += 1
    return selected[:6], {
        "ranked_candidates": [
            {
                "query": candidate["display_name"],
                "source": candidate.get("candidate_source"),
                "row_type": candidate.get("row_type"),
                "query_layer": candidate.get("query_layer"),
                "candidate_bucket": candidate.get("candidate_bucket"),
                "score": round(score, 2),
                "ranking_factors": factors,
            }
            for score, candidate, factors in ranked[:20]
        ],
        "rejected_similar_rows": rejected_similar[:20],
        "row_type_counts": dict(type_counts),
        "query_layer_counts": dict(layer_counts),
        "candidate_bucket_counts": dict(bucket_counts),
        "selected_rows": [
            {
                "query": item.get("display_name"),
                "candidate_bucket": item.get("candidate_bucket"),
                "query_layer": item.get("query_layer"),
                "score": item.get("selection_score"),
                "reason": "balanced_row_mix",
            }
            for item in selected[:6]
        ],
        "selected_row_scores": [item.get("selection_score") for item in selected[:6]],
        "lowest_selected_row_score": min(
            (float(item.get("selection_score", 0) or 0) for item in selected[:6]),
            default=0,
        ),
    }


def _select_pack(records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    scores: dict[str, int] = {}
    for pack_id, pack in SEGMENT_PACKS.items():
        score = 0
        for record in records:
            resolved_blob = " ".join(
                _record_field_text(record, field)
                for field in ("resolved_category", "resolved_family", "resolved_product_type")
            )
            structured_blob = " ".join(
                _record_field_text(record, field)
                for field in ("category", "family", "product_type", "subcategory")
            )
            title_blob = " ".join(
                _record_field_text(record, field)
                for field in ("product_title", "title")
            )
            for term in pack["match_terms"]:
                normalized_term = _normalize(term)
                if normalized_term and normalized_term in resolved_blob:
                    score += 4
                elif normalized_term and normalized_term in structured_blob:
                    score += 3
                elif normalized_term and normalized_term in title_blob:
                    score += 1
        scores[pack_id] = score
    pack_id, score = max(scores.items(), key=lambda item: item[1], default=("generic", 0))
    if score > 0:
        return pack_id, SEGMENT_PACKS[pack_id]
    return "generic", _build_generic_pack(records)


def _common_value(records: list[dict[str, Any]], *fields: str) -> str:
    values: Counter[str] = Counter()
    original: dict[str, str] = {}
    for record in records:
        for field in fields:
            value = _safe_text(record.get(field))
            normalized = _normalize(value)
            if normalized:
                values[normalized] += 1
                original.setdefault(normalized, value)
                break
    return original.get(values.most_common(1)[0][0], "") if values else ""


def _keywords(records: list[dict[str, Any]], fields: tuple[str, ...], limit: int = 5) -> list[str]:
    stop = {
        "and", "the", "for", "with", "from", "this", "that", "your", "our", "are",
        "product", "products", "brand", "walmart", "oz", "pack",
    }
    counts: Counter[str] = Counter()
    for record in records:
        for field in fields:
            counts.update(
                token
                for token in _record_field_text(record, field).split()
                if len(token) >= 4 and token not in stop
            )
    return [token for token, _ in counts.most_common(limit)]


def _build_generic_pack(records: list[dict[str, Any]]) -> dict[str, Any]:
    category = _common_value(records, "resolved_category", "category") or "Category"
    family = _common_value(records, "resolved_family", "family")
    product_type = _common_value(records, "resolved_product_type", "product_type", "subcategory")
    benefits = _keywords(records, ("key_features", "features", "description", "content_signals"), 4)
    titles = _keywords(records, ("product_title", "title"), 4)
    category_term = _normalize(category)
    family_term = _normalize(family)
    product_term = _normalize(product_type)
    benefit_terms = tuple(benefits) or tuple(titles[:2])
    use_terms = tuple(term for term in benefits + titles if term in {"daily", "family", "snack", "routine", "travel", "home"})
    audience_terms = tuple(term for term in benefits + titles if term in {"baby", "kids", "adult", "family", "women", "men"})
    ingredient_terms = tuple(term for term in benefits if term not in set(use_terms + audience_terms))
    phrase = _safe_text(product_type or family or category or "the category").lower()
    return {
        "category_phrase": phrase,
        "match_terms": (),
        "segments": (
            _segment("category_core", category or "Category Core", tuple(filter(None, (category_term, product_term)))),
            _segment("product_type", product_type or "Product Type", tuple(filter(None, (product_term, category_term)))),
            _segment("family_alignment", f"{family or category} Solutions", tuple(filter(None, (family_term, category_term)))),
            _segment("benefit_alignment", "Benefit-Led Options", benefit_terms),
            _segment("use_case_alignment", "Use-Case Solutions", use_terms or benefit_terms[:2]),
            _segment("audience_ingredient_alignment", "Audience & Ingredient Needs", audience_terms or ingredient_terms[:2] or tuple(titles[:2])),
        ),
    }


def _generic_fill_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generic = list(_build_generic_pack(records)["segments"])
    generic.extend(
        [
            _segment("content_coverage", "Content Coverage", ("benefit", "feature", "usage")),
            _segment("need_state_language", "Need-State Language", ("daily", "routine", "solution")),
            _segment("product_detail_language", "Product Detail Language", ("size", "ingredient", "material")),
        ]
    )
    return generic


def _definition_has_record_signal(definition: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    for record in records:
        if not isinstance(record, dict):
            continue
        scored = _score_record(record, definition, include_concept_support=False)
        if scored.get("matched_fields") or scored.get("weight", 0) > 0:
            return True
    return False


def _fallback_definition_rejection(definition: dict[str, Any]) -> str:
    normalized = _normalize(definition.get("display_name"))
    words = set(normalized.split())
    if normalized in {"skin care", "beauty", "beauty skin care", "baby care", "clean lifestyle", "daily skin routine"}:
        return "broad_fallback_row"
    if words and words <= {"beauty", "skin", "care"}:
        return "broad_fallback_row"
    if words & {"solutions", "options", "alignment", "coverage", "language"} and not (words & PRODUCT_ANCHOR_WORDS):
        return "generic_fallback_row"
    if words & MODIFIER_ANCHOR_WORDS and not (words & PRODUCT_ANCHOR_WORDS):
        return "incomplete_fallback_modifier"
    return ""


def _six_distinct_segments(
    selected_segments: Any,
    records: list[dict[str, Any]],
    warnings: list[str],
    ranked_selected_count: int | None = None,
) -> list[dict[str, Any]]:
    distinct: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    seen_clusters: set[str] = set()
    selected_count = 0
    selected_distinct_count = 0
    duplicate_or_blank_replaced = False
    selected_list = list(selected_segments or [])
    selected_limit = len(selected_list) if ranked_selected_count is None else ranked_selected_count
    for index, definition in enumerate([*selected_list, *_generic_fill_segments(records)]):
        if not isinstance(definition, dict):
            continue
        if index < selected_limit:
            selected_count += 1
        name_key = _normalize(definition.get("display_name"))
        id_key = _normalize(definition.get("segment_id"))
        cluster_key = _query_cluster_key(definition.get("display_name", ""))
        if not name_key or not id_key or name_key in seen_names or id_key in seen_ids or cluster_key in seen_clusters:
            duplicate_or_blank_replaced = True
            continue
        is_filler = index >= selected_limit
        is_raw_pack_filler = selected_limit <= index < len(selected_list)
        fallback_rejection = _fallback_definition_rejection(definition) if is_raw_pack_filler else ""
        if fallback_rejection:
            duplicate_or_blank_replaced = True
            continue
        if is_raw_pack_filler and len(distinct) >= 4 and not _definition_has_record_signal(definition, records):
            continue
        normalized_definition = dict(definition)
        normalized_definition["display_name"] = _display_query(definition.get("display_name"))
        distinct.append(normalized_definition)
        if index < selected_limit:
            selected_distinct_count += 1
        seen_names.add(name_key)
        seen_ids.add(id_key)
        seen_clusters.add(cluster_key)
        if len(distinct) == 6:
            break
    if len(distinct) < 6:
        warnings.append("Selected pack could not provide six distinct nonblank segments.")
    elif selected_distinct_count < selected_count or duplicate_or_blank_replaced:
        warnings.append("Duplicate or blank segment definitions were replaced with restrained generic rows.")
    elif selected_distinct_count < 6:
        warnings.append("Selected pack was supplemented with restrained generic rows to reach six segments.")
    return distinct[:6]


def _term_matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _normalize(term) and _normalize(term) in text]


def _term_variants(term: str) -> set[str]:
    normalized = _normalize(term)
    if not normalized:
        return set()
    variants = {normalized}
    words = normalized.split()
    singular_words = [word[:-1] if word.endswith("s") and not word.endswith("ss") else word for word in words]
    variants.add(" ".join(singular_words))
    if normalized.startswith("facial "):
        variants.add(normalized.replace("facial ", "face ", 1))
    if " face wash" in normalized or normalized.endswith("face wash"):
        variants.add(normalized.replace("face wash", "face cleanser"))
    return {value for value in variants if value}


def _concept_matches(text: str, terms: set[str]) -> list[str]:
    normalized = _normalize(text)
    matches: list[str] = []
    for term in terms:
        if any(variant and variant in normalized for variant in _term_variants(term)):
            matches.append(term)
    return list(dict.fromkeys(matches))


def _row_concept_terms(definition: dict[str, Any]) -> dict[str, set[str]]:
    display = _normalize(definition.get("display_name"))
    positives = {_normalize(term) for term in definition.get("positive_terms", ()) if _normalize(term)}
    supporting = {_normalize(term) for term in definition.get("supporting_terms", ()) if _normalize(term)}
    guide = {_normalize(term) for term in definition.get("guide_support", []) if _normalize(term)}
    query_words = set(display.split())
    product_terms = {
        term
        for term in (*positives, *supporting, display)
        if set(term.split()) & PRODUCT_ANCHOR_WORDS
    }
    modifier_terms = {
        term
        for term in (*positives, *guide, display)
        if set(term.split()) & MODIFIER_ANCHOR_WORDS
    }
    form_terms = {
        term
        for term in (*positives, *guide, display)
        if set(term.split()) & FORM_ANCHOR_WORDS
    }
    if query_words & MODIFIER_ANCHOR_WORDS:
        modifier_terms.update(query_words & MODIFIER_ANCHOR_WORDS)
    if query_words & FORM_ANCHOR_WORDS:
        form_terms.update(query_words & FORM_ANCHOR_WORDS)
    if query_words & PRODUCT_ANCHOR_WORDS:
        product_terms.update(query_words & PRODUCT_ANCHOR_WORDS)
    return {
        "product_type": {term for term in product_terms if term},
        "modifier": {term for term in modifier_terms if term},
        "form": {term for term in form_terms if term},
        "guide": guide,
    }


def _score_record(record: dict[str, Any], definition: dict[str, Any], *, include_concept_support: bool = True) -> dict[str, Any]:
    fields = definition["fields"]
    positives = tuple(definition["positive_terms"])
    supporting = tuple(definition["supporting_terms"])
    negatives = tuple(definition["negative_terms"])
    matched_fields: list[str] = []
    matched_terms: list[str] = []
    structured_match_count = 0
    text_match_count = 0
    title_match_count = 0
    guide_match_count = 0
    product_type_match_count = 0
    modifier_match_count = 0
    form_match_count = 0
    concept_matched_fields: list[str] = []
    concept_matched_terms: list[str] = []
    analyzed_fields: list[str] = []
    ocr = _ocr_text(record)
    ocr_terms = _term_matches(ocr, positives)
    ocr_context = _term_matches(ocr, supporting)
    concept_terms = _row_concept_terms(definition)

    for field in fields:
        text = _record_field_text(record, field)
        if not text:
            continue
        analyzed_fields.append(field)
        if _term_matches(text, negatives):
            return {
                "weight": 0.0,
                "matched_fields": [],
                "matched_terms": [],
                "ocr_only": False,
                "analyzed": True,
                "meaningful": False,
                "structured": False,
            }
        terms = _term_matches(text, positives)
        context = _term_matches(text, supporting)
        if terms and (not supporting or context or ocr_context or field in _STRUCTURED_FIELDS):
            matched_fields.append(field)
            matched_terms.extend(terms + context)
            if field in _STRUCTURED_FIELDS:
                structured_match_count += len(terms)
            elif field in {"product_title", "title"}:
                title_match_count += len(terms)
                text_match_count += len(terms)
            else:
                text_match_count += len(terms)
            if context:
                guide_match_count += len(context)
        if include_concept_support:
            product_matches = _concept_matches(text, concept_terms["product_type"])
            modifier_matches = _concept_matches(text, concept_terms["modifier"])
            form_matches = _concept_matches(text, concept_terms["form"])
            if product_matches or modifier_matches or form_matches:
                concept_matched_fields.append(field)
                concept_matched_terms.extend(product_matches + modifier_matches + form_matches)
                if product_matches:
                    product_type_match_count += len(product_matches)
                if modifier_matches:
                    modifier_match_count += len(modifier_matches)
                if form_matches:
                    form_match_count += len(form_matches)

    if ocr:
        analyzed_fields.append("image_analysis.ocr")
    ocr_match = bool(ocr_terms and (not supporting or ocr_context))
    ocr_corroborates = bool(ocr_context and (structured_match_count or text_match_count))
    if ocr_match or ocr_corroborates:
        matched_fields.append("image_analysis.ocr")
        matched_terms.extend(ocr_terms + ocr_context)

    if structured_match_count:
        weight = 1.0
    elif text_match_count:
        weight = 0.90 if (ocr_match or ocr_corroborates) else 0.85
    elif product_type_match_count and (modifier_match_count or form_match_count):
        weight = 0.55
    else:
        weight = 0.0
    matched_fields.extend(field for field in concept_matched_fields if field not in matched_fields)
    matched_terms.extend(term for term in concept_matched_terms if term not in matched_terms)
    non_ocr_fields = set(matched_fields) - {"image_analysis.ocr"}
    support_family_count = sum(
        1
        for value in (
            bool(structured_match_count or title_match_count),
            bool(product_type_match_count),
            bool(modifier_match_count),
            bool(form_match_count),
            bool(text_match_count),
            bool(guide_match_count),
            bool(ocr_match or ocr_corroborates),
            bool((definition.get("identity_support") or {}).get("term_family_support")),
        )
        if value
    )
    meaningful = bool(
        structured_match_count
        or text_match_count >= 2
        or len(non_ocr_fields) >= 2
        or (product_type_match_count and (modifier_match_count or form_match_count))
    )
    return {
        "weight": weight,
        "matched_fields": list(dict.fromkeys(matched_fields)),
        "matched_terms": list(dict.fromkeys(matched_terms)),
        "ocr_only": bool(ocr_match and not structured_match_count and not text_match_count),
        "analyzed": bool(analyzed_fields),
        "meaningful": meaningful,
        "structured": bool(structured_match_count),
        "score_components": {
            "direct_query_match": bool(structured_match_count or title_match_count),
            "structured_query_match": bool(structured_match_count),
            "title_query_match": bool(title_match_count),
            "title_product_type_support": bool(title_match_count or structured_match_count),
            "direct_phrase_support": bool(structured_match_count or title_match_count or text_match_count),
            "product_type_support": bool(product_type_match_count),
            "modifier_support": bool(modifier_match_count),
            "form_support": bool(form_match_count),
            "pdp_content_support": bool(text_match_count),
            "guide_keyword_support": bool(guide_match_count),
            "image_ocr_support": bool(ocr_match or ocr_corroborates),
            "search_framework_support": bool(
                definition.get("candidate_source") == "actual_search_evidence"
                or (definition.get("identity_support") or {}).get("term_family_support")
            ),
            "support_family_count": support_family_count,
            "field_support_count": len(non_ocr_fields),
            "concept_support_only": bool(
                not (structured_match_count or title_match_count or text_match_count)
                and product_type_match_count
                and (modifier_match_count or form_match_count)
            ),
            "weight": weight,
        },
    }


def _visibility_label(
    weighted_support: float,
    analyzed_count: int,
    meaningful_count: int,
    structured_support_count: int = 0,
    direct_fit_count: int = 0,
    pdp_support_count: int = 0,
    guide_support_count: int = 0,
    broad_category_only: bool = False,
    support_family_count: int = 0,
    concept_only_count: int = 0,
    field_support_count: int = 0,
    image_support_count: int = 0,
    shelf_summary: dict[str, Any] | None = None,
) -> str:
    if analyzed_count <= 0:
        return _shelf_visibility_tier(shelf_summary or {}) if shelf_summary else "Limited"
    ratio = weighted_support / analyzed_count
    if ratio >= 0.70:
        label = "Strong"
    elif ratio >= 0.40 or support_family_count >= 3:
        label = "Moderate"
    elif ratio >= 0.15 or support_family_count >= 2:
        label = "Partial"
    else:
        label = "Limited"

    if analyzed_count == 1:
        if meaningful_count <= 0:
            label = "Partial" if weighted_support > 0 else "Limited"
            shelf_tier = _shelf_visibility_tier(shelf_summary or {}) if shelf_summary else "Limited"
            return _max_label(label, shelf_tier) if shelf_summary and shelf_tier in {"Partial", "Moderate"} else label
        if label in {"Strong", "Moderate"} and structured_support_count <= 0 and field_support_count < 2:
            return "Partial"
        if label in {"Strong", "Moderate"} and image_support_count > 0 and structured_support_count <= 0 and field_support_count <= 2:
            return "Partial"
        return "Moderate" if label == "Strong" else label
    if analyzed_count == 2:
        if meaningful_count <= 0:
            return "Partial" if weighted_support > 0 else "Limited"
        if label == "Strong" and (meaningful_count < 2 or structured_support_count < 2):
            return "Moderate"
    if label == "Strong":
        aligned_signal_count = sum(1 for value in (direct_fit_count, pdp_support_count, guide_support_count) if value > 0)
        has_signal_context = any(value > 0 for value in (direct_fit_count, pdp_support_count, guide_support_count))
        if broad_category_only or (support_family_count > 0 and direct_fit_count <= 0) or concept_only_count > 0 or (has_signal_context and aligned_signal_count < 2):
            return "Moderate"
    if shelf_summary:
        shelf_tier = _shelf_visibility_tier(shelf_summary)
        if label == "Strong" and shelf_tier != "Strong":
            return "Moderate" if shelf_tier == "Moderate" else "Partial"
        if label == "Moderate" and shelf_tier == "Limited" and direct_fit_count <= 0:
            return "Partial"
        if label in {"Limited", "Partial"} and shelf_tier in {"Moderate", "Strong"}:
            return "Moderate"
    return label


def _max_label(*labels: str) -> str:
    return max(labels, key=_rating_rank)


def _validate_visibility_label(value: Any, warnings: list[str], context: str) -> str:
    label = _safe_text(value)
    if label in _VISIBILITY_SET:
        return label
    warnings.append(
        f"{context} produced invalid visibility '{label or '<empty>'}'; replaced with Limited."
    )
    return "Limited"


def _score_group(records: list[dict[str, Any]], definition: dict[str, Any]) -> dict[str, Any]:
    matches = [_score_record(record, definition) for record in records]
    shelf_summary = _search_shelf_summary(records, definition)
    analyzed = [match for match in matches if match.get("analyzed")]
    analyzed_count = len(analyzed)
    supporting = [match for match in matches if match["weight"] > 0]
    observed = [match for match in matches if match.get("matched_fields")]
    weighted_support = sum(float(match["weight"]) for match in supporting)
    meaningful_count = sum(1 for match in supporting if match.get("meaningful"))
    structured_support_count = sum(1 for match in supporting if match.get("structured"))
    ratio = weighted_support / analyzed_count if analyzed_count else 0.0
    warnings: list[str] = []
    if analyzed_count == 0:
        warnings.append("No applicable PDP fields were available for this segment.")
    elif analyzed_count <= 2:
        warnings.append(
            f"Only {analyzed_count} PDP(s) contained applicable evidence; small-sample safeguards were applied."
        )
    ocr_only_count = sum(1 for match in matches if match.get("ocr_only"))
    if ocr_only_count:
        warnings.append(
            f"{ocr_only_count} OCR-only match(es) were retained for traceability but did not count as support."
        )
    score_components = {
        "direct_query_fit": sum(1 for match in supporting if (match.get("score_components") or {}).get("direct_query_match")),
        "structured_query_fit": sum(1 for match in supporting if (match.get("score_components") or {}).get("structured_query_match")),
        "title_query_fit": sum(1 for match in supporting if (match.get("score_components") or {}).get("title_query_match")),
        "title_product_type_support": sum(
            1 for match in supporting if (match.get("score_components") or {}).get("title_product_type_support")
        ),
        "direct_phrase_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("direct_phrase_support")),
        "product_type_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("product_type_support")),
        "modifier_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("modifier_support")),
        "form_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("form_support")),
        "pdp_content_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("pdp_content_support")),
        "guide_keyword_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("guide_keyword_support")),
        "image_ocr_support": sum(1 for match in observed if (match.get("score_components") or {}).get("image_ocr_support")),
        "search_framework_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("search_framework_support")),
        "concept_support_only": sum(1 for match in supporting if (match.get("score_components") or {}).get("concept_support_only")),
        "field_support_count": max(
            ((match.get("score_components") or {}).get("field_support_count", 0) for match in supporting),
            default=0,
        ),
        "weighted_support": round(weighted_support, 2),
        "analyzed_count": analyzed_count,
        "meaningful_count": meaningful_count,
        "structured_support_count": structured_support_count,
        "search_shelf_signals": shelf_summary,
        "search_shelf_tier": _shelf_visibility_tier(shelf_summary),
    }
    support_families = {
        "direct_phrase_support": score_components["direct_phrase_support"] > 0,
        "product_type_support": score_components["product_type_support"] > 0,
        "modifier_support": score_components["modifier_support"] > 0,
        "form_support": score_components["form_support"] > 0,
        "pdp_content_support": score_components["pdp_content_support"] > 0,
        "guide_support": score_components["guide_keyword_support"] > 0,
        "image_or_pdp_visual_support": score_components["image_ocr_support"] > 0,
        "search_framework_support": score_components["search_framework_support"] > 0,
    }
    score_components["support_families"] = support_families
    score_components["support_family_count"] = sum(
        1
        for key, value in support_families.items()
        if value and key != "image_or_pdp_visual_support"
    )
    broad_category_only = bool(
        definition.get("candidate_bucket") == "category_anchor"
        and score_components["structured_query_fit"] > 0
        and score_components["title_query_fit"] == 0
        and score_components["pdp_content_support"] == 0
        and score_components["guide_keyword_support"] == 0
    )
    label = _validate_visibility_label(
        _visibility_label(
            weighted_support,
            analyzed_count,
            meaningful_count,
            structured_support_count,
            score_components["direct_query_fit"],
            score_components["pdp_content_support"],
            score_components["guide_keyword_support"],
            broad_category_only,
            score_components["support_family_count"],
            score_components["concept_support_only"],
            score_components["field_support_count"],
            score_components["image_ocr_support"],
            shelf_summary,
        ),
        warnings,
        str(definition.get("segment_id") or "segment"),
    )
    return {
        "label": label,
        "supporting_count": len(supporting),
        "analyzed_count": analyzed_count,
        "fraction": f"{len(supporting)}/{analyzed_count}",
        "alignment_ratio": ratio,
        "alignment_percentage": round(ratio * 100, 1),
        "matched_fields": list(dict.fromkeys(field for match in observed for field in match["matched_fields"])),
        "matched_terms": list(dict.fromkeys(term for match in observed for term in match["matched_terms"])),
        "ocr_only_count": ocr_only_count,
        "weighted_support": round(weighted_support, 2),
        "score_components": score_components,
        "label_reason": _label_reason(label, score_components, broad_category_only),
        "warnings": warnings,
    }


def _label_reason(label: str, components: dict[str, Any], broad_category_only: bool = False) -> str:
    shelf = components.get("search_shelf_signals") or {}
    shelf_tier = _safe_text(components.get("search_shelf_tier"))
    if shelf_tier and shelf_tier != "Limited":
        rank = shelf.get("best_rank")
        matches = int(shelf.get("visible_brand_match_count", 0) or 0)
        direct_rows = int(shelf.get("direct_search_rows", 0) or 0)
        pressure = int(shelf.get("sponsored_pressure_count", 0) or 0) + int(shelf.get("badge_pressure_count", 0) or 0)
        rank_phrase = f"best visible rank {rank}" if rank else "top-10 shelf presence"
        return (
            f"{label} because {rank_phrase}, {matches} visible brand match(es), "
            f"{direct_rows} direct search row(s), and {pressure} shelf pressure signal(s) support this row."
        )
    if broad_category_only:
        return "Capped below Strong because broad category membership was not supported by row-specific title, PDP, or guide signals."
    family_count = int(components.get("support_family_count", 0) or 0)
    if components.get("concept_support_only"):
        return (
            f"{label} because product-type and modifier/form evidence align, "
            "but direct phrase support was not strong enough for Strong."
        )
    if label == "Strong":
        return "Strong because direct row fit is supported by multiple aligned evidence sources."
    if label == "Moderate":
        return f"Moderate because {family_count} aligned evidence families support the row without meeting the strict Strong bar."
    if label == "Partial":
        return f"Partial because {family_count} evidence family or lighter indirect support exists, but row support is not yet broad."
    return "Limited because direct phrase, product-type, modifier, PDP, guide, and search-framework support were weak or mostly absent."


def _rating_rank(label: str) -> int:
    return {"Limited": 0, "Partial": 1, "Moderate": 2, "Strong": 3}.get(label, 0)


def build_slide6_visibility(
    primary_records: list[dict[str, Any]],
    competitor_records: list[dict[str, Any]],
    slide4_findings: dict[str, Any] | None = None,
    audit_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = [record for record in (primary_records or []) if isinstance(record, dict)]
    competitors = [record for record in (competitor_records or []) if isinstance(record, dict)]
    audit_metadata = audit_metadata or {}
    pack_id, pack = _select_pack(primary or competitors)
    identity = resolve_strategic_identity(
        primary or competitors,
        fallback_category=str(pack.get("category_phrase") or ""),
        fallback_product_type=str(pack.get("category_phrase") or ""),
    )
    category_phrase = _display_query(
        identity.get("product_type_display")
        or identity.get("family_display")
        or pack["category_phrase"]
    )
    client_label = _safe_text(
        audit_metadata.get("client_company_name")
        or audit_metadata.get("client_name")
        or "Client"
    )
    warnings: list[str] = []
    if not primary:
        warnings.append("No client PDP records were available; client alignment defaults to Limited.")
    if not competitors:
        warnings.append("No competitor PDP records were available; competitor alignment defaults to Limited.")
    if 0 < len(primary) < 3:
        warnings.append("Client evidence is sparse; ratings use conservative small-sample handling.")
    if 0 < len(competitors) < 3:
        warnings.append("Competitor evidence is sparse; ratings use conservative small-sample handling.")
    if pack_id == "generic":
        warnings.append("No controlled category pack matched; restrained generic segments were used.")

    segments: list[dict[str, Any]] = []
    candidate_records = [*primary, *competitors] or (primary or competitors)
    query_candidates, candidate_debug = _build_query_candidates(
        candidate_records,
        identity,
        pack,
        audit_metadata,
        competitors,
    )
    definitions, selection_debug = _select_final_query_rows(query_candidates, primary, competitors, warnings)
    if len(definitions) < 6:
        definitions = _six_distinct_segments(
            [*definitions, *pack.get("segments", ())],
            primary or competitors,
            warnings,
            ranked_selected_count=len(definitions),
        )
    for definition in definitions:
        client = _score_group(primary, definition)
        competitor = _score_group(competitors, definition)
        row_warnings = [
            *(f"Client: {warning}" for warning in client["warnings"]),
            *(f"Competitor: {warning}" for warning in competitor["warnings"]),
        ]
        competitor_label = _validate_visibility_label(
            competitor["label"],
            row_warnings,
            f"{definition['segment_id']} competitor",
        )
        client_visibility = _validate_visibility_label(
            client["label"],
            row_warnings,
            f"{definition['segment_id']} client",
        )
        segments.append(
            {
                "segment": definition["display_name"],
                "competitor_visibility": competitor_label,
                "client_visibility": client_visibility,
                "competitor_supporting_count": competitor["supporting_count"],
                "competitor_analyzed_count": competitor["analyzed_count"],
                "competitor_fraction": competitor["fraction"],
                "competitor_percentage": competitor["alignment_percentage"],
                "client_supporting_count": client["supporting_count"],
                "client_analyzed_count": client["analyzed_count"],
                "client_fraction": client["fraction"],
                "client_percentage": client["alignment_percentage"],
                "warnings": row_warnings,
                "debug": {
                    "segment_id": definition["segment_id"],
                    "selected_pack": pack_id,
                    "row_selection_reason": (
                        "Selected as a ranked shopper-query row using realism, product fit, evidence support, "
                        "diversity, benchmark usefulness, and adjacent discovery value."
                    ),
                    "candidate_source": definition.get("candidate_source", "segment_pack_fallback"),
                    "row_type": definition.get("row_type", "fallback"),
                    "query_layer": definition.get("query_layer", ""),
                    "candidate_bucket": definition.get("candidate_bucket", ""),
                    "identity_support": definition.get("identity_support", {}),
                    "ranking_factors": definition.get("ranking_factors", {}),
                    "selection_score": definition.get("selection_score"),
                    "guide_support_used": definition.get("guide_support", []),
                    "negative_terms": list(definition.get("negative_terms", ())),
                    "matched_fields": {
                        "competitor": competitor["matched_fields"],
                        "client": client["matched_fields"],
                    },
                    "matched_terms": {
                        "competitor": competitor["matched_terms"],
                        "client": client["matched_terms"],
                    },
                    "ocr_only_support": {
                        "competitor": competitor["ocr_only_count"],
                        "client": client["ocr_only_count"],
                    },
                    "weighted_support": {
                        "competitor": competitor["weighted_support"],
                        "client": client["weighted_support"],
                    },
                    "score_inputs": {
                        "competitor": competitor["score_components"],
                        "client": client["score_components"],
                    },
                    "label_reason": {
                        "competitor": competitor["label_reason"],
                        "client": client["label_reason"],
                    },
                    "reason": (
                        "Selected from query-style row candidates and rated from row-specific structured PDP fields "
                        "first, secondary title and content fields second, with OCR used only to corroborate "
                        "non-OCR evidence."
                    ),
                },
            }
        )

    competitor_wins = sum(
        _rating_rank(item["competitor_visibility"]) > _rating_rank(item["client_visibility"])
        for item in segments
    )
    client_wins = sum(
        _rating_rank(item["client_visibility"]) > _rating_rank(item["competitor_visibility"])
        for item in segments
    )
    intro = (
        f"Competitor and {client_label} PDP content is compared for alignment with {category_phrase} "
        "search paths using product-type, benefit, use-case, and supporting visual evidence."
    )
    if competitor_wins >= 4:
        takeaway = (
            f"Broader {category_phrase} product-type and need-state language supports stronger search-path "
            "coverage and long-term digital shelf alignment."
        )
    elif client_wins >= 4:
        takeaway = (
            f"{client_label} PDP content aligns with more core {category_phrase} search paths, with an "
            "opportunity to maintain that coverage through broader benefit and use-case language."
        )
    else:
        takeaway = (
            f"{client_label} and competitor PDP content shows comparable digital shelf alignment across "
            f"core {category_phrase} search paths, with targeted content-coverage opportunities remaining."
        )
    return {
        "category_phrase": category_phrase,
        "pack_id": pack_id,
        "intro": intro,
        "client_label": client_label,
        "segments": segments,
        "takeaway": takeaway,
        "warnings": warnings,
        "debug": {
            "client_record_count": len(primary),
            "competitor_record_count": len(competitors),
            "competitor_row_wins": competitor_wins,
            "client_row_wins": client_wins,
            "slide4_findings_available": bool(slide4_findings),
            "row_generation": candidate_debug,
            "row_selection": selection_debug,
            "segment_packs_role": "fallback_seed_only",
        },
    }
