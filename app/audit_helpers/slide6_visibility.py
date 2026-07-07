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
QUERY_ROW_TYPES = ("core", "attribute", "adjacent")
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
    if normalized in {"skin care", "beauty shelf", "category core", "product type"}:
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
    }


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


def _build_query_candidates(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    pack: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    product_data = _guide_product_data(identity)
    negative_keywords = tuple(_display_query(value) for value in _as_list(product_data.get("negative_keywords")))
    product = _singular_query(identity.get("product_type_display") or _common_value(records, "product_type", "subcategory"))
    family = _singular_query(identity.get("family_display") or "")
    category = _singular_query(identity.get("category_display") or _common_value(records, "category"))
    aliases = [_display_query(value) for value in _as_list(product_data.get("aliases"))]
    title_keywords = [_display_query(value) for value in _as_list(product_data.get("title_keywords"))]
    context_keywords = [_display_query(value) for value in _as_list(product_data.get("context_keywords"))]
    attributes = [
        _display_query(value)
        for value in _as_list(product_data.get("attributes"))
        if _normalize(value) and _normalize(value) not in ATTRIBUTE_STOPWORDS
    ]
    search_terms = _extract_search_evidence(records)
    title_phrases = _best_title_phrases(records, negative_keywords)
    pool: list[dict[str, Any]] = []

    def add(
        query: str,
        row_type: str,
        source: str,
        score: float,
        guide_support: tuple[str, ...] = (),
        positive_terms: tuple[str, ...] = (),
    ) -> None:
        ok, reason = _query_like(query, negative_keywords)
        if not ok:
            rejected.append({"query": _display_query(query), "source": source, "reason": reason})
            return
        base = product if product and product != "category" else query
        support = tuple(term for term in (product, family, category) if term and _normalize(term) != _normalize(query))
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
            )
        )

    rejected: list[dict[str, str]] = []
    for query in (product,):
        add(
            query,
            "core",
            "resolved_identity",
            10.0,
            positive_terms=(query, _safe_text(identity.get("product_type_display"))),
        )
    for query in [*aliases[:5], *title_keywords[:6]]:
        add(query, "core", "style_guide_title_or_alias", 9.0, ("aliases/title_keywords",))
    for query in search_terms:
        add(query, "core", "actual_search_evidence", 11.0)
    for query in title_phrases:
        add(query, "attribute", "actual_pdp_title_language", 7.5)

    product_head = product.split()[-1] if product else ""
    for term in [*context_keywords[:10], *attributes[:8]]:
        term_norm = _normalize(term)
        if not term_norm:
            continue
        if product_head and product_head not in term_norm and len(term_norm.split()) <= 2:
            query = f"{term} {product_head}"
        else:
            query = term
        add(query, "attribute", "style_guide_context_or_attribute", 7.0, ("context_keywords/attributes",))

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
        pool.append(
            _candidate(
                segment_name,
                source_type,
                "segment_pack_fallback_seed",
                positive_terms=(*segment_terms, segment_name),
                supporting_terms=segment_supporting,
                negative_terms=negative_keywords,
                base_score=5.5,
            )
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in pool:
        key = _query_cluster_key(item["display_name"])
        existing = deduped.get(key)
        if not existing or float(item.get("base_score", 0)) > float(existing.get("base_score", 0)):
            deduped[key] = item
        elif existing:
            existing.setdefault("merged_sources", []).append(item.get("candidate_source"))
    return list(deduped.values()), {
        "identity": {
            "category_key": identity.get("category_key"),
            "product_type_display": identity.get("product_type_display"),
            "family_display": identity.get("family_display"),
            "style_guide_path": identity.get("style_guide_path"),
            "style_product_type_score": identity.get("style_product_type_score"),
        },
        "guide_terms_used": {
            "aliases": aliases[:8],
            "title_keywords": title_keywords[:8],
            "context_keywords": context_keywords[:10],
            "attributes": attributes[:8],
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
    matches = [_score_record(record, definition) for record in records if isinstance(record, dict)]
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
        "resolved_identity": 4.0,
        "style_guide_title_or_alias": 3.5,
        "actual_pdp_title_language": 4.0,
        "style_guide_context_or_attribute": 2.0,
        "segment_pack_fallback_seed": 0.5,
    }.get(str(definition.get("candidate_source")), 1.0)
    evidence_support = client_summary["supporting_records"] + competitor_summary["supporting_records"]
    benchmark_usefulness = 1.0 if client_summary["supporting_records"] != competitor_summary["supporting_records"] else 0.0
    row_type_bonus = {"core": 4.0, "attribute": 3.0, "adjacent": 1.0}.get(str(definition.get("row_type")), 1.0)
    realism = _query_realism_score(str(definition.get("display_name")))
    quality_penalty, quality_reasons = _query_quality_penalty(definition, evidence_support)
    score = (
        float(definition.get("base_score", 0))
        + source_bonus
        + row_type_bonus
        + evidence_support * 3
        + benchmark_usefulness
        + realism
        - quality_penalty
    )
    factors = {
        "base_score": definition.get("base_score", 0),
        "source_bonus": source_bonus,
        "row_type_bonus": row_type_bonus,
        "evidence_support": evidence_support,
        "benchmark_usefulness": benchmark_usefulness,
        "shopper_query_realism": realism,
        "quality_penalty": quality_penalty,
        "quality_penalty_reasons": quality_reasons,
        "client_supporting_records": client_summary["supporting_records"],
        "competitor_supporting_records": competitor_summary["supporting_records"],
    }
    return score, factors


def _query_quality_penalty(definition: dict[str, Any], evidence_support: int) -> tuple[float, list[str]]:
    query = str(definition.get("display_name") or "")
    normalized = _normalize(query)
    words = normalized.split()
    source = str(definition.get("candidate_source") or "")
    row_type = str(definition.get("row_type") or "")
    reasons: list[str] = []
    penalty = 0.0
    weak_pairs = {
        "cleansing cleanser",
        "cleanser face",
        "facial cleanser face",
        "gentle facial",
        "wash cleanser",
        "wash sensitive",
        "hydrating face",
        "organic option",
        "flavor jam",
        "option cleanser",
        "product cleanser",
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
            "skin",
            "butter",
            "jam",
            "spread",
            "moisturizer",
            "cleaner",
        }
        if benefit_or_concern_terms & set(words) and not (product_or_concern_anchors & set(words)):
            penalty += 4.0
            reasons.append("benefit_fragment_without_product_anchor")
        elif not (benefit_or_concern_terms & set(words)):
            penalty += 5.0
            reasons.append("unsupported_title_fragment")
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
    rejected_similar: list[dict[str, str]] = []

    def can_add(candidate: dict[str, Any]) -> tuple[bool, str]:
        key = _query_cluster_key(candidate["display_name"])
        if key in selected_keys:
            return False, "near_duplicate"
        similarity_reason = _row_similarity_reason(key, selected_keys)
        if similarity_reason:
            return False, similarity_reason
        row_type = str(candidate.get("row_type"))
        if row_type == "core" and type_counts[row_type] >= 2:
            return False, "core_row_quota_met"
        if row_type == "attribute" and type_counts[row_type] >= 4:
            return False, "attribute_row_quota_met"
        if row_type == "adjacent" and type_counts[row_type] >= 1:
            return False, "adjacent_row_quota_met"
        penalty = float(((candidate.get("ranking_factors") or {}).get("quality_penalty", 0)) or 0)
        if penalty >= 8:
            return False, "weak_query_quality"
        if len(selected) >= 4:
            factors = candidate.get("ranking_factors") or {}
            evidence_support = int(factors.get("evidence_support", 0) or 0)
            source = str(candidate.get("candidate_source") or "")
            if source == "segment_pack_fallback_seed" and evidence_support <= 0 and penalty >= 5:
                return False, "weak_trailing_fallback_row"
        return True, "selected"

    for score, candidate, factors in ranked:
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
            penalty = float(factors.get("quality_penalty", 0) or 0)
            if penalty >= 8:
                rejected_similar.append({"query": candidate["display_name"], "reason": "weak_query_quality"})
                continue
            if len(selected) >= 4 and penalty >= 5 and int(factors.get("evidence_support", 0) or 0) <= 0:
                rejected_similar.append({"query": candidate["display_name"], "reason": "weak_trailing_fallback_row"})
                continue
            selected.append(candidate)
            selected_keys.add(key)
    return selected[:6], {
        "ranked_candidates": [
            {
                "query": candidate["display_name"],
                "source": candidate.get("candidate_source"),
                "row_type": candidate.get("row_type"),
                "score": round(score, 2),
                "ranking_factors": factors,
            }
            for score, candidate, factors in ranked[:20]
        ],
        "rejected_similar_rows": rejected_similar[:20],
        "row_type_counts": dict(type_counts),
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


def _six_distinct_segments(
    selected_segments: Any,
    records: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    distinct: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    selected_count = 0
    selected_distinct_count = 0
    selected_list = list(selected_segments or [])
    for index, definition in enumerate([*selected_list, *_generic_fill_segments(records)]):
        if not isinstance(definition, dict):
            continue
        if index < len(selected_list):
            selected_count += 1
        name_key = _normalize(definition.get("display_name"))
        id_key = _normalize(definition.get("segment_id"))
        if not name_key or not id_key or name_key in seen_names or id_key in seen_ids:
            continue
        distinct.append(definition)
        if index < len(selected_list):
            selected_distinct_count += 1
        seen_names.add(name_key)
        seen_ids.add(id_key)
        if len(distinct) == 6:
            break
    if len(distinct) < 6:
        warnings.append("Selected pack could not provide six distinct nonblank segments.")
    elif selected_distinct_count < selected_count:
        warnings.append("Duplicate or blank segment definitions were replaced with restrained generic rows.")
    elif selected_distinct_count < 6:
        warnings.append("Selected pack was supplemented with restrained generic rows to reach six segments.")
    return distinct[:6]


def _term_matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _normalize(term) and _normalize(term) in text]


def _score_record(record: dict[str, Any], definition: dict[str, Any]) -> dict[str, Any]:
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
    analyzed_fields: list[str] = []
    ocr = _ocr_text(record)
    ocr_terms = _term_matches(ocr, positives)
    ocr_context = _term_matches(ocr, supporting)

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
    else:
        weight = 0.0
    non_ocr_fields = set(matched_fields) - {"image_analysis.ocr"}
    meaningful = bool(
        structured_match_count
        or text_match_count >= 2
        or len(non_ocr_fields) >= 2
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
            "title_product_type_support": bool(title_match_count or structured_match_count),
            "pdp_content_support": bool(text_match_count),
            "guide_keyword_support": bool(guide_match_count),
            "image_ocr_support": bool(ocr_match or ocr_corroborates),
            "weight": weight,
        },
    }


def _visibility_label(
    weighted_support: float,
    analyzed_count: int,
    meaningful_count: int,
    structured_support_count: int = 0,
) -> str:
    if analyzed_count <= 0:
        return "Limited"
    ratio = weighted_support / analyzed_count
    if ratio >= 0.70:
        label = "Strong"
    elif ratio >= 0.40:
        label = "Moderate"
    elif ratio >= 0.15:
        label = "Partial"
    else:
        label = "Limited"

    if analyzed_count == 1:
        if meaningful_count <= 0:
            return "Partial" if weighted_support > 0 else "Limited"
        return "Moderate" if label == "Strong" else label
    if analyzed_count == 2:
        if meaningful_count <= 0:
            return "Partial" if weighted_support > 0 else "Limited"
        if label == "Strong" and (meaningful_count < 2 or structured_support_count < 2):
            return "Moderate"
    return label


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
        "title_product_type_support": sum(
            1 for match in supporting if (match.get("score_components") or {}).get("title_product_type_support")
        ),
        "pdp_content_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("pdp_content_support")),
        "guide_keyword_support": sum(1 for match in supporting if (match.get("score_components") or {}).get("guide_keyword_support")),
        "image_ocr_support": sum(1 for match in observed if (match.get("score_components") or {}).get("image_ocr_support")),
        "weighted_support": round(weighted_support, 2),
        "analyzed_count": analyzed_count,
        "meaningful_count": meaningful_count,
        "structured_support_count": structured_support_count,
    }
    label = _validate_visibility_label(
        _visibility_label(
            weighted_support,
            analyzed_count,
            meaningful_count,
            structured_support_count,
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
        "label_reason": _label_reason(label, score_components),
        "warnings": warnings,
    }


def _label_reason(label: str, components: dict[str, Any]) -> str:
    if label == "Strong":
        return "Strong because direct row fit is supported by multiple aligned evidence sources."
    if label == "Moderate":
        return "Moderate because meaningful row-specific support exists but is not dominant across the evidence set."
    if label == "Partial":
        return "Partial because support exists but is limited, indirect, or small-sample capped."
    return "Limited because little row-specific evidence was available for this side."


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
    query_candidates, candidate_debug = _build_query_candidates(primary or competitors, identity, pack)
    definitions, selection_debug = _select_final_query_rows(query_candidates, primary, competitors, warnings)
    if len(definitions) < 6:
        definitions = _six_distinct_segments([*definitions, *pack.get("segments", ())], primary or competitors, warnings)
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
