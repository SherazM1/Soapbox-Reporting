from __future__ import annotations

import re
from collections import Counter
from typing import Any


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
            else:
                text_match_count += len(terms)

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
        "warnings": warnings,
    }


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
    category_phrase = str(pack["category_phrase"])
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
    definitions = _six_distinct_segments(
        pack.get("segments", ()),
        primary or competitors,
        warnings,
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
                    "reason": (
                        "Selected from the controlled pack and rated from available structured PDP fields "
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
        },
    }
