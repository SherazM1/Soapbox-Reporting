from __future__ import annotations

import re
from statistics import median
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlparse

from app.audit_helpers.bullet_uniqueness import make_unique_bullet_text
from app.audit_helpers.strategic_cues import search_cue_context


CURRENT_BULLET_BANK = {
    "brand_presence": {
        "Strong": "Brand visibility anchors {shopping_context}",
        "Moderate": "Client items hold a visible shelf position",
        "Missing": "Client presence is limited in leading results",
        "Unknown": "Client visibility is not clear in this shelf",
    },
    "top_result_visibility": {
        "Strong": "Top results reinforce {shelf_navigation}",
        "Moderate": "Visibility is present across leading results",
        "Limited": "Top-of-page visibility is still constrained",
        "Missing": "Client presence sits lower in the search shelf",
    },
    "sponsored_competition": {
        "Strong": "Sponsored pressure shapes the search shelf",
        "Moderate": "Paid placements influence discovery",
        "Unknown": "Sponsored pressure is not fully visible",
        "Missing": "Sponsored status could not be fully verified",
    },
    "keyword_alignment": {
        "Strong": "Titles align with {product_type} search language",
        "Moderate": "Mixed alignment with shopper search language",
        "Limited": "Limited benefit-led keyword differentiation",
        "Missing": "Query-relevant title language can work harder",
    },
    "assortment_range": {
        "Strong": "Assortment supports wider shopper comparison",
        "Moderate": "Focused assortment supports core discovery",
        "Limited": "Assortment representation is narrow",
        "Missing": "Competitors show wider product-type coverage",
    },
    "benefit_intent_alignment": {
        "Strong": "Clear benefit-led product positioning",
        "Moderate": "Limited benefit differentiation",
        "Limited": "Reduced need-state messaging",
        "Missing": "Opportunity to strengthen intent-based positioning",
    },
    "review_authority": {
        "Strong": "Strong review authority supports visibility",
        "Moderate": "Review strength aligns with leading competitors",
        "Limited": "Competitors demonstrate stronger review authority",
        "Missing": "Limited reviews make shelf trust harder to build",
    },
    "badge_promotional_visibility": {
        "Strong": "Strong promotional shelf visibility",
        "Moderate": "Visible retail badges support differentiation",
        "Limited": "Limited promotional badge visibility",
        "Missing": "Competitors use stronger retail callouts",
    },
}

BENCHMARK_BULLET_BANK = {
    "keyword_alignment": {
        "Strong": "Titles match core {product_type} query intent",
        "Moderate": "Mixed alignment with shopper search language",
        "Limited": "Limited benefit-led keyword differentiation",
    },
    "benefit_intent_alignment": {
        "Strong": "Benefit-forward product positioning",
        "Moderate": "Wider intent-based discoverability",
        "Limited": "Benefit and use-case language expands relevance",
    },
    "assortment_range": {
        "Strong": "Assortment range expands {discovery_context}",
        "Moderate": "Expanded need-state assortment",
        "Limited": "Multiple brands compete across core search terms",
    },
    "review_authority": {
        "Strong": "Established review authority",
        "Moderate": "High-review products reinforce shopper confidence",
        "Limited": "Review strength varies across leading competitors",
    },
    "badge_promotional_visibility": {
        "Strong": "Strong promotional shelf visibility",
        "Moderate": "Retail badges strengthen product differentiation",
        "Limited": "Promotional shelf signals remain uneven",
    },
    "sponsored_competition": {
        "Strong": "Heavy sponsored competition shapes the search shelf",
        "Moderate": "Sponsored placements influence the search shelf",
        "Unknown": "Sponsored visibility may be shaping the search shelf",
    },
}

DIMENSION_ORDER = [
    "client_brand_presence",
    "top_result_visibility",
    "sponsored_competition",
    "keyword_alignment",
    "assortment_range",
    "benefit_intent_alignment",
    "review_authority",
    "badge_promotional_visibility",
]

BENCHMARK_DIMENSION_ORDER = [
    "keyword_alignment",
    "benefit_intent_alignment",
    "assortment_range",
    "review_authority",
    "badge_promotional_visibility",
    "sponsored_competition",
]

SEARCH_BULLET_FAMILIES = (
    "query_alignment",
    "shelf_breadth",
    "trust_authority",
    "side_specific",
)

MALFORMED_PHRASE_BLOCKLIST = (
    "enhanced to improve",
    "review and confidence signals",
    "enhanced active",
    "form, and dosage segmentation",
    "benchmark cue",
    "cue translation",
    "benchmark shelf range is still concentrated",
    "benchmark shelf range is limited",
    "shelf range is concentrated",
    "higher review threshold reaches",
    "presence is narrow for",
)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "into",
    "of",
    "the",
    "to",
    "with",
    "walmart",
    "products",
    "product",
    "items",
    "item",
    "search",
    "results",
    "category",
    "categories",
    "best",
    "buy",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _client_display_name(value: Any) -> str:
    name = _safe_text(value)
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    if (
        not name
        or normalized in {"test", "deck", "sample", "untitled"}
        or normalized.startswith("test")
        or " deck" in f" {normalized} "
        or " file" in f" {normalized} "
        or name.lower().endswith((".ppt", ".pptx", ".html", ".csv", ".xlsx"))
        or "\\" in name
        or "/" in name
    ):
        return "Client brand"
    return name


def _representative_valid_record(
    selected: dict[str, Any],
    valid_records: list[dict[str, Any]],
    search_framework: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = [record for record in valid_records if isinstance(record, dict)] or [selected]
    merged = dict(selected)
    merged_products: list[dict[str, Any]] = []
    search_terms: list[str] = []
    for record in records:
        search_terms.append(_resolve_search_term(record))
        merged_products.extend(_main_products(record))
    if merged_products:
        merged["orderedMainResultProducts"] = merged_products
    explicit_fields = (
        "brand_in_top_3",
        "brand_in_top_5",
        "brand_in_top_10",
        "first_brand_rank",
        "visible_brand_ranks",
        "brand_match_count_visible",
        "dominant_brand_names",
        "dominant_brand_count",
        "brand_share_top_10",
        "top_10_badge_count",
        "top_10_best_seller_count",
        "top_10_overall_pick_count",
        "top_10_sponsored_count",
        "top_5_review_counts",
        "top_5_avg_review_count",
        "top_5_avg_rating",
        "top_10_form_factors",
        "top_10_solution_types",
        "visible_use_case_terms",
        "result_form_diversity",
    )
    for field in explicit_fields:
        values = [
            _get_first(record, field, f"data.{field}", default="")
            for record in records
            if isinstance(record, dict)
        ]
        values = [value for value in values if value not in (None, "", [], {})]
        if not values:
            continue
        if field.startswith("brand_in_top_"):
            merged[field] = any(_to_bool(value) for value in values)
        elif field in {"first_brand_rank"}:
            ranks = [_to_int(value, 10**9) for value in values]
            valid_ranks = [rank for rank in ranks if 0 < rank < 10**9]
            if valid_ranks:
                merged[field] = min(valid_ranks)
        elif field in {
            "brand_match_count_visible",
            "dominant_brand_count",
            "top_10_badge_count",
            "top_10_best_seller_count",
            "top_10_overall_pick_count",
            "top_10_sponsored_count",
        }:
            merged[field] = sum(_to_int(value) for value in values)
        elif field in {"brand_share_top_10", "top_5_avg_review_count", "top_5_avg_rating", "result_form_diversity"}:
            merged[field] = max(_to_float(value) for value in values)
        else:
            combined: list[Any] = []
            for value in values:
                combined.extend(_coerce_list(value))
            merged[field] = list(dict.fromkeys(_safe_text(value) for value in combined if _safe_text(value)))
    term_counts: dict[str, int] = {}
    for term in search_terms:
        if term:
            term_counts[term] = term_counts.get(term, 0) + 1
    if term_counts:
        framework_terms = [
            _normalize_term(term).lower()
            for term in (search_framework or {}).get("query_terms", [])
            if _normalize_term(term)
        ]

        def term_rank(item: tuple[str, int]) -> tuple[int, int, int]:
            term, count = item
            normalized = _normalize_term(term).lower()
            framework_match = any(
                normalized == candidate
                or normalized in candidate
                or candidate in normalized
                for candidate in framework_terms
            )
            return (-count, 0 if framework_match else 1, search_terms.index(term))

        representative_term = sorted(term_counts.items(), key=term_rank)[0][0]
        merged["searchTerm"] = representative_term
    merged["_slide3_representative_debug"] = {
        "valid_capture_count": len(records),
        "source_rows": [_source_row(record) for record in records],
        "search_terms": search_terms,
        "product_count": len(merged_products),
        "selected_source_row_for_screenshot": _source_row(selected),
        "framework_terms_considered": (search_framework or {}).get("query_terms", [])[:8],
    }
    return merged


def _normalize_term(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    text = unquote_plus(text.replace("+", " "))
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_tokens(value: Any) -> list[str]:
    text = _normalize_term(value).lower()
    return [token for token in re.split(r"[^a-z0-9]+", text) if token and token not in STOP_WORDS]


SURFACED_TERM_BLOCKLIST = {
    "audience and ingredient needs",
    "audience ingredient needs",
    "ear acupressure seed",
    "ear acupressure seeds",
    "ear vaccaria seed",
    "taxonomy path",
    "vaccaria seed",
    "vaccaria seeds",
}
SURFACED_TERM_BLOCKED_TOKENS = {
    "acupressure",
    "audience",
    "taxonomy",
    "vaccaria",
}
SURFACED_TERM_INTERNAL_TOKENS = {
    "alignment",
    "benchmark",
    "coverage",
    "discoverability",
    "framework",
    "internal",
    "merchandising",
    "taxonomy",
    "visibility",
}
SURFACE_METADATA_RE = re.compile(r"\b\d{3,5}\s*x\s*\d{3,5}\b|\b[\w-]+\.(?:jpg|jpeg|png|webp)\b|\b(?:jpg|jpeg|png|webp)\b", re.I)
SURFACE_INTERNAL_RE = re.compile(
    r"\b(?:framework|resolution path|signal bucket|taxonomy path|inferred category node)\b",
    re.I,
)


def _sanitize_slide3_surface_text(text: Any, fallback: str = "Search language remains category-focused") -> str:
    clean = re.sub(r"\s+", " ", _safe_text(text)).strip()
    clean = SURFACE_METADATA_RE.sub("", clean)
    clean = SURFACE_INTERNAL_RE.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" .;:-")
    return clean or fallback


def _surface_term_supported(term: str, evidence: str) -> bool:
    tokens = [token for token in _normalize_tokens(term) if len(token) > 3]
    if not tokens:
        return False
    evidence_tokens = set(_normalize_tokens(evidence))
    return any(token in evidence_tokens for token in tokens)


def _safe_slide3_surface_term(term: Any, evidence: str = "") -> str:
    cleaned = _normalize_term(term).lower()
    normalized = " ".join(_normalize_tokens(cleaned))
    if not cleaned or cleaned in {"category search", "product type", "category"}:
        return ""
    if SURFACE_METADATA_RE.search(cleaned) or SURFACE_INTERNAL_RE.search(cleaned):
        return ""
    if cleaned in SURFACED_TERM_BLOCKLIST:
        return ""
    if any(token in normalized.split() for token in SURFACED_TERM_BLOCKED_TOKENS):
        return ""
    words = normalized.split()
    if len(words) > 5:
        return ""
    if " and " in f" {cleaned} " and not any(
        phrase in cleaned
        for phrase in ("jams jellies", "nut butters", "fragrance free", "normal to oily")
    ):
        return ""
    if len(words) >= 3 and any(token in words for token in SURFACED_TERM_INTERNAL_TOKENS):
        return ""
    if evidence and not _surface_term_supported(cleaned, evidence):
        return ""
    return cleaned


def _safe_slide3_fallback_phrase(evidence: str, fallback: str = "") -> str:
    normalized = " ".join(_normalize_tokens(evidence))
    if any(term in normalized for term in ("antacid", "heartburn", "acid reducer", "acid", "upset stomach", "stomach")):
        if "heartburn" in normalized:
            return "heartburn relief"
        if "acid" in normalized and "reducer" in normalized:
            return "acid reducer"
        if "stomach" in normalized:
            return "stomach relief"
        return "antacid"
    safe_fallback = _safe_slide3_surface_term(fallback)
    return safe_fallback or "over-the-counter medicine"


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return re.sub(r"[^a-z0-9]+", " ", _safe_text(value).lower()).strip() in {
        "1",
        "true",
        "yes",
        "y",
        "present",
        "detected",
    }


def _get_first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
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


def _source_row(record: dict[str, Any]) -> int:
    raw = _get_first(record, "sourceRow", "rowNumber", "data.sourceRow", "data.rowNumber", default=10**9)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 10**9


def _role(record: dict[str, Any]) -> str:
    role = _safe_text(
        _get_first(
            record,
            "role",
            "inputRole",
            "sourceRole",
            "originalRole",
            "data.role",
            "data.originalRole",
            default="",
        )
    ).strip()
    if role.lower() == "client":
        return "Current"
    if role.lower() == "competitor":
        return "Benchmark"
    return role


def _is_valid_capture(record: dict[str, Any]) -> bool:
    role = _role(record)
    if role not in {"Current", "Benchmark"}:
        return False
    if not any([_safe_text(_get_first(record, "searchTerm", "Search Term", "search term", "label", "data.searchTerm", "data.label", default="")), _safe_text(_get_first(record, "URL", "url", "searchUrl", "data.URL", "data.url", "data.searchUrl", default="")), _safe_text(_get_first(record, "screenshotDataUrl", "screenshotDataURL", "data.screenshotDataUrl", "data.screenshotDataURL", "screenshot.dataUrl", "data.screenshot.dataUrl", "screenshot", default=""))]):
        return False
    return True


def _select_earliest_valid(records: list[dict[str, Any]], role: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if _role(record) != role:
            continue
        if not _is_valid_capture(record):
            continue
        valid.append(record)
    valid.sort(key=lambda item: (_source_row(item), _safe_text(_get_first(item, "searchTerm", "Search Term", "search term", "label", default=""))))
    if not valid:
        return None, [], []
    selected = valid[0]
    warnings: list[str] = []
    if len(valid) > 1:
        warnings.append(
        f"{role} evidence included multiple valid captures; source row {_source_row(selected)} was used for the screenshot while bullets summarize the broader valid evidence set."
        )
    return selected, valid, warnings


def _resolve_search_term(record: dict[str, Any]) -> str:
    for key in ("searchTerm", "Search Term", "search term", "label", "data.searchTerm", "data.label", "data.search.term"):
        term = _normalize_term(_get_first(record, key, default=""))
        if term:
            return term
    url = _safe_text(_get_first(record, "URL", "url", "searchUrl", "data.URL", "data.url", "data.searchUrl", default=""))
    if url:
        parsed = urlparse(url)
        for key in ("q", "query", "search"):
            value = parse_qsl(parsed.query, keep_blank_values=True)
            for query_key, query_value in value:
                if query_key.lower() == key:
                    term = _normalize_term(query_value)
                    if term:
                        return term
    visible_text = _safe_text(
        _get_first(
            record,
            "ocrText",
            "visibleText",
            "pageText",
            "extractedText",
            "text",
            "data.ocrText",
            "data.visibleText",
            "data.pageText",
            "data.extractedText",
            "data.text",
            default="",
        )
    )
    if visible_text:
        return _normalize_term(visible_text.splitlines()[0] if visible_text else "") or "category search"
    return "category search"


def _category_phrase(search_term: str) -> str:
    term = _normalize_term(search_term)
    if not term:
        return "category search"
    words = [word for word in term.lower().split() if word not in STOP_WORDS]
    if not words:
        return "category search"
    phrase_words = words[:]
    if phrase_words[-1] in {"products", "product", "items", "item", "categories", "category"}:
        phrase_words = phrase_words[:-1]
    if not phrase_words:
        return "category search"
    if phrase_words[-1] in {"spread", "product", "item", "butter"}:
        phrase_words[-1] = f"{phrase_words[-1]}s"
    if phrase_words[-1] in {"search", "results"}:
        phrase_words = phrase_words[:-1]
    if not phrase_words:
        return "category search"
    return " ".join(phrase_words)


def _search_phrase_context(search_term: str) -> dict[str, str]:
    phrase = _category_phrase(search_term)
    if phrase == "category search":
        product_type = "category"
        category = "category"
    else:
        product_type = phrase
        category = phrase.replace(" spreads", "").replace(" spread", "")
    return {
        "product_type": product_type,
        "shopping_context": f"{category} search and discovery",
        "shelf_navigation": f"{product_type} shelf navigation",
        "discovery_context": f"{category} discovery and comparison",
        "shopping_journey": f"{product_type} shopping journey",
    }


def _client_brand_matches(client_brand: str, product: dict[str, Any]) -> bool:
    if not client_brand:
        return False
    brand = _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName"))
    title = _safe_text(product.get("title") or product.get("productTitle") or product.get("name"))
    if not brand and not title:
        return False
    client_pattern = re.escape(_normalize_term(client_brand).lower())
    if brand:
        if _normalize_term(brand).lower() == _normalize_term(client_brand).lower():
            return True
    if title:
        title_norm = _normalize_term(title).lower()
        if re.search(rf"(?:^|\W){client_pattern}(?:$|\W)", title_norm):
            return True
    return False


def _main_products(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "orderedMainResultProducts",
        "mainResultProducts",
        "products",
        "orderedProducts",
        "capturedProducts",
        "data.orderedMainResultProducts",
        "data.mainResultProducts",
        "data.products",
        "data.orderedProducts",
        "data.capturedProducts",
    ):
        raw = _get_first(record, key, default=[])
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]
    return []


def _top_result_visibility(client_positions: list[int]) -> str:
    if not client_positions:
        return "Missing"
    first_position = min(client_positions)
    if first_position <= 4:
        return "Strong"
    if first_position <= 12:
        return "Moderate"
    return "Limited"


def _sponsored_competition(products: list[dict[str, Any]], client_positions: list[int]) -> str:
    explicit_sponsored = [product for product in products if str(product.get("sponsored", "")).lower() in {"true", "1", "yes"}]
    if explicit_sponsored and (len(explicit_sponsored) >= 2 or client_positions):
        return "Strong"
    unknown = [product for product in products if str(product.get("sponsored", "")).lower() in {"unknown", "", "none"}]
    if explicit_sponsored:
        return "Moderate"
    if unknown:
        return "Unknown"
    return "Unknown"


def _keyword_alignment(search_term: str, products: list[dict[str, Any]]) -> str:
    tokens = set(_normalize_tokens(search_term))
    if not tokens or not products:
        return "Limited"
    matching = []
    for product in products:
        title = _safe_text(product.get("title") or product.get("productTitle") or product.get("name"))
        title_tokens = set(_normalize_tokens(title))
        if title_tokens & tokens:
            matching.append(title)
    if len(matching) >= 2:
        return "Strong"
    if matching:
        return "Moderate"
    return "Limited"


def _benefit_alignment(products: list[dict[str, Any]]) -> str:
    benefit_terms = {"low sugar", "natural", "sensitive", "clean", "protein", "organic", "dermatologist", "family", "hypoallergenic"}
    text_blob = " ".join(
        _safe_text(product.get("title") or product.get("productTitle") or product.get("name"))
        for product in products
    ).lower()
    if any(term in text_blob for term in benefit_terms):
        return "Strong"
    if products:
        return "Moderate"
    return "Limited"


def _assortment_range(products: list[dict[str, Any]]) -> str:
    brands = { _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName")) for product in products if _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName")) }
    if len(brands) >= 3:
        return "Strong"
    if len(brands) >= 2:
        return "Moderate"
    if products:
        return "Limited"
    return "Missing"


def _review_authority(products: list[dict[str, Any]], client_positions: list[int]) -> str:
    review_counts = []
    for product in products:
        value = product.get("reviewCount")
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        review_counts.append(count)
    if not review_counts:
        return "Missing"
    median_reviews = median(review_counts)
    if median_reviews >= 100 and client_positions:
        return "Strong"
    if median_reviews >= 25:
        return "Moderate"
    return "Limited"


def _badge_visibility(products: list[dict[str, Any]]) -> str:
    badges = []
    for product in products:
        badges.extend(_coerce_list(product.get("badges") or product.get("badge") or product.get("badgeText")))
    if badges:
        return "Strong"
    return "Limited"


def _record_int_list(record: dict[str, Any], key: str) -> list[int]:
    values = _coerce_list(_get_first(record, key, f"data.{key}", default=[]))
    ranks: list[int] = []
    for value in values:
        parsed = _to_int(value, -1)
        if parsed > 0:
            ranks.append(parsed)
    return ranks


def _record_text_list(record: dict[str, Any], key: str) -> list[str]:
    return list(
        dict.fromkeys(
            _normalize_term(value).lower()
            for value in _coerce_list(_get_first(record, key, f"data.{key}", default=[]))
            if _normalize_term(value)
        )
    )


def _explicit_shelf_evidence(record: dict[str, Any] | None, products: list[dict[str, Any]]) -> dict[str, Any]:
    record = record or {}
    visible_ranks = _record_int_list(record, "visible_brand_ranks")
    first_rank = _to_int(_get_first(record, "first_brand_rank", "data.first_brand_rank", default=0))
    if first_rank > 0:
        visible_ranks.append(first_rank)
    share = _to_float(_get_first(record, "brand_share_top_10", "data.brand_share_top_10", default=0.0))
    if share > 1:
        share = share / 100
    review_counts = [
        _to_int(value)
        for value in _coerce_list(_get_first(record, "top_5_review_counts", "data.top_5_review_counts", default=[]))
        if _to_int(value) > 0
    ]
    if not review_counts:
        review_counts = _review_counts(products)
    badge_count = sum(
        _to_int(_get_first(record, key, f"data.{key}", default=0))
        for key in ("top_10_badge_count", "top_10_best_seller_count", "top_10_overall_pick_count")
    )
    if not badge_count:
        badge_count = len(_badges(products))
    sponsored_count = _to_int(_get_first(record, "top_10_sponsored_count", "data.top_10_sponsored_count", default=0))
    if not sponsored_count:
        sponsored_count = sum(1 for product in products if str(product.get("sponsored", "")).lower() in {"true", "1", "yes"})
    form_terms = _record_text_list(record, "top_10_form_factors")
    solution_terms = _record_text_list(record, "top_10_solution_types")
    use_case_terms = _record_text_list(record, "visible_use_case_terms")
    dominant_brands = _record_text_list(record, "dominant_brand_names")
    if not dominant_brands:
        dominant_brands = [brand.lower() for brand in _top_brands(products)]
    return {
        "brand_in_top_3": _to_bool(_get_first(record, "brand_in_top_3", "data.brand_in_top_3", default=False)) or any(rank <= 3 for rank in visible_ranks),
        "brand_in_top_5": _to_bool(_get_first(record, "brand_in_top_5", "data.brand_in_top_5", default=False)) or any(rank <= 5 for rank in visible_ranks),
        "brand_in_top_10": _to_bool(_get_first(record, "brand_in_top_10", "data.brand_in_top_10", default=False)) or any(rank <= 10 for rank in visible_ranks),
        "first_brand_rank": min(visible_ranks) if visible_ranks else 0,
        "visible_brand_ranks": sorted(set(visible_ranks)),
        "brand_match_count_visible": _to_int(_get_first(record, "brand_match_count_visible", "data.brand_match_count_visible", default=0)),
        "brand_share_top_10": share,
        "dominant_brand_names": dominant_brands[:6],
        "dominant_brand_count": _to_int(_get_first(record, "dominant_brand_count", "data.dominant_brand_count", default=len(dominant_brands))),
        "badge_count": badge_count,
        "sponsored_count": sponsored_count,
        "review_counts": review_counts,
        "top_5_avg_review_count": _to_float(_get_first(record, "top_5_avg_review_count", "data.top_5_avg_review_count", default=0.0)),
        "top_5_avg_rating": _to_float(_get_first(record, "top_5_avg_rating", "data.top_5_avg_rating", default=0.0)),
        "form_terms": form_terms[:6],
        "solution_terms": solution_terms[:6],
        "use_case_terms": use_case_terms[:6],
        "result_form_diversity": _to_float(_get_first(record, "result_form_diversity", "data.result_form_diversity", default=0.0)),
    }


def _dimension_scores(search_term: str, products: list[dict[str, Any]], client_brand: str, record: dict[str, Any] | None = None) -> dict[str, str]:
    shelf = _explicit_shelf_evidence(record, products)
    client_positions = [int(product.get("position", 999)) for product in products if _client_brand_matches(client_brand, product) and _safe_text(product.get("position"))]
    if not client_brand:
        client_presence = "Unknown"
    elif shelf["brand_in_top_3"] or shelf["brand_match_count_visible"] >= 3 or shelf["brand_share_top_10"] >= 0.25:
        client_presence = "Strong"
    elif shelf["brand_in_top_10"] or shelf["brand_match_count_visible"] >= 1:
        client_presence = "Moderate"
    else:
        client_matches = [product for product in products if _client_brand_matches(client_brand, product)]
        if len(client_matches) >= 2:
            client_presence = "Strong"
        elif client_matches:
            client_presence = "Moderate"
        else:
            client_presence = "Missing"
    scores = {
        "client_brand_presence": client_presence,
        "top_result_visibility": (
            "Strong" if shelf["brand_in_top_3"] else "Moderate" if shelf["brand_in_top_10"] else _top_result_visibility(client_positions)
        ),
        "sponsored_competition": (
            "Strong" if shelf["sponsored_count"] >= 3 else "Moderate" if shelf["sponsored_count"] else _sponsored_competition(products, client_positions)
        ),
        "keyword_alignment": _keyword_alignment(search_term, products),
        "assortment_range": (
            "Strong" if len(set(shelf["dominant_brand_names"])) >= 3 or shelf["result_form_diversity"] >= 4 else _assortment_range(products)
        ),
        "benefit_intent_alignment": (
            "Strong" if len(set(shelf["solution_terms"] + shelf["use_case_terms"])) >= 3 else _benefit_alignment(products)
        ),
        "review_authority": (
            "Strong"
            if shelf["top_5_avg_review_count"] >= 500 or max(shelf["review_counts"] or [0]) >= 500
            else "Moderate"
            if shelf["top_5_avg_review_count"] >= 100 or max(shelf["review_counts"] or [0]) >= 100
            else _review_authority(products, client_positions)
        ),
        "badge_promotional_visibility": "Strong" if shelf["badge_count"] else _badge_visibility(products),
    }
    if client_presence == "Unknown":
        scores["client_brand_presence"] = "Unknown"
    return scores


def _fit_search_bullet(text: str) -> str:
    clean = re.sub(r"\s+", " ", _safe_text(text)).strip(" .;")
    words = clean.split()
    if len(words) > 9:
        clean = " ".join(words[:9]).rstrip(" ,;-")
    if len(clean) > 68:
        clean = clean[:65].rsplit(" ", 1)[0].rstrip(" ,;-")
    return clean


def _search_language_allowed(text: str) -> tuple[bool, str]:
    normalized = _safe_text(text).lower()
    if not normalized:
        return False, "blank"
    if any(blocked in normalized for blocked in MALFORMED_PHRASE_BLOCKLIST):
        return False, "malformed_or_strategy_phrase"
    titleish_tokens = [token for token in normalized.split() if token[:1].isupper()]
    if len(titleish_tokens) >= 6:
        return False, "raw_product_title_like"
    if "breadth" in normalized:
        return False, "blocked_range_language"
    if not any(term in normalized for term in ("query", "queries", "shelf", "review", "brand", "coverage", "presence", "badge", "sponsored", "authority", "alignment", "range", "variety", "assortment", "comparison", "adjacent", "trust", "discovery", "visibility", "results", "pressure", "relevance", "intent", "selection")):
        return False, "not_search_native"
    return True, "allowed"


def _score_rank(score: str) -> int:
    return {"Strong": 4, "Moderate": 3, "Limited": 2, "Unknown": 1, "Missing": 0}.get(score, 1)


def _visibility_rank(label: Any) -> int:
    return {"Strong": 4, "Moderate": 3, "Partial": 2, "Limited": 1}.get(_safe_text(label), 0)


def _clean_framework_term(value: Any) -> str:
    term = _normalize_term(value).lower()
    term = re.sub(r"\s+", " ", term).strip(" .;:-")
    if not _safe_slide3_surface_term(term):
        return ""
    return term


def _search_framework_from_slide6(slide6_visibility: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(slide6_visibility, dict):
        return {}
    row_generation = ((slide6_visibility.get("debug") or {}).get("row_generation") or {})
    identity = row_generation.get("shared_search_identity") or {}
    if not isinstance(identity, dict):
        identity = {}
    segments = [item for item in slide6_visibility.get("segments", []) or [] if isinstance(item, dict)]
    query_terms = [
        *identity.get("product_type_anchor_terms", [])[:4],
        *identity.get("family_anchor_terms", [])[:3],
        *identity.get("category_anchor_terms", [])[:3],
        *(item.get("segment") for item in segments[:6]),
    ]
    query_terms = list(dict.fromkeys(term for term in (_clean_framework_term(term) for term in query_terms) if term))

    def side_summary(side_key: str) -> dict[str, Any]:
        visibility_key = "client_visibility" if side_key == "current" else "competitor_visibility"
        ranked_rows: list[dict[str, Any]] = []
        for index, row in enumerate(segments):
            term = _clean_framework_term(row.get("segment"))
            if not term:
                continue
            debug = row.get("debug") or {}
            label = _safe_text(row.get(visibility_key))
            rank = _visibility_rank(label)
            ranked_rows.append(
                {
                    "term": term,
                    "label": label,
                    "rank": rank,
                    "family": debug.get("candidate_bucket") or debug.get("query_layer") or "search_path",
                    "index": index,
                }
            )
        ranked_rows.sort(key=lambda item: (-item["rank"], item["index"]))
        meaningful = [item for item in ranked_rows if item["rank"] >= _visibility_rank("Partial")]
        strong = [item for item in ranked_rows if item["rank"] >= _visibility_rank("Moderate")]
        return {
            "top_terms": [item["term"] for item in ranked_rows[:4]],
            "meaningful_terms": [item["term"] for item in meaningful[:5]],
            "strong_terms": [item["term"] for item in strong[:4]],
            "strong_path_count": len(strong),
            "meaningful_path_count": len(meaningful),
            "top_label": ranked_rows[0]["label"] if ranked_rows else "",
            "top_family": ranked_rows[0]["family"] if ranked_rows else "",
        }

    product_type = next((term for term in query_terms if term), "")
    return {
        "available": bool(query_terms or segments),
        "product_type": product_type,
        "query_terms": query_terms[:10],
        "modifier_terms": [
            term for term in (_clean_framework_term(term) for term in identity.get("modifier_terms", [])[:8]) if term
        ],
        "attribute_form_terms": [
            term for term in (_clean_framework_term(term) for term in identity.get("attribute_form_terms", [])[:8]) if term
        ],
        "current": side_summary("current"),
        "benchmark": side_summary("benchmark"),
        "source": "slide6_shared_search_framework",
    }


def _framework_product_type(search_framework: dict[str, Any] | None, fallback: str) -> str:
    term = _clean_framework_term((search_framework or {}).get("product_type"))
    return term or fallback


def _framework_side_summary(search_framework: dict[str, Any] | None, side: str) -> dict[str, Any]:
    if not isinstance(search_framework, dict):
        return {}
    summary = search_framework.get(side) or {}
    return summary if isinstance(summary, dict) else {}


def _framework_query_alignment_insight(side: str, summary: dict[str, Any], product_type: str) -> str:
    terms = [term for term in summary.get("strong_terms", []) if term]
    strongest = terms[0] if terms else product_type
    if summary.get("strong_path_count", 0) >= 2:
        if side == "benchmark":
            return f"Competitors track core {strongest} queries"
        return f"Core {strongest} queries carry the search story"
    if summary.get("meaningful_path_count", 0) >= 2:
        return f"Core {strongest} searches show clearer visibility"
    if side == "benchmark":
        return f"Competitor presence is thin across {product_type} searches"
    return f"Client visibility is thin across {product_type} searches"


def _framework_range_insight(side: str, summary: dict[str, Any], product_type: str) -> str:
    if summary.get("strong_path_count", 0) >= 3:
        if side == "benchmark":
            return "Benchmark visibility spans more shopper need states"
        return "Client visibility varies by shopper need"
    if summary.get("meaningful_path_count", 0) >= 3:
        if side == "benchmark":
            return "Benchmark visibility is clearest in core searches"
        return "Core searches show the clearest client visibility"
    if side == "benchmark":
        return "Benchmark visibility is focused on fewer search paths"
    return f"Discovery range is narrower across related {product_type} searches"


def _framework_differentiator_insight(side: str, summary: dict[str, Any], product_type: str) -> str:
    top_terms = [term for term in summary.get("top_terms", []) if term]
    top_term = top_terms[0] if top_terms else product_type
    if side == "benchmark":
        if summary.get("strong_path_count", 0) >= 2:
            return f"{top_term} adds clear competitive pressure"
        return f"{top_term} is the clearest benchmark search lane"
    if summary.get("strong_path_count", 0) >= 2:
        return f"{top_term} is the clearest client search lane"
    return f"{top_term} needs stronger client shelf support"


def _client_presence_insight(client_display: str, client_positions: list[int], product_type: str) -> str:
    if not client_positions:
        return f"{client_display} is missing from leading {product_type} results"
    first_position = min(client_positions)
    if first_position <= 4:
        return f"{client_display} has strong top-shelf presence"
    if first_position <= 12:
        return f"{client_display} has visible mid-shelf presence"
    return f"{client_display} sits deeper in the search shelf"


def _shelf_range_insight(side: str, brand_count: int, product_type: str) -> str:
    if side == "current":
        if brand_count >= 3:
            return "Brand variety gives shoppers a wider comparison set"
        if brand_count >= 2:
            return "Shelf variety supports basic shopper comparison"
        return f"Discovery range is narrower across related {product_type} searches"
    if brand_count >= 3:
        return "Competing brands create a wider comparison set"
    if brand_count >= 2:
        return "Benchmark results show moderate brand variety"
    return "Benchmark visibility is focused on fewer search paths"


def _review_depth_insight(side: str, review_counts: list[int]) -> str:
    if not review_counts:
        return "Review depth is limited on this shelf"
    review_peak = max(review_counts)
    review_midpoint = median(review_counts)
    if side == "benchmark":
        if review_peak >= 500 or review_midpoint >= 100:
            return "Review depth creates stronger shelf trust"
        return "Review strength is uneven across benchmark results"
    if review_midpoint >= 100:
        return "Review strength helps support shelf credibility"
    if review_midpoint >= 25:
        return "Review depth gives shoppers some trust context"
    return "Limited reviews make shelf trust harder to build"


def _query_alignment_insight(side: str, keyword_score: str, product_type: str) -> str:
    if side == "benchmark":
        if keyword_score in {"Strong", "Moderate"}:
            return f"Titles match core {product_type} query intent"
        return f"Leading titles vary across {product_type} searches"
    if keyword_score in {"Strong", "Moderate"}:
        return f"Titles connect clearly to {product_type} queries"
    return f"Titles need clearer {product_type} search language"


def _short_join(values: list[str], fallback: str, limit: int = 2) -> str:
    cleaned = [_normalize_term(value).lower() for value in values if _normalize_term(value)]
    cleaned = list(dict.fromkeys(cleaned))[:limit]
    if not cleaned:
        return fallback
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{cleaned[0]} and {cleaned[1]}"


def _presence_from_shelf(side: str, shelf: dict[str, Any], client_display: str, product_type: str) -> str:
    first_rank = int(shelf.get("first_brand_rank", 0) or 0)
    match_count = int(shelf.get("brand_match_count_visible", 0) or 0)
    if side == "current":
        if shelf.get("brand_in_top_3") or first_rank <= 3 and first_rank > 0:
            return f"{client_display} anchors top search presence"
        if shelf.get("brand_in_top_10") or match_count:
            return f"{client_display} remains visible but not shelf-leading"
        return f"{client_display} is harder to find on shelf"
    if shelf.get("brand_in_top_3") or first_rank <= 3 and first_rank > 0:
        return "Benchmark brands hold stronger top-shelf presence"
    if shelf.get("brand_in_top_10") or match_count:
        return "Benchmark brands remain visible across leading results"
    return f"Benchmark presence is lighter across {product_type} results"


def _intent_from_shelf(side: str, shelf: dict[str, Any], product_type: str, keyword_score: str) -> str:
    intent = _short_join(
        [*shelf.get("solution_terms", []), *shelf.get("use_case_terms", [])],
        product_type,
    )
    if side == "benchmark":
        if keyword_score in {"Strong", "Moderate"} or intent != product_type:
            return f"Competitive products map clearly to {intent} query intent"
        return f"Search relevance varies across {product_type} queries"
    if keyword_score in {"Strong", "Moderate"} and intent != product_type:
        return f"Query language connects to {intent} needs"
    if keyword_score in {"Strong", "Moderate"}:
        return f"Query language connects to core {product_type} intent"
    return "Title and benefit language can work harder"


def _trust_from_shelf(side: str, shelf: dict[str, Any], review_counts: list[int]) -> str:
    avg_reviews = float(shelf.get("top_5_avg_review_count", 0) or 0)
    avg_rating = float(shelf.get("top_5_avg_rating", 0) or 0)
    review_peak = max(review_counts or [0])
    if side == "benchmark":
        if avg_reviews >= 500 or review_peak >= 500:
            return "Review depth strengthens shelf trust"
        if avg_reviews >= 100 or review_peak >= 100 or avg_rating >= 4.5:
            return "Shelf trust reinforces benchmark credibility"
        return "Shelf trust varies across benchmark results"
    if avg_reviews >= 250 or review_peak >= 250:
        return "Review depth provides solid trust support"
    if avg_reviews >= 50 or review_peak >= 50 or avg_rating >= 4.5:
        return "Review depth provides modest trust support"
    return "Limited review depth makes trust harder to build"


def _pressure_from_shelf(side: str, shelf: dict[str, Any]) -> str:
    sponsored = int(shelf.get("sponsored_count", 0) or 0)
    badges = int(shelf.get("badge_count", 0) or 0)
    dominant = _short_join(shelf.get("dominant_brand_names", []), "competitor")
    if side == "benchmark":
        if sponsored:
            return "Sponsored visibility adds benchmark pressure"
        if badges:
            return "Competitive callouts sharpen shelf differentiation"
        return f"{dominant} brands shape the benchmark comparison set"
    if sponsored and badges:
        return "Sponsored and retail callouts increase competitive pressure"
    if sponsored:
        return "Sponsored placements increase competitive pressure"
    if badges:
        return "Retail callouts make the shelf harder to break through"
    return f"{dominant} keeps comparison pressure visible"


def _range_from_shelf(side: str, shelf: dict[str, Any], brand_count: int, product_type: str) -> str:
    forms = _short_join(shelf.get("form_terms", []), "")
    solutions = _short_join(shelf.get("solution_terms", []), "")
    if side == "benchmark":
        if forms:
            return f"Benchmark assortment spans {forms} formats"
        if solutions:
            return f"Competitive coverage extends across {solutions} needs"
        if brand_count >= 3:
            return "Benchmark brands create a wider comparison set"
        return f"Benchmark products are concentrated in core {product_type} searches"
    if forms:
        return f"Client visibility is clearest in {forms} formats"
    if brand_count >= 2:
        return "Shelf variety supports basic shopper comparison"
    return f"Client visibility is thinner across {product_type} searches"


def _candidate_overlap_key(text: str) -> set[str]:
    stop = STOP_WORDS | {
        "current",
        "client",
        "benchmark",
        "competitive",
        "stronger",
        "wider",
        "visible",
        "visibility",
        "leading",
        "results",
        "across",
        "near",
        "sets",
        "supports",
        "anchors",
    }
    return {token for token in _normalize_tokens(text) if token not in stop and len(token) > 2}


def _is_overlapping_bullet(left: str, right: str) -> bool:
    left_terms = _candidate_overlap_key(left)
    right_terms = _candidate_overlap_key(right)
    if not left_terms or not right_terms:
        return False
    shared = left_terms & right_terms
    return len(shared) >= 2 and len(shared) >= min(len(left_terms), len(right_terms)) * 0.5


def _add_search_candidate(
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, str]],
    *,
    side: str,
    text: str,
    family: str,
    dimension: str,
    score: str,
    rank: float,
    evidence: list[str],
    reason: str,
    search_term: str,
    product_type: str,
    framework_source: str = "",
) -> None:
    fitted = _sanitize_slide3_surface_text(_fit_search_bullet(text))
    allowed, guard_reason = _search_language_allowed(fitted)
    if not allowed:
        rejected.append({"text": fitted, "reason": guard_reason, "family": family})
        return
    candidates.append(
        {
            "text": fitted,
            "side": side,
            "family": family if family in SEARCH_BULLET_FAMILIES else "side_specific",
            "dimension": dimension,
            "score": score,
            "rank": rank + _score_rank(score),
            "evidence": evidence,
            "reason": reason,
            "search_term": search_term,
            "product_type": product_type,
            "framework_source": framework_source,
        }
    )


def _build_side_candidates(
    side: str,
    scores: dict[str, str],
    products: list[dict[str, Any]],
    client_brand: str,
    search_term: str,
    search_framework: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    brands = _top_brands(products)
    badges = _badges(products)
    review_counts = _review_counts(products)
    shelf = _explicit_shelf_evidence(record, products)
    client_positions = [
        _product_position(product, index)
        for index, product in enumerate(products, start=1)
        if _client_brand_matches(client_brand, product)
    ]
    client_display = _client_display_name(client_brand)
    median_reviews = int(median(review_counts)) if review_counts else 0
    max_reviews = max(review_counts) if review_counts else 0
    phrase_context = _search_phrase_context(search_term)
    evidence_blob = " ".join(
        [
            search_term,
            phrase_context.get("product_type", ""),
            _safe_text(record),
            *[
                _safe_text(_get_first(product, "title", "name", "productTitle", default=""))
                for product in products[:10]
                if isinstance(product, dict)
            ],
        ]
    )
    product_type = _safe_slide3_surface_term(
        _framework_product_type(search_framework, phrase_context["product_type"]),
        evidence_blob,
    ) or _safe_slide3_fallback_phrase(evidence_blob, phrase_context["product_type"])
    framework_summary = _framework_side_summary(search_framework, side)
    if framework_summary:
        framework_summary = dict(framework_summary)
        for key in ("top_terms", "meaningful_terms", "strong_terms"):
            framework_summary[key] = [
                term
                for term in (_safe_slide3_surface_term(term, evidence_blob) for term in framework_summary.get(key, []))
                if term
            ]
    framework_available = bool((search_framework or {}).get("available") and framework_summary)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    def add(**kwargs: Any) -> None:
        _add_search_candidate(
            candidates,
            rejected,
            side=side,
            search_term=search_term,
            product_type=product_type,
            **kwargs,
        )

    keyword_score = scores.get("keyword_alignment", "Limited")
    assortment_score = scores.get("assortment_range", "Limited")
    review_score = scores.get("review_authority", "Missing")
    sponsored_score = scores.get("sponsored_competition", "Unknown")

    if framework_available:
        add(
            text=_framework_query_alignment_insight(side, framework_summary, product_type),
            family="query_alignment",
            dimension="shared_search_query_alignment",
            score=framework_summary.get("top_label") or keyword_score,
            rank=94,
            evidence=framework_summary.get("strong_terms", [])[:3] or framework_summary.get("top_terms", [])[:3],
            reason="Slide 3 translated Slide 6 selected search paths into a query-alignment takeaway.",
            framework_source="slide6_shared_search_framework",
        )
        add(
            text=_framework_range_insight(side, framework_summary, product_type),
            family="shelf_breadth",
            dimension="shared_search_breadth",
            score=framework_summary.get("top_label") or assortment_score,
            rank=92,
            evidence=framework_summary.get("meaningful_terms", [])[:4] or framework_summary.get("top_terms", [])[:4],
            reason="Slide 3 summarized the representative range of Slide 6 search paths.",
            framework_source="slide6_shared_search_framework",
        )
        add(
            text=_framework_differentiator_insight(side, framework_summary, product_type),
            family="side_specific",
            dimension="shared_search_differentiator",
            score=framework_summary.get("top_label") or sponsored_score,
            rank=73,
            evidence=framework_summary.get("top_terms", [])[:3],
            reason="Slide 3 used the strongest Slide 6 row as a side-specific search takeaway.",
            framework_source="slide6_shared_search_framework",
        )

    if side == "current":
        if client_brand:
            add(
                text=_presence_from_shelf(side, shelf, client_display, product_type),
                family="side_specific",
                dimension="client_brand_presence",
                score=scores.get("client_brand_presence", "Unknown"),
                rank=102,
                evidence=(
                    [
                        f"first_rank={shelf.get('first_brand_rank') or min(client_positions)}",
                        f"brand_matches={shelf.get('brand_match_count_visible', 0)}",
                        f"query={search_term}",
                    ]
                    if client_positions or shelf.get("first_brand_rank") or shelf.get("brand_match_count_visible")
                    else ["client_not_found", f"query={search_term}"]
                ),
                reason="Current side translated client placement into a shelf-presence takeaway.",
            )
        add(
            text=_intent_from_shelf(side, shelf, product_type, keyword_score),
            family="query_alignment",
            dimension="keyword_alignment",
            score=keyword_score,
            rank=90,
            evidence=[keyword_score, f"query={search_term}", *shelf.get("solution_terms", [])[:2], *shelf.get("use_case_terms", [])[:2]],
            reason="Current side evaluated query, solution, use-case, title, and PDP language support.",
        )
        add(
            text=_range_from_shelf(side, shelf, len(brands), product_type),
            family="shelf_breadth",
            dimension="assortment_range",
            score=assortment_score,
            rank=82,
            evidence=[f"brands={len(brands)}", *brands[:3], *shelf.get("form_terms", [])[:2]],
            reason="Current side used brand count, form factors, and solution signals as comparison-range evidence.",
        )
        add(
            text=_trust_from_shelf(side, shelf, review_counts),
            family="trust_authority",
            dimension="review_authority",
            score=review_score,
            rank=86,
            evidence=[f"median_reviews={median_reviews}", f"avg_top5_reviews={shelf.get('top_5_avg_review_count', 0)}", f"avg_rating={shelf.get('top_5_avg_rating', 0)}"],
            reason="Current side used reviews and rating evidence as shopper-trust support.",
        )
        add(
            text=_pressure_from_shelf(side, shelf),
            family="side_specific",
            dimension="competitive_pressure",
            score=sponsored_score,
            rank=84,
            evidence=[f"sponsored={shelf.get('sponsored_count', 0)}", f"badges={shelf.get('badge_count', 0)}", *shelf.get("dominant_brand_names", [])[:3]],
            reason="Current side translated sponsored, badge, and dominant-brand signals into competitive pressure.",
        )
        if badges:
            add(
                text=f"{badges[0]} badge helps draw shelf attention",
                family="side_specific",
                dimension="badge_promotional_visibility",
                score=scores.get("badge_promotional_visibility", "Limited"),
                rank=74,
                evidence=badges[:3],
                reason="Current side kept promotional badge evidence as a distinct shelf signal.",
            )
        if sponsored_score in {"Strong", "Moderate"}:
            add(
                text="Sponsored placements add pressure on the shelf",
                family="side_specific",
                dimension="sponsored_competition",
                score=sponsored_score,
                rank=70,
                evidence=[sponsored_score],
                reason="Current side identified paid placement pressure in captured results.",
            )
        add(
            text="Benefit-led coverage is still limited on the shelf",
            family="side_specific",
            dimension="benefit_intent_alignment",
            score=scores.get("benefit_intent_alignment", "Limited"),
            rank=66,
            evidence=[scores.get("benefit_intent_alignment", "Limited")],
            reason="Current side used benefit and adjacent-intent coverage as a distinct fallback signal.",
        )
    else:
        add(
            text=_range_from_shelf(side, shelf, len(brands), product_type),
            family="shelf_breadth",
            dimension="assortment_range",
            score=assortment_score,
            rank=96,
            evidence=[f"brands={len(brands)}", *brands[:3], *shelf.get("form_terms", [])[:2], *shelf.get("solution_terms", [])[:2]],
            reason="Benchmark side translated brand, form, and solution spread into a comparison-set takeaway.",
        )
        add(
            text=_intent_from_shelf(side, shelf, product_type, keyword_score),
            family="query_alignment",
            dimension="keyword_alignment",
            score=keyword_score,
            rank=90,
            evidence=[keyword_score, f"query={search_term}", *shelf.get("solution_terms", [])[:2], *shelf.get("use_case_terms", [])[:2]],
            reason="Benchmark side evaluated query, solution, use-case, title, and PDP language support.",
        )
        add(
            text=_trust_from_shelf(side, shelf, review_counts),
            family="trust_authority",
            dimension="review_authority",
            score=review_score,
            rank=84,
            evidence=[f"max_reviews={max_reviews}", f"median_reviews={median_reviews}", f"avg_top5_reviews={shelf.get('top_5_avg_review_count', 0)}", f"avg_rating={shelf.get('top_5_avg_rating', 0)}"],
            reason="Benchmark side used reviews and rating evidence to describe shelf trust.",
        )
        add(
            text=_pressure_from_shelf(side, shelf),
            family="side_specific",
            dimension="visibility_drivers",
            score=sponsored_score,
            rank=82,
            evidence=[f"sponsored={shelf.get('sponsored_count', 0)}", f"badges={shelf.get('badge_count', 0)}", *shelf.get("dominant_brand_names", [])[:3]],
            reason="Benchmark side translated sponsored, badge, and dominant-brand signals into visibility drivers.",
        )
        if sponsored_score in {"Strong", "Moderate", "Unknown"}:
            add(
                text="Sponsored visibility adds competitive pressure",
                family="side_specific",
                dimension="sponsored_competition",
                score=sponsored_score,
                rank=76,
                evidence=[sponsored_score],
                reason="Benchmark side kept sponsored pressure as a distinct competitive signal.",
            )
        if badges:
            add(
                text="Retail badges help sharpen shelf choice",
                family="side_specific",
                dimension="badge_promotional_visibility",
                score=scores.get("badge_promotional_visibility", "Limited"),
                rank=72,
                evidence=badges[:3],
                reason="Benchmark side used retail badge evidence for shelf differentiation.",
            )
        add(
            text="Benefit-led coverage broadens search relevance",
            family="side_specific",
            dimension="benefit_intent_alignment",
            score=scores.get("benefit_intent_alignment", "Limited"),
            rank=68,
            evidence=[scores.get("benefit_intent_alignment", "Limited")],
            reason="Benchmark side used benefit and adjacent-intent coverage as a distinct fallback signal.",
        )

    ranked = sorted(candidates, key=lambda item: float(item.get("rank", 0)), reverse=True)
    return ranked, {
        "side": side,
        "search_term": search_term,
        "product_type": product_type,
        "brands": brands[:6],
        "review_summary": {
            "median": median_reviews,
            "max": max_reviews,
            "count": len(review_counts),
        },
        "client_positions": client_positions,
        "scores": scores,
        "explicit_shelf_evidence": shelf,
        "shared_search_framework_used": framework_available,
        "shared_search_framework_summary": framework_summary,
        "ranked_candidate_themes": [
            {
                "text": item["text"],
                "family": item["family"],
                "dimension": item["dimension"],
                "rank": item["rank"],
                "evidence": item["evidence"],
                "reason": item["reason"],
                "framework_source": item.get("framework_source", ""),
            }
            for item in ranked[:12]
        ],
        "rejected_candidates": rejected,
    }


def _select_side_candidates(
    candidates: list[dict[str, Any]],
    used_texts: set[str] | None = None,
    *,
    side: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    used = set(used_texts or set())
    rejected: list[dict[str, str]] = []
    family_order = (
        ("side_specific", "query_alignment", "shelf_breadth", "trust_authority")
        if side == "current"
        else ("shelf_breadth", "trust_authority", "query_alignment", "side_specific")
        if side == "benchmark"
        else SEARCH_BULLET_FAMILIES
    )

    def try_add(candidate: dict[str, Any], *, allow_second_family: bool) -> bool:
        text = candidate["text"]
        family = candidate["family"]
        normalized = make_unique_bullet_text(text, used, fallback_subject="search shelf")[0]
        if normalized != text:
            rejected.append({"text": text, "reason": "duplicate_text"})
            return False
        if family_counts.get(family, 0) >= 2:
            rejected.append({"text": text, "reason": "family_cap_reached"})
            return False
        if family_counts.get(family, 0) >= 1 and not allow_second_family:
            return False
        used.add(text)
        family_counts[family] = family_counts.get(family, 0) + 1
        selected.append(candidate)
        return True

    for family in family_order:
        for candidate in candidates:
            if candidate["family"] == family and try_add(candidate, allow_second_family=False):
                break
        if len(selected) == 4:
            break
    if len(selected) < 4:
        for candidate in candidates:
            if len(selected) == 4:
                break
            if candidate in selected:
                continue
            try_add(candidate, allow_second_family=True)

    return selected[:4], {
        "accepted_bullets": [item["text"] for item in selected[:4]],
        "lead_family_order": family_order,
        "family_counts": family_counts,
        "rejected_bullets": rejected[:12],
    }


def _candidate_to_debug(candidate: dict[str, Any], *, overlap_note: str = "") -> dict[str, Any]:
    return {
        "text": candidate["text"],
        "side": candidate["side"],
        "dimension": candidate["dimension"],
        "score": candidate["score"],
        "template_id": f"search_native_{candidate['dimension']}",
        "signals": candidate["evidence"],
        "supporting_count": 0,
        "reason": candidate["reason"],
        "bullet_family": candidate["family"],
        "evidence_summary": {
            "search_term": candidate["search_term"],
            "product_type": candidate["product_type"],
            "evidence": candidate["evidence"],
            "framework_source": candidate.get("framework_source", ""),
        },
        "overlap_note": overlap_note,
    }


def _apply_cross_side_overlap_suppression(current_payload: dict[str, Any], benchmark_payload: dict[str, Any]) -> dict[str, Any]:
    current_candidates = list(current_payload.get("candidate_themes", []) or [])
    benchmark_candidates = list(benchmark_payload.get("candidate_themes", []) or [])
    current_selected = list(current_payload.get("selected_candidates", []) or [])
    benchmark_selected = list(benchmark_payload.get("selected_candidates", []) or [])
    rejected_overlap: list[dict[str, Any]] = []

    def replace_candidate(side_name: str, selected: list[dict[str, Any]], candidates: list[dict[str, Any]], index: int, other_texts: list[str]) -> bool:
        current_texts = [item["text"] for pos, item in enumerate(selected) if pos != index]
        for candidate in candidates:
            if candidate in selected:
                continue
            if any(_is_overlapping_bullet(candidate["text"], text) for text in current_texts):
                continue
            if any(_is_overlapping_bullet(candidate["text"], text) for text in other_texts):
                continue
            rejected_text = selected[index]["text"]
            selected[index] = candidate
            rejected_overlap.append(
                {
                    "side": side_name,
                    "rejected": rejected_text,
                    "replacement": candidate["text"],
                    "reason": "Replaced overlapping bullet with next-best side-specific search insight.",
                }
            )
            return True
        return False

    for current_index, current_item in enumerate(list(current_selected)):
        for benchmark_index, benchmark_item in enumerate(list(benchmark_selected)):
            if not _is_overlapping_bullet(current_item["text"], benchmark_item["text"]):
                continue
            if current_item["family"] != benchmark_item["family"]:
                continue
            current_rank = float(current_item.get("rank", 0))
            benchmark_rank = float(benchmark_item.get("rank", 0))
            if abs(current_rank - benchmark_rank) >= 5:
                if current_rank >= benchmark_rank:
                    replace_candidate(
                        "benchmark",
                        benchmark_selected,
                        benchmark_candidates,
                        benchmark_index,
                        [item["text"] for item in current_selected],
                    )
                else:
                    replace_candidate(
                        "current",
                        current_selected,
                        current_candidates,
                        current_index,
                        [item["text"] for item in benchmark_selected],
                    )
            else:
                replaced = False
                protected_dimensions = {"shared_search_query_alignment", "shared_search_breadth"}
                current_protected = current_item.get("dimension") in protected_dimensions
                benchmark_protected = benchmark_item.get("dimension") in protected_dimensions
                if current_protected and benchmark_protected:
                    rejected_overlap.append(
                        {
                            "side": "shared",
                            "current": current_item["text"],
                            "benchmark": benchmark_item["text"],
                            "reason": "Shared framework themes kept because both sides intentionally selected them.",
                        }
                    )
                    continue
                if not current_protected:
                    replaced = replace_candidate(
                        "current",
                        current_selected,
                        current_candidates,
                        current_index,
                        [item["text"] for item in benchmark_selected],
                    )
                if not replaced and not benchmark_protected:
                    replaced = replace_candidate(
                        "benchmark",
                        benchmark_selected,
                        benchmark_candidates,
                        benchmark_index,
                        [item["text"] for item in current_selected],
                    )
                if not replaced:
                    rejected_overlap.append(
                        {
                            "side": "shared",
                            "current": current_item["text"],
                            "benchmark": benchmark_item["text"],
                            "reason": "Shared theme kept because both sides had similar evidence strength.",
                        }
                    )

    current_payload["selected_candidates"] = current_selected[:4]
    benchmark_payload["selected_candidates"] = benchmark_selected[:4]
    for payload in (current_payload, benchmark_payload):
        payload["bullets"] = [candidate["text"] for candidate in payload["selected_candidates"][:4]]
        payload["bullet_debug"] = [
            _candidate_to_debug(candidate)
            for candidate in payload["selected_candidates"][:4]
        ]
        payload.setdefault("debug", {})["accepted_bullets"] = payload["bullets"]
    return {
        "rejected_overlapping_bullets": rejected_overlap,
        "current_final_bullets": current_payload.get("bullets", []),
        "benchmark_final_bullets": benchmark_payload.get("bullets", []),
    }


def _dynamic_intro(current_payload: dict[str, Any], benchmark_payload: dict[str, Any]) -> str:
    current_scores = current_payload.get("dimension_scores", {}) or {}
    benchmark_scores = benchmark_payload.get("dimension_scores", {}) or {}
    current_debug = (current_payload.get("debug", {}) or {}).get("side_candidate_debug", {}) or {}
    benchmark_debug = (benchmark_payload.get("debug", {}) or {}).get("side_candidate_debug", {}) or {}
    current_shelf = current_debug.get("explicit_shelf_evidence", {}) or {}
    benchmark_shelf = benchmark_debug.get("explicit_shelf_evidence", {}) or {}
    current_visibility = _score_rank(current_scores.get("client_brand_presence", "Missing"))
    benchmark_trust = _score_rank(benchmark_scores.get("review_authority", "Missing"))
    current_trust = _score_rank(current_scores.get("review_authority", "Missing"))
    benchmark_range = _score_rank(benchmark_scores.get("assortment_range", "Missing"))
    current_range = _score_rank(current_scores.get("assortment_range", "Missing"))
    pressure = (
        int(benchmark_shelf.get("sponsored_count", 0) or 0)
        + int(benchmark_shelf.get("badge_count", 0) or 0)
        + int(current_shelf.get("sponsored_count", 0) or 0)
        + int(current_shelf.get("badge_count", 0) or 0)
    )
    category = (
        benchmark_payload.get("category_phrase")
        or current_payload.get("category_phrase")
        or "this category"
    )
    if current_visibility <= _score_rank("Limited") and (
        benchmark_trust > current_trust or benchmark_range > current_range or pressure
    ):
        return _sanitize_slide3_surface_text(
            f"Walmart search is highly competitive in {category}, with benchmark brands holding stronger visibility and trust signals across leading results."
        )
    if current_visibility >= _score_rank("Strong"):
        return _sanitize_slide3_surface_text(
            f"Walmart search shows the client holding strong visibility in {category}, while benchmark brands continue to shape comparison through trust and assortment signals."
        )
    return _sanitize_slide3_surface_text(
        f"Walmart search shows a competitive but mixed {category} shelf, where the client remains visible while benchmark brands reinforce discovery with stronger trust and promotional cues."
    )


def _select_bullets(
    side: str,
    scores: dict[str, str],
    products: list[dict[str, Any]],
    client_brand: str,
    search_term: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    bank = CURRENT_BULLET_BANK if side == "current" else BENCHMARK_BULLET_BANK
    order = DIMENSION_ORDER if side == "current" else BENCHMARK_DIMENSION_ORDER
    bullets: list[str] = []
    bullet_debug: list[dict[str, Any]] = []
    seen_dimensions: set[str] = set()
    used_texts: set[str] = set()
    brands = _top_brands(products)
    badges = _badges(products)
    review_counts = _review_counts(products)
    client_positions = [
        _product_position(product, index)
        for index, product in enumerate(products, start=1)
        if _client_brand_matches(client_brand, product)
    ]
    client_display = _client_display_name(client_brand)
    search_brand_phrase = ", ".join(brands[:3])
    median_reviews = int(median(review_counts)) if review_counts else 0
    phrase_context = _search_phrase_context(search_term)

    def fit(text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        words = clean.split()
        if len(words) > 9:
            clean = " ".join(words[:9]).rstrip(" ,;-")
        if len(clean) <= 68:
            return clean
        return clean[:65].rsplit(" ", 1)[0].rstrip(" ,;-")

    def add(text: str, dimension: str, score: str, signals: list[str], reason: str, template_id: str) -> None:
        if len(bullets) >= 4:
            return
        rendered = text.format(**phrase_context)
        unique_text, changed = make_unique_bullet_text(
            _sanitize_slide3_surface_text(fit(rendered)),
            used_texts,
            fallback_subject=f"{side} search",
        )
        bullets.append(unique_text)
        if changed:
            signals = [*signals, "duplicate_bullet_reworded"]
        bullet_debug.append(
            {
                "text": unique_text,
                "side": side,
                "dimension": dimension,
                "score": score,
                "template_id": template_id,
                "signals": signals,
                "supporting_count": len(products),
                "reason": reason,
            }
        )
        if dimension != "fallback":
            seen_dimensions.add(dimension)

    if side == "current":
        if client_positions:
            add(
                _client_presence_insight(client_display, client_positions, phrase_context["product_type"]),
                "client_brand_presence",
                scores.get("client_brand_presence", "Unknown"),
                [f"client_position={min(client_positions)}"],
                "Selected because the client brand was found in captured search products.",
                "current_client_position",
            )
        elif client_brand:
            add(
                f"{client_display} is missing from leading {phrase_context['shelf_navigation']}",
                "client_brand_presence",
                "Missing",
                ["client_not_found"],
                "Selected because the client brand was not found in captured products.",
                "current_client_missing",
            )
        if badges:
            add(
                f"{badges[0]} badge visibility shapes {phrase_context['shelf_navigation']}",
                "badge_promotional_visibility",
                scores.get("badge_promotional_visibility", "Limited"),
                badges[:3],
                "Selected because badges or promotional labels were detected.",
                "current_badge_visibility",
            )
        if median_reviews:
            add(
                _review_depth_insight(side, review_counts),
                "review_authority",
                scores.get("review_authority", "Limited"),
                [f"median_reviews={median_reviews}"],
                "Selected because review counts were captured in search results.",
                "current_review_context",
            )
    else:
        if search_brand_phrase:
            add(
                f"Competing brands broaden {phrase_context['discovery_context']}",
                "assortment_range",
                scores.get("assortment_range", "Limited"),
                brands[:3],
                "Selected because multiple benchmark brands were captured.",
                "benchmark_top_brands",
            )
        if median_reviews:
            add(
                _review_depth_insight(side, review_counts),
                "review_authority",
                scores.get("review_authority", "Limited"),
                [f"median_reviews={median_reviews}"],
                "Selected because benchmark products included review-count evidence.",
                "benchmark_review_context",
            )

    for dimension in order:
        if dimension in seen_dimensions:
            continue
        score = scores.get(dimension, "Unknown")
        if side == "current":
            if dimension == "client_brand_presence" and score not in {"Strong", "Moderate", "Missing", "Unknown"}:
                continue
            if dimension == "sponsored_competition" and score == "Missing":
                continue
            if dimension == "top_result_visibility" and score == "Missing" and not products:
                continue
        value = bank.get(dimension, {}).get(score)
        if not value:
            continue
        add(
            value,
            dimension,
            score,
            [score],
            f"Score {score} for {dimension.replace('_', ' ')} was selected from the controlled bullet bank.",
            f"{side}_{dimension}",
        )
        seen_dimensions.add(dimension)
        if len(bullets) >= 4:
            break
    while len(bullets) < 4:
        fallback_text = (
            f"Clarify benefit cues across {phrase_context['shopping_journey']}"
            if side == "current"
            else f"Wider {phrase_context['product_type']} cues support discovery"
        )
        add(
            fallback_text,
            "fallback",
            "Unknown",
            ["controlled_fallback"],
            "Fallback bullet added to meet the four-bullet requirement.",
            f"{side}_fallback",
        )
    return bullets[:4], bullet_debug[:4]


def _product_position(product: dict[str, Any], fallback: int) -> int:
    raw = product.get("position") or product.get("rank") or product.get("index") or fallback
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _top_brands(products: list[dict[str, Any]]) -> list[str]:
    brands: list[str] = []
    for product in products:
        brand = _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName"))
        if brand:
            brands.append(brand)
    return list(dict.fromkeys(brands))[:6]


def _badges(products: list[dict[str, Any]]) -> list[str]:
    badges: list[str] = []
    for product in products:
        for value in _coerce_list(product.get("badges") or product.get("badge") or product.get("badgeText")):
            text = _safe_text(value.get("text") if isinstance(value, dict) else value)
            if text:
                badges.append(text)
    return list(dict.fromkeys(badges))[:8]


def _review_counts(products: list[dict[str, Any]]) -> list[int]:
    counts: list[int] = []
    for product in products:
        try:
            counts.append(int(product.get("reviewCount") or product.get("reviews") or product.get("ratingCount")))
        except (TypeError, ValueError):
            continue
    return counts


def _build_side_payload(
    record: dict[str, Any] | None,
    side: str,
    client_name: str,
    search_framework: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if record is None:
        return {
            "source_row": None,
            "search_term": "",
            "category_phrase": "",
            "screenshot": "",
            "bullets": [],
            "dimension_scores": {},
            "bullet_debug": [],
            "warnings": [],
        }

    search_term = _resolve_search_term(record)
    category_phrase = _category_phrase(search_term)
    screenshot = _safe_text(
        _get_first(
            record,
            "screenshotDataUrl",
            "screenshotDataURL",
            "data.screenshotDataUrl",
            "data.screenshotDataURL",
            "screenshot.dataUrl",
            "data.screenshot.dataUrl",
            "screenshot",
            default="",
        )
    )
    products = _main_products(record)
    scores = _dimension_scores(search_term, products, client_name, record)
    cue_context = search_cue_context(
        search_term,
        products,
        client_brand=client_name,
        side=side,
    )
    candidate_themes, side_debug = _build_side_candidates(side, scores, products, client_name, search_term, search_framework, record)
    selected_candidates, selection_debug = _select_side_candidates(candidate_themes, side=side)
    bullets = [candidate["text"] for candidate in selected_candidates[:4]]
    bullet_debug = [_candidate_to_debug(candidate) for candidate in selected_candidates[:4]]
    if len(bullets) != 4:
        bullets, bullet_debug = _select_bullets(side, scores, products, client_name, search_term)
    client_products = [
        {
            "position": _product_position(product, index),
            "title": _safe_text(product.get("title") or product.get("productTitle") or product.get("name")),
            "brand": _safe_text(product.get("brand") or product.get("productBrand") or product.get("brandName")),
        }
        for index, product in enumerate(products, start=1)
        if _client_brand_matches(client_name, product)
    ]
    return {
        "source_row": _source_row(record),
        "search_term": search_term,
        "category_phrase": category_phrase,
        "screenshot": screenshot,
        "bullets": bullets,
        "dimension_scores": scores,
        "bullet_debug": bullet_debug,
        "candidate_themes": candidate_themes,
        "selected_candidates": selected_candidates,
        "product_count": len(products),
        "client_brand": client_name,
        "client_products": client_products,
        "top_brands": _top_brands(products),
        "badges": _badges(products),
        "review_counts": _review_counts(products),
        "warnings": [],
        "strategic_cues": cue_context.get("debug", {}),
        "debug": {
            "side_built_separately": True,
            "side_candidate_debug": side_debug,
            "selection_debug": selection_debug,
            "strategic_cue_debug": cue_context.get("debug", {}),
            "shared_search_framework_used": side_debug.get("shared_search_framework_used", False),
        },
    }


def build_slide3_search_benchmark(
    search_evidence: Any,
    client_name: str = "",
    slide6_visibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(search_evidence, dict):
        current_records = list(search_evidence.get("current", []) or [])
        benchmark_records = list(search_evidence.get("benchmark", []) or [])
    else:
        current_records = []
        benchmark_records = []

    current_record, current_valid_records, current_warnings = _select_earliest_valid(current_records, "Current")
    benchmark_record, benchmark_valid_records, benchmark_warnings = _select_earliest_valid(benchmark_records, "Benchmark")
    search_framework = _search_framework_from_slide6(slide6_visibility)

    client_label = _safe_text(client_name or "")
    current_representative = (
        _representative_valid_record(current_record, current_valid_records, search_framework)
        if current_record is not None
        else None
    )
    benchmark_representative = (
        _representative_valid_record(benchmark_record, benchmark_valid_records, search_framework)
        if benchmark_record is not None
        else None
    )
    current_payload = _build_side_payload(current_representative, "current", client_label, search_framework)
    benchmark_payload = _build_side_payload(benchmark_representative, "benchmark", client_label, search_framework)
    overlap_debug = _apply_cross_side_overlap_suppression(current_payload, benchmark_payload)

    if current_record is None:
        current_payload["warnings"].append("Missing Current evidence; left side was left unchanged.")
    if benchmark_record is None:
        benchmark_payload["warnings"].append("Missing Benchmark evidence; right side was left unchanged.")
    current_payload["warnings"].extend(current_warnings)
    benchmark_payload["warnings"].extend(benchmark_warnings)

    intro = _dynamic_intro(current_payload, benchmark_payload)

    warnings = [*current_payload["warnings"], *benchmark_payload["warnings"]]
    if current_record is None and benchmark_record is None:
        warnings.append("Slide 3 could not be populated because both Current and Benchmark evidence were unavailable.")

    return {
        "current": current_payload,
        "benchmark": benchmark_payload,
        "intro": intro,
        "warnings": warnings,
        "debug": {
            "current_source_row": current_payload.get("source_row"),
            "benchmark_source_row": benchmark_payload.get("source_row"),
            "current_search_term": current_payload.get("search_term"),
            "benchmark_search_term": benchmark_payload.get("search_term"),
            "representative_evidence": {
                "current": (current_representative or {}).get("_slide3_representative_debug", {}),
                "benchmark": (benchmark_representative or {}).get("_slide3_representative_debug", {}),
            },
            "side_build_order": [
                "current_candidates_built",
                "benchmark_candidates_built",
                "cross_side_overlap_suppression_applied",
            ],
            "current_ranked_candidate_themes": (
                current_payload.get("debug", {})
                .get("side_candidate_debug", {})
                .get("ranked_candidate_themes", [])
            ),
            "benchmark_ranked_candidate_themes": (
                benchmark_payload.get("debug", {})
                .get("side_candidate_debug", {})
                .get("ranked_candidate_themes", [])
            ),
            "overlap_suppression": overlap_debug,
            "shared_search_framework": search_framework,
        },
    }
