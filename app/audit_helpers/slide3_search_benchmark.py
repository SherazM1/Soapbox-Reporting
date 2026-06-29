from __future__ import annotations

import re
from statistics import median
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlparse

from app.audit_helpers.bullet_uniqueness import make_unique_bullet_text


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
    "assortment_breadth": {
        "Strong": "Client assortment supports broader comparison",
        "Moderate": "Focused assortment supports core discovery",
        "Limited": "Assortment representation is narrow",
        "Missing": "Competitors show broader product-type coverage",
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
        "Missing": "Limited review evidence across visible Client products",
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
        "Strong": "Benchmark titles mirror {product_type} intent",
        "Moderate": "Mixed alignment with shopper search language",
        "Limited": "Limited benefit-led keyword differentiation",
    },
    "benefit_intent_alignment": {
        "Strong": "Benefit-forward product positioning",
        "Moderate": "Broader intent-based discoverability",
        "Limited": "Benefit and use-case language broadens relevance",
    },
    "assortment_breadth": {
        "Strong": "Assortment breadth expands {discovery_context}",
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
        "Unknown": "Sponsored competition shapes the search shelf",
    },
}

DIMENSION_ORDER = [
    "client_brand_presence",
    "top_result_visibility",
    "sponsored_competition",
    "keyword_alignment",
    "assortment_breadth",
    "benefit_intent_alignment",
    "review_authority",
    "badge_promotional_visibility",
]

BENCHMARK_DIMENSION_ORDER = [
    "keyword_alignment",
    "benefit_intent_alignment",
    "assortment_breadth",
    "review_authority",
    "badge_promotional_visibility",
    "sponsored_competition",
]

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


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


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
        f"{role} evidence included multiple valid captures; selected source row {_source_row(selected)} as the earliest valid source row."
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


def _assortment_breadth(products: list[dict[str, Any]]) -> str:
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


def _dimension_scores(search_term: str, products: list[dict[str, Any]], client_brand: str) -> dict[str, str]:
    client_positions = [int(product.get("position", 999)) for product in products if _client_brand_matches(client_brand, product) and _safe_text(product.get("position"))]
    if not client_brand:
        client_presence = "Unknown"
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
        "top_result_visibility": _top_result_visibility(client_positions),
        "sponsored_competition": _sponsored_competition(products, client_positions),
        "keyword_alignment": _keyword_alignment(search_term, products),
        "assortment_breadth": _assortment_breadth(products),
        "benefit_intent_alignment": _benefit_alignment(products),
        "review_authority": _review_authority(products, client_positions),
        "badge_promotional_visibility": _badge_visibility(products),
    }
    if client_presence == "Unknown":
        scores["client_brand_presence"] = "Unknown"
    return scores


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
        unique_text, changed = make_unique_bullet_text(fit(rendered), used_texts, fallback_subject=f"{side} search")
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
                f"{client_brand} appears near position {min(client_positions)} in {phrase_context['shopping_context']}",
                "client_brand_presence",
                scores.get("client_brand_presence", "Unknown"),
                [f"client_position={min(client_positions)}"],
                "Selected because the client brand was found in captured search products.",
                "current_client_position",
            )
        elif client_brand:
            add(
                f"{client_brand} is not visible in leading {phrase_context['shelf_navigation']}",
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
                f"Median review count near {median_reviews} sets trust context",
                "review_authority",
                scores.get("review_authority", "Limited"),
                [f"median_reviews={median_reviews}"],
                "Selected because review counts were captured in search results.",
                "current_review_context",
            )
    else:
        if search_brand_phrase:
            add(
                f"{search_brand_phrase} broaden {phrase_context['discovery_context']}",
                "assortment_breadth",
                scores.get("assortment_breadth", "Limited"),
                brands[:3],
                "Selected because multiple benchmark brands were captured.",
                "benchmark_top_brands",
            )
        if median_reviews:
            add(
                f"Review counts near {median_reviews} vary benchmark authority",
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
            else f"Benchmark results broaden {phrase_context['product_type']} cues"
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


def _build_side_payload(record: dict[str, Any] | None, side: str, client_name: str) -> dict[str, Any]:
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
    scores = _dimension_scores(search_term, products, client_name)
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
        "product_count": len(products),
        "client_brand": client_name,
        "client_products": client_products,
        "top_brands": _top_brands(products),
        "badges": _badges(products),
        "review_counts": _review_counts(products),
        "warnings": [],
    }


def build_slide3_search_benchmark(search_evidence: Any, client_name: str = "") -> dict[str, Any]:
    if isinstance(search_evidence, dict):
        current_records = list(search_evidence.get("current", []) or [])
        benchmark_records = list(search_evidence.get("benchmark", []) or [])
    else:
        current_records = []
        benchmark_records = []

    current_record, _, current_warnings = _select_earliest_valid(current_records, "Current")
    benchmark_record, _, benchmark_warnings = _select_earliest_valid(benchmark_records, "Benchmark")

    client_label = _safe_text(client_name or "")
    current_payload = _build_side_payload(current_record, "current", client_label)
    benchmark_payload = _build_side_payload(benchmark_record, "benchmark", client_label)

    if current_record is None:
        current_payload["warnings"].append("Missing Current evidence; left side was left unchanged.")
    if benchmark_record is None:
        benchmark_payload["warnings"].append("Missing Benchmark evidence; right side was left unchanged.")
    current_payload["warnings"].extend(current_warnings)
    benchmark_payload["warnings"].extend(benchmark_warnings)

    intro = (
        "Walmart search results within the "
        f"{current_payload['category_phrase'] or 'the category'} and "
        f"{benchmark_payload['category_phrase'] or 'the category'} categories reveal a highly competitive environment where brands are increasingly using educational content, benefit-led messaging, and lifestyle positioning to drive shopper engagement and conversion."
    )

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
        },
    }
