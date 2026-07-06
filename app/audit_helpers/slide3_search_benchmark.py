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
    if not any(term in normalized for term in ("query", "queries", "shelf", "review", "brand", "position", "coverage", "presence", "badge", "sponsored", "authority", "alignment", "breadth", "adjacent")):
        return False, "not_search_native"
    return True, "allowed"


def _score_rank(score: str) -> int:
    return {"Strong": 4, "Moderate": 3, "Limited": 2, "Unknown": 1, "Missing": 0}.get(score, 1)


def _candidate_overlap_key(text: str) -> set[str]:
    stop = STOP_WORDS | {
        "current",
        "client",
        "benchmark",
        "competitive",
        "stronger",
        "broader",
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
) -> None:
    fitted = _fit_search_bullet(text)
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
        }
    )


def _build_side_candidates(
    side: str,
    scores: dict[str, str],
    products: list[dict[str, Any]],
    client_brand: str,
    search_term: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    brands = _top_brands(products)
    badges = _badges(products)
    review_counts = _review_counts(products)
    client_positions = [
        _product_position(product, index)
        for index, product in enumerate(products, start=1)
        if _client_brand_matches(client_brand, product)
    ]
    median_reviews = int(median(review_counts)) if review_counts else 0
    max_reviews = max(review_counts) if review_counts else 0
    phrase_context = _search_phrase_context(search_term)
    product_type = phrase_context["product_type"]
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
    assortment_score = scores.get("assortment_breadth", "Limited")
    review_score = scores.get("review_authority", "Missing")
    sponsored_score = scores.get("sponsored_competition", "Unknown")

    if side == "current":
        if client_positions and client_brand:
            add(
                text=f"{client_brand} appears near position {min(client_positions)} for {product_type}",
                family="side_specific",
                dimension="client_brand_presence",
                score=scores.get("client_brand_presence", "Unknown"),
                rank=96,
                evidence=[f"client_position={min(client_positions)}", f"query={search_term}"],
                reason="Current side prioritized actual client placement in captured results.",
            )
        elif client_brand:
            add(
                text=f"{client_brand} presence is narrow for {product_type}",
                family="side_specific",
                dimension="client_brand_presence",
                score="Missing",
                rank=92,
                evidence=["client_not_found", f"query={search_term}"],
                reason="Current side prioritized missing client presence in leading results.",
            )
        add(
            text=(
                f"Current titles align with {product_type} queries"
                if keyword_score in {"Strong", "Moderate"}
                else f"Current query alignment is thin for {product_type}"
            ),
            family="query_alignment",
            dimension="keyword_alignment",
            score=keyword_score,
            rank=88,
            evidence=[keyword_score, f"query={search_term}"],
            reason="Current side evaluated title/query fit against captured products.",
        )
        add(
            text=(
                f"Current shelf breadth spans {len(brands)} visible brands"
                if len(brands) >= 2
                else f"Current shelf breadth is narrow for {product_type}"
            ),
            family="shelf_breadth",
            dimension="assortment_breadth",
            score=assortment_score,
            rank=82,
            evidence=[f"brands={len(brands)}", *brands[:3]],
            reason="Current side used captured brand count as shelf-breadth evidence.",
        )
        add(
            text=(
                f"Visible review threshold is near {median_reviews}"
                if median_reviews
                else f"Review authority is limited on current shelf"
            ),
            family="trust_authority",
            dimension="review_authority",
            score=review_score,
            rank=78,
            evidence=[f"median_reviews={median_reviews}"] if median_reviews else ["review_counts_missing"],
            reason="Current side used median review count as trust evidence.",
        )
        if badges:
            add(
                text=f"{badges[0]} badge shapes current shelf attention",
                family="side_specific",
                dimension="badge_promotional_visibility",
                score=scores.get("badge_promotional_visibility", "Limited"),
                rank=74,
                evidence=badges[:3],
                reason="Current side kept promotional badge evidence as a distinct shelf signal.",
            )
        if sponsored_score in {"Strong", "Moderate"}:
            add(
                text=f"Sponsored pressure competes with current shelf presence",
                family="side_specific",
                dimension="sponsored_competition",
                score=sponsored_score,
                rank=70,
                evidence=[sponsored_score],
                reason="Current side identified paid placement pressure in captured results.",
            )
        add(
            text="Benefit-led coverage remains limited on current shelf",
            family="side_specific",
            dimension="benefit_intent_alignment",
            score=scores.get("benefit_intent_alignment", "Limited"),
            rank=66,
            evidence=[scores.get("benefit_intent_alignment", "Limited")],
            reason="Current side used benefit and adjacent-intent coverage as a distinct fallback signal.",
        )
    else:
        add(
            text=(
                f"Benchmark titles match {product_type} query intent"
                if keyword_score in {"Strong", "Moderate"}
                else f"Benchmark query fit varies for {product_type}"
            ),
            family="query_alignment",
            dimension="keyword_alignment",
            score=keyword_score,
            rank=90,
            evidence=[keyword_score, f"query={search_term}"],
            reason="Benchmark side evaluated competitor title/query fit separately.",
        )
        add(
            text=(
                f"Benchmark shelf spans {len(brands)} competing brands"
                if len(brands) >= 2
                else f"Benchmark shelf breadth is still concentrated"
            ),
            family="shelf_breadth",
            dimension="assortment_breadth",
            score=assortment_score,
            rank=88,
            evidence=[f"brands={len(brands)}", *brands[:3]],
            reason="Benchmark side prioritized competing brand breadth.",
        )
        add(
            text=(
                f"Higher review threshold reaches {max_reviews}"
                if max_reviews >= 100
                else f"Benchmark review authority remains uneven"
            ),
            family="trust_authority",
            dimension="review_authority",
            score=review_score,
            rank=84,
            evidence=[f"max_reviews={max_reviews}", f"median_reviews={median_reviews}"] if review_counts else ["review_counts_missing"],
            reason="Benchmark side used review ceiling and median to describe authority.",
        )
        if sponsored_score in {"Strong", "Moderate", "Unknown"}:
            add(
                text=f"Sponsored pressure is visible on benchmark shelf",
                family="side_specific",
                dimension="sponsored_competition",
                score=sponsored_score,
                rank=76,
                evidence=[sponsored_score],
                reason="Benchmark side kept sponsored pressure as a distinct competitive signal.",
            )
        if badges:
            add(
                text=f"Retail badges sharpen benchmark shelf choice",
                family="side_specific",
                dimension="badge_promotional_visibility",
                score=scores.get("badge_promotional_visibility", "Limited"),
                rank=72,
                evidence=badges[:3],
                reason="Benchmark side used retail badge evidence for shelf differentiation.",
            )
        add(
            text="Benefit-led coverage broadens benchmark results",
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
        "ranked_candidate_themes": [
            {
                "text": item["text"],
                "family": item["family"],
                "dimension": item["dimension"],
                "rank": item["rank"],
                "evidence": item["evidence"],
                "reason": item["reason"],
            }
            for item in ranked[:12]
        ],
        "rejected_candidates": rejected,
    }


def _select_side_candidates(candidates: list[dict[str, Any]], used_texts: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    used = set(used_texts or set())
    rejected: list[dict[str, str]] = []

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

    for family in SEARCH_BULLET_FAMILIES:
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
    cue_context = search_cue_context(
        search_term,
        products,
        client_brand=client_name,
        side=side,
    )
    candidate_themes, side_debug = _build_side_candidates(side, scores, products, client_name, search_term)
    selected_candidates, selection_debug = _select_side_candidates(candidate_themes)
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
        },
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
    overlap_debug = _apply_cross_side_overlap_suppression(current_payload, benchmark_payload)

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
        },
    }
