from __future__ import annotations

import re
from typing import Any

from app.audit_helpers.bullet_uniqueness import dedupe_bullet_debug, normalize_bullet_text
from app.audit_helpers.strategic_cues import brand_shop_cue_context, translate_cues


DIMENSION_PRIORITY = (
    "brand_presentation",
    "lifestyle_merchandising",
    "category_segmentation",
    "product_discovery",
    "educational_storytelling",
    "video_rich_media",
    "cross_category_navigation",
)
SCORE_ORDER = {"Missing": 0, "Limited": 1, "Present": 2, "Strong": 3}
VALID_STATUSES = {"success", "successful", "partial", "partially successful", "partially_successful"}
IMPORTANCE_BULLETS = {
    "brand_presentation": (
        "importance_brand_01",
        "Dedicated brand experiences strengthen shopper education",
    ),
    "lifestyle_merchandising": (
        "importance_lifestyle_01",
        "Rich merchandising creates a more immersive shopping journey",
    ),
    "category_segmentation": (
        "importance_category_01",
        "Brand Shops create centralized discovery destinations",
    ),
    "product_discovery": (
        "importance_product_01",
        "Brand-led navigation connects products with shopper needs",
    ),
    "educational_storytelling": (
        "importance_education_01",
        "Dedicated brand experiences strengthen shopper education",
    ),
    "video_rich_media": (
        "importance_video_01",
        "Rich merchandising creates a more immersive shopping journey",
    ),
    "cross_category_navigation": (
        "importance_cross_01",
        "Cross-category pathways support broader brand exploration",
    ),
}
CLIENT_OPPORTUNITY_BULLETS = {
    "brand_presentation": (
        "client_opportunity_brand_01",
        "Opportunity to establish a dedicated Walmart brand experience",
    ),
    "lifestyle_merchandising": (
        "client_opportunity_lifestyle_01",
        "Opportunity to build a more immersive Walmart destination",
    ),
    "category_segmentation": (
        "client_opportunity_category_01",
        "Opportunity to create category-led brand navigation",
    ),
    "product_discovery": (
        "client_opportunity_product_01",
        "Opportunity to centralize product discovery and education",
    ),
    "educational_storytelling": (
        "client_opportunity_education_01",
        "Opportunity to centralize product discovery and education",
    ),
    "video_rich_media": (
        "client_opportunity_video_01",
        "Opportunity to build a more immersive Walmart destination",
    ),
    "cross_category_navigation": (
        "client_opportunity_cross_01",
        "Opportunity to create category-led brand navigation",
    ),
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _safe_text(value).lower()).strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
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
    raw = _first(
        record,
        "sourceRow",
        "rowNumber",
        "data.sourceRow",
        "data.rowNumber",
        "_combined_source_index",
        default=10**9,
    )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 10**9


def _normalize_role_value(value: Any) -> str:
    role = _safe_text(value).lower()
    if role == "current":
        return "client"
    if role == "benchmark":
        return "competitor"
    return role


def _explicit_role(record: dict[str, Any]) -> str:
    role = _safe_text(
        _first(
            record,
            "role",
            "inputRole",
            "sourceRole",
            "originalRole",
            "input.role",
            "source.role",
            "data.role",
        )
    )
    return _normalize_role_value(role)


def _original_role(record: dict[str, Any]) -> str:
    return _normalize_role_value(
        _first(record, "originalRole", "data.originalRole", "source.originalRole")
    )


def _record_brand(record: dict[str, Any]) -> str:
    return _safe_text(
        _first(
            record,
            "inputBrandName",
            "brandName",
            "extractedBrandName",
            "brand",
            "data.inputBrandName",
            "data.brandName",
            "data.extractedBrandName",
            "data.brand",
        )
    )


def _resolve_role(
    record: dict[str, Any],
    *,
    side_fallback: str,
    client_name: str = "",
) -> tuple[str, str]:
    explicit = _explicit_role(record)
    if explicit in {"client", "competitor"}:
        return explicit, "explicit role"
    original = _original_role(record)
    if original in {"client", "competitor"}:
        return original, "originalRole"
    brand = _normalize(_record_brand(record))
    client = _normalize(client_name)
    if brand and client and (brand in client or client in brand):
        return "client", "brand/client-name matching"
    return side_fallback, "side fallback"


def _screenshot(record: dict[str, Any]) -> str:
    return _safe_text(
        _first(
            record,
            "fullPageScreenshotDataUrl",
            "fullPageScreenshotDataURL",
            "data.fullPageScreenshotDataUrl",
            "data.fullPageScreenshotDataURL",
            "screenshot.fullPageDataUrl",
            "screenshot.fullPageDataURL",
            "data.screenshot.fullPageDataUrl",
            "data.screenshot.fullPageDataURL",
            "screenshotDataUrl",
            "screenshotDataURL",
            "data.screenshotDataUrl",
            "data.screenshotDataURL",
            "screenshot.dataUrl",
            "screenshot.dataURL",
            "data.screenshot.dataUrl",
            "data.screenshot.dataURL",
            "screenshot",
            "data.screenshot",
        )
    )


def _modules(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _first(
        record,
        "modules",
        "structuredModules",
        "moduleDetails",
        "data.modules",
        "data.structuredModules",
        "data.moduleDetails",
        default=[],
    )
    return [item for item in _as_list(raw) if isinstance(item, dict)]


def _status(record: dict[str, Any]) -> str:
    return _normalize(_first(record, "extractionStatus", "status", "data.extractionStatus", "data.status"))


def _is_valid_capture(record: dict[str, Any], expected_role: str) -> bool:
    return (
        _resolve_role(record, side_fallback=expected_role.lower())[0] == expected_role.lower()
        and _status(record) in VALID_STATUSES
        and _screenshot(record).lower().startswith("data:image/")
        and bool(_modules(record))
    )


def _record_texts(record: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = _first(record, key, default=[])
        for item in _as_list(raw):
            if isinstance(item, dict):
                text = _safe_text(
                    _first(item, "text", "title", "heading", "label", "description", "name")
                )
            else:
                text = _safe_text(item)
            if text:
                values.append(text)
    return list(dict.fromkeys(values))


def _module_type(module: dict[str, Any]) -> str:
    return _safe_text(_first(module, "type", "moduleType", "name"))


def _evidence(record: dict[str, Any]) -> dict[str, Any]:
    modules = _modules(record)
    module_types = [_module_type(module) for module in modules if _module_type(module)]
    explicit_module_types = _as_list(_first(record, "moduleTypes", "data.moduleTypes", default=[]))
    module_types = list(dict.fromkeys([*module_types, *[_safe_text(value) for value in explicit_module_types if _safe_text(value)]]))
    normalized_types = [_normalize(value) for value in module_types]
    headings = _record_texts(
        record,
        "editorialHeadings",
        "headings",
        "titles",
        "data.editorialHeadings",
        "data.headings",
        "data.titles",
    )
    descriptions = _record_texts(
        record,
        "descriptions",
        "editorialDescriptions",
        "promotionalCopy",
        "copy",
        "data.descriptions",
        "data.editorialDescriptions",
        "data.promotionalCopy",
        "data.copy",
    )
    module_headings = [
        text
        for module in modules
        for text in _record_texts(module, "heading", "title", "description", "copy")
    ]
    categories = _record_texts(
        record,
        "categoryNavigation",
        "categoryNavigationItems",
        "categories",
        "data.categoryNavigation",
        "data.categoryNavigationItems",
        "data.categories",
    )
    destination_links = _as_list(
        _first(
            record,
            "destinationLinks",
            "links",
            "navigationLinks",
            "data.destinationLinks",
            "data.links",
            "data.navigationLinks",
            default=[],
        )
    )
    normalized_links: list[dict[str, str]] = []
    for link in destination_links:
        if isinstance(link, dict):
            label = _safe_text(_first(link, "label", "text", "title", "name"))
            url = _safe_text(_first(link, "url", "href", "destination"))
        else:
            label = _safe_text(link)
            url = _safe_text(link)
        if label or url:
            normalized_links.append({"label": label, "url": url})
    videos = _record_texts(record, "videoTitles", "videos", "data.videoTitles", "data.videos")
    for module in modules:
        if "video" in _normalize(_module_type(module)):
            title = _safe_text(_first(module, "title", "heading", "name"))
            if title:
                videos.append(title)
    product_count = _first(record, "productCount", "data.productCount", default=None)
    if product_count in (None, ""):
        products = _as_list(_first(record, "products", "productTiles", "data.products", "data.productTiles", default=[]))
        product_count = len(products)
    try:
        product_count = int(product_count or 0)
    except (TypeError, ValueError):
        product_count = 0
    video_present = bool(_first(record, "videoPresent", "hasVideo", "data.videoPresent", "data.hasVideo", default=False))
    return {
        "modules": modules,
        "module_count": int(_first(record, "moduleCount", "data.moduleCount", default=len(modules)) or len(modules)),
        "module_types": module_types,
        "normalized_types": normalized_types,
        "headings": list(dict.fromkeys([*headings, *module_headings])),
        "descriptions": descriptions,
        "categories": categories,
        "links": normalized_links,
        "videos": list(dict.fromkeys(videos)),
        "video_present": video_present,
        "product_count": product_count,
    }


def _contains_type(evidence: dict[str, Any], *terms: str) -> list[str]:
    matches: list[str] = []
    for raw, normalized in zip(evidence["module_types"], evidence["normalized_types"]):
        if any(_normalize(term) in normalized for term in terms):
            matches.append(raw)
    return list(dict.fromkeys(matches))


def _score_dimensions(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hero = _contains_type(evidence, "hero pov", "heropov", "skinny banner", "skinnybanner", "hero", "banner")
    editorial = _contains_type(evidence, "pov", "editorial", "story", "lifestyle", "content card")
    product_modules = _contains_type(evidence, "item carousel", "itemcarousel", "product carousel", "product grid", "product")
    video_modules = _contains_type(evidence, "video player", "videoplayer", "video")
    rich_modules = _contains_type(evidence, "pov", "editorial", "hero", "banner", "content card")
    headings = evidence["headings"]
    descriptions = evidence["descriptions"]
    categories = evidence["categories"]
    links = evidence["links"]
    product_count = int(evidence["product_count"])
    educational_terms = {
        "benefit",
        "routine",
        "ingredient",
        "usage",
        "use",
        "solution",
        "hydration",
        "moisture",
        "care",
        "guide",
        "learn",
        "how",
    }
    educational_text = []
    for text in [*headings, *descriptions]:
        tokens = set(_normalize(text).split())
        if len(tokens) >= 3 and tokens & educational_terms:
            educational_text.append(text)
    category_destinations = list(dict.fromkeys(_normalize(value) for value in categories if _normalize(value)))
    link_destinations = list(
        dict.fromkeys(
            _normalize(link.get("label") or link.get("url"))
            for link in links
            if _normalize(link.get("label") or link.get("url"))
        )
    )

    def result(score: str, signals: list[str], reason: str) -> dict[str, Any]:
        return {
            "score": score,
            "signals": list(dict.fromkeys(signal for signal in signals if signal)),
            "supporting_count": len(list(dict.fromkeys(signal for signal in signals if signal))),
            "reason": reason,
        }

    branded_copy_count = len(headings) + len(descriptions)
    if hero and (branded_copy_count or len(rich_modules) >= 2):
        brand = result("Strong", [*hero, *headings[:2]], "A hero/banner and supporting branded content were detected.")
    elif hero:
        brand = result("Present", hero, "One meaningful hero or banner module was detected.")
    elif evidence["modules"]:
        brand = result("Limited", evidence["module_types"][:2], "The page is structured but lacks reliable hero/banner evidence.")
    else:
        brand = result("Missing", [], "No reliable branded hero or banner evidence was detected.")

    if len(editorial) >= 2 or (editorial and hero and headings):
        lifestyle = result("Strong", [*editorial, *headings[:2]], "Multiple lifestyle or editorial storytelling signals were detected.")
    elif editorial:
        lifestyle = result("Present", editorial, "One meaningful lifestyle or editorial module was detected.")
    elif rich_modules:
        lifestyle = result("Limited", rich_modules[:2], "Static rich content exists with limited lifestyle story support.")
    else:
        lifestyle = result("Missing", [], "No reliable lifestyle merchandising evidence was detected.")

    category_count = len(category_destinations)
    if category_count >= 4:
        category = result("Strong", categories[:4], f"{category_count} distinct category destinations were detected.")
    elif category_count >= 2:
        category = result("Present", categories[:3], f"{category_count} category destinations were detected.")
    elif category_count == 1:
        category = result("Limited", categories[:1], "Only one category destination was detected.")
    else:
        category = result("Missing", [], "No category-navigation evidence was detected.")

    if len(product_modules) >= 2 or (product_modules and product_count >= 8):
        product = result("Strong", [*product_modules, f"{product_count} products"], "Multiple product pathways or a substantial product module were detected.")
    elif product_modules:
        product = result("Present", product_modules, "One meaningful product module was detected.")
    elif product_count > 0:
        product = result("Limited", [f"{product_count} products"], "Products are present without a clear discovery module.")
    else:
        product = result("Missing", [], "No reliable product-discovery evidence was detected.")

    if len(educational_text) >= 3:
        education = result("Strong", educational_text[:4], "Multiple headings or descriptions provide shopper education.")
    elif educational_text:
        education = result("Present", educational_text[:2], "One meaningful educational or editorial section was detected.")
    elif descriptions:
        education = result("Limited", descriptions[:2], "Promotional copy exists with limited educational depth.")
    else:
        education = result("Missing", [], "No educational content evidence was detected.")

    if (video_modules or evidence["video_present"]) and rich_modules:
        video = result(
            "Strong",
            [*video_modules, "video present" if evidence["video_present"] else "", *rich_modules[:2]],
            "Video and another rich editorial module were detected.",
        )
    elif video_modules or evidence["videos"] or evidence["video_present"]:
        video = result(
            "Present",
            [*video_modules, *evidence["videos"][:2], "video present" if evidence["video_present"] else ""],
            "Video content was detected.",
        )
    elif rich_modules:
        video = result("Limited", rich_modules[:2], "Static rich content exists without video.")
    else:
        video = result("Missing", [], "No reliable rich-media evidence was detected.")

    link_count = len(link_destinations)
    category_like_count = len(set([*category_destinations, *link_destinations]))
    if category_like_count >= 4 and link_count >= 2:
        cross = result("Strong", [*categories[:3], *(link["label"] or link["url"] for link in links[:3])], "Multiple links span distinct category or need-state pathways.")
    elif link_count >= 2:
        cross = result("Present", [(link["label"] or link["url"]) for link in links[:3]], "More than one destination pathway was detected.")
    elif link_count == 1:
        cross = result("Limited", [(links[0]["label"] or links[0]["url"])], "Navigation is limited to one destination pathway.")
    else:
        cross = result("Missing", [], "No meaningful cross-category navigation evidence was detected.")

    return {
        "brand_presentation": brand,
        "lifestyle_merchandising": lifestyle,
        "category_segmentation": category,
        "product_discovery": product,
        "educational_storytelling": education,
        "video_rich_media": video,
        "cross_category_navigation": cross,
    }


def _short_phrase(values: list[str], max_items: int = 3) -> str:
    cleaned: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", _safe_text(value)).strip(" -|")
        if text and len(text) <= 28 and _normalize(text) not in {_normalize(item) for item in cleaned}:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{cleaned[0]}, {cleaned[1]}, and {cleaned[2]}"


def _category_context(categories: list[str]) -> str:
    phrase = _short_phrase(categories, 2)
    if not phrase:
        return "category discovery"
    return f"{phrase} discovery"


def _navigation_context(categories: list[str]) -> str:
    phrase = _short_phrase(categories, 2)
    if not phrase:
        return "shop navigation"
    return f"{phrase} navigation"


def _fit_bullet(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 11:
        text = " ".join(words[:11]).rstrip(" ,;-")
    if len(text) <= 78:
        return text
    return text[:75].rsplit(" ", 1)[0].rstrip(" ,;-") + "..."


def _mirrored_bullet_key(text: str) -> str:
    normalized = normalize_bullet_text(text)
    tokens = [
        token
        for token in normalized.split()
        if token
        and token
        not in {
            "client",
            "competitor",
            "brand",
            "shop",
            "shops",
            "walmart",
            "supports",
            "support",
            "shopper",
            "shoppers",
        }
    ]
    return " ".join(tokens)


def _bullet_token_set(text: str) -> set[str]:
    return {
        token
        for token in _mirrored_bullet_key(text).split()
        if len(token) > 3
    }


def _has_mirrored_structure(text: str, client_token_sets: list[set[str]]) -> bool:
    candidate = _bullet_token_set(text)
    if not candidate:
        return False
    for client_tokens in client_token_sets:
        if not client_tokens:
            continue
        overlap = len(candidate & client_tokens)
        smaller = max(1, min(len(candidate), len(client_tokens)))
        if overlap / smaller >= 0.45:
            return True
    return False


def _alternate_competitor_bullet(
    item: dict[str, Any],
    *,
    brand_name: str,
    evidence: dict[str, Any],
) -> str:
    brand = brand_name or "Competitor"
    categories = _short_phrase(evidence.get("categories", []) or [], 2)
    topic = _short_phrase([*evidence.get("headings", []), *evidence.get("descriptions", [])], 1)
    video_title = _short_phrase(evidence.get("videos", []) or [], 1)
    module_count = int(evidence.get("module_count", 0) or 0)
    product_count = int(evidence.get("product_count", 0) or 0)
    dimension = item.get("dimension")
    alternatives = {
        "brand_presentation": f"{brand} anchors the benchmark with {module_count or 'multiple'} branded sections",
        "lifestyle_merchandising": f"{brand} frames shopping occasions through {topic or 'editorial'} content",
        "category_segmentation": f"{brand} turns {_navigation_context(evidence.get('categories', []) or [])} into entry points",
        "product_discovery": f"{brand} gives shoppers {product_count or 'multiple'} discovery paths",
        "educational_storytelling": f"{brand} uses {topic or 'editorial copy'} for shopper education",
        "video_rich_media": f"{brand} adds rich-media depth through {video_title or 'video content'}",
        "cross_category_navigation": f"{brand} links {_category_context(evidence.get('categories', []) or [])} paths",
    }
    return _fit_bullet(alternatives.get(dimension, f"{brand} evidence creates a distinct benchmark cue"))


def _reduce_cross_side_mirroring(
    client_side: dict[str, Any] | None,
    competitor_side: dict[str, Any] | None,
) -> None:
    if not isinstance(client_side, dict) or not isinstance(competitor_side, dict):
        return
    client_keys = {
        _mirrored_bullet_key(text)
        for text in client_side.get("bullets", []) or []
        if _mirrored_bullet_key(text)
    }
    client_token_sets = [
        _bullet_token_set(text)
        for text in client_side.get("bullets", []) or []
    ]
    used = set(client_keys)
    rewritten: list[str] = []
    competitor_debug = list(competitor_side.get("bullet_debug", []) or [])
    evidence = {
        "module_count": competitor_side.get("evidence", {}).get("module_count", 0),
        "categories": competitor_side.get("evidence", {}).get("categories", []),
        "headings": competitor_side.get("evidence", {}).get("headings", []),
        "descriptions": [],
        "videos": competitor_side.get("evidence", {}).get("video_titles", []),
        "product_count": competitor_side.get("evidence", {}).get("product_count", 0),
    }
    for index, item in enumerate(competitor_debug):
        text = _safe_text(item.get("text"))
        key = _mirrored_bullet_key(text)
        if key and (key in used or _has_mirrored_structure(text, client_token_sets)):
            text = _alternate_competitor_bullet(
                item,
                brand_name=competitor_side.get("brand_name", ""),
                evidence=evidence,
            )
            item["text"] = text
            item["signals"] = list(item.get("signals", []) or []) + ["cross_side_mirror_reworded"]
            item["reason"] = (
                _safe_text(item.get("reason"))
                + " Reworded to avoid mirroring the client Brand Shop bullets."
            ).strip()
        candidate_key = _mirrored_bullet_key(text)
        suffix = 2
        while candidate_key and candidate_key in used and suffix <= 6:
            marker = f"Benchmark {suffix}"
            text = _fit_bullet(f"{marker} {text}")
            item["text"] = text
            candidate_key = _mirrored_bullet_key(text)
            suffix += 1
        if candidate_key and candidate_key in used:
            text = _fit_bullet(f"Distinct benchmark cue {suffix}")
            item["text"] = text
            candidate_key = _mirrored_bullet_key(text)
        if candidate_key:
            used.add(candidate_key)
        rewritten.append(text)
        competitor_debug[index] = item
    competitor_side["bullet_debug"] = competitor_debug[:5]
    competitor_side["bullets"] = rewritten[:5]


def _bullet_for_dimension(
    *,
    side: str,
    bullet_type: str,
    dimension: str,
    dimension_data: dict[str, Any],
    brand_name: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    score = dimension_data["score"]
    categories = _short_phrase(evidence["categories"])
    category_context = _category_context(evidence["categories"])
    navigation_context = _navigation_context(evidence["categories"])
    topic = _short_phrase([*evidence["headings"], *evidence["descriptions"]], 1)
    video_title = _short_phrase(evidence["videos"], 1)
    module_count = int(evidence.get("module_count", 0) or len(evidence.get("module_types", []) or []))
    product_count = int(evidence.get("product_count", 0) or 0)
    module_phrase = _short_phrase(evidence.get("module_types", []) or [], 2)
    strength_templates = {
        "brand_presentation": (
            "brand_strength_01",
            f"{brand_name} uses {module_phrase} to establish shop context" if brand_name and module_phrase else "Branded sections establish shop context",
        ),
        "lifestyle_merchandising": (
            "lifestyle_strength_02",
            f"{topic} connects the shop to shopper occasions" if topic else "Lifestyle merchandising builds occasion context",
        ),
        "category_segmentation": (
            "category_strength_02",
            f"{navigation_context.title()} broadens discovery" if categories else "Clear pathways simplify shop exploration",
        ),
        "product_discovery": (
            "product_strength_01",
            f"{product_count} products create clearer discovery paths" if product_count else "Product pathways clarify shopper entry points",
        ),
        "educational_storytelling": (
            "education_strength_02",
            f"{topic} content deepens shopper education" if topic else "Education modules clarify product benefits",
        ),
        "video_rich_media": (
            "video_strength_02",
            f"{video_title} adds richer brand education" if video_title else "Rich media supports engaging product discovery",
        ),
        "cross_category_navigation": (
            "cross_strength_02",
            f"{category_context.title()} encourages broader exploration" if categories else "Multiple pathways encourage broader exploration",
        ),
    }
    opportunity_templates = {
        "brand_presentation": ("brand_opportunity_01", "Opening modules can carry more brand-storytelling work"),
        "lifestyle_merchandising": ("lifestyle_opportunity_01", "Lifestyle content can connect products to occasions"),
        "category_segmentation": ("category_opportunity_02", "Navigation can make varieties easier to compare"),
        "product_discovery": ("product_opportunity_01", "Discovery paths can surface more variants and use cases"),
        "educational_storytelling": ("education_opportunity_01", "Education can deepen benefit and usage storytelling"),
        "video_rich_media": ("video_opportunity_01", "Rich-media gaps leave room for deeper education"),
        "cross_category_navigation": ("cross_opportunity_01", "Cross-category paths can connect routines and basket ideas"),
    }
    restrained_benchmark_templates = {
        "brand_presentation": ("brand_benchmark_basic", f"{module_count} modules establish basic shop identity"),
        "lifestyle_merchandising": ("lifestyle_benchmark_basic", "Static editorial content provides light brand context"),
        "category_segmentation": ("category_benchmark_basic", "Category grouping creates a basic exploration path"),
        "product_discovery": ("product_benchmark_basic", "Product presentation gives shoppers a starting point"),
        "educational_storytelling": ("education_benchmark_basic", "Available copy gives shoppers basic product context"),
        "video_rich_media": ("video_benchmark_basic", "Static rich content supports product browsing"),
        "cross_category_navigation": ("cross_benchmark_basic", "Available links create a basic navigation path"),
    }
    if bullet_type == "opportunity":
        template_id, text = opportunity_templates[dimension]
    elif score in {"Limited", "Missing"}:
        template_id, text = restrained_benchmark_templates[dimension]
    else:
        template_id, text = strength_templates[dimension]
    return {
        "text": _fit_bullet(text),
        "side": side,
        "type": bullet_type,
        "dimension": dimension,
        "score": score,
        "template_id": template_id,
        "signals": list(dimension_data.get("signals", []) or []),
        "supporting_count": int(dimension_data.get("supporting_count", 0) or 0),
        "reason": dimension_data.get("reason", ""),
    }


def _client_bullets(
    dimensions: dict[str, dict[str, Any]],
    brand_name: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    priority_index = {dimension: index for index, dimension in enumerate(DIMENSION_PRIORITY)}
    strengths = sorted(
        DIMENSION_PRIORITY,
        key=lambda dimension: (
            -SCORE_ORDER[dimensions[dimension]["score"]],
            -int(dimensions[dimension]["supporting_count"]),
            priority_index[dimension],
        ),
    )[:3]
    remaining = [dimension for dimension in DIMENSION_PRIORITY if dimension not in strengths]
    opportunities = sorted(
        remaining,
        key=lambda dimension: (
            SCORE_ORDER[dimensions[dimension]["score"]],
            int(dimensions[dimension]["supporting_count"]),
            priority_index[dimension],
        ),
    )[:2]
    bullets = [
        *(
            _bullet_for_dimension(
                side="client",
                bullet_type="strength",
                dimension=dimension,
                dimension_data=dimensions[dimension],
                brand_name=brand_name,
                evidence=evidence,
            )
            for dimension in strengths
        ),
        *(
            _bullet_for_dimension(
                side="client",
                bullet_type="opportunity",
                dimension=dimension,
                dimension_data=dimensions[dimension],
                brand_name=brand_name,
                evidence=evidence,
            )
            for dimension in opportunities
        ),
    ]
    return dedupe_bullet_debug(bullets, fallback_subject="client brand shop")[0][:5]


def _competitor_bullets(
    dimensions: dict[str, dict[str, Any]],
    brand_name: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    priority_index = {dimension: index for index, dimension in enumerate(DIMENSION_PRIORITY)}
    selected = sorted(
        DIMENSION_PRIORITY,
        key=lambda dimension: (
            -SCORE_ORDER[dimensions[dimension]["score"]],
            -int(dimensions[dimension]["supporting_count"]),
            priority_index[dimension],
        ),
    )[:5]
    bullets = [
        _bullet_for_dimension(
            side="competitor",
            bullet_type="strength",
            dimension=dimension,
            dimension_data=dimensions[dimension],
            brand_name=brand_name,
            evidence=evidence,
        )
        for dimension in selected
    ]
    return dedupe_bullet_debug(bullets, fallback_subject="competitor brand shop")[0][:5]


def _no_brand_shop_bullets(
    dimensions: dict[str, dict[str, Any]],
    brand_name: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    priority_index = {dimension: index for index, dimension in enumerate(DIMENSION_PRIORITY)}
    supported = [
        dimension
        for dimension in DIMENSION_PRIORITY
        if int(dimensions[dimension].get("supporting_count", 0) or 0) > 0
    ]
    unsupported = [dimension for dimension in DIMENSION_PRIORITY if dimension not in supported]
    ranked = sorted(
        supported,
        key=lambda dimension: (
            -SCORE_ORDER[dimensions[dimension]["score"]],
            -int(dimensions[dimension]["supporting_count"]),
            priority_index[dimension],
        ),
    )
    if len(ranked) < 4:
        ranked.extend(
            sorted(
                unsupported,
                key=lambda dimension: (
                    -SCORE_ORDER[dimensions[dimension]["score"]],
                    priority_index[dimension],
                ),
            )
        )
    strength_dimensions = ranked[:3]
    strength_bullets = [
        _bullet_for_dimension(
            side="competitor",
            bullet_type="competitor_strength",
            dimension=dimension,
            dimension_data=dimensions[dimension],
            brand_name=brand_name,
            evidence=evidence,
        )
        for dimension in strength_dimensions
    ]
    strongest_dimension = strength_dimensions[0]
    importance_id, importance_text = IMPORTANCE_BULLETS[strongest_dimension]
    opportunity_dimension = next(
        (
            dimension
            for dimension in DIMENSION_PRIORITY
            if dimension not in strength_dimensions
        ),
        strongest_dimension,
    )
    opportunity_id, opportunity_text = CLIENT_OPPORTUNITY_BULLETS[opportunity_dimension]
    importance = {
        "text": _fit_bullet(importance_text),
        "side": "competitor",
        "type": "importance",
        "dimension": strongest_dimension,
        "score": dimensions[strongest_dimension]["score"],
        "template_id": importance_id,
        "signals": list(dimensions[strongest_dimension].get("signals", []) or []),
        "supporting_count": int(
            dimensions[strongest_dimension].get("supporting_count", 0) or 0
        ),
        "reason": (
            "Selected because this was the strongest evidence-supported competitor capability."
        ),
    }
    client_opportunity = {
        "text": _fit_bullet(opportunity_text),
        "side": "client",
        "type": "client_opportunity",
        "dimension": opportunity_dimension,
        "score": "Missing",
        "template_id": opportunity_id,
        "signals": [],
        "supporting_count": 0,
        "reason": (
            "Selected because the user confirmed the Client does not have a Walmart Brand Shop."
        ),
    }
    return dedupe_bullet_debug(
        [*strength_bullets, importance, client_opportunity],
        fallback_subject="no brand shop",
    )[0][:5]


def _brand_shop_cue_bullets(record: dict[str, Any], side: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = brand_shop_cue_context(record)
    if side == "client":
        for candidate in context.get("candidate_cues", []):
            if candidate.get("cue") in {
                "discovery_pathways",
                "shopper_education",
                "usage_storytelling",
                "conversion_guidance",
            }:
                candidate["classification"] = "opportunity"
        first_bullets, first_debug = translate_cues(
            context,
            slide_key="slide5",
            count=1,
            preferred_order=("strength", "context", "opportunity", "pressure"),
            side=side,
        )
        opportunity_bullets, opportunity_debug = translate_cues(
            context,
            slide_key="slide5",
            count=5,
            preferred_order=("opportunity", "context", "strength", "pressure"),
            side=side,
        )
        bullets = []
        debug = []
        used: set[str] = set()
        for text, item in [*zip(first_bullets, first_debug), *zip(opportunity_bullets, opportunity_debug)]:
            key = normalize_bullet_text(text)
            if key in used:
                continue
            used.add(key)
            bullets.append(text)
            debug.append(item)
            if len(bullets) >= 5:
                break
    else:
        bullets, debug = translate_cues(
            context,
            slide_key="slide5",
            count=5,
            preferred_order=("pressure", "strength", "opportunity", "context"),
            side=side,
        )
    return [
        {
            "text": item["text"],
            "side": side,
            "type": item.get("classification", "context"),
            "dimension": item.get("cue", "cue"),
            "score": item.get("classification", "context"),
            "template_id": f"cue_{item.get('cue')}",
            "signals": [item.get("classification", ""), item.get("cue", "")],
            "supporting_count": 0,
            "reason": item.get("reason", ""),
            "cue_debug": item,
        }
        for item in debug
    ], context.get("debug", {})


def _build_side(record: dict[str, Any], side: str, role_path: str) -> dict[str, Any]:
    evidence = _evidence(record)
    dimensions = _score_dimensions(evidence)
    brand_name = _record_brand(record)
    cue_bullets, cue_context_debug = _brand_shop_cue_bullets(record, side)
    fallback_bullets = (
        _client_bullets(dimensions, brand_name, evidence)
        if side == "client"
        else _competitor_bullets(dimensions, brand_name, evidence)
    )
    bullet_debug = cue_bullets if len(cue_bullets) == 5 else fallback_bullets
    warnings: list[str] = []
    if sum(SCORE_ORDER[data["score"]] >= 2 for data in dimensions.values()) < 2:
        warnings.append(
            "Brand Shop evidence is weak; restrained capability and opportunity language was used."
        )
    return {
        "source_row": _source_row(record),
        "brand_name": brand_name,
        "screenshot": _screenshot(record),
        "screenshot_source": "fullPageScreenshotDataUrl"
        if _safe_text(
            _first(
                record,
                "fullPageScreenshotDataUrl",
                "fullPageScreenshotDataURL",
                "data.fullPageScreenshotDataUrl",
                "data.fullPageScreenshotDataURL",
            )
        )
        else "screenshotDataUrl",
        "role_resolution_path": role_path,
        "bullets": [item["text"] for item in bullet_debug],
        "dimension_scores": {
            dimension: data["score"] for dimension, data in dimensions.items()
        },
        "dimension_debug": dimensions,
        "bullet_debug": bullet_debug,
        "strategic_cues": cue_context_debug,
        "warnings": warnings,
        "evidence": {
            "module_count": evidence["module_count"],
            "module_types": evidence["module_types"],
            "categories": evidence["categories"],
            "headings": evidence["headings"],
            "video_titles": evidence["videos"],
            "product_count": evidence["product_count"],
        },
    }


def build_slide5_brand_shop(
    client_evidence: list[dict[str, Any]] | None,
    competitor_evidence: list[dict[str, Any]] | None,
    *,
    client_has_brand_shop: bool = True,
    client_name: str = "",
) -> dict[str, Any]:
    role_debug: list[dict[str, Any]] = []
    classified = {"client": [], "competitor": []}
    for fallback, records in (
        ("client", client_evidence or []),
        ("competitor", competitor_evidence or []),
    ):
        for record in records:
            if not isinstance(record, dict):
                continue
            resolved_role, path = _resolve_role(
                record,
                side_fallback=fallback,
                client_name=client_name,
            )
            role_debug.append(
                {
                    "source_row": _source_row(record),
                    "brand_name": _record_brand(record),
                    "resolved_role": resolved_role,
                    "path": path,
                    "side_fallback": fallback,
                }
            )
            if resolved_role in classified:
                copied = dict(record)
                copied["_slide5_role_path"] = path
                classified[resolved_role].append(copied)
    clients = [
        record
        for record in classified["client"]
        if isinstance(record, dict) and _is_valid_capture(record, "Client")
    ]
    competitors = [
        record
        for record in classified["competitor"]
        if isinstance(record, dict) and _is_valid_capture(record, "Competitor")
    ]
    clients.sort(key=_source_row)
    competitors.sort(key=_source_row)
    warnings: list[str] = []
    if len(clients) > 1:
        warnings.append(
            f"Multiple valid Client Brand Shops were available; source row {_source_row(clients[0])} was selected."
        )
    if len(competitors) > 1:
        warnings.append(
            f"Multiple valid Competitor Brand Shops were available; source row {_source_row(competitors[0])} was selected."
        )
    if client_has_brand_shop and not clients:
        warnings.append("No valid Client Brand Shop evidence was available; the Client side will remain unchanged.")
    if not competitors:
        warnings.append("No valid Competitor Brand Shop evidence was available; the Competitor side will remain unchanged.")
    if not client_has_brand_shop and clients:
        warnings.append(
            "Client Brand Shop evidence was uploaded, but No Brand Shop mode was selected; "
            "the user selection was honored."
        )
    client_side = _build_side(clients[0], "client", clients[0].get("_slide5_role_path", "")) if clients and client_has_brand_shop else None
    competitor_side = _build_side(competitors[0], "competitor", competitors[0].get("_slide5_role_path", "")) if competitors else None
    if client_has_brand_shop:
        _reduce_cross_side_mirroring(client_side, competitor_side)
    if competitor_side and not client_has_brand_shop:
        no_brand_debug = _no_brand_shop_bullets(
            competitor_side["dimension_debug"],
            competitor_side["brand_name"],
            {
                "categories": competitor_side["evidence"]["categories"],
                "headings": competitor_side["evidence"]["headings"],
                "descriptions": [],
                "videos": competitor_side["evidence"]["video_titles"],
            },
        )
        competitor_side["bullets"] = [item["text"] for item in no_brand_debug]
        competitor_side["bullet_debug"] = no_brand_debug
    return {
        "mode": "standard" if client_has_brand_shop else "no_brand_shop",
        "client_has_brand_shop": bool(client_has_brand_shop),
        "client": client_side,
        "competitor": competitor_side,
        "warnings": warnings,
        "debug": {
            "valid_client_capture_count": len(clients),
            "valid_competitor_capture_count": len(competitors),
            "unused_client_source_rows": [_source_row(record) for record in clients[1:]],
            "unused_competitor_source_rows": [_source_row(record) for record in competitors[1:]],
            "role_resolution": role_debug,
            "screenshot_source_by_side": {
                "client": (client_side or {}).get("screenshot_source"),
                "competitor": (competitor_side or {}).get("screenshot_source"),
            },
            "module_evidence_by_side": {
                "client": (client_side or {}).get("evidence"),
                "competitor": (competitor_side or {}).get("evidence"),
            },
            "strategic_cues_by_side": {
                "client": (client_side or {}).get("strategic_cues"),
                "competitor": (competitor_side or {}).get("strategic_cues"),
            },
            "final_bullets_by_side": {
                "client": (client_side or {}).get("bullets"),
                "competitor": (competitor_side or {}).get("bullets"),
            },
            "warnings": warnings,
        },
    }
