from __future__ import annotations

import re
from typing import Any

from app.audit_helpers.bullet_uniqueness import dedupe_bullet_debug, normalize_bullet_text
from app.audit_helpers.audit_language_resolver import strategic_bullet_text
from app.audit_helpers.strategic_cue_engine import aggregate_strategic_cues


DIMENSION_PRIORITY = (
    "brand_presentation",
    "lifestyle_merchandising",
    "category_segmentation",
    "product_discovery",
    "educational_storytelling",
    "video_rich_media",
    "cross_category_navigation",
)
SLIDE5_TARGET_BULLET_COUNT = 7
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


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    normalized = _normalize(value)
    return normalized in {"1", "true", "yes", "y", "present", "detected"}


def _record_int(record: dict[str, Any], key: str, default: int = 0) -> int:
    return _to_int(_first(record, key, f"data.{key}", default=default), default)


def _record_bool(record: dict[str, Any], key: str) -> bool:
    return _to_bool(_first(record, key, f"data.{key}", default=False))


def _score_level(value: Any) -> str:
    normalized = _normalize(value)
    if normalized in {"strong", "high", "advanced", "robust"}:
        return "Strong"
    if normalized in {"present", "medium", "moderate", "developed"}:
        return "Present"
    if normalized in {"limited", "low", "light", "basic", "weak"}:
        return "Limited"
    if normalized in {"missing", "none", "absent", "0"}:
        return "Missing"
    numeric = _to_float(value, default=-1)
    if 0 <= numeric <= 1:
        if numeric >= 0.75:
            return "Strong"
        if numeric >= 0.45:
            return "Present"
        if numeric > 0:
            return "Limited"
        return "Missing"
    if numeric >= 8:
        return "Strong"
    if numeric >= 5:
        return "Present"
    if numeric >= 1:
        return "Limited"
    return ""


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
    visible_product_count = _record_int(record, "visible_product_count", -1)
    captured_product_count = _record_int(record, "captured_product_count", -1)
    product_count = _first(record, "productCount", "data.productCount", default=None)
    if product_count in (None, ""):
        products = _as_list(_first(record, "products", "productTiles", "data.products", "data.productTiles", default=[]))
        product_count = len(products)
    product_count = _to_int(product_count)
    if visible_product_count >= 0:
        product_count = visible_product_count
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
        "page_experience_type": _normalize(_first(record, "page_experience_type", "data.page_experience_type")),
        "visible_product_count": max(0, visible_product_count),
        "distinct_product_count": _record_int(record, "distinct_product_count"),
        "duplicate_card_count": _record_int(record, "duplicate_card_count"),
        "captured_product_count": max(0, captured_product_count),
        "captured_distinct_product_count": _record_int(record, "captured_distinct_product_count"),
        "captured_duplicate_card_count": _record_int(record, "captured_duplicate_card_count"),
        "hero_present": _record_bool(record, "hero_present"),
        "hero_copy_present": _record_bool(record, "hero_copy_present"),
        "hero_cta_present": _record_bool(record, "hero_cta_present"),
        "nav_link_count": _record_int(record, "nav_link_count"),
        "section_heading_count": _record_int(record, "section_heading_count"),
        "product_module_count": _record_int(record, "product_module_count"),
        "assortment_module_count": _record_int(record, "assortment_module_count"),
        "featured_product_count": _record_int(record, "featured_product_count"),
        "multi_product_section_count": _record_int(record, "multi_product_section_count"),
        "editorial_module_count": _record_int(record, "editorial_module_count"),
        "copy_block_count": _record_int(record, "copy_block_count"),
        "faq_module_count": _record_int(record, "faq_module_count"),
        "education_module_count": _record_int(record, "education_module_count"),
        "benefit_callout_count": _record_int(record, "benefit_callout_count"),
        "how_to_use_present": _record_bool(record, "how_to_use_present"),
        "routine_or_regimen_present": _record_bool(record, "routine_or_regimen_present"),
        "video_module_count": _record_int(record, "video_module_count"),
        "generic_module_count": _record_int(record, "generic_module_count"),
        "custom_module_count": _record_int(record, "custom_module_count"),
        "shop_depth_score": _first(record, "shop_depth_score", "data.shop_depth_score", default=""),
        "education_depth_score": _first(record, "education_depth_score", "data.education_depth_score", default=""),
        "navigation_depth_score": _first(record, "navigation_depth_score", "data.navigation_depth_score", default=""),
        "assortment_depth_score": _first(record, "assortment_depth_score", "data.assortment_depth_score", default=""),
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
    page_type = _safe_text(evidence.get("page_experience_type"))
    visible_count = int(evidence.get("visible_product_count", 0) or 0)
    distinct_count = int(evidence.get("distinct_product_count", 0) or 0)
    nav_link_count = int(evidence.get("nav_link_count", 0) or 0)
    section_heading_count = int(evidence.get("section_heading_count", 0) or 0)
    product_module_count = int(evidence.get("product_module_count", 0) or 0)
    assortment_module_count = int(evidence.get("assortment_module_count", 0) or 0)
    multi_product_count = int(evidence.get("multi_product_section_count", 0) or 0)
    editorial_count = int(evidence.get("editorial_module_count", 0) or 0)
    copy_block_count = int(evidence.get("copy_block_count", 0) or 0)
    education_module_count = int(evidence.get("education_module_count", 0) or 0)
    faq_count = int(evidence.get("faq_module_count", 0) or 0)
    benefit_count = int(evidence.get("benefit_callout_count", 0) or 0)
    video_module_count = int(evidence.get("video_module_count", 0) or 0)
    custom_count = int(evidence.get("custom_module_count", 0) or 0)
    generic_count = int(evidence.get("generic_module_count", 0) or 0)
    has_hero = bool(evidence.get("hero_present") or hero)
    has_hero_guidance = bool(evidence.get("hero_copy_present") or evidence.get("hero_cta_present"))
    has_usage_guidance = bool(evidence.get("how_to_use_present") or evidence.get("routine_or_regimen_present"))
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

    shop_score = _score_level(evidence.get("shop_depth_score"))
    brand_signals = [
        page_type,
        "hero present" if has_hero else "",
        "hero copy" if evidence.get("hero_copy_present") else "",
        "hero cta" if evidence.get("hero_cta_present") else "",
        f"{custom_count} custom modules" if custom_count else "",
        f"{generic_count} generic modules" if generic_count else "",
        f"shop depth {evidence.get('shop_depth_score')}" if evidence.get("shop_depth_score") not in ("", None) else "",
    ]
    if has_hero and has_hero_guidance and (custom_count >= 3 or shop_score == "Strong"):
        brand = result("Strong", brand_signals, "Hero guidance and custom structure support a stronger branded entry point.")
    elif has_hero and (has_hero_guidance or custom_count >= 1 or shop_score in {"Present", "Strong"}):
        brand = result("Present", brand_signals, "Branded presence is visible, but guidance or custom structure is not deep enough for Strong.")
    elif has_hero or page_type in {"brand browse", "brand_browse", "hybrid"} or custom_count or evidence["modules"]:
        brand = result("Limited", brand_signals or evidence["module_types"][:2], "The page has branded-shop presence, but the entry experience is browse-led or lightly developed.")
    else:
        brand = result("Missing", [], "No reliable branded entry evidence was detected.")

    lifestyle_signals = [
        f"{editorial_count} editorial modules" if editorial_count else "",
        f"{copy_block_count} copy blocks" if copy_block_count else "",
        "hero copy" if evidence.get("hero_copy_present") else "",
        *editorial[:2],
        *headings[:2],
    ]
    if editorial_count >= 2 and copy_block_count >= 2 and has_hero_guidance:
        lifestyle = result("Strong", lifestyle_signals, "Editorial modules, copy, and hero guidance create a more developed merchandising story.")
    elif editorial_count or copy_block_count >= 2 or (editorial and has_hero_guidance):
        lifestyle = result("Present", lifestyle_signals, "Some editorial or copy support is present without deep lifestyle merchandising.")
    elif editorial or rich_modules or has_hero:
        lifestyle = result("Limited", lifestyle_signals or rich_modules[:2], "Static branded content provides light context without a developed lifestyle journey.")
    else:
        lifestyle = result("Missing", [], "No reliable lifestyle merchandising evidence was detected.")

    category_count = len(category_destinations)
    navigation_score = _score_level(evidence.get("navigation_depth_score"))
    category_signals = [
        f"{nav_link_count} navigation links" if nav_link_count else "",
        f"{section_heading_count} section headings" if section_heading_count else "",
        *categories[:4],
        f"navigation depth {evidence.get('navigation_depth_score')}" if evidence.get("navigation_depth_score") not in ("", None) else "",
    ]
    if nav_link_count >= 5 and category_count >= 3 and navigation_score in {"Present", "Strong"}:
        category = result("Strong", category_signals, "Navigation is supported by multiple actual pathways and category destinations.")
    elif nav_link_count >= 2 or (category_count >= 2 and navigation_score in {"Present", "Strong"}):
        category = result("Present", category_signals, "Discovery pathways are present, though not extensive enough for Strong.")
    elif nav_link_count == 1 or category_count == 1 or section_heading_count:
        category = result("Limited", category_signals, "The page has light structure, but headings alone do not create strong navigation.")
    else:
        category = result("Missing", [], "No category-navigation evidence was detected.")

    assortment_score = _score_level(evidence.get("assortment_depth_score"))
    assortment_count = visible_count or product_count
    product_signal_count = product_module_count or len(product_modules)
    product_signals = [
        f"{assortment_count} visible products" if assortment_count else "",
        f"{distinct_count} distinct products" if distinct_count else "",
        f"{product_signal_count} product modules" if product_signal_count else "",
        f"{assortment_module_count} assortment modules" if assortment_module_count else "",
        f"{multi_product_count} multi-product sections" if multi_product_count else "",
        f"assortment depth {evidence.get('assortment_depth_score')}" if evidence.get("assortment_depth_score") not in ("", None) else "",
    ]
    if assortment_count >= 12 and distinct_count >= 8 and (product_signal_count >= 2 or multi_product_count >= 2 or assortment_score == "Strong"):
        product = result("Strong", product_signals, "Visible assortment and product modules support broader product discovery.")
    elif assortment_count >= 6 or distinct_count >= 4 or product_signal_count >= 1 or assortment_score in {"Present", "Strong"}:
        product = result("Present", product_signals, "The page offers meaningful assortment depth without broad variant coverage.")
    elif assortment_count > 0 or product_modules or assortment_score == "Limited":
        product = result("Limited", product_signals or [f"{product_count} products"], "Visible assortment is narrow or lightly supported.")
    else:
        product = result("Missing", [], "No reliable product-discovery evidence was detected.")

    education_score = _score_level(evidence.get("education_depth_score"))
    education_signal_count = (
        education_module_count
        + faq_count
        + benefit_count
        + int(has_usage_guidance)
    )
    education_signals = [
        f"{education_module_count} education modules" if education_module_count else "",
        f"{faq_count} FAQ modules" if faq_count else "",
        f"{benefit_count} benefit callouts" if benefit_count else "",
        "how-to-use guidance" if evidence.get("how_to_use_present") else "",
        "routine guidance" if evidence.get("routine_or_regimen_present") else "",
        *educational_text[:3],
        f"education depth {evidence.get('education_depth_score')}" if evidence.get("education_depth_score") not in ("", None) else "",
    ]
    if education_signal_count >= 3 and education_score in {"Present", "Strong"}:
        education = result("Strong", education_signals, "Multiple explicit guidance signals support stronger shopper education.")
    elif education_signal_count >= 1 or education_score in {"Present", "Strong"}:
        education = result("Present", education_signals, "Shopper guidance is present without multiple education layers.")
    elif educational_text or descriptions or copy_block_count:
        education = result("Limited", education_signals or descriptions[:2], "Copy is present, but education and usage guidance are limited.")
    else:
        education = result("Missing", [], "No educational content evidence was detected.")

    explicit_video = video_module_count > 0 or evidence["video_present"] or bool(evidence["videos"])
    if explicit_video and (video_module_count >= 2 or rich_modules):
        video = result(
            "Strong",
            [f"{video_module_count} video modules" if video_module_count else "", *video_modules, "video present" if evidence["video_present"] else "", *rich_modules[:2]],
            "Video and another rich editorial module were detected.",
        )
    elif explicit_video:
        video = result(
            "Present",
            [f"{video_module_count} video modules" if video_module_count else "", *video_modules, *evidence["videos"][:2], "video present" if evidence["video_present"] else ""],
            "Video content was detected.",
        )
    elif rich_modules:
        video = result("Limited", rich_modules[:2], "Static rich content exists without video.")
    else:
        video = result("Missing", [], "No reliable rich-media evidence was detected.")

    link_count = len(link_destinations)
    category_like_count = len(set([*category_destinations, *link_destinations]))
    cross_signals = [
        f"{nav_link_count} navigation links" if nav_link_count else "",
        *categories[:3],
        *(link["label"] or link["url"] for link in links[:3]),
        f"navigation depth {evidence.get('navigation_depth_score')}" if evidence.get("navigation_depth_score") not in ("", None) else "",
    ]
    if nav_link_count >= 5 and category_like_count >= 4 and navigation_score in {"Present", "Strong"}:
        cross = result("Strong", cross_signals, "Multiple links span distinct category or need-state pathways.")
    elif nav_link_count >= 2 or link_count >= 2 or (category_like_count >= 3 and navigation_score in {"Present", "Strong"}):
        cross = result("Present", cross_signals, "More than one destination pathway was detected.")
    elif nav_link_count == 1 or link_count == 1 or category_count == 1:
        cross = result("Limited", cross_signals, "Navigation is limited to one destination pathway.")
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


def _beauty_context(evidence: dict[str, Any]) -> bool:
    blob = _normalize(
        " ".join(
            [
                *evidence.get("categories", [])[:8],
                *evidence.get("headings", [])[:8],
                *evidence.get("descriptions", [])[:8],
            ]
        )
    )
    return any(
        term in blob
        for term in (
            "beauty",
            "skin care",
            "skincare",
            "face care",
            "body care",
            "hair care",
            "hydration",
            "moisture",
        )
    )


def _wrong_category_language_reason(text: str, evidence: dict[str, Any]) -> str:
    if not _beauty_context(evidence):
        return ""
    normalized = _normalize(text)
    tokens = set(normalized.split())
    pet_phrases = {"breed size", "life stage"}
    pet_tokens = {"animal", "breed", "cat", "dog", "kitten", "lifestage", "pet", "puppy"}
    if any(phrase in normalized for phrase in pet_phrases) or bool(tokens & pet_tokens):
        return "beauty_context_blocked_pet_language"
    return ""


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
        "brand_presentation": f"Cohesive visual identity across {module_count or 'multiple'} sections",
        "lifestyle_merchandising": f"Lifestyle-led shopping occasions through {topic or 'editorial'} content",
        "category_segmentation": f"Structured {_navigation_context(evidence.get('categories', []) or [])} entry points",
        "product_discovery": f"Enhanced product discovery across {product_count or 'multiple'} items",
        "educational_storytelling": f"Benefit-forward education through {topic or 'editorial copy'}",
        "video_rich_media": f"Rich-media depth through {video_title or 'video content'}",
        "cross_category_navigation": f"Clear {_category_context(evidence.get('categories', []) or [])} navigation",
        "visual_identity": f"Cohesive branded journey across {module_count or 'multiple'} sections",
        "category_grouping": f"Organized {_navigation_context(evidence.get('categories', []) or [])} category entry points",
        "discovery_pathways": f"Expanded assortment discovery across {product_count or 'multiple'} items",
        "shopper_education": f"Benefit-forward shopper education through {topic or 'editorial copy'}",
        "usage_storytelling": f"Lifestyle-led brand storytelling through {topic or 'editorial content'}",
        "assortment_segmentation": f"Clear assortment segmentation across {product_count or 'multiple'} products",
        "conversion_guidance": "Benefit-led guidance supports product comparison",
        "benefit_communication": f"Clear benefit messaging through {topic or 'brand copy'}",
    }
    return _fit_bullet(alternatives.get(dimension, "Focused Brand Shop merchandising"))


def _visible_assortment_count(evidence: dict[str, Any]) -> int:
    return int(evidence.get("visible_product_count", 0) or evidence.get("product_count", 0) or 0)


def _dimension_bucket_text(
    *,
    side: str,
    bullet_type: str,
    dimension: str,
    score: str,
    evidence: dict[str, Any],
) -> tuple[str, str]:
    weak = score in {"Missing", "Limited"} or bullet_type in {"opportunity", "client_opportunity"}
    competitor = side == "competitor" and bullet_type not in {"opportunity", "client_opportunity"}
    visible_count = _visible_assortment_count(evidence)
    nav_count = int(evidence.get("nav_link_count", 0) or 0)
    education_count = (
        int(evidence.get("education_module_count", 0) or 0)
        + int(evidence.get("faq_module_count", 0) or 0)
        + int(evidence.get("benefit_callout_count", 0) or 0)
        + int(bool(evidence.get("how_to_use_present")))
        + int(bool(evidence.get("routine_or_regimen_present")))
    )
    hero_visual_only = bool(evidence.get("hero_present")) and not (
        evidence.get("hero_copy_present") or evidence.get("hero_cta_present")
    )
    page_type = _safe_text(evidence.get("page_experience_type")).replace("_", "-")
    templates = {
        "brand_presentation": (
            "brand_presence",
            "Limited branded-shop structure keeps the experience browse-led"
            if weak or hero_visual_only
            else "More intentional branded presentation creates a clearer shop entry"
            if competitor
            else "Clearer branded entry development supports a stronger first impression",
        ),
        "lifestyle_merchandising": (
            "brand_presence",
            "Light branded entry development keeps the shop mostly product-led"
            if weak
            else "More developed brand context gives the shop stronger first-impression support"
            if competitor
            else "More developed brand context can make the shop feel less browse-led",
        ),
        "category_segmentation": (
            "navigation",
            "Light navigation structure creates a more self-directed shopping journey"
            if weak or nav_count < 2
            else "Clearer discovery pathways make the shop easier to explore"
            if competitor
            else "Clearer discovery pathways would make browsing feel more guided",
        ),
        "cross_category_navigation": (
            "navigation",
            "Limited discovery support leaves shoppers to connect pathways themselves"
            if weak or nav_count < 2
            else "More guided browsing helps shoppers move across shop sections"
            if competitor
            else "More guided browsing can connect shop sections more clearly",
        ),
        "product_discovery": (
            "assortment",
            "Narrow visible assortment limits browse depth and variant discovery"
            if weak or visible_count <= 5
            else "Broader visible assortment creates stronger product discovery"
            if competitor
            else "Broader visible assortment would improve product and variant discovery",
        ),
        "educational_storytelling": (
            "education",
            "Limited educational support leaves shoppers with less guided product context"
            if weak or education_count <= 1
            else "More developed benefit storytelling provides stronger shopper guidance"
            if competitor
            else "Clearer benefit storytelling would strengthen shopper education",
        ),
        "video_rich_media": (
            "education",
            "Light rich-media support keeps product guidance mostly static"
            if weak
            else "Video storytelling adds clearer product context for shoppers"
            if competitor
            else "Richer product storytelling can make shopper guidance easier to absorb",
        ),
    }
    template_id, text = templates[dimension]
    if page_type and dimension == "brand_presentation" and weak and "browse" in page_type:
        text = "Browse-led brand presence signals light branded entry development"
    return template_id, text


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
        "visible_product_count": competitor_side.get("evidence", {}).get("visible_product_count", 0),
        "nav_link_count": competitor_side.get("evidence", {}).get("nav_link_count", 0),
        "education_module_count": competitor_side.get("evidence", {}).get("education_module_count", 0),
        "faq_module_count": competitor_side.get("evidence", {}).get("faq_module_count", 0),
        "benefit_callout_count": competitor_side.get("evidence", {}).get("benefit_callout_count", 0),
        "how_to_use_present": competitor_side.get("evidence", {}).get("how_to_use_present", False),
        "routine_or_regimen_present": competitor_side.get("evidence", {}).get("routine_or_regimen_present", False),
    }
    reworded_any = False
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
            reworded_any = True
        candidate_key = _mirrored_bullet_key(text)
        suffix = 2
        while candidate_key and candidate_key in used and suffix <= 6:
            marker = ("Clear", "Structured", "Enhanced", "Cohesive", "Balanced")[min(suffix - 2, 4)]
            text = _fit_bullet(re.sub(r"^\w+\s+", f"{marker} ", text, count=1))
            item["text"] = text
            candidate_key = _mirrored_bullet_key(text)
            suffix += 1
        if candidate_key and candidate_key in used:
            text = _fit_bullet("Distinct merchandising advantage")
            item["text"] = text
            candidate_key = _mirrored_bullet_key(text)
        if candidate_key:
            used.add(candidate_key)
        rewritten.append(text)
        competitor_debug[index] = item
    if not reworded_any and competitor_debug and client_side.get("bullets"):
        index = next(
            (
                idx
                for idx, item in enumerate(competitor_debug)
                if item.get("dimension")
                in {
                    "product_discovery",
                    "educational_storytelling",
                    "category_segmentation",
                    "cross_category_navigation",
                    "discovery_pathways",
                    "shopper_education",
                    "category_grouping",
                }
            ),
            0,
        )
        item = competitor_debug[index]
        text = _alternate_competitor_bullet(
            item,
            brand_name=competitor_side.get("brand_name", ""),
            evidence=evidence,
        )
        item["text"] = text
        item["signals"] = list(item.get("signals", []) or []) + ["cross_side_mirror_reworded"]
        item["reason"] = (
            _safe_text(item.get("reason"))
            + " Reworded to keep competitor Brand Shop bullets directionally distinct from the client side."
        ).strip()
        rewritten[index] = text
        competitor_debug[index] = item
    competitor_side["bullet_debug"] = competitor_debug[:SLIDE5_TARGET_BULLET_COUNT]
    competitor_side["bullets"] = rewritten[:SLIDE5_TARGET_BULLET_COUNT]


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
    bucket, text = _dimension_bucket_text(
        side=side,
        bullet_type=bullet_type,
        dimension=dimension,
        score=score,
        evidence=evidence,
    )
    template_id = f"strategic_cue_{bucket}_{bullet_type}_{dimension}"
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
    )[:4]
    remaining = [dimension for dimension in DIMENSION_PRIORITY if dimension not in strengths]
    opportunities = sorted(
        remaining,
        key=lambda dimension: (
            SCORE_ORDER[dimensions[dimension]["score"]],
            int(dimensions[dimension]["supporting_count"]),
            priority_index[dimension],
        ),
    )[:3]
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
    return dedupe_bullet_debug(bullets, fallback_subject="client brand shop")[0][:SLIDE5_TARGET_BULLET_COUNT]


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
    )[:SLIDE5_TARGET_BULLET_COUNT]
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
    return dedupe_bullet_debug(bullets, fallback_subject="competitor brand shop")[0][:SLIDE5_TARGET_BULLET_COUNT]


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
    if len(ranked) < 5:
        ranked.extend(
            sorted(
                unsupported,
                key=lambda dimension: (
                    -SCORE_ORDER[dimensions[dimension]["score"]],
                    priority_index[dimension],
                ),
            )
        )
    strength_dimensions = ranked[:5]
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
    )[0][:SLIDE5_TARGET_BULLET_COUNT]


def _slide5_context_from_record(record: dict[str, Any], side: str) -> dict[str, Any]:
    return aggregate_strategic_cues(
        [],
        brand_shop_evidence={side: [record]},
        fallback_category="Brand Shop",
        fallback_product_type="Brand Shop",
    )


def _slide5_topic_phrase(evidence: dict[str, Any]) -> str:
    blob = " ".join([*evidence.get("headings", []), *evidence.get("descriptions", [])]).lower()
    if "hydration" in blob or "moisture" in blob:
        return "Hydration-led storytelling"
    if "routine" in blob:
        return "Routine-led storytelling"
    if "solution" in blob:
        return "Solution-led storytelling"
    return "Educational storytelling"


def _slide5_evidence_terms(evidence: dict[str, Any]) -> dict[str, str]:
    categories = evidence.get("categories", []) or []
    category_context = _category_context(categories)
    navigation_context = _navigation_context(categories)
    product_count = int(evidence.get("product_count", 0) or 0)
    module_count = int(evidence.get("module_count", 0) or 0)
    topic = _slide5_topic_phrase(evidence)
    return {
        "category_grouping": (
            f"Structured {category_context} category grouping"
            if len(categories) >= 3
            else f"Clear {navigation_context} category focus"
        ),
        "discovery": (
            "Broad assortment depth supports product discovery"
            if product_count >= 10
            else "Focused assortment supports guided discovery"
            if product_count > 0
            else "Clear discovery pathways support shopper navigation"
        ),
        "cross_category": f"Cross-category pathways expand {navigation_context}",
        "education": f"{topic} builds shopper education",
        "story": f"{topic} deepens the branded journey",
        "visual": (
            "Cohesive visual identity anchors the Brand Shop"
            if module_count >= 5
            else "Clear visual identity supports brand recognition"
        ),
    }


def _slide5_candidate_text(
    cue_key: str,
    evidence: dict[str, Any],
    classification: str,
    identity: dict[str, Any],
) -> str:
    terms = _slide5_evidence_terms(evidence)
    if cue_key == "visual_identity":
        return terms["visual"]
    if cue_key == "benefit_communication":
        return "Benefit-forward messaging supports shopper confidence"
    if cue_key == "product_positioning":
        return "Clear product positioning supports brand relevance"
    if cue_key == "discoverability":
        return "Brand-led navigation strengthens shopper discovery"
    if cue_key == "review_or_trust_signals" and classification == "opportunity":
        return "Opportunity to strengthen conversion confidence"
    if bool(evidence.get("video_present") or evidence.get("videos")) and cue_key == "usage_storytelling":
        return "Rich media deepens the branded shopping journey"
    return strategic_bullet_text(
        {"cue_key": cue_key, "classification": classification},
        identity,
        slide_key="slide5",
        evidence_terms=terms,
    )


def _slide5_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, float]:
    class_rank = {"strength": 0, "opportunity": 1, "context": 2, "pressure": 3}
    cue_rank = {
        "visual_identity": 0,
        "category_grouping": 1,
        "discovery_pathways": 2,
        "shopper_education": 3,
        "usage_storytelling": 4,
        "cross_category_navigation": 5,
        "assortment_segmentation": 6,
        "conversion_guidance": 7,
        "benefit_communication": 8,
        "discoverability": 9,
    }
    return (
        class_rank.get(candidate.get("classification"), 9),
        cue_rank.get(candidate.get("cue_key"), 99),
        -float(candidate.get("coverage_ratio", 0) or 0),
        -float(candidate.get("strength_ratio", 0) or 0),
    )


def _brand_shop_cue_bullets(
    record: dict[str, Any],
    side: str,
    evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = _slide5_context_from_record(record, side)
    candidates = [
        candidate
        for candidate in context.get("candidate_cues", [])
        if "slide5_brand_shop" in (candidate.get("slide_objective_tags") or [])
    ]
    debug_items: list[dict[str, Any]] = []
    used: set[str] = set()
    def visible_count() -> int:
        return sum(1 for item in debug_items if not item.get("_rejected"))

    for candidate in sorted(candidates, key=_slide5_candidate_sort_key):
        if visible_count() >= SLIDE5_TARGET_BULLET_COUNT:
            break
        text = _fit_bullet(
            _slide5_candidate_text(
                candidate.get("cue_key", ""),
                evidence,
                candidate.get("classification", "context"),
                context.get("identity", {}),
            )
        )
        wrong_category_reason = _wrong_category_language_reason(text, evidence)
        if wrong_category_reason:
            debug_items.append(
                {
                    "text": "",
                    "side": side,
                    "type": "rejected",
                    "dimension": candidate.get("cue_key", "cue"),
                    "score": candidate.get("classification", "context"),
                    "template_id": f"strategic_cue_{candidate.get('cue_key')}",
                    "signals": [wrong_category_reason],
                    "supporting_count": 0,
                    "reason": "Rejected because the wording did not match the Brand Shop category context.",
                    "cue_debug": candidate,
                    "_rejected": True,
                }
            )
            continue
        key = normalize_bullet_text(text)
        if not text or key in used:
            continue
        used.add(key)
        debug_items.append(
            {
                "text": text,
                "side": side,
                "type": candidate.get("classification", "context"),
                "dimension": candidate.get("cue_key", "cue"),
                "score": candidate.get("classification", "context"),
                "template_id": f"strategic_cue_{candidate.get('cue_key')}",
                "signals": [
                    candidate.get("classification", ""),
                    candidate.get("cue_key", ""),
                    *candidate.get("evidence_sources", [])[:2],
                ],
                "supporting_count": len(candidate.get("matched_guide_rules", []) or []),
                "reason": candidate.get("debug_reason", "Selected from strategic cue engine output."),
                "cue_debug": candidate,
            }
        )
        if visible_count() >= SLIDE5_TARGET_BULLET_COUNT:
            break
    fallback_texts = [
        "Cohesive visual identity anchors the Brand Shop",
        "Structured category grouping supports navigation",
        "Broad assortment depth supports product discovery",
        "Educational storytelling builds shopper confidence",
        "Cross-category pathways support broader exploration",
        "Rich-media cues make the shop feel more immersive",
        "Conversion guidance helps shoppers compare products",
    ]
    for text in fallback_texts:
        if visible_count() >= SLIDE5_TARGET_BULLET_COUNT:
            break
        key = normalize_bullet_text(text)
        if key in used:
            continue
        used.add(key)
        debug_items.append(
            {
                "text": _fit_bullet(text),
                "side": side,
                "type": "context",
                "dimension": "fallback",
                "score": "context",
                "template_id": "strategic_cue_fallback",
                "signals": ["fallback"],
                "supporting_count": 0,
                "reason": "Controlled fallback used to preserve Slide 5 bullet count.",
                "cue_debug": {},
            }
        )
    visible_items = [item for item in debug_items if not item.get("_rejected")]
    rejected_items = [item for item in debug_items if item.get("_rejected")]
    return visible_items[:SLIDE5_TARGET_BULLET_COUNT], {
        **(context.get("debug", {}) or {}),
        "wrong_category_rejections": rejected_items,
    }


def _cue_bullets_have_strong_support(
    cue_bullets: list[dict[str, Any]],
    dimensions: dict[str, dict[str, Any]],
) -> bool:
    if len(cue_bullets) < SLIDE5_TARGET_BULLET_COUNT:
        return False
    visible = [item for item in cue_bullets if _safe_text(item.get("text"))]
    if len(visible) < SLIDE5_TARGET_BULLET_COUNT:
        return False
    rejected = [item for item in cue_bullets if item.get("_rejected")]
    if rejected:
        return False
    supported_cues = sum(1 for item in visible if int(item.get("supporting_count", 0) or 0) > 0)
    evidence_supported_dimensions = sum(
        1
        for data in dimensions.values()
        if SCORE_ORDER.get(data.get("score", "Missing"), 0) >= SCORE_ORDER["Present"]
    )
    strategic_types = {"strength", "opportunity", "context", "pressure"}
    valid_type_count = sum(1 for item in visible if item.get("type") in strategic_types)
    return (
        supported_cues >= 3
        and evidence_supported_dimensions >= 3
        and valid_type_count >= SLIDE5_TARGET_BULLET_COUNT - 1
    )


def _build_side(record: dict[str, Any], side: str, role_path: str) -> dict[str, Any]:
    evidence = _evidence(record)
    dimensions = _score_dimensions(evidence)
    brand_name = _record_brand(record)
    cue_bullets, cue_context_debug = _brand_shop_cue_bullets(record, side, evidence)
    fallback_bullets = (
        _client_bullets(dimensions, brand_name, evidence)
        if side == "client"
        else _competitor_bullets(dimensions, brand_name, evidence)
    )
    use_cue_bullets = _cue_bullets_have_strong_support(cue_bullets, dimensions)
    bullet_debug = cue_bullets if use_cue_bullets else fallback_bullets
    warnings: list[str] = []
    if sum(SCORE_ORDER[data["score"]] >= 2 for data in dimensions.values()) < 2:
        warnings.append(
            "Brand Shop evidence is weak; restrained capability and opportunity language was used."
        )
    if cue_bullets and not use_cue_bullets:
        warnings.append(
            "Strategic cue bullets were available but fallback Brand Shop bullets were used because cue support was incomplete or low-confidence."
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
            "page_experience_type": evidence["page_experience_type"],
            "visible_product_count": evidence["visible_product_count"],
            "distinct_product_count": evidence["distinct_product_count"],
            "duplicate_card_count": evidence["duplicate_card_count"],
            "captured_product_count": evidence["captured_product_count"],
            "captured_distinct_product_count": evidence["captured_distinct_product_count"],
            "captured_duplicate_card_count": evidence["captured_duplicate_card_count"],
            "hero_present": evidence["hero_present"],
            "hero_copy_present": evidence["hero_copy_present"],
            "hero_cta_present": evidence["hero_cta_present"],
            "nav_link_count": evidence["nav_link_count"],
            "section_heading_count": evidence["section_heading_count"],
            "product_module_count": evidence["product_module_count"],
            "assortment_module_count": evidence["assortment_module_count"],
            "featured_product_count": evidence["featured_product_count"],
            "multi_product_section_count": evidence["multi_product_section_count"],
            "editorial_module_count": evidence["editorial_module_count"],
            "copy_block_count": evidence["copy_block_count"],
            "faq_module_count": evidence["faq_module_count"],
            "education_module_count": evidence["education_module_count"],
            "benefit_callout_count": evidence["benefit_callout_count"],
            "how_to_use_present": evidence["how_to_use_present"],
            "routine_or_regimen_present": evidence["routine_or_regimen_present"],
            "video_module_count": evidence["video_module_count"],
            "generic_module_count": evidence["generic_module_count"],
            "custom_module_count": evidence["custom_module_count"],
            "shop_depth_score": evidence["shop_depth_score"],
            "education_depth_score": evidence["education_depth_score"],
            "navigation_depth_score": evidence["navigation_depth_score"],
            "assortment_depth_score": evidence["assortment_depth_score"],
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
            "render_targets": {
                "target_bullet_count": SLIDE5_TARGET_BULLET_COUNT,
                "final_bullet_counts": {
                    "client": len((client_side or {}).get("bullets") or []),
                    "competitor": len((competitor_side or {}).get("bullets") or []),
                },
            },
            "warnings": warnings,
        },
    }
