from __future__ import annotations

import json
import re
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from audit_models import create_cached_pdp_record, make_image, make_key_feature, make_reviews_summary


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
}


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_pdp_url(raw_url: str) -> tuple[str | None, str | None]:
    url = (raw_url or "").strip()
    if not url:
        return None, "Empty URL"
    if not re.match(r"^https?://", url, flags=re.I):
        url = f"https://{url}"
    try:
        parsed = urlparse(url)
    except Exception:
        return None, "Invalid URL format"
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, "URL must include a valid domain"

    query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
    filtered_query = [(k, v) for k, v in query_pairs if k.lower() not in TRACKING_QUERY_KEYS]
    normalized = urlunparse(
        (
            "https",
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(filtered_query),
            "",
        )
    )
    return normalized, None


def fetch_pdp_html(url: str, timeout_sec: int = 15) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            try:
                html = raw.decode(charset, errors="replace")
            except Exception:
                html = raw.decode("utf-8", errors="replace")
            return html, errors
    except Exception as exc:
        errors.append(f"Fetch error: {exc}")
        return None, errors


def _clean_text(text: str) -> str:
    t = unescape(re.sub(r"\s+", " ", (text or "")).strip())
    return re.sub(r"\s+\|\s+.*$", "", t).strip()


def _strip_tags(text: str) -> str:
    no_tags = re.sub(r"(?is)<[^>]+>", " ", text or "")
    return _clean_text(no_tags)


def _extract_script_json_ld(html: str) -> list[Any]:
    blocks = re.findall(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html or "",
    )
    parsed: list[Any] = []
    for block in blocks:
        payload = block.strip()
        if not payload:
            continue
        try:
            parsed.append(json.loads(payload))
        except Exception:
            # Some pages inject invalid trailing commas/comments; keep parser conservative.
            continue
    return parsed


def _iter_product_like_nodes(obj: Any):
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_product_like_nodes(item)
    elif isinstance(obj, dict):
        type_value = obj.get("@type")
        if (
            type_value == "Product"
            or (isinstance(type_value, list) and "Product" in type_value)
            or "aggregateRating" in obj
            or "sku" in obj
        ):
            yield obj
        for value in obj.values():
            yield from _iter_product_like_nodes(value)


def _first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text or "", flags=re.I | re.S)
    return m.group(1).strip() if m else ""


def parse_title(html: str, json_ld_nodes: list[Any]) -> str:
    for node in json_ld_nodes:
        for p in _iter_product_like_nodes(node):
            name = _clean_text(str(p.get("name", "") or ""))
            if name:
                return name

    h1 = _first_match(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if h1:
        return _strip_tags(h1)

    og_title = _first_match(
        r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html,
    )
    if og_title:
        return _clean_text(og_title)

    page_title = _first_match(r"(?is)<title[^>]*>(.*?)</title>", html)
    return _clean_text(page_title)


def _parse_json_ld_images(json_ld_nodes: list[Any]) -> list[str]:
    urls: list[str] = []
    for node in json_ld_nodes:
        for p in _iter_product_like_nodes(node):
            image_data = p.get("image")
            if isinstance(image_data, str):
                urls.append(image_data)
            elif isinstance(image_data, list):
                for img in image_data:
                    if isinstance(img, str):
                        urls.append(img)
                    elif isinstance(img, dict):
                        u = img.get("url")
                        if isinstance(u, str):
                            urls.append(u)
    return urls


def _parse_html_images(html: str) -> list[str]:
    urls: list[str] = []
    meta_ogs = re.findall(
        r'(?is)<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\'](.*?)["\']',
        html,
    )
    urls.extend(meta_ogs)

    img_attrs = re.findall(r"(?is)<img[^>]+>", html)
    for tag in img_attrs:
        for attr in ("src", "data-src", "data-original", "data-image-src"):
            m = re.search(fr'(?i){attr}=["\'](.*?)["\']', tag)
            if m:
                urls.append(m.group(1))
                break
    return urls


def _normalize_image_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return f"https:{u}"
    if re.match(r"^https?://", u, flags=re.I):
        return u
    return ""


def parse_images(html: str, json_ld_nodes: list[Any]) -> list[dict[str, Any]]:
    raw_urls = _parse_json_ld_images(json_ld_nodes) + _parse_html_images(html)
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in raw_urls:
        url = _normalize_image_url(raw)
        if not url:
            continue
        lower = url.lower()
        if ".svg" in lower or "sprite" in lower or "logo" in lower:
            continue
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    normalized = normalized[:12]
    return [make_image(i, u, is_hero=(i == 0)) for i, u in enumerate(normalized)]


def parse_description(html: str, json_ld_nodes: list[Any]) -> tuple[str, list[str], list[str]]:
    body = ""
    bullets: list[str] = []
    labels: list[str] = []

    for node in json_ld_nodes:
        for p in _iter_product_like_nodes(node):
            desc = _clean_text(str(p.get("description", "") or ""))
            if desc:
                body = desc
                break
        if body:
            break

    if not body:
        meta_desc = _first_match(
            r'(?is)<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            html,
        )
        if meta_desc:
            body = _clean_text(meta_desc)

    section_matches = re.findall(
        r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>\s*(.*?)(?=<h[1-6][^>]*>|$)",
        html,
    )
    for heading_html, content_html in section_matches:
        heading = _strip_tags(heading_html).lower()
        if not heading:
            continue
        if any(
            k in heading
            for k in ("description", "product details", "about this item", "overview")
        ):
            label_text = _strip_tags(heading_html)
            if label_text:
                labels.append(label_text)
            if not body:
                first_p = _first_match(r"(?is)<p[^>]*>(.*?)</p>", content_html)
                if first_p:
                    body = _strip_tags(first_p)
            lis = re.findall(r"(?is)<li[^>]*>(.*?)</li>", content_html)
            for li in lis:
                txt = _strip_tags(li)
                if txt and txt not in bullets:
                    bullets.append(txt)

    return body, bullets[:10], labels[:5]


def parse_key_features(html: str, json_ld_nodes: list[Any]) -> tuple[list[dict[str, Any]], str]:
    features: list[str] = []
    section_label = ""

    section_matches = re.findall(
        r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>\s*(.*?)(?=<h[1-6][^>]*>|$)",
        html,
    )
    for heading_html, content_html in section_matches:
        heading = _strip_tags(heading_html).lower()
        if any(k in heading for k in ("key features", "about this item", "features", "highlights")):
            section_label = _strip_tags(heading_html)
            lis = re.findall(r"(?is)<li[^>]*>(.*?)</li>", content_html)
            for li in lis:
                txt = _strip_tags(li)
                if txt:
                    features.append(txt)
            if features:
                break

    if not features:
        # Fallback to first meaningful list in page.
        list_blocks = re.findall(r"(?is)<ul[^>]*>(.*?)</ul>", html)
        for block in list_blocks:
            lis = [_strip_tags(li) for li in re.findall(r"(?is)<li[^>]*>(.*?)</li>", block)]
            lis = [x for x in lis if x]
            if len(lis) >= 2:
                features = lis
                if not section_label:
                    section_label = "Key Features"
                break

    model_features = [make_key_feature(i + 1, text) for i, text in enumerate(features[:12])]
    return model_features, section_label


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value)
    s = re.sub(r"[^0-9.]+", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value)
    s = re.sub(r"[^0-9]+", "", s)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def parse_reviews(html: str, json_ld_nodes: list[Any]) -> dict[str, Any]:
    avg = None
    ratings = None
    reviews = None
    for node in json_ld_nodes:
        for p in _iter_product_like_nodes(node):
            aggr = p.get("aggregateRating", {})
            if isinstance(aggr, dict):
                avg = _to_float(aggr.get("ratingValue")) or avg
                ratings = _to_int(aggr.get("ratingCount")) or ratings
                reviews = _to_int(aggr.get("reviewCount")) or reviews

    if avg is None:
        m = re.search(r"([0-9](?:\.[0-9])?)\s*out of\s*5", html, flags=re.I)
        if m:
            avg = _to_float(m.group(1))
    if ratings is None:
        m = re.search(r"([0-9][0-9,\.]*)\s+ratings", html, flags=re.I)
        if m:
            ratings = _to_int(m.group(1))
    if reviews is None:
        m = re.search(r"([0-9][0-9,\.]*)\s+reviews", html, flags=re.I)
        if m:
            reviews = _to_int(m.group(1))

    return make_reviews_summary(average_rating=avg, ratings_count=ratings, review_count=reviews)


def parse_identity_fields(json_ld_nodes: list[Any]) -> tuple[str, str, str]:
    item_id = ""
    brand = ""
    category = ""
    for node in json_ld_nodes:
        for p in _iter_product_like_nodes(node):
            if not item_id:
                item_id = _clean_text(str(p.get("sku", "") or p.get("productID", "") or ""))
            if not brand:
                b = p.get("brand", "")
                if isinstance(b, dict):
                    brand = _clean_text(str(b.get("name", "")))
                else:
                    brand = _clean_text(str(b))
            if not category:
                category = _clean_text(str(p.get("category", "") or ""))
    return item_id, brand, category


def _compute_extraction_status(
    *,
    title: str,
    desc_body: str,
    desc_bullets: list[str],
    key_features: list[dict[str, Any]],
    images: list[dict[str, Any]],
    reviews: dict[str, Any],
) -> tuple[str, list[str]]:
    errors: list[str] = []
    title_ok = bool(title)
    desc_ok = bool(desc_body or desc_bullets)
    features_ok = bool(key_features)
    images_ok = bool(images)
    reviews_ok = bool(
        reviews.get("average_rating") is not None
        or reviews.get("ratings_count") is not None
        or reviews.get("review_count") is not None
    )

    if not title_ok:
        errors.append("Missing title")
    if not desc_ok:
        errors.append("Missing description content")
    if not features_ok:
        errors.append("Missing key features")
    if not images_ok:
        errors.append("Missing images")
    if not reviews_ok:
        errors.append("Missing review summary (non-blocking)")

    core_components = sum([1 if title_ok else 0, 1 if images_ok else 0, 1 if desc_ok else 0, 1 if features_ok else 0])
    strong_core = title_ok and images_ok and (desc_ok or features_ok)

    if strong_core and core_components >= 3:
        return "success", errors
    if core_components >= 2 or reviews_ok:
        return "partial", errors
    return "fail", errors


def build_cached_record_from_html(
    *,
    html: str,
    source_url: str,
    source_type: str,
    client_name: str,
    retailer: str,
) -> dict[str, Any]:
    json_ld_nodes = _extract_script_json_ld(html)
    current_title = parse_title(html, json_ld_nodes)
    images = parse_images(html, json_ld_nodes)
    desc_body, desc_bullets, section_labels = parse_description(html, json_ld_nodes)
    key_features, key_feature_label = parse_key_features(html, json_ld_nodes)
    reviews = parse_reviews(html, json_ld_nodes)
    item_id, brand, category = parse_identity_fields(json_ld_nodes)

    status, errors = _compute_extraction_status(
        title=current_title,
        desc_body=desc_body,
        desc_bullets=desc_bullets,
        key_features=key_features,
        images=images,
        reviews=reviews,
    )

    return create_cached_pdp_record(
        client_name=client_name,
        retailer=retailer,
        source_url=source_url,
        source_type=source_type,
        item_id=item_id,
        brand=brand,
        product_title=current_title or "Untitled Product",
        category=category,
        subcategory="",
        current_title=current_title,
        current_description_body=desc_body,
        current_description_bullets=desc_bullets,
        description_section_labels=section_labels,
        current_key_features=key_features,
        key_features_section_label=key_feature_label or "Key Features",
        images=images,
        reviews_summary=reviews,
        extraction_status=status,
        extraction_errors=errors,
        fetched_at=_utc_now_iso(),
        last_used_at=_utc_now_iso(),
    )


def extract_cached_record_from_url(
    *,
    raw_url: str,
    source_type: str,
    client_name: str,
    retailer: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    normalized_url, normalize_error = normalize_pdp_url(raw_url)
    if normalize_error or not normalized_url:
        return None, [normalize_error or "URL normalization failed"]

    html, fetch_errors = fetch_pdp_html(normalized_url)
    errors.extend(fetch_errors)
    if not html:
        return create_cached_pdp_record(
            client_name=client_name,
            retailer=retailer,
            source_url=normalized_url,
            source_type=source_type,
            extraction_status="fail",
            extraction_errors=errors or ["Failed to fetch PDP HTML"],
            fetched_at=_utc_now_iso(),
            last_used_at=_utc_now_iso(),
        ), errors

    record = build_cached_record_from_html(
        html=html,
        source_url=normalized_url,
        source_type=source_type,
        client_name=client_name,
        retailer=retailer,
    )
    return record, errors


def extract_primary_cached_record(
    *,
    raw_url: str,
    client_name: str,
    retailer: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    return extract_cached_record_from_url(
        raw_url=raw_url,
        source_type="primary",
        client_name=client_name,
        retailer=retailer,
    )


def extract_competitor_cached_record(
    *,
    raw_url: str,
    client_name: str,
    retailer: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    return extract_cached_record_from_url(
        raw_url=raw_url,
        source_type="competitor",
        client_name=client_name,
        retailer=retailer,
    )
