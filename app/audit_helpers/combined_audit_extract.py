from __future__ import annotations

import io
import json
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from audit_models import (
    create_cached_pdp_record,
    make_image,
    make_key_feature,
    make_reviews_summary,
)


SUPPORTED_SCHEMA_VERSIONS = {"2.0", "2.2"}
LATEST_SUPPORTED_SCHEMA_VERSION = "2.2"


class _AuditDataScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture = False
        self._parts: list[str] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attr_map = {str(key).lower(): str(value or "") for key, value in attrs}
        if attr_map.get("id") == "soapbox-audit-data":
            self._capture = True
            self.found = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._capture:
            self._capture = False

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    @property
    def payload(self) -> str:
        return "".join(self._parts).strip()


def _read_uploaded_text(uploaded_file: Any) -> tuple[str, list[str]]:
    if uploaded_file is None:
        return "", ["No combined HTML report was uploaded."]
    try:
        raw = (
            uploaded_file.getvalue()
            if hasattr(uploaded_file, "getvalue")
            else uploaded_file.read()
        )
    except Exception as exc:
        return "", [f"Could not read combined HTML report: {exc}"]
    if isinstance(raw, str):
        return raw, []
    if not raw:
        return "", ["The combined HTML report is empty."]
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return bytes(raw).decode(encoding), []
        except UnicodeDecodeError:
            continue
    return bytes(raw).decode("utf-8", errors="replace"), [
        "The combined HTML report contained invalid UTF-8; replacement characters were used."
    ]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_role(value: Any) -> str:
    role = _safe_text(value).lower().replace("_", " ").replace("-", " ")
    role = " ".join(role.split())
    return {
        "client": "Client",
        "primary": "Client",
        "competitor": "Competitor",
        "benchmark": "Benchmark",
        "current": "Current",
        "current search": "Current",
        "benchmark search": "Benchmark",
        "client brand shop": "Client",
        "competitor brand shop": "Competitor",
    }.get(role, "")


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


def _messages(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _ordered_records(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    value: Any = []
    for key in keys:
        if key in payload:
            value = payload.get(key)
            break
    if isinstance(value, dict):
        value = _first(value, "records", "items", "captures", "evidence", default=[])
    records: list[dict[str, Any]] = []
    for source_index, item in enumerate(_as_list(value)):
        if isinstance(item, dict):
            nested_data = item.get("data")
            copied = dict(nested_data) if isinstance(nested_data, dict) else {}
            copied.update({key: value for key, value in item.items() if key != "data"})
            if isinstance(nested_data, dict):
                copied["data"] = nested_data
            copied.setdefault("_combined_source_index", source_index)
            records.append(copied)
    return records


def _drop_failed_empty_records(
    records: list[dict[str, Any]],
    *,
    evidence_label: str,
    warnings: list[Any],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        status = _safe_text(
            _first(record, "status", "extractionStatus")
        ).lower()
        nested_data = record.get("data")
        usable_data = isinstance(nested_data, dict) and any(
            value not in (None, "", [], {})
            for value in nested_data.values()
        )
        usable_data = usable_data or any(
            _first(
                record,
                key,
                default="",
            )
            not in (None, "", [], {})
            for key in (
                "productId",
                "productTitle",
                "searchTerm",
                "brandName",
                "modules",
                "screenshotDataUrl",
            )
        )
        if status in {"failed", "fail", "error"} and not usable_data:
            warnings.append(
                {
                    "source": evidence_label,
                    "row": _first(record, "sourceRow", "rowNumber", default=index),
                    "message": f"{evidence_label} record failed with no usable nested data and was skipped.",
                }
            )
            continue
        kept.append(record)
    return kept


def _separate_by_role(
    records: list[dict[str, Any]],
    *,
    allowed: tuple[str, ...],
    evidence_label: str,
    warnings: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    separated = {role: [] for role in allowed}
    for index, record in enumerate(records, start=1):
        role = _normalized_role(
            _first(
                record,
                "role",
                "inputRole",
                "sourceRole",
                "auditRole",
                "searchRole",
                "brandShopRole",
                "input.role",
                "source.role",
            )
        )
        if not role:
            warnings.append(
                {
                    "source": evidence_label,
                    "row": _first(record, "sourceRow", "rowNumber", default=index),
                    "message": f"{evidence_label} record has a blank or unsupported role and was not assigned.",
                }
            )
            continue
        if role not in allowed:
            warnings.append(
                {
                    "source": evidence_label,
                    "row": _first(record, "sourceRow", "rowNumber", default=index),
                    "message": f"{evidence_label} role '{role}' is not valid for this evidence type.",
                }
            )
            continue
        separated[role].append(record)
    return separated


def parse_combined_audit_html(uploaded_file: Any) -> dict[str, Any]:
    html_text, read_errors = _read_uploaded_text(uploaded_file)
    result: dict[str, Any] = {
        "schema_version": "",
        "is_legacy": False,
        "client_pdps": [],
        "competitor_pdps": [],
        "current_searches": [],
        "benchmark_searches": [],
        "client_brand_shops": [],
        "competitor_brand_shops": [],
        "warnings": [],
        "errors": list(read_errors),
        "raw_payload": {},
    }
    if not html_text:
        return result

    parser = _AuditDataScriptParser()
    try:
        parser.feed(html_text)
    except Exception as exc:
        result["errors"].append(f"Could not inspect combined HTML report: {exc}")
        return result
    if not parser.found:
        result["is_legacy"] = True
        result["errors"].append(
            "The uploaded file is a legacy PDP-only report because "
            "#soapbox-audit-data is missing. Use the existing legacy upload workflow."
        )
        return result
    try:
        payload = json.loads(parser.payload)
    except Exception as exc:
        result["errors"].append(f"Embedded soapbox-audit-data JSON is malformed: {exc}")
        return result
    if not isinstance(payload, dict):
        result["errors"].append("Embedded soapbox-audit-data JSON must be an object.")
        return result

    result["raw_payload"] = payload
    schema_version = _safe_text(
        payload.get("schemaVersion") or payload.get("schema_version")
    )
    result["schema_version"] = schema_version
    if not schema_version or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        result["errors"].append(
            f"Unsupported combined audit schema version '{schema_version or '<missing>'}'. "
            f"Expected one of: 2.0, 2.2."
        )
        return result

    result["warnings"].extend(_messages(payload, "warnings"))
    result["errors"].extend(_messages(payload, "errors"))
    pdps = _ordered_records(payload, "pdpEvidence", "pdp_evidence")
    searches = _ordered_records(payload, "searchEvidence", "search_evidence")
    brand_shops = _ordered_records(
        payload,
        "brandShopEvidence",
        "brand_shop_evidence",
    )
    pdps = _drop_failed_empty_records(
        pdps,
        evidence_label="PDP",
        warnings=result["warnings"],
    )
    searches = _drop_failed_empty_records(
        searches,
        evidence_label="Search",
        warnings=result["warnings"],
    )
    brand_shops = _drop_failed_empty_records(
        brand_shops,
        evidence_label="Brand Shop",
        warnings=result["warnings"],
    )
    for evidence_label, records in (
        ("PDP", pdps),
        ("Search", searches),
        ("Brand Shop", brand_shops),
    ):
        for fallback_row, record in enumerate(records, start=1):
            row = _first(record, "sourceRow", "rowNumber", default=fallback_row)
            for warning in _as_list(record.get("warnings")):
                result["warnings"].append(
                    {"source": evidence_label, "row": row, "message": warning}
                )
            for error in _as_list(record.get("errors")):
                result["errors"].append(
                    {"source": evidence_label, "row": row, "message": error}
                )

    pdp_roles = _separate_by_role(
        pdps,
        allowed=("Client", "Competitor"),
        evidence_label="PDP",
        warnings=result["warnings"],
    )
    search_roles = _separate_by_role(
        searches,
        allowed=("Current", "Benchmark"),
        evidence_label="Search",
        warnings=result["warnings"],
    )
    shop_roles = _separate_by_role(
        brand_shops,
        allowed=("Client", "Competitor"),
        evidence_label="Brand Shop",
        warnings=result["warnings"],
    )
    result["client_pdps"] = pdp_roles["Client"]
    result["competitor_pdps"] = pdp_roles["Competitor"]
    result["current_searches"] = search_roles["Current"]
    result["benchmark_searches"] = search_roles["Benchmark"]
    result["client_brand_shops"] = shop_roles["Client"]
    result["competitor_brand_shops"] = shop_roles["Competitor"]

    if not searches:
        result["warnings"].append(
            "No Search evidence was included; Slide 3 cannot yet be populated."
        )
    if not brand_shops:
        result["warnings"].append(
            "No Brand Shop evidence was included; Slide 5 cannot yet be populated."
        )
    return result


def _image_values(record: dict[str, Any]) -> list[str]:
    raw_images = _first(
        record,
        "images",
        "imageUrls",
        "imageURLs",
        "product.images",
        "productDetails.images",
        default=[],
    )
    values: list[str] = []
    for image in _as_list(raw_images):
        if isinstance(image, dict):
            value = _first(
                image,
                "url",
                "src",
                "dataUrl",
                "dataURL",
                "screenshotDataUrl",
            )
        else:
            value = image
        text = _safe_text(value)
        if text:
            values.append(text)
    return values


def _text_values(record: dict[str, Any], *keys: str) -> list[str]:
    value = _first(record, *keys, default=[])
    values: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = _safe_text(
                _first(item, "text", "value", "label", "description")
            )
        else:
            text = _safe_text(item)
        if text:
            values.append(text)
    return values


def combined_pdp_to_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {
        "Product URL": _first(
            record,
            "productUrl",
            "productURL",
            "sourceUrl",
            "url",
            "product.url",
            "productDetails.url",
            "data.url",
        ),
        "Product ID": _first(
            record,
            "productId",
            "productID",
            "itemId",
            "itemID",
            "sku",
            "product.id",
            "productDetails.productId",
            "data.productId",
        ),
        "Product Title": _first(
            record,
            "productTitle",
            "title",
            "name",
            "product.title",
            "productDetails.title",
            "data.productTitle",
        ),
        "Brand": _first(
            record,
            "brand",
            "brandName",
            "product.brand",
            "inputBrandName",
            "input.brandName",
            "data.brand",
        ),
        "Category": _first(
            record,
            "resolvedCategory",
            "category",
            "product.category",
            "productDetails.category",
            "categoryPathName",
            "data.categoryPathName",
        ),
        "Product Type": _first(
            record,
            "resolvedProductType",
            "productType",
            "subcategory",
            "resolvedFamily",
            "family",
            "productDetails.productType",
            "data.productType",
        ),
        "Description Body": _first(
            record,
            "description",
            "descriptionBody",
            "product.description",
            "productDetails.description",
            "data.descriptionBody",
        ),
        "Average Rating": _first(record, "averageRating", "ratings.average", "data.averageRating"),
        "Ratings Count": _first(record, "ratingsCount", "ratings.count", "data.ratingsCount"),
        "Review Count": _first(record, "reviewCount", "reviews.count", "data.reviewCount"),
        "Image Count": _first(record, "imageCount", "data.imageCount"),
    }
    for index, image in enumerate(_image_values(record), start=1):
        row[f"Image {index}"] = image
    bullets = _text_values(
        record,
        "descriptionBullets",
        "bullets",
        "product.descriptionBullets",
        "productDetails.descriptionBullets",
        "data.descriptionBullets",
    )
    features = _text_values(
        record,
        "keyFeatures",
        "features",
        "product.keyFeatures",
        "productDetails.keyFeatures",
        "data.keyFeatures",
    )
    for index, value in enumerate(bullets, start=1):
        row[f"Description Bullet {index}"] = value
    for index, value in enumerate(features, start=1):
        row[f"Key Feature {index}"] = value
    return pd.DataFrame([row])


def _number(value: Any, converter: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return converter(value)
    except (TypeError, ValueError):
        return None


def map_schema2_pdp_to_cached_record(
    source_record: dict[str, Any],
    *,
    role: str,
    client_name: str = "",
    retailer: str = "",
) -> tuple[dict[str, Any] | None, list[str]]:
    """Map one schema 2.x PDP envelope directly into the existing cached model."""
    row = _first(source_record, "sourceRow", "rowNumber", default="?")
    item_id = _safe_text(
        _first(source_record, "productId", "productID", "itemId", "data.productId")
    )
    product_title = _safe_text(
        _first(source_record, "productTitle", "title", "data.productTitle")
    )
    errors: list[str] = []
    if not item_id:
        errors.append(f"Row {row}: missing required schema 2.0 productId.")
    if not product_title:
        errors.append(f"Row {row}: missing required schema 2.0 productTitle.")
    if errors:
        return None, errors

    raw_images = _first(source_record, "images", "data.images", default=[])
    images: list[dict[str, Any]] = []
    for position, raw_image in enumerate(_as_list(raw_images)):
        image = raw_image if isinstance(raw_image, dict) else {}
        url = _safe_text(
            _first(image, "url", "src", "dataUrl", "dataURL")
            if image
            else raw_image
        )
        if not url:
            continue
        raw_index = image.get("index", position)
        try:
            image_index = int(raw_index)
        except (TypeError, ValueError):
            image_index = position
        mapped = make_image(image_index, url, is_hero=bool(image.get("isHero", position == 0)))
        mapped["width"] = _number(image.get("width"), int)
        mapped["height"] = _number(image.get("height"), int)
        if mapped["width"] and mapped["height"]:
            mapped["dimensions"] = f"{mapped['width']} x {mapped['height']}"
            mapped["dimensions_text"] = f"{mapped['width']} W x {mapped['height']} H"
        else:
            mapped["dimensions"] = ""
            mapped["dimensions_text"] = ""
        mapped["show_dimensions_in_powerpoint"] = False
        images.append(mapped)

    description_bullets = _text_values(
        source_record,
        "descriptionBullets",
        "data.descriptionBullets",
    )
    key_feature_values = _text_values(
        source_record,
        "keyFeatures",
        "data.keyFeatures",
    )
    key_features = [
        make_key_feature(index + 1, text)
        for index, text in enumerate(key_feature_values)
    ]
    source_type = "primary" if role == "Client" else "competitor"
    status = _safe_text(
        _first(source_record, "status", "extractionStatus", default="success")
    )
    row_errors = [
        _safe_text(value)
        for value in _as_list(source_record.get("errors"))
        if _safe_text(value)
    ]
    record = create_cached_pdp_record(
        client_name=client_name,
        retailer=retailer,
        source_url=_safe_text(_first(source_record, "url", "productUrl", "data.url")),
        source_type=source_type,
        item_id=item_id,
        brand=_safe_text(
            _first(
                source_record,
                "brand",
                "data.brand",
                "brandName",
                "inputBrandName",
            )
        ),
        product_title=product_title,
        category=_safe_text(
            _first(
                source_record,
                "categoryPathName",
                "resolvedCategory",
                "category",
                "data.categoryPathName",
            )
        ),
        subcategory=_safe_text(
            _first(
                source_record,
                "productType",
                "resolvedProductType",
                "data.productType",
            )
        ),
        current_title=product_title,
        current_description_body=_safe_text(
            _first(source_record, "descriptionBody", "data.descriptionBody")
        ),
        current_description_bullets=description_bullets,
        current_key_features=key_features,
        images=images,
        reviews_summary=make_reviews_summary(
            average_rating=_number(
                _first(source_record, "averageRating", "data.averageRating", default=None),
                float,
            ),
            ratings_count=_number(
                _first(source_record, "ratingsCount", "data.ratingsCount", default=None),
                int,
            ),
            review_count=_number(
                _first(source_record, "reviewCount", "data.reviewCount", default=None),
                int,
            ),
        ),
        extraction_status=status or "success",
        extraction_errors=row_errors,
    )
    attach_combined_evidence_to_record(record, source_record)
    record.update(
        {
            "Product ID": item_id,
            "Product Title": product_title,
            "Product URL": record["source_url"],
            "Brand": record["brand"],
            "Category": record["category"],
            "Product Type": record["subcategory"],
            "Image Count": record["image_count"],
            "Description Body": record["current_description_body"],
            "Average Rating": record["reviews_summary"].get("average_rating"),
            "Review Count": record["reviews_summary"].get("review_count"),
            "Role": role,
            "Input Brand Name": _safe_text(_first(source_record, "inputBrandName")),
            "Seller Name": record["ingest_metadata"].get("seller"),
            "Sold by Walmart": record["ingest_metadata"].get("sold_by_walmart"),
            "Shipped by Walmart": record["ingest_metadata"].get("shipped_by_walmart"),
            "Enhanced Brand Content": record["ingest_metadata"].get(
                "enhanced_brand_content_status"
            ),
            "product_id": item_id,
            "product_url": record["source_url"],
            "product_type": record["subcategory"],
            "description_body": record["current_description_body"],
            "average_rating": record["reviews_summary"].get("average_rating"),
            "review_count": record["reviews_summary"].get("review_count"),
            "role": role,
            "input_brand_name": _safe_text(_first(source_record, "inputBrandName")),
        }
    )
    for position, image in enumerate(images, start=1):
        record[f"Image {position}"] = image["url"]
    return record, []


def attach_combined_evidence_to_record(
    cached_record: dict[str, Any],
    source_record: dict[str, Any],
) -> None:
    ingest_metadata = cached_record.setdefault("ingest_metadata", {})
    ingest_metadata.update(
        {
            "combined_schema_version": (
                _safe_text(
                    source_record.get("schemaVersion")
                    or source_record.get("schema_version")
                )
                or LATEST_SUPPORTED_SCHEMA_VERSION
            ),
            "combined_source_index": source_record.get("_combined_source_index"),
            "source_row": _first(source_record, "sourceRow", "rowNumber"),
            "role": _first(
                source_record,
                "role",
                "inputRole",
                "sourceRole",
                "input.role",
            ),
            "original_role": _first(source_record, "originalRole"),
            "input_brand_name": _first(
                source_record,
                "inputBrandName",
                "inputBrand",
                "requestedBrand",
                "input.brandName",
            ),
            "seller": _first(source_record, "seller", "sellerName", "data.sellerName"),
            "sold_by_walmart": _first(
                source_record,
                "soldByWalmart",
                "fulfillment.soldByWalmart",
                "data.soldByWalmart",
            ),
            "shipped_by_walmart": _first(
                source_record,
                "shippedByWalmart",
                "fulfillment.shippedByWalmart",
                "data.shippedByWalmart",
            ),
            "enhanced_brand_content_status": _first(
                source_record,
                "enhancedBrandContentStatus",
                "enhancedContent.status",
                "enhancedBrandContentPresent",
                "data.enhancedBrandContentPresent",
            ),
            "reported_image_count": _first(source_record, "imageCount", "data.imageCount"),
            "row_warnings": list(_as_list(source_record.get("warnings"))),
            "row_errors": list(_as_list(source_record.get("errors"))),
            "combined_structured_evidence": source_record,
        }
    )
    cached_record["resolved_category"] = _safe_text(
        _first(source_record, "resolvedCategory", "categoryPathName", "data.categoryPathName")
    )
    cached_record["resolved_family"] = _safe_text(
        _first(source_record, "resolvedFamily", "family")
    )
    cached_record["resolved_product_type"] = _safe_text(
        _first(source_record, "resolvedProductType", "productType", "data.productType")
    )
    extraction_status = _safe_text(
        _first(source_record, "status", "extractionStatus")
    )
    if extraction_status:
        cached_record["extraction_status"] = extraction_status
    cached_record["extraction_errors"] = [
        _safe_text(value)
        for value in _as_list(source_record.get("errors"))
        if _safe_text(value)
    ]
    raw_images = _first(source_record, "images", "data.images", default=[])
    normalized_images: list[dict[str, Any]] = []
    for position, image in enumerate(_as_list(raw_images)):
        if not isinstance(image, dict):
            url = _safe_text(image)
            image = {}
        else:
            url = _safe_text(_first(image, "url", "src", "dataUrl", "dataURL"))
        if not url:
            continue
        raw_index = image.get("index", position)
        try:
            image_index = int(raw_index)
        except (TypeError, ValueError):
            image_index = position
        normalized_images.append(
            {
                "index": image_index,
                "url": url,
                "is_hero": bool(image.get("isHero", image_index == 0)),
                "width": image.get("width"),
                "height": image.get("height"),
                "dimensions": (
                    f"{image.get('width')} x {image.get('height')}"
                    if image.get("width") and image.get("height")
                    else ""
                ),
                "dimensions_text": (
                    f"{image.get('width')} W x {image.get('height')} H"
                    if image.get("width") and image.get("height")
                    else ""
                ),
                "show_dimensions_in_powerpoint": False,
            }
        )
    if normalized_images:
        cached_record["images"] = normalized_images
        cached_record["image_count"] = len(normalized_images)


def reset_combined_audit_state(state: dict[str, Any]) -> None:
    preserved_keys = {
        "audit_client_name",
        "audit_retailer",
        "audit_date",
        "audit_template_version",
        "audit_combined_extract_upload",
    }
    dynamic_prefixes = (
        "audit_v2_primary_",
        "audit_v2_comp_",
        "audit_competitor_",
    )
    for key in list(state):
        if key in preserved_keys:
            continue
        if key.startswith(dynamic_prefixes):
            state.pop(key, None)
    state["audit_primary_entries"] = []
    state["audit_competitor_entries"] = []
    state["audit_cached_pdp_records"] = {}
    state["audit_search_evidence"] = {
        "current": [],
        "benchmark": [],
        "all": [],
    }
    state["audit_brand_shop_evidence"] = {
        "client": [],
        "competitor": [],
        "all": [],
    }
    state["audit_competitor_assignments"] = []
    state["audit_competitor_image_orders"] = {}
    state["audit_generated"] = False
    state["audit_results_seeded_for"] = []
    state["audit_result_record"] = {}
    state["audit_export_plan"] = {}
    state["audit_generated_export_plan"] = {}
    state["audit_cleaned_export_plan"] = {}
    state["audit_active_export_plan_source"] = "generated"
    state["audit_slide_cleanup_status"] = {
        "has_run": False,
        "succeeded": False,
        "active": False,
        "slides_cleaned": [],
        "slides_skipped": [],
        "warnings": [],
    }
    state.pop("audit_combined_extract_result", None)
    state["audit_ppt_bytes"] = None
    state["audit_ppt_filename"] = ""
