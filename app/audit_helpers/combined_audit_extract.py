from __future__ import annotations

import io
import json
from html.parser import HTMLParser
from typing import Any

import pandas as pd


SUPPORTED_SCHEMA_VERSION = "2.0"


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
            copied = dict(item)
            copied.setdefault("_combined_source_index", source_index)
            records.append(copied)
    return records


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
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        result["errors"].append(
            f"Unsupported combined audit schema version '{schema_version or '<missing>'}'. "
            f"Expected {SUPPORTED_SCHEMA_VERSION}."
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
        ),
        "Product Title": _first(
            record,
            "productTitle",
            "title",
            "name",
            "product.title",
            "productDetails.title",
        ),
        "Brand": _first(
            record,
            "brand",
            "brandName",
            "product.brand",
            "inputBrandName",
            "input.brandName",
        ),
        "Category": _first(
            record,
            "resolvedCategory",
            "category",
            "product.category",
            "productDetails.category",
        ),
        "Product Type": _first(
            record,
            "resolvedProductType",
            "productType",
            "subcategory",
            "resolvedFamily",
            "family",
            "productDetails.productType",
        ),
        "Description Body": _first(
            record,
            "description",
            "descriptionBody",
            "product.description",
            "productDetails.description",
        ),
        "Average Rating": _first(record, "averageRating", "ratings.average"),
        "Ratings Count": _first(record, "ratingsCount", "ratings.count"),
        "Review Count": _first(record, "reviewCount", "reviews.count"),
    }
    for index, image in enumerate(_image_values(record), start=1):
        row[f"Image {index}"] = image
    bullets = _text_values(
        record,
        "descriptionBullets",
        "bullets",
        "product.descriptionBullets",
        "productDetails.descriptionBullets",
    )
    features = _text_values(
        record,
        "keyFeatures",
        "features",
        "product.keyFeatures",
        "productDetails.keyFeatures",
    )
    for index, value in enumerate(bullets, start=1):
        row[f"Description Bullet {index}"] = value
    for index, value in enumerate(features, start=1):
        row[f"Key Feature {index}"] = value
    return pd.DataFrame([row])


def attach_combined_evidence_to_record(
    cached_record: dict[str, Any],
    source_record: dict[str, Any],
) -> None:
    ingest_metadata = cached_record.setdefault("ingest_metadata", {})
    ingest_metadata.update(
        {
            "combined_schema_version": SUPPORTED_SCHEMA_VERSION,
            "combined_source_index": source_record.get("_combined_source_index"),
            "source_row": _first(source_record, "sourceRow", "rowNumber"),
            "role": _first(
                source_record,
                "role",
                "inputRole",
                "sourceRole",
                "input.role",
            ),
            "input_brand_name": _first(
                source_record,
                "inputBrandName",
                "inputBrand",
                "requestedBrand",
                "input.brandName",
            ),
            "seller": _first(source_record, "seller", "sellerName"),
            "sold_by_walmart": _first(
                source_record,
                "soldByWalmart",
                "fulfillment.soldByWalmart",
            ),
            "shipped_by_walmart": _first(
                source_record,
                "shippedByWalmart",
                "fulfillment.shippedByWalmart",
            ),
            "enhanced_brand_content_status": _first(
                source_record,
                "enhancedBrandContentStatus",
                "enhancedContent.status",
            ),
            "combined_structured_evidence": source_record,
        }
    )
    cached_record["resolved_category"] = _safe_text(
        _first(source_record, "resolvedCategory")
    )
    cached_record["resolved_family"] = _safe_text(
        _first(source_record, "resolvedFamily", "family")
    )
    cached_record["resolved_product_type"] = _safe_text(
        _first(source_record, "resolvedProductType", "productType")
    )


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
    state.pop("audit_combined_extract_result", None)
    state["audit_ppt_bytes"] = None
    state["audit_ppt_filename"] = ""
