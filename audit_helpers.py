from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from audit_analyze import analyze_primary_record
from audit_extract import extract_cached_record_from_url, normalize_pdp_url
from audit_models import (
    create_audit_result_record,
    create_cached_pdp_record,
    create_competitor_graphics_assignment,
    create_product_audit_entry,
    create_selected_primary_image,
    make_image,
    make_key_feature,
    make_reviews_summary,
    normalize_title,
    title_length,
)


def _token_from_url(url: str, fallback: str) -> str:
    clean = (url or "").strip().rstrip("/")
    tail = clean.split("/")[-1] if clean else fallback
    token = re.sub(r"[^0-9a-z]+", "-", tail.lower()).strip("-")
    return token[:28] if token else fallback


def _mock_image_urls(seed: str, count: int = 4) -> list[str]:
    base = abs(hash(seed)) % 9000
    return [f"https://picsum.photos/seed/soapbox-{base + i}/320/320" for i in range(count)]


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def create_empty_cached_pdp_record(**kwargs: Any) -> dict[str, Any]:
    return create_cached_pdp_record(**kwargs)


def create_demo_cached_pdp_record(
    *,
    source_url: str,
    index: int,
    source_type: str = "primary",
    client_name: str = "",
    retailer: str = "",
) -> dict[str, Any]:
    token = _token_from_url(source_url, f"{source_type}-{index + 1}")
    title_prefix = "Soapbox Product" if source_type == "primary" else "Competitor Product"
    title = f"{title_prefix} {index + 1} - {token.replace('-', ' ').title()}"
    images = _mock_image_urls(f"{source_type}-{token}", count=4 if source_type == "primary" else 3)
    key_features = [
        make_key_feature(1, "Mock feature one"),
        make_key_feature(2, "Mock feature two"),
        make_key_feature(3, "Mock feature three"),
        make_key_feature(4, "Mock feature four"),
    ]
    description_body = (
        f"{title} is a mock extracted PDP description used for workflow testing. "
        "This placeholder validates review and editing behavior."
    )
    description_bullets = ["Mock detail one", "Mock detail two"]
    image_models = [make_image(i, u, is_hero=(i == 0)) for i, u in enumerate(images)]
    reviews = make_reviews_summary(
        average_rating=float(f"{4.0 + (index % 8) * 0.1:.1f}"),
        ratings_count=95 + index * 13,
        review_count=75 + index * 11,
    )

    return create_cached_pdp_record(
        client_name=client_name,
        retailer=retailer,
        source_url=source_url,
        source_type=source_type,
        item_id=f"ITM-{210000 + index}" if source_type == "primary" else f"CMP-{310000 + index}",
        brand="Mock Brand" if source_type == "primary" else f"Brand {index + 1}",
        product_title=title,
        category="Dairy",
        subcategory="Yogurt",
        current_title=title,
        current_description_body=description_body,
        current_description_bullets=description_bullets,
        description_section_labels=["Product Details", "Description"],
        current_key_features=key_features,
        key_features_section_label="Key Features",
        images=image_models,
        reviews_summary=reviews,
        extraction_status="mock_extracted",
    )


def create_mock_cached_pdp_record(
    *,
    source_url: str,
    index: int,
    source_type: str = "primary",
    client_name: str = "",
    retailer: str = "",
) -> dict[str, Any]:
    # Backward-compatible alias for earlier scaffold naming.
    return create_demo_cached_pdp_record(
        source_url=source_url,
        index=index,
        source_type=source_type,
        client_name=client_name,
        retailer=retailer,
    )


def create_empty_product_audit_entry(record_id: str, product_title: str = "", item_id: str = "") -> dict[str, Any]:
    return create_product_audit_entry(record_id=record_id, product_title=product_title, item_id=item_id)


def create_product_audit_entry_from_record(record: dict[str, Any]) -> dict[str, Any]:
    images = record.get("images", [])
    selected_primary_image = None
    if images:
        selected = images[0]
        selected_primary_image = create_selected_primary_image(
            record["record_id"],
            int(selected.get("index", 0)),
            selected.get("url", ""),
        )

    entry = create_product_audit_entry(
        record_id=record["record_id"],
        product_title=record.get("product_title", ""),
        item_id=record.get("item_id", ""),
        selected_primary_image=selected_primary_image,
        rule_findings=analyze_primary_record(record),
        status="loaded",
    )
    entry["cached_record"] = record
    return entry


def create_mock_product_audit_entry_from_record(record: dict[str, Any]) -> dict[str, Any]:
    # Backward-compatible alias for earlier scaffold references.
    return create_product_audit_entry_from_record(record)


def create_empty_audit_result_record(
    *,
    client_name: str = "",
    retailer: str = "",
    audit_date_value: date | None = None,
) -> dict[str, Any]:
    audit_date_text = audit_date_value.isoformat() if audit_date_value else ""
    return create_audit_result_record(
        client_name=client_name,
        retailer=retailer,
        audit_date=audit_date_text,
    )


def create_empty_competitor_assignment(record_id: str = "", image_index: int = 0, url: str = "") -> dict[str, Any]:
    return create_competitor_graphics_assignment(
        record_id=record_id,
        image_index=image_index,
        url=url,
        display_order=0,
    )


def initialize_auditing_session_state(state: dict[str, Any]) -> None:
    defaults = {
        "audit_generated": False,
        "audit_primary_source_method": "Single PDP URL",
        "audit_competitor_source_method": "Single PDP URL",
        "audit_primary_entries": [],
        "audit_competitor_entries": [],
        "audit_competitor_image_orders": {},
        "audit_competitor_assignments": [],
        "audit_cached_pdp_records": {},
        "audit_results_seeded_for": [],
        "audit_result_record": create_empty_audit_result_record(),
        "audit_export_plan": {},
        "audit_batch_preview": pd.DataFrame(),
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def urls_from_uploaded_dataframe(df_uploaded: pd.DataFrame, selected_col: str) -> list[str]:
    if df_uploaded.empty or selected_col not in df_uploaded.columns:
        return []
    return [str(v).strip() for v in df_uploaded[selected_col].dropna().tolist() if str(v).strip()]


def build_demo_primary_entries_from_urls(
    urls: list[str],
    *,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for i, url in enumerate(urls):
        record = create_demo_cached_pdp_record(
            source_url=url,
            index=i,
            source_type="primary",
            client_name=client_name,
            retailer=retailer,
        )
        entry = create_product_audit_entry_from_record(record)
        entries.append(entry)
        records_by_id[record["record_id"]] = record
    return entries, records_by_id


def build_primary_entries_from_urls(
    urls: list[str],
    *,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    # Backward-compatible alias for old mock-only naming.
    return build_demo_primary_entries_from_urls(urls, client_name=client_name, retailer=retailer)


def find_cached_record_by_normalized_url(
    cached_records_by_id: dict[str, dict[str, Any]],
    normalized_url: str,
    source_type: str | None = None,
) -> dict[str, Any] | None:
    if not normalized_url:
        return None
    for record in cached_records_by_id.values():
        if (record.get("source_url") or "").strip().lower() == normalized_url.strip().lower():
            if source_type and record.get("source_type") != source_type:
                continue
            if record.get("extraction_status") in {"success", "partial"}:
                return record
    return None


def process_primary_pdp_urls_real(
    *,
    urls: list[str],
    cached_records_by_id: dict[str, dict[str, Any]] | None,
    client_name: str,
    retailer: str,
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    new_records: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    cached_records_by_id = dict(cached_records_by_id or {})

    normalized_urls: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        normalized, normalize_error = normalize_pdp_url(raw)
        if normalize_error or not normalized:
            errors.append(f"URL skipped ({raw}): {normalize_error or 'invalid'}")
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)

    for idx, normalized_url in enumerate(normalized_urls[:max_count]):
        cached = find_cached_record_by_normalized_url(
            cached_records_by_id,
            normalized_url,
            source_type="primary",
        )
        if cached is not None:
            cached["last_used_at"] = _utc_now_iso()
            record = cached
        else:
            record, extract_errors = extract_cached_record_from_url(
                raw_url=normalized_url,
                source_type="primary",
                client_name=client_name,
                retailer=retailer,
            )
            if extract_errors:
                errors.extend([f"{normalized_url}: {e}" for e in extract_errors])
            if record is None:
                continue
            new_records[record["record_id"]] = record

        entry = create_product_audit_entry_from_record(record)
        entry["entry_id"] = f"primary-{idx + 1}-{entry.get('record_id', 'unknown')}"
        entries.append(entry)

    return entries, new_records, errors


def process_primary_pdp_urls(
    *,
    urls: list[str],
    cached_records_by_id: dict[str, dict[str, Any]] | None,
    client_name: str,
    retailer: str,
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    # Backward-compatible alias for earlier naming.
    return process_primary_pdp_urls_real(
        urls=urls,
        cached_records_by_id=cached_records_by_id,
        client_name=client_name,
        retailer=retailer,
        max_count=max_count,
    )


def process_competitor_pdp_urls_real(
    *,
    urls: list[str],
    cached_records_by_id: dict[str, dict[str, Any]] | None,
    client_name: str,
    retailer: str,
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    new_records: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    cached_records_by_id = dict(cached_records_by_id or {})

    normalized_urls: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        normalized, normalize_error = normalize_pdp_url(raw)
        if normalize_error or not normalized:
            errors.append(f"URL skipped ({raw}): {normalize_error or 'invalid'}")
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)

    for normalized_url in normalized_urls[:max_count]:
        cached = find_cached_record_by_normalized_url(
            cached_records_by_id,
            normalized_url,
            source_type="competitor",
        )
        if cached is not None:
            cached["last_used_at"] = _utc_now_iso()
            record = cached
        else:
            record, extract_errors = extract_cached_record_from_url(
                raw_url=normalized_url,
                source_type="competitor",
                client_name=client_name,
                retailer=retailer,
            )
            if extract_errors:
                errors.extend([f"{normalized_url}: {e}" for e in extract_errors])
            if record is None:
                continue
            new_records[record["record_id"]] = record
        records.append(record)

    return records, new_records, errors


def process_competitor_pdp_urls(
    *,
    urls: list[str],
    cached_records_by_id: dict[str, dict[str, Any]] | None,
    client_name: str,
    retailer: str,
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    # Backward-compatible alias for future call sites.
    return process_competitor_pdp_urls_real(
        urls=urls,
        cached_records_by_id=cached_records_by_id,
        client_name=client_name,
        retailer=retailer,
        max_count=max_count,
    )


def build_demo_competitor_records_from_urls(
    urls: list[str],
    *,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for i, url in enumerate(urls):
        record = create_demo_cached_pdp_record(
            source_url=url,
            index=i,
            source_type="competitor",
            client_name=client_name,
            retailer=retailer,
        )
        records.append(record)
        records_by_id[record["record_id"]] = record
    return records, records_by_id


def build_competitor_records_from_urls(
    urls: list[str],
    *,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    # Competitor extraction remains placeholder in this phase.
    return build_demo_competitor_records_from_urls(urls, client_name=client_name, retailer=retailer)


def update_record_tier1_derived_fields(record: dict[str, Any]) -> None:
    current_title = record.get("current_title", "")
    record["normalized_title"] = normalize_title(current_title)
    record["title_length"] = title_length(current_title)

    body = record.get("current_description_body", "")
    bullets = record.get("current_description_bullets", [])
    if body or bullets:
        combined = body.strip()
        bullet_lines = [f"- {b.strip()}" for b in bullets if str(b).strip()]
        record["current_description_combined"] = "\n".join([x for x in [combined, *bullet_lines] if x]).strip()
    else:
        record["current_description_combined"] = ""

    record["image_count"] = len(record.get("images", []))


def build_competitor_assignments(
    competitor_records: list[dict[str, Any]],
    display_orders: dict[str, int],
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for record in competitor_records:
        for image in record.get("images", []):
            key = f"{record['record_id']}|{image['index']}"
            order = int(display_orders.get(key, 0))
            if order > 0:
                assignments.append(
                    create_competitor_graphics_assignment(
                        record_id=record["record_id"],
                        image_index=int(image["index"]),
                        url=image["url"],
                        display_order=order,
                    )
                )
    assignments.sort(key=lambda x: (x["display_order"], x["record_id"], x["image_index"]))
    return assignments
