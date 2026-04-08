from __future__ import annotations

import base64
import html as html_lib
import io
import re
from urllib.parse import unquote
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


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("Could not decode CSV content.")


def _looks_like_csv(text: str) -> bool:
    sample = (text or "").strip().splitlines()[:3]
    if not sample:
        return False
    return any("," in line for line in sample)


def _unescape_js_string_literal(value: str) -> str:
    text = html_lib.unescape(value or "")
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return (
            text.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )


def _decode_embedded_csv_candidates_from_html(html_text: str) -> list[str]:
    candidates: list[str] = []
    if not html_text:
        return candidates

    specific_csv_encoded = re.compile(
        r"""(?is)\bcsvEncoded\b\s*[:=]\s*(?:decodeURIComponent\()?\s*([\"'`])(.*?)\1\s*\)?"""
    )
    for _, raw_literal in specific_csv_encoded.findall(html_text):
        literal = _unescape_js_string_literal(raw_literal).strip()
        if not literal:
            continue
        candidates.append(literal)
        decoded = unquote(literal)
        if decoded != literal:
            candidates.append(decoded)
        if re.fullmatch(r"[A-Za-z0-9+/=\s]+", literal) and len(literal) >= 40:
            try:
                decoded_b64 = base64.b64decode(literal, validate=True).decode("utf-8", errors="ignore")
                if decoded_b64:
                    candidates.append(decoded_b64)
            except Exception:
                pass

    decode_uri_calls = re.compile(r"""(?is)decodeURIComponent\(\s*([\"'`])(.*?)\1\s*\)""")
    for _, raw_literal in decode_uri_calls.findall(html_text):
        literal = _unescape_js_string_literal(raw_literal).strip()
        if not literal:
            continue
        candidates.append(unquote(literal))

    atob_calls = re.compile(r"""(?is)atob\(\s*([\"'`])([A-Za-z0-9+/=\s]+)\1\s*\)""")
    for _, raw_b64 in atob_calls.findall(html_text):
        try:
            decoded = base64.b64decode(raw_b64, validate=True).decode("utf-8", errors="ignore")
            if decoded:
                candidates.append(decoded)
        except Exception:
            continue

    assignment_pattern = re.compile(
        r"""(?is)\b([a-zA-Z0-9_]*csv[a-zA-Z0-9_]*)\b\s*[:=]\s*([\"'`])(.*?)\2"""
    )
    for _, _, raw_literal in assignment_pattern.findall(html_text):
        literal = _unescape_js_string_literal(raw_literal).strip()
        if not literal:
            continue
        candidates.append(literal)
        if "%" in literal:
            decoded = unquote(literal)
            if decoded != literal:
                candidates.append(decoded)
        if re.fullmatch(r"[A-Za-z0-9+/=\s]+", literal) and len(literal) >= 40:
            try:
                decoded_b64 = base64.b64decode(literal, validate=True).decode("utf-8", errors="ignore")
                if decoded_b64:
                    candidates.append(decoded_b64)
            except Exception:
                pass

    data_uri_pattern = re.compile(r"(?is)data:text/csv;base64,([A-Za-z0-9+/=\s]+)")
    for raw_b64 in data_uri_pattern.findall(html_text):
        try:
            decoded_b64 = base64.b64decode(raw_b64, validate=True).decode("utf-8", errors="ignore")
            if decoded_b64:
                candidates.append(decoded_b64)
        except Exception:
            continue

    return candidates


def _looks_like_audit_extract_df(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    requiredish = {"product url", "product id", "product title"}
    if requiredish.issubset(cols):
        return True
    return bool(
        any(re.match(r"^image\s*\d+$", c, flags=re.IGNORECASE) for c in df.columns)
        or any(re.match(r"^description bullet\s*\d+$", c, flags=re.IGNORECASE) for c in df.columns)
        or any(re.match(r"^key feature\s*\d+$", c, flags=re.IGNORECASE) for c in df.columns)
    )


def _parse_html_audit_extract(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    messages: list[str] = []
    html_text = raw.decode("utf-8", errors="ignore")
    if not html_text.strip():
        return pd.DataFrame(), ["Uploaded HTML file appears empty."]

    for candidate in _decode_embedded_csv_candidates_from_html(html_text):
        if not _looks_like_csv(candidate):
            continue
        try:
            df = _read_csv_bytes(candidate.encode("utf-8", errors="ignore"))
            if not df.empty and _looks_like_audit_extract_df(df):
                messages.append("Parsed embedded CSV payload from HTML report.")
                return df, messages
        except Exception:
            continue

    try:
        tables = pd.read_html(io.StringIO(html_text))
        for table in tables:
            if isinstance(table, pd.DataFrame) and not table.empty and _looks_like_audit_extract_df(table):
                messages.append("Embedded CSV payload not found; parsed visible HTML table instead.")
                return table, messages
        return pd.DataFrame(), ["No usable table data found in HTML report."]
    except Exception as exc:
        return pd.DataFrame(), [f"Could not parse HTML report table data: {exc}"]


def parse_audit_extract_upload_to_dataframe(uploaded: Any) -> tuple[pd.DataFrame, list[str]]:
    """Parse uploaded audit extract files (CSV/XLSX/XLS/HTML) into a DataFrame."""
    if uploaded is None:
        return pd.DataFrame(), ["No file was uploaded."]

    filename = str(getattr(uploaded, "name", "") or "").lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""

    try:
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    except Exception as exc:
        return pd.DataFrame(), [f"Could not read uploaded file bytes: {exc}"]

    if not raw:
        return pd.DataFrame(), ["Uploaded file is empty."]

    try:
        if ext == "csv":
            return _read_csv_bytes(raw), []
        if ext in {"xlsx", "xls"}:
            return pd.read_excel(io.BytesIO(raw)), []
        if ext in {"html", "htm"}:
            return _parse_html_audit_extract(raw)

        # Fallback sniffing for unknown extension.
        text_head = raw[:512].decode("utf-8", errors="ignore").lower()
        if "<html" in text_head or "<table" in text_head:
            return _parse_html_audit_extract(raw)
        try:
            return _read_csv_bytes(raw), []
        except Exception:
            return pd.read_excel(io.BytesIO(raw)), []
    except Exception as exc:
        return pd.DataFrame(), [f"Could not parse uploaded audit extract file: {exc}"]


def _coerce_columns(columns_or_row: pd.DataFrame | pd.Series | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(columns_or_row, pd.DataFrame):
        return [str(c) for c in columns_or_row.columns]
    if isinstance(columns_or_row, pd.Series):
        return [str(c) for c in columns_or_row.index.tolist()]
    return [str(c) for c in columns_or_row]


def _sorted_numbered_columns(
    columns_or_row: pd.DataFrame | pd.Series | list[str] | tuple[str, ...],
    prefix: str,
) -> list[str]:
    columns = _coerce_columns(columns_or_row)
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(\d+)$", re.IGNORECASE)
    indexed: list[tuple[int, str]] = []
    for col in columns:
        match = pattern.match(col.strip())
        if match:
            indexed.append((int(match.group(1)), col))
    indexed.sort(key=lambda x: x[0])
    return [col for _, col in indexed]


def detect_image_columns(
    columns_or_row: pd.DataFrame | pd.Series | list[str] | tuple[str, ...],
) -> list[str]:
    """Return all Image N columns in numeric order."""
    return _sorted_numbered_columns(columns_or_row, "Image")


def detect_description_bullet_columns(
    columns_or_row: pd.DataFrame | pd.Series | list[str] | tuple[str, ...],
) -> list[str]:
    """Return all Description Bullet N columns in numeric order."""
    return _sorted_numbered_columns(columns_or_row, "Description Bullet")


def detect_key_feature_columns(
    columns_or_row: pd.DataFrame | pd.Series | list[str] | tuple[str, ...],
) -> list[str]:
    """Return all Key Feature(s) N columns in numeric order."""
    single = _sorted_numbered_columns(columns_or_row, "Key Feature")
    plural = _sorted_numbered_columns(columns_or_row, "Key Features")
    seen: set[str] = set()
    merged: list[str] = []
    for col in [*single, *plural]:
        key = col.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(col)
    merged.sort(
        key=lambda c: int(
            re.search(r"(\d+)$", c.strip()).group(1)  # type: ignore[union-attr]
        )
        if re.search(r"(\d+)$", c.strip())
        else 0
    )
    return merged


def _cell_as_text(row: pd.Series, col: str) -> str:
    if col not in row.index:
        return ""
    value = row[col]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _cell_as_float(row: pd.Series, col: str) -> float | None:
    text = _cell_as_text(row, col)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _cell_as_int(row: pd.Series, col: str) -> int | None:
    text = _cell_as_text(row, col)
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except ValueError:
        return None


def extract_image_urls_from_sheet_row(row: pd.Series) -> list[str]:
    urls: list[str] = []
    for col in detect_image_columns(row):
        value = _cell_as_text(row, col)
        if value:
            urls.append(value)
    return urls


def extract_description_bullets_from_sheet_row(row: pd.Series) -> list[str]:
    bullets: list[str] = []
    for col in detect_description_bullet_columns(row):
        value = _cell_as_text(row, col)
        if value:
            bullets.append(value)
    return bullets


def extract_key_features_from_sheet_row(row: pd.Series) -> list[str]:
    bullets: list[str] = []
    for col in detect_key_feature_columns(row):
        value = _cell_as_text(row, col)
        if value:
            bullets.append(value)
    return bullets


def extract_feature_bullets_from_sheet_row(row: pd.Series) -> list[str]:
    description_bullets = extract_description_bullets_from_sheet_row(row)
    if description_bullets:
        return description_bullets
    return extract_key_features_from_sheet_row(row)


def validate_audit_extract_sheet_columns(df_uploaded: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Lightweight schema check for extension Audit Extract Sheets."""
    errors: list[str] = []
    warnings: list[str] = []
    required_cols = ["Product URL", "Product ID", "Product Title"]
    missing = [c for c in required_cols if c not in df_uploaded.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")

    if not detect_image_columns(df_uploaded):
        warnings.append("No Image N columns were found. Records will be ingested without images.")
    if not detect_description_bullet_columns(df_uploaded) and not detect_key_feature_columns(df_uploaded):
        warnings.append(
            "No Description Bullet N or Key Feature N columns were found. Bullet/key-feature fields will be empty."
        )

    return errors, warnings


def _validate_required_sheet_row_fields(row: pd.Series, row_number: int) -> list[str]:
    row_errors: list[str] = []
    for col in ["Product URL", "Product ID", "Product Title"]:
        if not _cell_as_text(row, col):
            row_errors.append(f"Row {row_number}: missing required value for '{col}'.")
    return row_errors


def map_sheet_row_to_cached_record(
    *,
    row: pd.Series,
    source_type: str,
    client_name: str = "",
    retailer: str = "",
) -> dict[str, Any]:
    """Map a single Audit Extract Sheet row into a normalized cached PDP record."""
    image_urls = extract_image_urls_from_sheet_row(row)
    images = [make_image(i, url, is_hero=(i == 0)) for i, url in enumerate(image_urls)]

    bullets = extract_feature_bullets_from_sheet_row(row)
    key_features = [make_key_feature(i + 1, text) for i, text in enumerate(bullets)]

    reviews = make_reviews_summary(
        average_rating=_cell_as_float(row, "Average Rating"),
        ratings_count=_cell_as_int(row, "Ratings Count"),
        review_count=_cell_as_int(row, "Review Count"),
    )

    ingest_metadata = {
        "title_count": _cell_as_text(row, "Title Count"),
        "title_notes": _cell_as_text(row, "Title Notes"),
        "description_count": _cell_as_text(row, "Description Count"),
        "description_notes": _cell_as_text(row, "Description Notes"),
        "description_bullet_count": _cell_as_text(row, "Description Bullet Count"),
        "description_bullet_notes": _cell_as_text(row, "Description Bullet Notes"),
        "content_score": _cell_as_text(row, "Content Score"),
    }

    record = create_cached_pdp_record(
        client_name=client_name,
        retailer=retailer,
        source_url=_cell_as_text(row, "Product URL"),
        source_type=source_type,
        item_id=_cell_as_text(row, "Product ID"),
        brand=_cell_as_text(row, "Brand"),
        product_title=_cell_as_text(row, "Product Title"),
        category=_cell_as_text(row, "Category"),
        subcategory=_cell_as_text(row, "Product Type"),
        current_title=_cell_as_text(row, "Product Title"),
        current_description_body=_cell_as_text(row, "Description Body"),
        current_description_bullets=bullets,
        current_key_features=key_features,
        images=images,
        reviews_summary=reviews,
        extraction_status="sheet_ingested",
        extraction_errors=[],
        ingest_metadata=ingest_metadata,
    )
    update_record_tier1_derived_fields(record)
    return record


def process_primary_audit_extract_sheet(
    *,
    df_uploaded: pd.DataFrame,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    errors, warnings = validate_audit_extract_sheet_columns(df_uploaded)
    all_messages = list(errors) + [f"Warning: {w}" for w in warnings]
    if errors or df_uploaded.empty:
        return [], {}, all_messages

    entries: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for idx, (_, row) in enumerate(df_uploaded.iterrows()):
        row_number = idx + 2
        row_errors = _validate_required_sheet_row_fields(row, row_number)
        if row_errors:
            all_messages.extend(row_errors)
            continue
        try:
            record = map_sheet_row_to_cached_record(
                row=row,
                source_type="primary",
                client_name=client_name,
                retailer=retailer,
            )
        except Exception as exc:
            all_messages.append(f"Row {row_number}: failed to ingest sheet row ({exc}).")
            continue
        entry = create_product_audit_entry_from_record(record)
        entry["entry_id"] = f"primary-sheet-{idx + 1}-{entry.get('record_id', 'unknown')}"
        entries.append(entry)
        records_by_id[record["record_id"]] = record

    return entries, records_by_id, all_messages


def process_competitor_audit_extract_sheet(
    *,
    df_uploaded: pd.DataFrame,
    client_name: str = "",
    retailer: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    errors, warnings = validate_audit_extract_sheet_columns(df_uploaded)
    all_messages = list(errors) + [f"Warning: {w}" for w in warnings]
    if errors or df_uploaded.empty:
        return [], {}, all_messages

    records: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for idx, (_, row) in enumerate(df_uploaded.iterrows()):
        row_number = idx + 2
        row_errors = _validate_required_sheet_row_fields(row, row_number)
        if row_errors:
            all_messages.extend(row_errors)
            continue
        try:
            record = map_sheet_row_to_cached_record(
                row=row,
                source_type="competitor",
                client_name=client_name,
                retailer=retailer,
            )
        except Exception as exc:
            all_messages.append(f"Row {row_number}: failed to ingest sheet row ({exc}).")
            continue
        records.append(record)
        records_by_id[record["record_id"]] = record

    return records, records_by_id, all_messages


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
            if 1 <= order <= 10:
                assignment = create_competitor_graphics_assignment(
                    record_id=record["record_id"],
                    image_index=int(image["index"]),
                    url=image["url"],
                    display_order=order,
                )
                assignment["product_title"] = record.get("product_title", "")
                assignment["brand"] = record.get("brand", "")
                assignment["item_id"] = record.get("item_id", "")
                assignment["source_url"] = record.get("source_url", "")
                assignments.append(assignment)
    assignments.sort(key=lambda x: (x["display_order"], x["record_id"], x["image_index"]))
    return assignments
