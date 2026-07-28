from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.quote_metadata import add_calendar_months


DRAFT_SCHEMA_VERSION = 1

QUOTE_METADATA_KEYS = {
    "quote_title": "photo_pricing_quote_title",
    "reference_number": "photo_pricing_reference_number",
    "quote_created_date": "photo_pricing_quote_created_date",
    "quote_expiration_date": "photo_pricing_quote_expiration_date",
    "expiration_overridden": "photo_pricing_expiration_overridden",
}

PRICING_DEFAULTS: dict[str, Any] = {
    "on_model_image_quantity": 0,
    "on_model_detail_quantity": 0,
    "laydown_silo_type": "else/default",
    "laydown_silo_quantity": 0,
    "color_corrections_quantity": 0,
    "post_production_hours": "0.00",
    "model_hours_mode": "adult",
    "adult_model_hours": "0.00",
    "kid_model_hours": "0.00",
    "model_fitting_quantity": 0,
    "ai_generation_quantity": 0,
    "account_management_mode": "automatic",
    "manual_account_management_amount": "0.00",
}

PRICING_STATE_KEYS = {
    "on_model_image_quantity": "photo_pricing_on_model_image_quantity",
    "on_model_detail_quantity": "photo_pricing_on_model_detail_quantity",
    "laydown_silo_type": "photo_pricing_laydown_silo_type",
    "laydown_silo_quantity": "photo_pricing_laydown_silo_quantity",
    "color_corrections_quantity": "photo_pricing_color_corrections_quantity",
    "post_production_hours": "photo_pricing_post_production_hours",
    "model_hours_mode": "photo_pricing_model_type",
    "model_fitting_quantity": "photo_pricing_model_fitting_quantity",
    "ai_generation_quantity": "photo_pricing_ai_generation_quantity",
    "account_management_mode": "photo_pricing_account_management_mode",
    "manual_account_management_amount": "photo_pricing_manual_account_management_amount",
}

PROJECT_FIELDS = (
    "project_name",
    "on_model",
    "laydown_detail",
    "color_correct",
    "post",
    "model_hours",
)

COMMENT_KEYS = {
    "estimate_subject": "photo_pricing_comments_estimate_subject",
    "subtitle_line": "photo_pricing_comments_subtitle_line",
    "custom_notes": "photo_pricing_comments_custom_notes",
}


def _text(value: Any, fallback: str = "") -> str:
    cleaned = str(value or "").strip()
    return cleaned or fallback


def _date(value: Any, fallback: date | None = None) -> date:
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            pass
    return fallback or date.today()


def _date_text(value: Any) -> str:
    return _date(value).isoformat()


def _decimal_text(value: Any) -> str:
    try:
        return str(Decimal(str(value or "0")).quantize(Decimal("0.01")))
    except Exception:
        return "0.00"


def _int_value(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except Exception:
        return 0


def _project_entry_from_state(state: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "project_name": _text(state.get(f"photo_pricing_comments_project_name_{index}")),
        "on_model": _decimal_text(state.get(f"photo_pricing_comments_on_model_{index}")),
        "laydown_detail": _decimal_text(state.get(f"photo_pricing_comments_laydown_detail_{index}")),
        "color_correct": _decimal_text(state.get(f"photo_pricing_comments_color_correct_{index}")),
        "post": _decimal_text(state.get(f"photo_pricing_comments_post_{index}")),
        "model_hours": _decimal_text(state.get(f"photo_pricing_comments_model_hours_{index}")),
    }


def _project_entries_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = state.get("photo_pricing_project_rows") or [{}]
    return [_project_entry_from_state(state, index) for index, _row in enumerate(rows)]


def _project_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    comments = payload.get("comments", {}) or {}
    raw_entries = comments.get("project_entries") or [{}]
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        raw = raw or {}
        entries.append(
            {
                "project_name": _text(raw.get("project_name")),
                "on_model": _decimal_text(raw.get("on_model")),
                "laydown_detail": _decimal_text(raw.get("laydown_detail")),
                "color_correct": _decimal_text(raw.get("color_correct")),
                "post": _decimal_text(raw.get("post")),
                "model_hours": _decimal_text(raw.get("model_hours")),
            }
        )
    return entries or [{field: ("" if field == "project_name" else "0.00") for field in PROJECT_FIELDS}]


def build_draft_name(payload: dict[str, Any], client_company_name: Any = None) -> str:
    metadata = payload.get("quote_metadata", {}) or {}
    contacts = payload.get("contacts", {}) or {}
    title = _text(metadata.get("quote_title"), "Untitled Quote")
    company = _text(client_company_name or contacts.get("client_company_name"), "No Client")
    created = _date_text(metadata.get("quote_created_date"))
    return f"{title} — {company} — {created}"


def serialize_draft_payload(
    state: dict[str, Any],
    *,
    selected_client: ClientContact | None = None,
    selected_internal: InternalContact | None = None,
) -> dict[str, Any]:
    created = _date(state.get("photo_pricing_quote_created_date"))
    expires = _date(
        state.get("photo_pricing_quote_expiration_date"),
        add_calendar_months(created, 3),
    )
    model_mode = _text(state.get("photo_pricing_model_type"), "adult")
    adult_hours = state.get(
        "photo_pricing_adult_model_hours" if model_mode == "both" else "photo_pricing_adult_model_hours_single",
        state.get("photo_pricing_adult_model_hours", 0),
    )
    kid_hours = state.get(
        "photo_pricing_kid_model_hours" if model_mode == "both" else "photo_pricing_kid_model_hours_single",
        state.get("photo_pricing_kid_model_hours", 0),
    )

    return {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "quote_metadata": {
            "quote_title": _text(state.get("photo_pricing_quote_title")),
            "reference_number": _text(state.get("photo_pricing_reference_number")),
            "quote_created_date": created.isoformat(),
            "quote_expiration_date": expires.isoformat(),
            "expiration_overridden": bool(state.get("photo_pricing_expiration_overridden", False)),
        },
        "contacts": {
            "client_contact_id": getattr(selected_client, "id", None) or state.get("photo_pricing_client_contact_id"),
            "internal_contact_id": getattr(selected_internal, "id", None) or state.get("photo_pricing_internal_contact_id"),
        },
        "pricing": {
            "on_model_image_quantity": _int_value(state.get("photo_pricing_on_model_image_quantity")),
            "on_model_detail_quantity": _int_value(state.get("photo_pricing_on_model_detail_quantity")),
            "laydown_silo_type": _text(state.get("photo_pricing_laydown_silo_type"), "else/default"),
            "laydown_silo_quantity": _int_value(state.get("photo_pricing_laydown_silo_quantity")),
            "color_corrections_quantity": _int_value(state.get("photo_pricing_color_corrections_quantity")),
            "post_production_hours": _decimal_text(state.get("photo_pricing_post_production_hours")),
            "model_hours_mode": model_mode if model_mode in {"adult", "kid", "both"} else "adult",
            "adult_model_hours": _decimal_text(adult_hours),
            "kid_model_hours": _decimal_text(kid_hours),
            "model_fitting_quantity": _int_value(state.get("photo_pricing_model_fitting_quantity")),
            "ai_generation_quantity": _int_value(state.get("photo_pricing_ai_generation_quantity")),
            "account_management_mode": _text(state.get("photo_pricing_account_management_mode"), "automatic"),
            "manual_account_management_amount": _decimal_text(state.get("photo_pricing_manual_account_management_amount")),
        },
        "comments": {
            "estimate_subject": _text(state.get("photo_pricing_comments_estimate_subject")),
            "subtitle_line": _text(state.get("photo_pricing_comments_subtitle_line")),
            "project_entries": _project_entries_from_state(state),
            "custom_notes": _text(state.get("photo_pricing_comments_custom_notes")),
        },
    }


def normalize_draft_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload or {}
    metadata = payload.get("quote_metadata", {}) or {}
    contacts = payload.get("contacts", {}) or {}
    pricing = payload.get("pricing", {}) or {}
    comments = payload.get("comments", {}) or {}
    created = _date(metadata.get("quote_created_date"))

    normalized_pricing = dict(PRICING_DEFAULTS)
    normalized_pricing.update({key: pricing.get(key, default) for key, default in PRICING_DEFAULTS.items()})
    normalized_pricing["post_production_hours"] = _decimal_text(normalized_pricing["post_production_hours"])
    normalized_pricing["adult_model_hours"] = _decimal_text(normalized_pricing["adult_model_hours"])
    normalized_pricing["kid_model_hours"] = _decimal_text(normalized_pricing["kid_model_hours"])
    normalized_pricing["manual_account_management_amount"] = _decimal_text(
        normalized_pricing["manual_account_management_amount"]
    )

    return {
        "schema_version": int(payload.get("schema_version") or DRAFT_SCHEMA_VERSION),
        "quote_metadata": {
            "quote_title": _text(metadata.get("quote_title")),
            "reference_number": _text(metadata.get("reference_number")),
            "quote_created_date": created.isoformat(),
            "quote_expiration_date": _date(metadata.get("quote_expiration_date"), add_calendar_months(created, 3)).isoformat(),
            "expiration_overridden": bool(metadata.get("expiration_overridden", False)),
        },
        "contacts": {
            "client_contact_id": contacts.get("client_contact_id") or None,
            "internal_contact_id": contacts.get("internal_contact_id") or None,
        },
        "pricing": normalized_pricing,
        "comments": {
            "estimate_subject": _text(comments.get("estimate_subject")),
            "subtitle_line": _text(comments.get("subtitle_line")),
            "project_entries": _project_entries(payload),
            "custom_notes": _text(comments.get("custom_notes")),
        },
    }


def restore_draft_payload_to_state(
    payload: dict[str, Any],
    state: dict[str, Any],
    *,
    available_client_ids: set[str] | None = None,
    available_internal_ids: set[str] | None = None,
) -> list[str]:
    normalized = normalize_draft_payload(payload)
    metadata = normalized["quote_metadata"]
    contacts = normalized["contacts"]
    pricing = normalized["pricing"]
    comments = normalized["comments"]
    warnings: list[str] = []

    state["photo_pricing_quote_title"] = metadata["quote_title"]
    state["photo_pricing_reference_number"] = metadata["reference_number"]
    state["photo_pricing_quote_created_date"] = _date(metadata["quote_created_date"])
    state["photo_pricing_quote_expiration_date"] = _date(metadata["quote_expiration_date"])
    state["photo_pricing_expiration_overridden"] = bool(metadata["expiration_overridden"])
    state["photo_pricing_previous_created_date"] = state["photo_pricing_quote_created_date"]

    client_id = contacts.get("client_contact_id")
    if client_id and (available_client_ids is None or client_id in available_client_ids):
        state["photo_pricing_client_contact_id"] = client_id
    elif client_id:
        state.pop("photo_pricing_client_contact_id", None)
        warnings.append("Saved client contact is no longer available.")

    internal_id = contacts.get("internal_contact_id")
    if internal_id and (available_internal_ids is None or internal_id in available_internal_ids):
        state["photo_pricing_internal_contact_id"] = internal_id
    elif internal_id:
        state.pop("photo_pricing_internal_contact_id", None)
        warnings.append("Saved internal contact is no longer available.")

    for payload_key, state_key in PRICING_STATE_KEYS.items():
        value = pricing.get(payload_key, PRICING_DEFAULTS[payload_key])
        if payload_key in {
            "post_production_hours",
            "manual_account_management_amount",
        }:
            value = float(Decimal(str(value or "0")))
        state[state_key] = value

    adult_hours = float(Decimal(str(pricing.get("adult_model_hours") or "0")))
    kid_hours = float(Decimal(str(pricing.get("kid_model_hours") or "0")))
    state["photo_pricing_adult_model_hours"] = adult_hours
    state["photo_pricing_adult_model_hours_single"] = adult_hours
    state["photo_pricing_kid_model_hours"] = kid_hours
    state["photo_pricing_kid_model_hours_single"] = kid_hours

    for key, state_key in COMMENT_KEYS.items():
        state[state_key] = comments.get(key, "")

    entries = comments.get("project_entries") or [{}]
    state["photo_pricing_project_rows"] = [{} for _entry in entries]
    _clear_project_widget_keys(state)
    for index, entry in enumerate(entries):
        state[f"photo_pricing_comments_project_name_{index}"] = entry.get("project_name", "")
        for field in PROJECT_FIELDS[1:]:
            state[f"photo_pricing_comments_{field}_{index}"] = float(Decimal(str(entry.get(field) or "0")))

    return warnings


def apparel_inputs_from_draft_payload(payload: dict[str, Any]) -> ApparelInputs:
    pricing = normalize_draft_payload(payload)["pricing"]
    mode = pricing.get("model_hours_mode", "adult")
    adult_hours = float(Decimal(str(pricing.get("adult_model_hours") or "0")))
    kid_hours = float(Decimal(str(pricing.get("kid_model_hours") or "0")))
    return ApparelInputs(
        on_model_image_quantity=_int_value(pricing.get("on_model_image_quantity")),
        on_model_detail_quantity=_int_value(pricing.get("on_model_detail_quantity")),
        laydown_silo_type=_text(pricing.get("laydown_silo_type"), "else/default"),
        laydown_silo_quantity=_int_value(pricing.get("laydown_silo_quantity")),
        color_corrections_quantity=_int_value(pricing.get("color_corrections_quantity")),
        post_production_hours=float(Decimal(str(pricing.get("post_production_hours") or "0"))),
        model_type=mode if mode in {"adult", "kid", "both"} else "adult",
        model_hours=kid_hours if mode == "kid" else adult_hours,
        adult_model_hours=adult_hours,
        kid_model_hours=kid_hours,
        model_fitting_quantity=_int_value(pricing.get("model_fitting_quantity")),
        ai_generation_quantity=_int_value(pricing.get("ai_generation_quantity")),
        account_management_mode=_text(pricing.get("account_management_mode"), "automatic"),
        manual_account_management_amount=float(Decimal(str(pricing.get("manual_account_management_amount") or "0"))),
    )


def _clear_project_widget_keys(state: dict[str, Any]) -> None:
    prefixes = tuple(f"photo_pricing_comments_{field}_" for field in PROJECT_FIELDS)
    for key in list(state.keys()):
        if key.startswith(prefixes):
            state.pop(key, None)
