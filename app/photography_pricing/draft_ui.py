from __future__ import annotations

from typing import Any

import streamlit as st

from app.contact_management.contact_ui import safe_list_active_client_contacts, safe_list_active_internal_contacts
from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing import draft_repository
from app.photography_pricing.draft_models import QuoteDraft
from app.photography_pricing.draft_service import (
    build_draft_name,
    reset_quote_form_state,
    restore_draft_payload_to_state,
    serialize_draft_payload,
)

ACTIVE_DRAFT_ID_KEY = "photo_pricing_active_draft_id"
ACTIVE_DRAFT_NAME_KEY = "photo_pricing_active_draft_name"
SELECTED_DRAFT_ID_KEY = "photo_pricing_selected_draft_id"
DRAFT_NOTICE_KEY = "photo_pricing_draft_notice"
PENDING_DRAFT_LOAD_KEY = "photo_pricing_pending_draft_load"
PENDING_SELECTION_KEY = "photo_pricing_pending_draft_selection"

# Old session keys are cleared locally; no historical database records are changed.
ACTIVE_VERSION_KEY = "photo_pricing_active_version_number"
DELETE_CONFIRM_DRAFT_ID_KEY = "photo_pricing_delete_confirm_draft_id"
LEGACY_LOADED_DRAFT_ID_KEY = "photo_pricing_loaded_draft_id"
LEGACY_LOADED_VERSION_KEY = "photo_pricing_loaded_draft_version"
LEGACY_OPEN_DRAFT_ID_KEY = "photo_pricing_open_draft_id"


def _draft_label(draft: QuoteDraft) -> str:
    return draft.draft_name


def _safe_list_drafts() -> list[QuoteDraft]:
    try:
        return draft_repository.list_drafts()
    except Exception:
        st.warning("Quote drafts are unavailable. Check the database connection and schema.")
        return []


def _migrate_legacy_draft_state(state: dict[str, Any]) -> None:
    if ACTIVE_DRAFT_ID_KEY not in state and state.get(LEGACY_LOADED_DRAFT_ID_KEY):
        state[ACTIVE_DRAFT_ID_KEY] = state[LEGACY_LOADED_DRAFT_ID_KEY]
    if SELECTED_DRAFT_ID_KEY not in state and state.get(LEGACY_OPEN_DRAFT_ID_KEY):
        state[SELECTED_DRAFT_ID_KEY] = state[LEGACY_OPEN_DRAFT_ID_KEY]
    for key in (ACTIVE_VERSION_KEY, LEGACY_LOADED_VERSION_KEY, "photo_pricing_draft_version_note",
                "photo_pricing_restore_version_number", DELETE_CONFIRM_DRAFT_ID_KEY):
        state.pop(key, None)


def _set_active_draft(
    draft_id: str | None,
    draft_name: str | None = None,
    *,
    sync_selected_draft: bool = False,
) -> None:
    if draft_id:
        st.session_state[ACTIVE_DRAFT_ID_KEY] = draft_id
        if sync_selected_draft:
            st.session_state[SELECTED_DRAFT_ID_KEY] = draft_id
    else:
        st.session_state.pop(ACTIVE_DRAFT_ID_KEY, None)
    st.session_state.pop(ACTIVE_VERSION_KEY, None)
    if draft_name:
        st.session_state[ACTIVE_DRAFT_NAME_KEY] = draft_name
    elif not draft_id:
        st.session_state.pop(ACTIVE_DRAFT_NAME_KEY, None)


def _start_new_quote() -> None:
    """Create a blank draft before widgets render; persist it when Save is clicked."""
    reset_quote_form_state(st.session_state)
    _set_active_draft(None)
    for key in (
        PENDING_DRAFT_LOAD_KEY, PENDING_SELECTION_KEY, DELETE_CONFIRM_DRAFT_ID_KEY,
        LEGACY_LOADED_DRAFT_ID_KEY, LEGACY_LOADED_VERSION_KEY, LEGACY_OPEN_DRAFT_ID_KEY,
        "photo_pricing_draft_warnings", "photo_pricing_draft_name",
        "photo_pricing_draft_version_note", "photo_pricing_restore_version_number",
    ):
        st.session_state.pop(key, None)
    st.session_state[DRAFT_NOTICE_KEY] = "New draft created. Fill in the quote and click Save Draft."


def _contact_id_sets() -> tuple[set[str], set[str]]:
    clients = {contact.id for contact in safe_list_active_client_contacts()}
    internals = {contact.id for contact in safe_list_active_internal_contacts()}
    return clients, internals


def _queue_draft_load(draft_id: str, payload: dict[str, Any], draft_name: str) -> None:
    st.session_state[PENDING_DRAFT_LOAD_KEY] = {
        "draft_id": draft_id, "draft_name": draft_name, "payload": payload,
    }
    st.rerun()


def apply_pending_draft_restore() -> None:
    pending = st.session_state.pop(PENDING_DRAFT_LOAD_KEY, None)
    if not pending:
        return
    client_ids, internal_ids = _contact_id_sets()
    warnings = restore_draft_payload_to_state(
        pending.get("payload", {}), st.session_state,
        available_client_ids=client_ids, available_internal_ids=internal_ids,
    )
    _set_active_draft(pending.get("draft_id"), pending.get("draft_name"), sync_selected_draft=True)
    st.session_state["photo_pricing_draft_warnings"] = warnings
    st.session_state.pop("photo_pricing_generated_pdf", None)
    st.session_state.pop("photo_pricing_generated_pdf_inputs", None)


def _current_payload(
    selected_client: ClientContact | None,
    selected_internal: InternalContact | None,
) -> dict[str, Any]:
    return serialize_draft_payload(
        st.session_state, selected_client=selected_client, selected_internal=selected_internal,
    )


def render_drafts_section(
    *,
    selected_client: ClientContact | None,
    selected_internal: InternalContact | None,
) -> None:
    _migrate_legacy_draft_state(st.session_state)
    pending_selection = st.session_state.pop(PENDING_SELECTION_KEY, None)
    if pending_selection:
        st.session_state[SELECTED_DRAFT_ID_KEY] = pending_selection
    with st.expander("Drafts"):
        notice = st.session_state.pop(DRAFT_NOTICE_KEY, None)
        if notice:
            st.success(notice)
        for warning in st.session_state.pop("photo_pricing_draft_warnings", []):
            st.warning(warning)

        active_id = st.session_state.get(ACTIVE_DRAFT_ID_KEY)
        active_name = st.session_state.get(ACTIVE_DRAFT_NAME_KEY) or active_id
        st.caption(f"Editing draft: {active_name}" if active_id else "New draft: not saved yet")
        st.caption("Create Draft starts a blank quote. Select a saved draft and click Edit Draft to change it. "
                   "Save Draft saves your work.")

        drafts = _safe_list_drafts()
        draft_by_id = {draft.id: draft for draft in drafts}
        selected_id = None
        if drafts:
            if st.session_state.get(SELECTED_DRAFT_ID_KEY) not in draft_by_id:
                st.session_state[SELECTED_DRAFT_ID_KEY] = active_id if active_id in draft_by_id else drafts[0].id
            selected_id = st.selectbox(
                "Saved Drafts", list(draft_by_id),
                format_func=lambda draft_id: _draft_label(draft_by_id[draft_id]), key=SELECTED_DRAFT_ID_KEY,
            )
        else:
            st.caption("No saved drafts are available yet.")

        create_col, edit_col, save_col = st.columns(3)
        with create_col:
            st.button("Create Draft", key="photo_pricing_start_new_draft", on_click=_start_new_quote)
        with edit_col:
            if st.button("Edit Draft", key="photo_pricing_open_draft", disabled=selected_id is None):
                try:
                    payload = draft_repository.get_editable_draft_payload(str(selected_id))
                    if payload is None:
                        st.error("Selected draft has no saved contents.")
                    else:
                        _queue_draft_load(str(selected_id), payload, draft_by_id[selected_id].draft_name)
                except Exception:
                    st.error("Draft could not be opened.")
        with save_col:
            if st.button("Save Draft", key="photo_pricing_save_draft"):
                payload = _current_payload(selected_client, selected_internal)
                try:
                    if active_id:
                        draft_repository.save_editable_draft(str(active_id), payload=payload)
                        st.session_state[DRAFT_NOTICE_KEY] = "Draft saved."
                    else:
                        draft = draft_repository.create_editable_draft(
                            draft_name=build_draft_name(payload, getattr(selected_client, "company_name", None)),
                            payload=payload,
                            client_contact_id=payload["contacts"].get("client_contact_id"),
                            internal_contact_id=payload["contacts"].get("internal_contact_id"),
                        )
                        _set_active_draft(draft.id, draft.draft_name)
                        st.session_state[PENDING_SELECTION_KEY] = draft.id
                        st.session_state[DRAFT_NOTICE_KEY] = "Draft saved."
                    st.rerun()
                except Exception:
                    st.error("Draft could not be saved.")
