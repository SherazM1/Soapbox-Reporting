from __future__ import annotations

from typing import Any

import streamlit as st

from app.contact_management.contact_ui import safe_list_active_client_contacts, safe_list_active_internal_contacts
from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing import draft_repository
from app.photography_pricing.draft_models import QuoteDraft, QuoteDraftVersion
from app.photography_pricing.draft_service import (
    build_draft_name,
    restore_draft_payload_to_state,
    serialize_draft_payload,
)

ACTIVE_DRAFT_ID_KEY = "photo_pricing_active_draft_id"
ACTIVE_VERSION_KEY = "photo_pricing_active_version_number"
ACTIVE_DRAFT_NAME_KEY = "photo_pricing_active_draft_name"
SELECTED_DRAFT_ID_KEY = "photo_pricing_selected_draft_id"
PENDING_DRAFT_LOAD_KEY = "photo_pricing_pending_draft_load"

LEGACY_LOADED_DRAFT_ID_KEY = "photo_pricing_loaded_draft_id"
LEGACY_LOADED_VERSION_KEY = "photo_pricing_loaded_draft_version"
LEGACY_OPEN_DRAFT_ID_KEY = "photo_pricing_open_draft_id"


def _draft_label(draft: QuoteDraft) -> str:
    return f"{draft.draft_name} (v{draft.latest_version_number})"


def _version_label(version: QuoteDraftVersion) -> str:
    note = f" - {version.version_note}" if version.version_note else ""
    return f"v{version.version_number} - {version.created_at}{note}"


def _safe_list_drafts() -> list[QuoteDraft]:
    try:
        return draft_repository.list_drafts()
    except Exception:
        st.warning("Quote drafts are unavailable. Check the database connection and schema.")
        return []


def _safe_list_versions(draft_id: str) -> list[QuoteDraftVersion]:
    try:
        return draft_repository.list_versions(draft_id)
    except Exception:
        st.warning("Version history is unavailable.")
        return []


def _migrate_legacy_draft_state(state: dict[str, Any]) -> None:
    if ACTIVE_DRAFT_ID_KEY not in state and state.get(LEGACY_LOADED_DRAFT_ID_KEY):
        state[ACTIVE_DRAFT_ID_KEY] = state[LEGACY_LOADED_DRAFT_ID_KEY]
    if ACTIVE_VERSION_KEY not in state and state.get(LEGACY_LOADED_VERSION_KEY) is not None:
        state[ACTIVE_VERSION_KEY] = state[LEGACY_LOADED_VERSION_KEY]
    if SELECTED_DRAFT_ID_KEY not in state and state.get(LEGACY_OPEN_DRAFT_ID_KEY):
        state[SELECTED_DRAFT_ID_KEY] = state[LEGACY_OPEN_DRAFT_ID_KEY]


def _set_active_draft(
    draft_id: str | None,
    version_number: int | None,
    draft_name: str | None = None,
    *,
    sync_draft_name_input: bool = False,
) -> None:
    if draft_id:
        st.session_state[ACTIVE_DRAFT_ID_KEY] = draft_id
        st.session_state[SELECTED_DRAFT_ID_KEY] = draft_id
    else:
        st.session_state.pop(ACTIVE_DRAFT_ID_KEY, None)

    if version_number is not None:
        st.session_state[ACTIVE_VERSION_KEY] = version_number
    else:
        st.session_state.pop(ACTIVE_VERSION_KEY, None)

    if draft_name:
        st.session_state[ACTIVE_DRAFT_NAME_KEY] = draft_name
        if sync_draft_name_input:
            st.session_state["photo_pricing_draft_name"] = draft_name
    elif not draft_id:
        st.session_state.pop(ACTIVE_DRAFT_NAME_KEY, None)


def _contact_id_sets() -> tuple[set[str], set[str]]:
    clients = {contact.id for contact in safe_list_active_client_contacts()}
    internals = {contact.id for contact in safe_list_active_internal_contacts()}
    return clients, internals


def _queue_draft_load(version: QuoteDraftVersion, draft_name: str | None = None) -> None:
    st.session_state[PENDING_DRAFT_LOAD_KEY] = {
        "draft_id": version.draft_id,
        "draft_name": draft_name,
        "version_number": version.version_number,
        "payload": version.payload,
    }
    st.rerun()


def apply_pending_draft_restore() -> None:
    pending = st.session_state.pop(PENDING_DRAFT_LOAD_KEY, None)
    if not pending:
        return

    client_ids, internal_ids = _contact_id_sets()
    warnings = restore_draft_payload_to_state(
        pending.get("payload", {}),
        st.session_state,
        available_client_ids=client_ids,
        available_internal_ids=internal_ids,
    )
    _set_active_draft(
        pending.get("draft_id"),
        pending.get("version_number"),
        pending.get("draft_name"),
        sync_draft_name_input=True,
    )
    st.session_state["photo_pricing_draft_warnings"] = warnings
    st.session_state.pop("photo_pricing_generated_pdf", None)


def _draft_name_for_save(
    payload: dict[str, Any],
    selected_client: ClientContact | None,
    requested_name: str,
) -> str:
    return (requested_name or "").strip() or build_draft_name(
        payload,
        getattr(selected_client, "company_name", None),
    )


def _current_payload(
    selected_client: ClientContact | None,
    selected_internal: InternalContact | None,
) -> dict[str, Any]:
    return serialize_draft_payload(
        st.session_state,
        selected_client=selected_client,
        selected_internal=selected_internal,
    )


def render_drafts_section(
    *,
    selected_client: ClientContact | None,
    selected_internal: InternalContact | None,
) -> None:
    _migrate_legacy_draft_state(st.session_state)

    with st.expander("Drafts"):
        for warning in st.session_state.pop("photo_pricing_draft_warnings", []):
            st.warning(warning)

        active_status = st.empty()

        def render_active_status() -> None:
            active_draft_id = st.session_state.get(ACTIVE_DRAFT_ID_KEY)
            active_version = st.session_state.get(ACTIVE_VERSION_KEY)
            active_name = st.session_state.get(ACTIVE_DRAFT_NAME_KEY)
            if active_draft_id:
                active_label = active_name or active_draft_id
                active_status.caption(f"Active draft: {active_label} (v{active_version})")
            else:
                active_status.caption("Active draft: Unsaved new quote")

        render_active_status()
        active_draft_id = st.session_state.get(ACTIVE_DRAFT_ID_KEY)

        draft_name = st.text_input(
            "Draft Name",
            key="photo_pricing_draft_name",
            placeholder="Optional; defaults to quote title, company, and date",
        )

        version_note = st.text_input(
            "Version Note",
            key="photo_pricing_draft_version_note",
            placeholder="Optional",
        )

        save_cols = st.columns(3)
        with save_cols[0]:
            if st.button("Save New Draft", key="photo_pricing_save_draft"):
                payload = _current_payload(selected_client, selected_internal)
                try:
                    draft, version = draft_repository.create_draft(
                        draft_name=_draft_name_for_save(payload, selected_client, draft_name),
                        payload=payload,
                        client_contact_id=payload["contacts"].get("client_contact_id"),
                        internal_contact_id=payload["contacts"].get("internal_contact_id"),
                        saved_by_contact_id=payload["contacts"].get("internal_contact_id"),
                        version_note=version_note or None,
                    )
                    _set_active_draft(draft.id, version.version_number, draft.draft_name)
                    active_draft_id = draft.id
                    render_active_status()
                    st.success(f"Draft saved as version {version.version_number}.")
                except Exception:
                    st.error("Draft could not be saved.")

        with save_cols[1]:
            if st.button("Save Version", key="photo_pricing_save_new_version", disabled=not bool(active_draft_id)):
                payload = _current_payload(selected_client, selected_internal)
                try:
                    version = draft_repository.create_version(
                        str(active_draft_id),
                        payload=payload,
                        saved_by_contact_id=payload["contacts"].get("internal_contact_id"),
                        version_note=version_note or None,
                    )
                    st.session_state[ACTIVE_VERSION_KEY] = version.version_number
                    render_active_status()
                    st.success(f"Version {version.version_number} saved.")
                except Exception:
                    st.error("New version could not be saved.")

        with save_cols[2]:
            if st.button("Start New Quote", key="photo_pricing_start_new_draft", disabled=not bool(active_draft_id)):
                _set_active_draft(None, None)
                active_draft_id = None
                render_active_status()
                st.session_state.pop("photo_pricing_generated_pdf", None)
                st.info("Current form is no longer attached to a saved draft.")

        drafts = _safe_list_drafts()
        if drafts:
            selected_options = [draft.id for draft in drafts]
            if st.session_state.get(SELECTED_DRAFT_ID_KEY) not in selected_options:
                st.session_state[SELECTED_DRAFT_ID_KEY] = selected_options[0]

            selected_draft_id = st.selectbox(
                "Open Draft",
                selected_options,
                format_func=lambda draft_id: _draft_label(next((draft for draft in drafts if draft.id == draft_id), drafts[0])),
                key=SELECTED_DRAFT_ID_KEY,
            )
            selected_draft = next((draft for draft in drafts if draft.id == selected_draft_id), drafts[0])
            open_cols = st.columns(2)
            with open_cols[0]:
                if st.button("Load Draft", key="photo_pricing_open_draft"):
                    try:
                        version = draft_repository.get_latest_version(str(selected_draft_id))
                        if version is None:
                            st.error("Selected draft has no saved versions.")
                        else:
                            _queue_draft_load(version, selected_draft.draft_name)
                    except Exception:
                        st.error("Draft could not be opened.")

            versions = _safe_list_versions(str(selected_draft_id))
            if versions:
                st.markdown("##### Version History")
                for version in versions:
                    st.caption(_version_label(version))
                version_numbers = [version.version_number for version in versions]
                if st.session_state.get("photo_pricing_restore_version_number") not in version_numbers:
                    st.session_state["photo_pricing_restore_version_number"] = version_numbers[0]
                restore_version_number = st.selectbox(
                    "Restore Version",
                    version_numbers,
                    key="photo_pricing_restore_version_number",
                )
                with open_cols[1]:
                    if st.button("Restore Version", key="photo_pricing_restore_version"):
                        try:
                            restored = draft_repository.restore_version_as_latest(
                                str(selected_draft_id),
                                int(restore_version_number),
                                saved_by_contact_id=st.session_state.get("photo_pricing_internal_contact_id"),
                                version_note=version_note or "Restored from older version",
                            )
                            _queue_draft_load(restored, selected_draft.draft_name)
                        except Exception:
                            st.error("Version could not be restored.")
