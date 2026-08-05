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


def _set_loaded_draft(draft_id: str | None, version_number: int | None) -> None:
    if draft_id:
        st.session_state["photo_pricing_loaded_draft_id"] = draft_id
        st.session_state["photo_pricing_open_draft_id"] = draft_id
    else:
        st.session_state.pop("photo_pricing_loaded_draft_id", None)

    if version_number is not None:
        st.session_state["photo_pricing_loaded_draft_version"] = version_number
    else:
        st.session_state.pop("photo_pricing_loaded_draft_version", None)


def _contact_id_sets() -> tuple[set[str], set[str]]:
    clients = {contact.id for contact in safe_list_active_client_contacts()}
    internals = {contact.id for contact in safe_list_active_internal_contacts()}
    return clients, internals


def _apply_version(version: QuoteDraftVersion) -> None:
    st.session_state["photo_pricing_pending_draft_restore"] = {
        "draft_id": version.draft_id,
        "version_number": version.version_number,
        "payload": version.payload,
    }
    st.rerun()


def apply_pending_draft_restore() -> None:
    pending = st.session_state.pop("photo_pricing_pending_draft_restore", None)
    if not pending:
        return

    client_ids, internal_ids = _contact_id_sets()
    warnings = restore_draft_payload_to_state(
        pending.get("payload", {}),
        st.session_state,
        available_client_ids=client_ids,
        available_internal_ids=internal_ids,
    )
    _set_loaded_draft(pending.get("draft_id"), pending.get("version_number"))
    st.session_state["photo_pricing_draft_warnings"] = warnings
    st.session_state.pop("photo_pricing_generated_pdf", None)


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
    with st.expander("Drafts"):
        for warning in st.session_state.pop("photo_pricing_draft_warnings", []):
            st.warning(warning)

        loaded_draft_id = st.session_state.get("photo_pricing_loaded_draft_id")
        loaded_version = st.session_state.get("photo_pricing_loaded_draft_version")
        if loaded_draft_id:
            st.caption(f"Loaded draft: {loaded_draft_id} v{loaded_version}")

        version_note = st.text_input(
            "Version Note",
            key="photo_pricing_draft_version_note",
            placeholder="Optional",
        )

        save_cols = st.columns(3)
        with save_cols[0]:
            if st.button("Save As New Draft", key="photo_pricing_save_draft"):
                payload = _current_payload(selected_client, selected_internal)
                try:
                    draft, version = draft_repository.create_draft(
                        draft_name=build_draft_name(
                            payload,
                            getattr(selected_client, "company_name", None),
                        ),
                        payload=payload,
                        client_contact_id=payload["contacts"].get("client_contact_id"),
                        internal_contact_id=payload["contacts"].get("internal_contact_id"),
                        saved_by_contact_id=payload["contacts"].get("internal_contact_id"),
                        version_note=version_note or None,
                    )
                    _set_loaded_draft(draft.id, version.version_number)
                    loaded_draft_id = draft.id
                    st.success(f"Draft saved as version {version.version_number}.")
                except Exception:
                    st.error("Draft could not be saved.")

        with save_cols[1]:
            if st.button("Save New Version", key="photo_pricing_save_new_version", disabled=not bool(loaded_draft_id)):
                payload = _current_payload(selected_client, selected_internal)
                try:
                    version = draft_repository.create_version(
                        str(loaded_draft_id),
                        payload=payload,
                        saved_by_contact_id=payload["contacts"].get("internal_contact_id"),
                        version_note=version_note or None,
                    )
                    st.session_state["photo_pricing_loaded_draft_version"] = version.version_number
                    st.success(f"Version {version.version_number} saved.")
                except Exception:
                    st.error("New version could not be saved.")

        with save_cols[2]:
            if st.button("Start New Draft", key="photo_pricing_start_new_draft", disabled=not bool(loaded_draft_id)):
                _set_loaded_draft(None, None)
                loaded_draft_id = None
                st.session_state.pop("photo_pricing_generated_pdf", None)
                st.info("Current form is no longer attached to a saved draft.")

        drafts = _safe_list_drafts()
        if drafts:
            selected_options = [draft.id for draft in drafts]
            if st.session_state.get("photo_pricing_open_draft_id") not in selected_options:
                st.session_state["photo_pricing_open_draft_id"] = selected_options[0]

            selected_draft_id = st.selectbox(
                "Open Draft",
                selected_options,
                format_func=lambda draft_id: _draft_label(next((draft for draft in drafts if draft.id == draft_id), drafts[0])),
                key="photo_pricing_open_draft_id",
            )
            open_cols = st.columns(2)
            with open_cols[0]:
                if st.button("Open Draft", key="photo_pricing_open_draft"):
                    try:
                        version = draft_repository.get_latest_version(str(selected_draft_id))
                        if version is None:
                            st.error("Selected draft has no saved versions.")
                        else:
                            _apply_version(version)
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
                            _apply_version(restored)
                        except Exception:
                            st.error("Version could not be restored.")
