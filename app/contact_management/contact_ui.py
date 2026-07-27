from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st

from app.contact_management.database import ContactManagementError
from app.contact_management.models import ClientContact, InternalContact
from app.contact_management.repositories import (
    ContactValidationError,
    create_client_contact,
    create_internal_contact,
    deactivate_client_contact,
    deactivate_internal_contact,
    get_client_contact,
    get_internal_contact,
    list_active_client_contacts,
    list_active_internal_contacts,
    list_all_client_contacts,
    list_all_internal_contacts,
    reactivate_client_contact,
    reactivate_internal_contact,
    update_client_contact,
    update_internal_contact,
)


def _handle_contact_error(exc: Exception) -> None:
    if isinstance(exc, ContactValidationError):
        st.error(str(exc))
    elif isinstance(exc, ContactManagementError):
        st.error("Contact database operation failed.")
    else:
        st.error("Contact operation failed.")


def _rerun_after_success(message: str) -> None:
    st.session_state["contact_management_flash"] = message
    st.rerun()


def _contact_by_id(contacts: Iterable[ClientContact | InternalContact], contact_id: str | None):
    if not contact_id:
        return None
    return next((contact for contact in contacts if contact.id == contact_id), None)


def safe_list_active_client_contacts() -> list[ClientContact]:
    try:
        return list_active_client_contacts()
    except Exception:
        return []


def safe_list_active_internal_contacts() -> list[InternalContact]:
    try:
        return list_active_internal_contacts()
    except Exception:
        return []


def resolve_client_contact(contact_id: str | None) -> Optional[ClientContact]:
    if not contact_id:
        return None
    try:
        return get_client_contact(contact_id)
    except Exception:
        return None


def resolve_internal_contact(contact_id: str | None) -> Optional[InternalContact]:
    if not contact_id:
        return None
    try:
        return get_internal_contact(contact_id)
    except Exception:
        return None


def contact_payload(contact: ClientContact | InternalContact | None) -> dict[str, str]:
    if contact is None:
        return {}
    if isinstance(contact, ClientContact):
        return {
            "id": contact.id,
            "company_name": contact.company_name,
            "full_name": contact.full_name,
            "email": contact.email,
        }
    return {
        "id": contact.id,
        "name": contact.name,
        "title": contact.title,
        "email": contact.email,
    }


def render_client_contact_select() -> Optional[ClientContact]:
    contacts = safe_list_active_client_contacts()
    if not contacts:
        st.warning("No active client contacts are available.")
        return None

    active_ids = {contact.id for contact in contacts}
    if st.session_state.get("photo_pricing_client_contact_id") not in active_ids:
        st.session_state.pop("photo_pricing_client_contact_id", None)

    selected_id = st.selectbox(
        "Client Contact",
        [contact.id for contact in contacts],
        format_func=lambda contact_id: (_contact_by_id(contacts, contact_id) or contacts[0]).dropdown_label,
        key="photo_pricing_client_contact_id",
    )
    return _contact_by_id(contacts, selected_id)


def render_internal_contact_select() -> Optional[InternalContact]:
    contacts = safe_list_active_internal_contacts()
    if not contacts:
        st.warning("No active internal contacts are available.")
        return None

    active_ids = {contact.id for contact in contacts}
    if st.session_state.get("photo_pricing_internal_contact_id") not in active_ids:
        st.session_state.pop("photo_pricing_internal_contact_id", None)

    selected_id = st.selectbox(
        "Internal Contact",
        [contact.id for contact in contacts],
        format_func=lambda contact_id: (_contact_by_id(contacts, contact_id) or contacts[0]).dropdown_label,
        key="photo_pricing_internal_contact_id",
        label_visibility="collapsed",
    )
    return _contact_by_id(contacts, selected_id)


def render_contact_management() -> None:
    with st.expander("Contact Management"):
        flash = st.session_state.pop("contact_management_flash", "")
        if flash:
            st.success(flash)
        client_tab, internal_tab = st.tabs(["Client Contacts", "Internal Contacts"])
        with client_tab:
            _render_client_management()
        with internal_tab:
            _render_internal_management()


def _render_client_management() -> None:
    st.markdown("##### Add Client Contact")
    add_cols = st.columns(4)
    with add_cols[0]:
        company_name = st.text_input("Company Name", key="contact_mgmt_client_add_company")
    with add_cols[1]:
        first_name = st.text_input("First Name", key="contact_mgmt_client_add_first")
    with add_cols[2]:
        last_name = st.text_input("Last Name", key="contact_mgmt_client_add_last")
    with add_cols[3]:
        email = st.text_input("Email", key="contact_mgmt_client_add_email")
    if st.button("Add Client Contact", key="contact_mgmt_client_add"):
        try:
            create_client_contact(
                company_name=company_name,
                first_name=first_name,
                last_name=last_name,
                email=email,
            )
            _rerun_after_success("Client contact added.")
        except Exception as exc:
            _handle_contact_error(exc)

    st.markdown("##### Manage Client Contacts")
    try:
        contacts = list_all_client_contacts()
    except Exception as exc:
        _handle_contact_error(exc)
        return
    if not contacts:
        st.caption("No client contacts found.")
        return

    selected_id = st.selectbox(
        "Client contact to manage",
        [contact.id for contact in contacts],
        format_func=lambda contact_id: (_contact_by_id(contacts, contact_id) or contacts[0]).dropdown_label,
        key="contact_mgmt_client_manage_id",
    )
    selected = _contact_by_id(contacts, selected_id)
    if selected is None:
        return

    edit_cols = st.columns(4)
    with edit_cols[0]:
        edit_company = st.text_input("Edit Company Name", value=selected.company_name, key="contact_mgmt_client_edit_company")
    with edit_cols[1]:
        edit_first = st.text_input("Edit First Name", value=selected.first_name, key="contact_mgmt_client_edit_first")
    with edit_cols[2]:
        edit_last = st.text_input("Edit Last Name", value=selected.last_name, key="contact_mgmt_client_edit_last")
    with edit_cols[3]:
        edit_email = st.text_input("Edit Email", value=selected.email, key="contact_mgmt_client_edit_email")

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Update Client Contact", key="contact_mgmt_client_update"):
            try:
                update_client_contact(
                    selected.id,
                    hubspot_record_id=selected.hubspot_record_id,
                    company_name=edit_company,
                    first_name=edit_first,
                    last_name=edit_last,
                    email=edit_email,
                    active=selected.active,
                )
                _rerun_after_success("Client contact updated.")
            except Exception as exc:
                _handle_contact_error(exc)
    with action_cols[1]:
        if selected.active and st.button("Deactivate Client Contact", key="contact_mgmt_client_deactivate"):
            try:
                deactivate_client_contact(selected.id)
                st.session_state.pop("photo_pricing_client_contact_id", None)
                _rerun_after_success("Client contact deactivated.")
            except Exception as exc:
                _handle_contact_error(exc)
    with action_cols[2]:
        if not selected.active and st.button("Reactivate Client Contact", key="contact_mgmt_client_reactivate"):
            try:
                reactivate_client_contact(selected.id)
                _rerun_after_success("Client contact reactivated.")
            except Exception as exc:
                _handle_contact_error(exc)


def _render_internal_management() -> None:
    st.markdown("##### Add Internal Contact")
    add_cols = st.columns(3)
    with add_cols[0]:
        name = st.text_input("Name", key="contact_mgmt_internal_add_name")
    with add_cols[1]:
        title = st.text_input("Title", key="contact_mgmt_internal_add_title")
    with add_cols[2]:
        email = st.text_input("Email", key="contact_mgmt_internal_add_email")
    if st.button("Add Internal Contact", key="contact_mgmt_internal_add"):
        try:
            create_internal_contact(name=name, title=title, email=email)
            _rerun_after_success("Internal contact added.")
        except Exception as exc:
            _handle_contact_error(exc)

    st.markdown("##### Manage Internal Contacts")
    try:
        contacts = list_all_internal_contacts()
    except Exception as exc:
        _handle_contact_error(exc)
        return
    if not contacts:
        st.caption("No internal contacts found.")
        return

    selected_id = st.selectbox(
        "Internal contact to manage",
        [contact.id for contact in contacts],
        format_func=lambda contact_id: (_contact_by_id(contacts, contact_id) or contacts[0]).dropdown_label,
        key="contact_mgmt_internal_manage_id",
    )
    selected = _contact_by_id(contacts, selected_id)
    if selected is None:
        return

    edit_cols = st.columns(3)
    with edit_cols[0]:
        edit_name = st.text_input("Edit Name", value=selected.name, key="contact_mgmt_internal_edit_name")
    with edit_cols[1]:
        edit_title = st.text_input("Edit Title", value=selected.title, key="contact_mgmt_internal_edit_title")
    with edit_cols[2]:
        edit_email = st.text_input("Edit Email", value=selected.email, key="contact_mgmt_internal_edit_email")

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Update Internal Contact", key="contact_mgmt_internal_update"):
            try:
                update_internal_contact(
                    selected.id,
                    name=edit_name,
                    title=edit_title,
                    email=edit_email,
                    active=selected.active,
                )
                _rerun_after_success("Internal contact updated.")
            except Exception as exc:
                _handle_contact_error(exc)
    with action_cols[1]:
        if selected.active and st.button("Deactivate Internal Contact", key="contact_mgmt_internal_deactivate"):
            try:
                deactivate_internal_contact(selected.id)
                st.session_state.pop("photo_pricing_internal_contact_id", None)
                _rerun_after_success("Internal contact deactivated.")
            except Exception as exc:
                _handle_contact_error(exc)
    with action_cols[2]:
        if not selected.active and st.button("Reactivate Internal Contact", key="contact_mgmt_internal_reactivate"):
            try:
                reactivate_internal_contact(selected.id)
                _rerun_after_success("Internal contact reactivated.")
            except Exception as exc:
                _handle_contact_error(exc)
