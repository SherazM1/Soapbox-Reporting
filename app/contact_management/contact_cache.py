from __future__ import annotations

import streamlit as st

from app.contact_management import repositories
from app.contact_management.models import ClientContact, InternalContact

# Exceptions propagate to the existing UI handlers and are never cached.


@st.cache_data(ttl=30, show_spinner=False)
def list_active_client_contacts() -> list[ClientContact]:
    return repositories.list_active_client_contacts()


@st.cache_data(ttl=30, show_spinner=False)
def list_all_client_contacts() -> list[ClientContact]:
    return repositories.list_all_client_contacts()


@st.cache_data(ttl=30, show_spinner=False)
def list_active_internal_contacts() -> list[InternalContact]:
    return repositories.list_active_internal_contacts()


@st.cache_data(ttl=30, show_spinner=False)
def list_all_internal_contacts() -> list[InternalContact]:
    return repositories.list_all_internal_contacts()


def invalidate_client_contacts() -> None:
    """Clear both list views after a successful mutation."""
    list_active_client_contacts.clear()
    list_all_client_contacts.clear()


def invalidate_internal_contacts() -> None:
    """Clear both list views after a successful mutation."""
    list_active_internal_contacts.clear()
    list_all_internal_contacts.clear()

