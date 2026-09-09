"""Browser-local contact finder using the installed Streamlit component API."""
from pathlib import Path

import streamlit as st

from app.contact_management.models import ClientContact


_component = st.components.v2.component(
    "photography_client_autocomplete",
    html='''<label for="contact-search">Client Contact Search</label>
    <input id="contact-search" role="combobox" aria-autocomplete="list"
      aria-controls="contact-results" aria-expanded="false" autocomplete="off"
      placeholder="Search by company, name, or email...">
    <div id="contact-results" role="listbox" hidden></div>''',
    css='''label { display:block; margin-bottom:.4rem; font-size:.875rem; }
    input { box-sizing:border-box; width:100%; padding:.65rem; border:1px solid #8886;
      border-radius:.5rem; font:inherit; color:var(--st-text-color);
      background:var(--st-secondary-background-color); }
    #contact-results { max-height:240px; overflow-y:auto; border:1px solid #8886;
      border-radius:.4rem; background:var(--st-background-color); }
    [role=option] { padding:.6rem; cursor:pointer; overflow-wrap:anywhere; }
    [role=option]:hover, [aria-selected=true] { background:var(--st-secondary-background-color); }
    .empty { padding:.6rem; }''',
    js=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
)


def autocomplete_data(contacts: list[ClientContact], selected_id: str) -> dict:
    ordered = sorted((c for c in contacts if c.active), key=lambda c: (
        c.full_name.casefold(), c.company_name.casefold(), c.email.casefold(), c.id))
    fields = [(c.first_name, c.last_name, c.full_name, c.company_name, c.email) for c in ordered]
    characters = set("".join(value for row in fields for value in row))
    characters.update("".join(char.upper() + char.lower() for char in list(characters)))
    return {
        "selected_id": selected_id,
        "folds": {char: char.casefold() for char in characters},
        "contacts": [{"id": c.id, "label": c.dropdown_label,
                      "fields": [value.casefold() for value in row]}
                     for c, row in zip(ordered, fields)],
    }


def apply_autocomplete_selection(value: object, contacts: list[ClientContact]) -> None:
    if isinstance(value, str) and any(c.active and c.id == value for c in contacts):
        st.session_state["photo_pricing_client_contact_id"] = value


def render_client_autocomplete(contacts: list[ClientContact], selected_id: str) -> None:
    revision = st.session_state.get("photo_pricing_client_search_revision", 0)
    key = f"photo_pricing_client_autocomplete_{revision}"

    def select_contact() -> None:
        apply_autocomplete_selection(st.session_state[key].get("selected"), contacts)

    _component(key=key, data=autocomplete_data(contacts, selected_id), on_selected_change=select_contact)
