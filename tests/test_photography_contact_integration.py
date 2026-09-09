import unittest
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from app.contact_management.contact_ui import contact_payload, resolve_client_contact, resolve_internal_contact
from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing.apparel_estimator import _build_page1_header_payload, _header_payload_errors
from app.photography_pricing.comments_builder import build_page1_comments_payload
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pdf_generator import build_page1_header_items, generate_page2_pricing_pdf
from app.photography_pricing.pdf_generator import (
    GOTHAM_MEDIUM,
    PAGE1_HEADER_MIN_FONT_SIZE,
    PAGE1_HEADER_RIGHT_MAX_WIDTH,
    PAGE1_HEADER_SMALL_FONT_SIZE,
    PAGE1_HEADER_TITLE_TOP_Y,
    PAGE1_HEADER_TITLE_X,
    PAGE1_LOGO_REGION,
    _register_gotham_fonts,
    _wrap_fitted_text,
)
from app.photography_pricing.pdf_mapper import build_page2_pricing_payload
from app.photography_pricing.quote_builder import build_apparel_quote
from app.photography_pricing.quote_metadata import QuoteMetadata
from app.photography_pricing.quote_metadata import add_calendar_months, generate_reference_number


class PhotographyContactIntegrationTests(unittest.TestCase):
    def test_contact_payloads_resolve_dropdown_values_by_stable_id(self) -> None:
        client = ClientContact(
            id="client-1",
            hubspot_record_id=None,
            company_name="Acme",
            first_name="Ada",
            last_name="Lovelace",
            email="ada@example.com",
        )
        internal = InternalContact(
            id="internal-1",
            name="Grace Hopper",
            title="Creative Lead",
            email="grace@example.com",
        )

        with patch("app.contact_management.contact_ui.get_client_contact", return_value=client):
            self.assertEqual(client, resolve_client_contact("client-1"))
        with patch("app.contact_management.contact_ui.get_internal_contact", return_value=internal):
            self.assertEqual(internal, resolve_internal_contact("internal-1"))

        self.assertEqual(
            {
                "id": "client-1",
                "company_name": "Acme",
                "full_name": "Ada Lovelace",
                "email": "ada@example.com",
            },
            contact_payload(client),
        )
        self.assertEqual(
            {
                "id": "internal-1",
                "name": "Grace Hopper",
                "title": "Creative Lead",
                "email": "grace@example.com",
            },
            contact_payload(internal),
        )

    def test_reference_number_format(self) -> None:
        reference = generate_reference_number(datetime(2026, 7, 12, 9, 8, 7), suffix="a1b2")

        self.assertEqual("20260712-090807-A1B2", reference)
        self.assertRegex(generate_reference_number(datetime(2026, 7, 12, 9, 8, 7)), r"^20260712-090807-[A-Z0-9]{4}$")

    def test_expiration_date_uses_calendar_months(self) -> None:
        self.assertEqual(date(2026, 10, 12), add_calendar_months(date(2026, 7, 12), 3))
        self.assertEqual(date(2026, 4, 30), add_calendar_months(date(2026, 1, 31), 3))

    def test_page1_header_mapper_receives_selected_contacts_and_metadata(self) -> None:
        items = build_page1_header_items(
            {
                "quote_metadata": {
                    "quote_title": "Holiday Apparel Photography Quote",
                    "reference_number": "20260712-090807-A1B2",
                    "quote_created_date": "2026-07-12",
                    "quote_expiration_date": "2026-10-12",
                },
                "selected_client": {
                    "company_name": "Acme",
                    "full_name": "Ada Lovelace",
                    "email": "ada@example.com",
                },
                "selected_internal": {
                    "name": "Grace Hopper",
                    "title": "Creative Lead",
                    "email": "grace@example.com",
                },
            }
        )
        text_values = [item.text for item in items]

        self.assertIn("Holiday Apparel Photography Quote", text_values)
        self.assertIn("Acme", text_values)
        self.assertIn("Ada Lovelace", text_values)
        self.assertIn("ada@example.com", text_values)
        self.assertIn("Reference: 20260712-090807-A1B2", text_values)
        self.assertIn("Quote created: July 12, 2026", text_values)
        self.assertIn("Quote expires: October 12, 2026", text_values)
        self.assertIn("Quote created by: Grace Hopper", text_values)
        self.assertIn("Creative Lead", text_values)
        self.assertIn("grace@example.com", text_values)

    def test_ui_metadata_payload_reaches_pdf_header_mapper(self) -> None:
        client = ClientContact(
            id="client-1",
            hubspot_record_id=None,
            company_name="Acme",
            first_name="Ada",
            last_name="Lovelace",
            email="ada@example.com",
        )
        internal_payload = {
            "id": "internal-1",
            "name": "Grace Hopper",
            "title": "Creative Lead",
            "email": "grace@example.com",
        }
        metadata = QuoteMetadata(
            quote_title="UI Entered Quote Title",
            reference_number="UI-REF-1234",
            quote_created_date=date(2026, 7, 12),
            quote_expiration_date=date(2026, 10, 12),
        )

        payload = _build_page1_header_payload(metadata, client, internal_payload)
        items = build_page1_header_items(payload)
        text_values = [item.text for item in items]

        self.assertIn("UI Entered Quote Title", text_values)
        self.assertIn("Reference: UI-REF-1234", text_values)
        self.assertIn("Quote created: July 12, 2026", text_values)
        self.assertIn("Quote expires: October 12, 2026", text_values)
        self.assertIn("Quote created by: Grace Hopper", text_values)

    def test_quote_title_uses_intended_non_logo_coordinate(self) -> None:
        items = build_page1_header_items({"quote_metadata": {"quote_title": "Holiday Apparel Quote"}})
        title = items[0]

        self.assertEqual(PAGE1_HEADER_TITLE_X, title.x)
        self.assertEqual(PAGE1_HEADER_TITLE_TOP_Y, title.top_y)
        self.assertFalse(title.x < PAGE1_LOGO_REGION[2] and title.top_y < PAGE1_LOGO_REGION[3])

    def test_no_header_coordinate_intersects_logo_region(self) -> None:
        items = build_page1_header_items(
            {
                "quote_metadata": {
                    "quote_title": "Holiday Apparel Quote",
                    "reference_number": "20260712-090807-A1B2",
                    "quote_created_date": "2026-07-12",
                    "quote_expiration_date": "2026-10-12",
                },
                "selected_client": {
                    "company_name": "Acme",
                    "full_name": "Ada Lovelace",
                    "email": "ada@example.com",
                },
                "selected_internal": {
                    "name": "Grace Hopper",
                    "title": "Creative Lead",
                    "email": "grace@example.com",
                },
            }
        )
        logo_left, logo_top, logo_right, logo_bottom = PAGE1_LOGO_REGION

        for item in items:
            intersects_logo = item.x < logo_right and item.top_y < logo_bottom
            self.assertFalse(intersects_logo, item)

    def test_right_side_values_have_distinct_coordinates(self) -> None:
        items = build_page1_header_items(
            {
                "quote_metadata": {
                    "reference_number": "20260712-090807-A1B2",
                    "quote_created_date": "2026-07-12",
                    "quote_expiration_date": "2026-10-12",
                },
                "selected_internal": {
                    "name": "Grace Hopper",
                    "title": "Creative Lead",
                    "email": "grace@example.com",
                },
            }
        )
        right_texts = {
            "Reference: 20260712-090807-A1B2",
            "Quote created: July 12, 2026",
            "Quote expires: October 12, 2026",
            "Quote created by: Grace Hopper",
            "Creative Lead",
            "grace@example.com",
        }
        right_items = [item for item in items if item.text in right_texts]

        self.assertEqual(6, len(right_items))
        self.assertEqual(6, len({(item.x, item.top_y) for item in right_items}))
        created = next(item for item in right_items if item.text.startswith("Quote created:"))
        expires = next(item for item in right_items if item.text.startswith("Quote expires:"))
        self.assertLess(created.top_y, expires.top_y)

    def test_right_side_labels_are_rendered_as_complete_label_value_lines(self) -> None:
        items = build_page1_header_items(
            {
                "quote_metadata": {
                    "reference_number": "20260712-090807-A1B2",
                    "quote_created_date": "2026-07-12",
                    "quote_expiration_date": "2026-10-12",
                },
                "selected_internal": {"name": "Grace Hopper"},
            }
        )
        text_values = [item.text for item in items]

        self.assertIn("Reference: 20260712-090807-A1B2", text_values)
        self.assertIn("Quote created: July 12, 2026", text_values)
        self.assertIn("Quote expires: October 12, 2026", text_values)
        self.assertIn("Quote created by: Grace Hopper", text_values)

    def test_page1_header_payload_validation_rejects_missing_fields(self) -> None:
        errors = _header_payload_errors(
            {
                "quote_metadata": {
                    "quote_title": "",
                    "reference_number": "",
                    "quote_created_date": "2026-07-12",
                    "quote_expiration_date": "2026-10-12",
                },
                "selected_client": {
                    "company_name": "",
                    "full_name": "Ada Lovelace",
                    "email": "ada@example.com",
                },
                "selected_internal": {
                    "name": "Grace Hopper",
                    "title": "",
                    "email": "grace@example.com",
                },
            }
        )

        self.assertIn("Missing quote title for the page 1 header.", errors)
        self.assertIn("Missing reference number for the page 1 header.", errors)
        self.assertIn("Missing client company for the page 1 header.", errors)
        self.assertIn("Missing internal contact title for the page 1 header.", errors)

    def test_numeric_only_company_value_is_accepted(self) -> None:
        items = build_page1_header_items({"selected_client": {"company_name": "123456789"}})

        self.assertIn("123456789", [item.text for item in items])

    def test_long_internal_title_fits_without_ellipsis_when_space_permits(self) -> None:
        _register_gotham_fonts()
        lines, _font_size = _wrap_fitted_text(
            "Vice President, Accounts & Studio Operations",
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
            2,
        )

        self.assertLessEqual(len(lines), 2)
        self.assertNotIn("...", " ".join(lines))

    def test_existing_comments_payload_rendering_remains_unchanged(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={
                "id": "internal-1",
                "name": "Grace Hopper",
                "title": "Creative Lead",
                "email": "grace@example.com",
            },
            estimate_subject="Apparel Refresh",
            subtitle_line="Spring27 - Bangladesh",
            project_entries=[{"project_name": "Project A", "on_model": 2}],
            custom_notes="Rush timing requested.",
        )

        self.assertIn("Comments from Grace Hopper", payload.rendered_comments_block)
        self.assertIn("Photography Estimate for Apparel Refresh:", payload.rendered_comments_block)
        self.assertIn("Spring27 - Bangladesh", payload.rendered_comments_block)
        self.assertTrue(payload.rendered_comments_block.endswith("1 project="))

    def test_page2_pricing_mapper_output_remains_unchanged(self) -> None:
        quote = build_apparel_quote(
            ApparelInputs(
                on_model_image_quantity=10,
                on_model_detail_quantity=5,
                laydown_silo_type="shoes",
                laydown_silo_quantity=4,
                color_corrections_quantity=3,
                post_production_hours=2.0,
                model_type="kid",
                model_hours=1.5,
                model_fitting_enabled=True,
                ai_generation_quantity=2,
            )
        )

        payload = build_page2_pricing_payload(quote)
        rows = {row.code: row for row in payload.rows}

        self.assertEqual("$4,592.50", payload.subtotal)
        self.assertEqual("$4,592.50", payload.total)
        self.assertEqual("10", rows["on_model_image"].quantity)
        self.assertEqual("$240.00", rows["on_model_image"].unit_price)
        self.assertEqual("$2,400.00", rows["on_model_image"].total)
        self.assertEqual("$175.00", rows["account_management"].total)

    @unittest.skipUnless(Path("templates/photographytemplate.pdf").exists(), "photography PDF template is not present")
    def test_generated_pdf_contains_required_header_strings_and_preserves_pages(self) -> None:
        quote = build_apparel_quote(ApparelInputs(on_model_image_quantity=1))
        comments = build_page1_comments_payload(
            selected_internal_contact={
                "id": "internal-1",
                "name": "Grace Hopper",
                "title": "Creative Lead",
                "email": "grace@example.com",
            },
            estimate_subject="Apparel Refresh",
            subtitle_line="Spring27",
            project_entries=[{"project_name": "Project A", "on_model": 1, "on_model_detail": 2}],
            custom_notes="",
        ).to_payload()
        header = {
            "quote_metadata": {
                "quote_title": "UI Entered Quote Title",
                "reference_number": "UI-REF-1234",
                "quote_created_date": "2026-07-12",
                "quote_expiration_date": "2026-10-12",
            },
            "selected_client": {
                "company_name": "Acme",
                "full_name": "Ada Lovelace",
                "email": "ada@example.com",
            },
            "selected_internal": {
                "name": "Grace Hopper",
                "title": "Creative Lead",
                "email": "grace@example.com",
            },
        }

        pdf_bytes = generate_page2_pricing_pdf(quote, page1_comments_payload=comments, page1_header_payload=header)
        reader = PdfReader(BytesIO(pdf_bytes))
        text = reader.pages[0].extract_text() or ""

        self.assertEqual(4, len(reader.pages))
        self.assertIn("UI Entered Quote Title", text)
        self.assertIn("Reference: UI-REF-1234", text)
        self.assertIn("Quote created: July 12, 2026", text)
        self.assertIn("Quote expires: October 12, 2026", text)
        self.assertIn("Quote created by: Grace Hopper", text)
        self.assertIn("On Model Details= 2", text)


class ClientContactSearchTests(unittest.TestCase):
    def test_ranked_matches_and_stable_order(self):
        from app.contact_management.contact_ui import rank_client_contacts

        def contact(identifier, first, last, company="Acme", email="person@example.test", active=True):
            return ClientContact(identifier, None, company, first, last, email, active)

        contacts = [
            contact("email", "Amy", "Owen", email="l@example.test"),
            contact("company", "Amy", "Owen", company="London"),
            contact("contains", "Alan", "Owen"),
            contact("last", "Arthur", "Lin"),
            contact("lisa", "Lisa", "Gao"),
            contact("laura", "Laura", "Xu"),
            contact("lilian", "Lilian", "Zhou"),
            contact("inactive", "Lara", "Smith", active=False),
        ]
        self.assertEqual(contacts[:-1], rank_client_contacts(contacts, ""))
        self.assertEqual(["laura", "lilian", "lisa", "last", "contains", "company", "email"],
                         [c.id for c in rank_client_contacts(contacts, "L")])
        self.assertEqual(["lilian", "lisa", "last"], [c.id for c in rank_client_contacts(contacts, "Li")])
        self.assertEqual(rank_client_contacts(contacts, "li"), rank_client_contacts(reversed(contacts), " LI "))
        self.assertEqual([], rank_client_contacts(contacts, "no match"))
        self.assertEqual([contacts[4]], rank_client_contacts(contacts, "Lisa Gao"))

    def test_search_selection_no_match_and_draft_restore(self):
        from streamlit.testing.v1 import AppTest
        from app.contact_management.models import ClientContact, InternalContact
        from app.photography_pricing import draft_ui
        from app.photography_pricing.draft_service import serialize_draft_payload

        contacts = [
            ClientContact("client-a", None, "Alpha Company", "Ada", "Lovelace", "ada@alpha.test"),
            ClientContact("client-b", None, "Beta Studio", "Grace", "Hopper", "grace@beta.test"),
            ClientContact("client-c", None, "Gamma", "Alan", "Turing", "alan@gamma.test"),
        ]
        internal = InternalContact("internal-a", "Producer", "Lead", "producer@test.com")
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from app.photography_pricing.draft_ui import apply_pending_draft_restore\n"
            "from app.contact_management.contact_ui import render_client_contact_select, render_internal_contact_select, contact_payload\n"
            "from app.contact_management.client_autocomplete import apply_autocomplete_selection\n"
            "from app.contact_management.contact_ui import safe_list_active_client_contacts\n"
            "st.button('Autocomplete choice', on_click=lambda: apply_autocomplete_selection('client-b', safe_list_active_client_contacts()))\n"
            "apply_pending_draft_restore()\n"
            "st.session_state['test_client_payload'] = contact_payload(render_client_contact_select())\n"
            "render_internal_contact_select()\n"
        )
        with (
            patch("app.contact_management.client_autocomplete._component"),
            patch("app.contact_management.contact_ui.safe_list_active_client_contacts", return_value=contacts),
            patch("app.contact_management.contact_ui.safe_list_active_internal_contacts", return_value=[internal]),
            patch.object(draft_ui, "_contact_id_sets", return_value=({c.id for c in contacts}, {internal.id})),
        ):
            app.run()
            labels = [c.dropdown_label for c in contacts]
            self.assertEqual(labels, app.selectbox(key="photo_pricing_client_contact_id").options)
            app.button[0].click().run()
            self.assertEqual(0, len(app.exception))
            self.assertEqual("client-b", app.selectbox(key="photo_pricing_client_contact_id").value)
            self.assertEqual("Grace Hopper", app.session_state["test_client_payload"]["full_name"])
            app.selectbox(key="photo_pricing_client_contact_id").select("client-a").run()
            app.run()
            self.assertEqual("client-a", app.session_state["photo_pricing_client_contact_id"])
            payload = serialize_draft_payload({"photo_pricing_client_contact_id": "client-c"})
            self.assertNotIn("photo_pricing_client_contact_search", str(payload))
            app.session_state[draft_ui.PENDING_DRAFT_LOAD_KEY] = {
                "draft_id": "draft-c", "version_number": 1, "payload": payload,
            }
            app.run()
            self.assertEqual(0, len(app.exception))
            self.assertEqual(1, app.session_state["photo_pricing_client_search_revision"])
            self.assertEqual("client-c", app.session_state["photo_pricing_client_contact_id"])
            self.assertEqual(labels, app.selectbox(key="photo_pricing_client_contact_id").options)
            self.assertEqual("internal-a", app.session_state["photo_pricing_internal_contact_id"])

    def test_autocomplete_rejects_unknown_or_inactive_ids(self):
        from app.contact_management.client_autocomplete import apply_autocomplete_selection
        from app.contact_management.models import ClientContact
        state = {"photo_pricing_client_contact_id": "existing"}
        inactive = ClientContact("inactive", None, "A", "B", "C", "d@test.com", False)
        with patch("streamlit.session_state", state):
            for value in (None, "arbitrary text", "inactive", {"id": "existing"}):
                apply_autocomplete_selection(value, [inactive])
                self.assertEqual("existing", state["photo_pricing_client_contact_id"])


if __name__ == "__main__":
    unittest.main()
