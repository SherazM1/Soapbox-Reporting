import re
import unittest
from datetime import date, datetime
from unittest.mock import patch

from app.contact_management.contact_ui import contact_payload, resolve_client_contact, resolve_internal_contact
from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing.comments_builder import build_page1_comments_payload
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pdf_generator import build_page1_header_items
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
        self.assertIn("20260712-090807-A1B2", text_values)
        self.assertIn("July 12, 2026", text_values)
        self.assertIn("October 12, 2026", text_values)
        self.assertIn("Grace Hopper", text_values)
        self.assertIn("Creative Lead", text_values)
        self.assertIn("grace@example.com", text_values)

    def test_quote_title_uses_intended_non_logo_coordinate(self) -> None:
        items = build_page1_header_items({"quote_metadata": {"quote_title": "Holiday Apparel Quote"}})
        title = items[0]

        self.assertEqual(PAGE1_HEADER_TITLE_X, title.x)
        self.assertEqual(PAGE1_HEADER_TITLE_TOP_Y, title.top_y)
        self.assertGreaterEqual(title.x, PAGE1_LOGO_REGION[2])

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
            "20260712-090807-A1B2",
            "July 12, 2026",
            "October 12, 2026",
            "Grace Hopper",
            "Creative Lead",
            "grace@example.com",
        }
        right_items = [item for item in items if item.text in right_texts]

        self.assertEqual(6, len(right_items))
        self.assertEqual(6, len({(item.x, item.top_y) for item in right_items}))

    def test_numeric_only_company_value_is_suppressed(self) -> None:
        items = build_page1_header_items({"selected_client": {"company_name": "123456789"}})

        self.assertNotIn("123456789", [item.text for item in items])

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


if __name__ == "__main__":
    unittest.main()
