import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pdf_generator import PRICING_ROW_SLOT_TOP_Y, _page2_overlay, generate_page2_pricing_pdf
from app.photography_pricing.pdf_mapper import Page2PricingPayload, PdfPricingRow, build_page2_pricing_payload
from app.photography_pricing.quote_builder import build_apparel_quote


def _row_codes(payload: Page2PricingPayload) -> list[str]:
    return [row.code for row in payload.rows]


def _write_pdf(path: Path, page_text: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    for text in page_text:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


class PhotographyPdfPhase1Tests(unittest.TestCase):
    def test_zero_value_rows_are_omitted(self) -> None:
        payload = build_page2_pricing_payload(build_apparel_quote(ApparelInputs()))

        self.assertEqual(["account_management"], _row_codes(payload))
        self.assertEqual("$175.00", payload.rows[0].total)

    def test_packed_row_order_uses_required_logical_order(self) -> None:
        quote = build_apparel_quote(
            ApparelInputs(
                on_model_image_quantity=1,
                on_model_detail_quantity=1,
                laydown_silo_quantity=1,
                color_corrections_quantity=1,
                post_production_hours=1,
                model_type="both",
                kid_model_hours=1,
                adult_model_hours=1,
                model_fitting_quantity=1,
                ai_generation_quantity=1,
            )
        )

        self.assertEqual(
            [
                "on_model_image",
                "on_model_detail",
                "laydown_silo",
                "color_corrections",
                "post_production",
                "kid_model_hours",
                "adult_model_hours",
                "model_fitting",
                "ai_generation",
                "account_management",
            ],
            _row_codes(build_page2_pricing_payload(quote)),
        )

    def test_adult_only_model_hours_map_to_adult_row(self) -> None:
        payload = build_page2_pricing_payload(
            build_apparel_quote(ApparelInputs(model_type="adult", adult_model_hours=2))
        )

        self.assertIn("adult_model_hours", _row_codes(payload))
        self.assertNotIn("kid_model_hours", _row_codes(payload))
        adult_row = next(row for row in payload.rows if row.code == "adult_model_hours")
        self.assertEqual("2", adult_row.quantity)
        self.assertEqual("$230.00", adult_row.unit_price)
        self.assertEqual("$460.00", adult_row.total)

    def test_kid_only_model_hours_map_to_kid_row(self) -> None:
        payload = build_page2_pricing_payload(build_apparel_quote(ApparelInputs(model_type="kid", kid_model_hours=2.5)))

        self.assertIn("kid_model_hours", _row_codes(payload))
        self.assertNotIn("adult_model_hours", _row_codes(payload))
        kid_row = next(row for row in payload.rows if row.code == "kid_model_hours")
        self.assertEqual("2.5", kid_row.quantity)
        self.assertEqual("$105.00", kid_row.unit_price)
        self.assertEqual("$262.50", kid_row.total)

    def test_both_model_hours_map_to_separate_rows(self) -> None:
        payload = build_page2_pricing_payload(
            build_apparel_quote(ApparelInputs(model_type="both", kid_model_hours=3, adult_model_hours=2))
        )

        self.assertIn("kid_model_hours", _row_codes(payload))
        self.assertIn("adult_model_hours", _row_codes(payload))
        self.assertLess(_row_codes(payload).index("kid_model_hours"), _row_codes(payload).index("adult_model_hours"))

    def test_both_model_hours_omits_zero_side(self) -> None:
        payload = build_page2_pricing_payload(
            build_apparel_quote(ApparelInputs(model_type="both", kid_model_hours=4, adult_model_hours=0))
        )

        self.assertIn("kid_model_hours", _row_codes(payload))
        self.assertNotIn("adult_model_hours", _row_codes(payload))

    def test_model_fitting_quantity_maps_quantity_unit_price_and_total(self) -> None:
        payload = build_page2_pricing_payload(build_apparel_quote(ApparelInputs(model_fitting_quantity=3)))
        row = next(row for row in payload.rows if row.code == "model_fitting")

        self.assertEqual("3", row.quantity)
        self.assertEqual("$50.00", row.unit_price)
        self.assertEqual("$150.00", row.total)

    def test_manual_account_management_uses_manual_amount(self) -> None:
        payload = build_page2_pricing_payload(
            build_apparel_quote(
                ApparelInputs(
                    on_model_image_quantity=10,
                    account_management_mode="manual",
                    manual_account_management_amount=425,
                )
            )
        )
        row = next(row for row in payload.rows if row.code == "account_management")

        self.assertEqual("$425.00", row.total)

    def test_subtotal_and_total_are_preserved(self) -> None:
        payload = build_page2_pricing_payload(
            build_apparel_quote(
                ApparelInputs(
                    on_model_image_quantity=10,
                    account_management_mode="manual",
                    manual_account_management_amount=425,
                )
            )
        )

        self.assertEqual("$2,825.00", payload.subtotal)
        self.assertEqual("$2,825.00", payload.total)

    def test_overlay_draws_remaining_rows_in_sequential_slots(self) -> None:
        payload = Page2PricingPayload(
            rows=(
                PdfPricingRow("on_model_image", "First Row", "1", "$2.00", "$2.00"),
                PdfPricingRow("model_fitting", "Second Row", "3", "$4.00", "$12.00"),
            ),
            subtotal="$14.00",
            total="$14.00",
        )
        overlay = PdfReader(_page2_overlay(1785, 2526, payload))
        positions: dict[str, float] = {}

        def visitor(text, cm, tm, font_dict, font_size):
            clean = " ".join((text or "").split())
            if clean in {"First Row", "Second Row"}:
                positions[clean] = round(2526 - float(tm[5]), 2)

        overlay.pages[0].extract_text(visitor_text=visitor)

        self.assertEqual(round(PRICING_ROW_SLOT_TOP_Y[0], 2), positions["First Row"])
        self.assertEqual(round(PRICING_ROW_SLOT_TOP_Y[1], 2), positions["Second Row"])

    def test_final_pdf_order_uses_new_pricing_page_and_skips_old_template_page_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main_template = tmp_path / "main.pdf"
            pricing_template = tmp_path / "pricing.pdf"
            _write_pdf(
                main_template,
                [
                    "MAIN PAGE 1",
                    "On-model detail Model Fitting AI Gene OLD PAGE 2",
                    "MAIN PAGE 3",
                    "MAIN PAGE 4",
                ],
            )
            _write_pdf(pricing_template, ["NEW PAGE 2"])

            pdf_bytes = generate_page2_pricing_pdf(
                build_apparel_quote(ApparelInputs(on_model_image_quantity=1)),
                template_path=main_template,
                pricing_template_path=pricing_template,
            )
            reader = PdfReader(BytesIO(pdf_bytes))
            texts = [page.extract_text() or "" for page in reader.pages]

        self.assertEqual(4, len(reader.pages))
        self.assertIn("MAIN PAGE 1", texts[0])
        self.assertIn("NEW PAGE 2", texts[1])
        self.assertIn("MAIN PAGE 3", texts[2])
        self.assertIn("MAIN PAGE 4", texts[3])
        self.assertFalse(any("OLD PAGE 2" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
