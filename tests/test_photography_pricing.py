import unittest

from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pricing_rules import account_management_fee
from app.photography_pricing.quote_builder import build_apparel_quote, image_count_for_account_management


class PhotographyPricingTests(unittest.TestCase):
    def test_account_management_tiers(self) -> None:
        self.assertEqual(account_management_fee(0), 175.00)
        self.assertEqual(account_management_fee(34), 175.00)
        self.assertEqual(account_management_fee(35), 350.00)
        self.assertEqual(account_management_fee(64), 350.00)
        self.assertEqual(account_management_fee(65), 750.00)

    def test_apparel_quote_uses_locked_rates_and_image_count_basis(self) -> None:
        inputs = ApparelInputs(
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

        quote = build_apparel_quote(inputs)
        lines = {line.code: line for line in quote.line_items}

        self.assertEqual(image_count_for_account_management(inputs), 24)
        self.assertEqual(quote.derived_account_management_fee, 175.00)
        self.assertEqual(lines["on_model_image"].total, 2400.00)
        self.assertEqual(lines["on_model_detail"].total, 725.00)
        self.assertEqual(lines["laydown_silo"].unit_price, 75.00)
        self.assertEqual(lines["laydown_silo"].total, 300.00)
        self.assertEqual(lines["color_corrections"].total, 135.00)
        self.assertEqual(lines["post_production"].total, 350.00)
        self.assertEqual(lines["model_hours"].unit_price, 105.00)
        self.assertEqual(lines["model_hours"].total, 157.50)
        self.assertEqual(lines["model_fitting"].quantity, 1)
        self.assertEqual(lines["model_fitting"].total, 50.00)
        self.assertEqual(lines["ai_generation"].total, 300.00)
        self.assertEqual(lines["account_management"].total, 175.00)
        self.assertEqual(quote.subtotal, 4592.50)
        self.assertEqual(quote.total, 4592.50)

    def test_payload_shape_is_normalized_for_future_pdf_mapping(self) -> None:
        payload = build_apparel_quote(ApparelInputs(on_model_image_quantity=35)).to_payload()

        self.assertEqual(payload["selected_job_type"], "Apparel")
        self.assertEqual(payload["derived_total_image_count"], 35)
        self.assertEqual(payload["derived_account_management_fee"], 350.00)
        self.assertIn("apparel_inputs", payload)
        self.assertIn("line_items", payload)
        self.assertEqual(
            {"code", "label", "quantity", "unit_price", "total"},
            set(payload["line_items"][0].keys()),
        )


if __name__ == "__main__":
    unittest.main()
