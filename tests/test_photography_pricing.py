import unittest

from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pricing_rules import MODEL_FITTING_FLAT_FEE, account_management_fee
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
            model_fitting_quantity=1,
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

    def test_model_fitting_quantity_zero_has_no_charge(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_fitting_quantity=0))
        line = {line.code: line for line in quote.line_items}["model_fitting"]

        self.assertEqual(line.quantity, 0)
        self.assertEqual(line.unit_price, MODEL_FITTING_FLAT_FEE)
        self.assertEqual(line.total, 0)

    def test_model_fitting_quantity_one_uses_flat_fee(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_fitting_quantity=1))
        line = {line.code: line for line in quote.line_items}["model_fitting"]

        self.assertEqual(line.quantity, 1)
        self.assertEqual(line.total, 50.00)

    def test_model_fitting_quantity_multiple_multiplies_flat_fee(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_fitting_quantity=3))
        line = {line.code: line for line in quote.line_items}["model_fitting"]

        self.assertEqual(line.quantity, 3)
        self.assertEqual(line.total, 150.00)
        self.assertEqual(quote.subtotal, 325.00)
        self.assertEqual(quote.total, 325.00)

    def test_adult_only_model_hours_use_adult_rate(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_type="adult", adult_model_hours=2.0))
        line = {line.code: line for line in quote.line_items}["model_hours"]

        self.assertEqual(line.quantity, 2.0)
        self.assertEqual(line.unit_price, 230.00)
        self.assertEqual(line.total, 460.00)

    def test_kid_only_model_hours_use_kid_rate(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_type="kid", kid_model_hours=2.5))
        line = {line.code: line for line in quote.line_items}["model_hours"]

        self.assertEqual(line.quantity, 2.5)
        self.assertEqual(line.unit_price, 105.00)
        self.assertEqual(line.total, 262.50)

    def test_both_model_hours_create_separate_adult_and_kid_lines(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_type="both", adult_model_hours=2.0, kid_model_hours=3.0))
        lines = {line.code: line for line in quote.line_items}

        self.assertEqual(lines["adult_model_hours"].label, "Adult model hours")
        self.assertEqual(lines["adult_model_hours"].quantity, 2.0)
        self.assertEqual(lines["adult_model_hours"].unit_price, 230.00)
        self.assertEqual(lines["adult_model_hours"].total, 460.00)
        self.assertEqual(lines["kid_model_hours"].label, "Kid model hours")
        self.assertEqual(lines["kid_model_hours"].quantity, 3.0)
        self.assertEqual(lines["kid_model_hours"].unit_price, 105.00)
        self.assertEqual(lines["kid_model_hours"].total, 315.00)
        self.assertEqual(quote.subtotal, 950.00)
        self.assertEqual(quote.total, 950.00)

    def test_both_model_hours_keep_zero_side_available(self) -> None:
        quote = build_apparel_quote(ApparelInputs(model_type="both", adult_model_hours=0.0, kid_model_hours=4.0))
        lines = {line.code: line for line in quote.line_items}

        self.assertIn("adult_model_hours", lines)
        self.assertEqual(lines["adult_model_hours"].quantity, 0.0)
        self.assertEqual(lines["adult_model_hours"].total, 0.00)
        self.assertEqual(lines["kid_model_hours"].total, 420.00)
        self.assertEqual(quote.subtotal, 595.00)
        self.assertEqual(quote.total, 595.00)

    def test_automatic_account_management_remains_default(self) -> None:
        quote = build_apparel_quote(ApparelInputs(on_model_image_quantity=35))
        line = {line.code: line for line in quote.line_items}["account_management"]

        self.assertEqual(quote.account_management_mode, "automatic")
        self.assertEqual(quote.derived_account_management_fee, 350.00)
        self.assertEqual(quote.manual_account_management_amount, 0.00)
        self.assertEqual(quote.account_management_amount_used, 350.00)
        self.assertEqual(line.total, 350.00)

    def test_manual_account_management_overrides_amount_used_and_totals(self) -> None:
        quote = build_apparel_quote(
            ApparelInputs(
                on_model_image_quantity=10,
                account_management_mode="manual",
                manual_account_management_amount=425.00,
            )
        )
        lines = {line.code: line for line in quote.line_items}

        self.assertEqual(quote.derived_account_management_fee, 175.00)
        self.assertEqual(quote.account_management_mode, "manual")
        self.assertEqual(quote.manual_account_management_amount, 425.00)
        self.assertEqual(quote.account_management_amount_used, 425.00)
        self.assertEqual(lines["account_management"].total, 425.00)
        self.assertEqual(quote.subtotal, 2825.00)
        self.assertEqual(quote.total, 2825.00)

    def test_manual_account_management_negative_amount_is_clamped_to_zero(self) -> None:
        quote = build_apparel_quote(
            ApparelInputs(
                account_management_mode="manual",
                manual_account_management_amount=-10.00,
            )
        )
        line = {line.code: line for line in quote.line_items}["account_management"]

        self.assertEqual(quote.manual_account_management_amount, 0.00)
        self.assertEqual(quote.account_management_amount_used, 0.00)
        self.assertEqual(line.total, 0.00)

    def test_payload_shape_is_normalized_for_future_pdf_mapping(self) -> None:
        payload = build_apparel_quote(ApparelInputs(on_model_image_quantity=35)).to_payload()

        self.assertEqual(payload["selected_job_type"], "Apparel")
        self.assertEqual(payload["derived_total_image_count"], 35)
        self.assertEqual(payload["derived_account_management_fee"], 350.00)
        self.assertEqual(payload["account_management_mode"], "automatic")
        self.assertEqual(payload["manual_account_management_amount"], 0.00)
        self.assertEqual(payload["account_management_amount_used"], 350.00)
        self.assertIn("apparel_inputs", payload)
        self.assertEqual(payload["apparel_inputs"]["model_fitting_quantity"], 0)
        self.assertIn("line_items", payload)
        self.assertEqual(
            {"code", "label", "quantity", "unit_price", "total"},
            set(payload["line_items"][0].keys()),
        )


if __name__ == "__main__":
    unittest.main()
