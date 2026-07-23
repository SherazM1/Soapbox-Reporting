import unittest
from pathlib import Path

from app.creative_pricing.config import (
    ASSET_OPTIONS,
    COPY_SCOPE_OPTIONS,
    INTERNAL_WORK_OPTIONS,
    MARKET_TIERS,
    PRICING_NOT_CONFIGURED_LABEL,
    REVISION_TIERS,
    TOOL_PLATFORM_OPTIONS,
)
from app.creative_pricing.models import CreativePricingInputs, DeliverableInputs
from app.creative_pricing.summary_builder import build_creative_pricing_summary


ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "streamlitapp.py"


class CreativePricingTests(unittest.TestCase):
    def test_config_contains_requested_workflow_options(self) -> None:
        self.assertIn("PDP Graphic Copy", [option.label for option in COPY_SCOPE_OPTIONS])
        self.assertIn("Product Imagery", [option.label for option in ASSET_OPTIONS])
        self.assertIn("Complex", [option.label for option in REVISION_TIERS])
        self.assertIn("Other Internal Work", [option.label for option in INTERNAL_WORK_OPTIONS])
        self.assertIn("AI Separate Cost", [option.label for option in TOOL_PLATFORM_OPTIONS])
        self.assertEqual(["Local", "Regional", "National"], [option.label for option in MARKET_TIERS])

    def test_summary_is_user_facing_and_missing_price_safe(self) -> None:
        sections = build_creative_pricing_summary(
            CreativePricingInputs(
                deliverables=DeliverableInputs(pdp_count=2, brief_count=1, ad_count=3),
                copy_provided="No",
                copy_scopes=("PDP Graphic Copy", "Ad Copy"),
                assets_provided="Partial",
                missing_assets=("Artwork", "Brand Guidelines"),
                revision_mode="Manual",
                manual_revision_amount=250.0,
                internal_work=("Other Internal Work",),
                other_internal_work="Concept planning",
                tool_platforms=("Adobe", "Other Tool / Platform"),
                other_tool_platform="Licensing portal",
                market_tier="National",
            )
        )

        rendered = "\n".join(
            f"{section.title}: {row.label}: {row.value}: {row.pricing}"
            for section in sections
            for row in section.rows
        )

        self.assertIn("Number of PDPs: 2", rendered)
        self.assertIn("Copy scope: PDP Graphic Copy, Ad Copy", rendered)
        self.assertIn("Missing assets: Artwork, Brand Guidelines", rendered)
        self.assertIn("Revision detail: $250.00 manual amount", rendered)
        self.assertIn("Other internal work: Concept planning", rendered)
        self.assertIn("Other tool/platform: Licensing portal", rendered)
        self.assertIn("Tier: National", rendered)
        self.assertIn(PRICING_NOT_CONFIGURED_LABEL, rendered)
        self.assertNotIn("payload", rendered.lower())
        self.assertNotIn("json", rendered.lower())

    def test_hub_exposes_creative_pricing_without_changing_existing_routes(self) -> None:
        source = STREAMLIT_APP.read_text(encoding="utf-8")

        self.assertIn('VIEW_CREATIVE_PRICING = "creative_pricing"', source)
        self.assertIn('<h3>Creative Pricing</h3>', source)
        self.assertIn('href="?hub=creative_pricing"', source)
        self.assertIn("render_creative_pricing()", source)
        self.assertIn('VIEW_PHOTOGRAPHY_PRICING = "photography_pricing"', source)
        self.assertIn("render_photography_pricing()", source)


if __name__ == "__main__":
    unittest.main()

