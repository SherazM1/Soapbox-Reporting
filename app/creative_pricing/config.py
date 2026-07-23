from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


PriceValue = Optional[float]


@dataclass(frozen=True)
class PricedOption:
    code: str
    label: str
    price: PriceValue = None


DELIVERABLE_PRICES: dict[str, PriceValue] = {
    "pdp": None,
    "brief": None,
    "ad": None,
}

COPY_SCOPE_OPTIONS: tuple[PricedOption, ...] = (
    PricedOption("pdp_graphic_copy", "PDP Graphic Copy"),
    PricedOption("brief_copy", "Brief Copy"),
    PricedOption("ad_copy", "Ad Copy"),
    PricedOption("mixed_multiple", "Mixed / Multiple"),
    PricedOption("full_copy_package", "Full Copy Package"),
)

ASSET_OPTIONS: tuple[PricedOption, ...] = (
    PricedOption("themes", "Themes"),
    PricedOption("color_palettes", "Color Palettes"),
    PricedOption("artwork", "Artwork"),
    PricedOption("product_imagery", "Product Imagery"),
    PricedOption("brand_guidelines", "Brand Guidelines"),
)

REVISION_TIERS: tuple[PricedOption, ...] = (
    PricedOption("simple", "Simple"),
    PricedOption("moderate", "Moderate"),
    PricedOption("complex", "Complex"),
)

INTERNAL_WORK_OPTIONS: tuple[PricedOption, ...] = (
    PricedOption("photography_support", "Photography Support"),
    PricedOption("mood_board_collaboration", "Mood Board Collaboration"),
    PricedOption("multi_team_collaboration", "Multi-Team Collaboration"),
    PricedOption("ecom_team_support", "Ecom Team Support"),
    PricedOption("retail_media_team_support", "Retail Media Team Support"),
    PricedOption("graphic_team_concepting", "Graphic Team Concepting"),
    PricedOption("other_internal_work", "Other Internal Work"),
)

TOOL_PLATFORM_OPTIONS: tuple[PricedOption, ...] = (
    PricedOption("adobe", "Adobe"),
    PricedOption("i_stock", "I-Stock"),
    PricedOption("illustrator", "Illustrator"),
    PricedOption("photoshop", "Photoshop"),
    PricedOption("indesign", "InDesign"),
    PricedOption("nanobanana", "Nanobanana"),
    PricedOption("ai_separate_cost", "AI Separate Cost"),
    PricedOption("other_tool_platform", "Other Tool / Platform"),
)

MARKET_TIERS: tuple[PricedOption, ...] = (
    PricedOption("local", "Local"),
    PricedOption("regional", "Regional"),
    PricedOption("national", "National"),
)

PRICING_NOT_CONFIGURED_LABEL = "Pricing not configured yet"

