from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.creative_pricing.config import PRICING_NOT_CONFIGURED_LABEL
from app.creative_pricing.models import CreativePricingInputs


@dataclass(frozen=True)
class SummaryRow:
    label: str
    value: str
    pricing: str = PRICING_NOT_CONFIGURED_LABEL


@dataclass(frozen=True)
class SummarySection:
    title: str
    rows: tuple[SummaryRow, ...]


def _join_selected(values: Iterable[str]) -> str:
    selected = [value for value in values if value]
    return ", ".join(selected) if selected else "None selected"


def build_creative_pricing_summary(inputs: CreativePricingInputs) -> tuple[SummarySection, ...]:
    deliverables = inputs.deliverables
    copy_rows = [SummaryRow("Copy provided", inputs.copy_provided)]
    if inputs.copy_provided == "No":
        copy_rows.append(SummaryRow("Copy scope", _join_selected(inputs.copy_scopes)))

    asset_rows = [SummaryRow("Assets provided", inputs.assets_provided)]
    if inputs.assets_provided == "Yes":
        asset_rows.append(SummaryRow("Provided assets", _join_selected(inputs.provided_assets)))
    else:
        asset_rows.append(SummaryRow("Missing assets", _join_selected(inputs.missing_assets)))

    revision_value = (
        inputs.fixed_revision_tier
        if inputs.revision_mode == "Fixed"
        else f"${inputs.manual_revision_amount:,.2f} manual amount"
    )

    internal_rows = [SummaryRow("Selected work", _join_selected(inputs.internal_work))]
    if "Other Internal Work" in inputs.internal_work:
        internal_rows.append(SummaryRow("Other internal work", inputs.other_internal_work or "Not specified"))
        if inputs.other_internal_amount:
            internal_rows.append(
                SummaryRow("Other internal amount", f"${inputs.other_internal_amount:,.2f}", "Manual amount")
            )

    tool_rows = [SummaryRow("Selected tools/platforms", _join_selected(inputs.tool_platforms))]
    if "Other Tool / Platform" in inputs.tool_platforms:
        tool_rows.append(SummaryRow("Other tool/platform", inputs.other_tool_platform or "Not specified"))

    return (
        SummarySection(
            "Deliverables",
            (
                SummaryRow("Number of PDPs", str(deliverables.pdp_count)),
                SummaryRow("Number of Briefs", str(deliverables.brief_count)),
                SummaryRow("Number of Ads", str(deliverables.ad_count)),
            ),
        ),
        SummarySection("Copy", tuple(copy_rows)),
        SummarySection("Assets", tuple(asset_rows)),
        SummarySection("Revisions", (SummaryRow("Revision handling", inputs.revision_mode), SummaryRow("Revision detail", revision_value))),
        SummarySection("Internal Work Costs", tuple(internal_rows)),
        SummarySection("External Work / Tools / Platforms", tuple(tool_rows)),
        SummarySection("Market Tier", (SummaryRow("Tier", inputs.market_tier),)),
    )

