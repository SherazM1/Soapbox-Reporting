from __future__ import annotations

import pandas as pd
import streamlit as st

from app.creative_pricing.config import (
    ASSET_OPTIONS,
    COPY_SCOPE_OPTIONS,
    INTERNAL_WORK_OPTIONS,
    MARKET_TIERS,
    REVISION_TIERS,
    TOOL_PLATFORM_OPTIONS,
)
from app.creative_pricing.models import CreativePricingInputs, DeliverableInputs
from app.creative_pricing.summary_builder import build_creative_pricing_summary


def _labels(options) -> list[str]:
    return [option.label for option in options]


def _render_summary(inputs: CreativePricingInputs) -> None:
    st.header("Summary")
    st.caption("Pricing values can be added later in the Creative Pricing config.")

    for section in build_creative_pricing_summary(inputs):
        with st.expander(section.title, expanded=True):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Item": row.label, "Selection": row.value, "Pricing": row.pricing}
                        for row in section.rows
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_creative_pricing() -> None:
    st.title("Creative Pricing")
    st.caption("Initial setup workspace for creative deliverables, inputs, and future pricing configuration.")

    if st.button("Back to Hub", key="creative_pricing_back_to_hub"):
        st.session_state["hub_view"] = "home"
        st.rerun()

    st.header("Deliverables")
    col_pdp, col_brief, col_ad = st.columns(3)
    with col_pdp:
        pdp_count = st.number_input("Number of PDPs", min_value=0, step=1, value=0, key="creative_pdp_count")
    with col_brief:
        brief_count = st.number_input("Number of Briefs", min_value=0, step=1, value=0, key="creative_brief_count")
    with col_ad:
        ad_count = st.number_input("Number of Ads", min_value=0, step=1, value=0, key="creative_ad_count")

    st.header("Copy")
    copy_provided = st.radio("Did they provide copy?", ["Yes", "No"], horizontal=True, key="creative_copy_provided")
    copy_scopes: list[str] = []
    if copy_provided == "No":
        copy_scopes = st.multiselect(
            "Copy requirements",
            _labels(COPY_SCOPE_OPTIONS),
            key="creative_copy_scopes",
        )

    st.header("Assets")
    assets_provided = st.radio(
        "Did they provide assets?",
        ["Yes", "No", "Partial"],
        horizontal=True,
        key="creative_assets_provided",
    )
    provided_assets: list[str] = []
    missing_assets: list[str] = []
    if assets_provided == "Yes":
        provided_assets = st.multiselect(
            "Assets provided for workflow context",
            _labels(ASSET_OPTIONS),
            key="creative_provided_assets",
        )
    else:
        missing_assets = st.multiselect(
            "Assets missing",
            _labels(ASSET_OPTIONS),
            key="creative_missing_assets",
        )

    st.header("Revisions")
    revision_mode = st.radio(
        "Revision mode",
        ["Fixed", "Manual"],
        horizontal=True,
        key="creative_revision_mode",
    )
    fixed_revision_tier = "Simple"
    manual_revision_amount = 0.0
    if revision_mode == "Fixed":
        fixed_revision_tier = st.selectbox(
            "Fixed revision tier",
            _labels(REVISION_TIERS),
            key="creative_fixed_revision_tier",
        )
    else:
        manual_revision_amount = st.number_input(
            "Manual revision amount",
            min_value=0.0,
            step=25.0,
            value=0.0,
            format="%.2f",
            key="creative_manual_revision_amount",
        )

    st.header("Internal Work Costs")
    internal_work = st.multiselect(
        "Internal work / added manpower / collaboration",
        _labels(INTERNAL_WORK_OPTIONS),
        key="creative_internal_work",
    )
    other_internal_work = ""
    other_internal_amount = 0.0
    if "Other Internal Work" in internal_work:
        other_internal_work = st.text_input(
            "Other internal work details",
            key="creative_other_internal_work",
        )
        other_internal_amount = st.number_input(
            "Optional other internal work amount",
            min_value=0.0,
            step=25.0,
            value=0.0,
            format="%.2f",
            key="creative_other_internal_amount",
        )

    st.header("External Work / Tools / Platforms")
    tool_platforms = st.multiselect(
        "Tools/platforms used",
        _labels(TOOL_PLATFORM_OPTIONS),
        key="creative_tool_platforms",
    )
    other_tool_platform = ""
    if "Other Tool / Platform" in tool_platforms:
        other_tool_platform = st.text_input(
            "Other tool/platform details",
            key="creative_other_tool_platform",
        )

    st.header("Market Tier")
    market_tier = st.selectbox("Market tier", _labels(MARKET_TIERS), key="creative_market_tier")

    inputs = CreativePricingInputs(
        deliverables=DeliverableInputs(
            pdp_count=int(pdp_count),
            brief_count=int(brief_count),
            ad_count=int(ad_count),
        ),
        copy_provided=copy_provided,
        copy_scopes=tuple(copy_scopes),
        assets_provided=assets_provided,
        provided_assets=tuple(provided_assets),
        missing_assets=tuple(missing_assets),
        revision_mode=revision_mode,
        fixed_revision_tier=fixed_revision_tier,
        manual_revision_amount=float(manual_revision_amount),
        internal_work=tuple(internal_work),
        other_internal_work=other_internal_work,
        other_internal_amount=float(other_internal_amount),
        tool_platforms=tuple(tool_platforms),
        other_tool_platform=other_tool_platform,
        market_tier=market_tier,
    )
    _render_summary(inputs)

