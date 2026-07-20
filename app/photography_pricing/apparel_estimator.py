from typing import Any

import pandas as pd
import streamlit as st

from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.pricing_rules import (
    AI_GENERATION_MARKUP_RATE,
    COLOR_CORRECTIONS_RATE,
    MODEL_FITTING_FLAT_FEE,
    ON_MODEL_DETAIL_RATE,
    ON_MODEL_IMAGE_RATE,
    POST_PRODUCTION_HOURLY_RATE,
    account_management_tier_label,
    laydown_silo_rate,
    model_hourly_rate,
)
from app.photography_pricing.quote_builder import build_apparel_quote


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _quantity_input(label: str, key: str, help_text: str | None = None) -> int:
    return int(st.number_input(label, min_value=0, step=1, value=0, key=key, help=help_text))


def _hours_input(label: str, key: str) -> float:
    return float(st.number_input(label, min_value=0.0, step=0.25, value=0.0, key=key, format="%.2f"))


def _line_table_rows(quote_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for line in quote_payload["line_items"]:
        rows.append(
            {
                "Item": line["label"],
                "Quantity": line["quantity"],
                "Locked base rate": _money(line["unit_price"]),
                "Row total": _money(line["total"]),
            }
        )
    return rows


def _render_apparel_inputs() -> ApparelInputs:
    st.subheader("Apparel Estimator")

    image_col, production_col = st.columns(2)

    with image_col:
        st.markdown("#### Image services")
        on_model_image_quantity = _quantity_input(
            "On-model image quantity",
            "photo_pricing_on_model_image_quantity",
        )
        st.caption(f"Locked rate: {_money(ON_MODEL_IMAGE_RATE)} per image")

        on_model_detail_quantity = _quantity_input(
            "On-model detail quantity",
            "photo_pricing_on_model_detail_quantity",
        )
        st.caption(f"Locked rate: {_money(ON_MODEL_DETAIL_RATE)} per image")

        laydown_silo_type = st.selectbox(
            "Laydown silo type",
            ["else/default", "shoes"],
            key="photo_pricing_laydown_silo_type",
        )
        laydown_silo_quantity = _quantity_input(
            "Laydown silo quantity",
            "photo_pricing_laydown_silo_quantity",
        )
        st.caption(f"Locked rate: {_money(laydown_silo_rate(laydown_silo_type))} per image")

        color_corrections_quantity = _quantity_input(
            "Color corrections from existing images quantity",
            "photo_pricing_color_corrections_quantity",
        )
        st.caption(f"Locked rate: {_money(COLOR_CORRECTIONS_RATE)} per image")

        ai_generation_quantity = _quantity_input(
            "AI generation markup quantity",
            "photo_pricing_ai_generation_quantity",
        )
        st.caption(f"Locked rate: {_money(AI_GENERATION_MARKUP_RATE)} per image")

    with production_col:
        st.markdown("#### Production services")
        post_production_hours = _hours_input(
            "Post production hourly time",
            "photo_pricing_post_production_hours",
        )
        st.caption(f"Locked rate: {_money(POST_PRODUCTION_HOURLY_RATE)} per hour")

        model_type = st.radio(
            "Model hours type",
            ["adult", "kid"],
            horizontal=True,
            key="photo_pricing_model_type",
        )
        model_hours = _hours_input("Model hours", "photo_pricing_model_hours")
        st.caption(f"Locked rate: {_money(model_hourly_rate(model_type))} per hour")

        model_fitting_enabled = st.checkbox(
            "Model fitting",
            key="photo_pricing_model_fitting_enabled",
        )
        st.caption(f"Locked flat fee: {_money(MODEL_FITTING_FLAT_FEE)}")

    return ApparelInputs(
        on_model_image_quantity=on_model_image_quantity,
        on_model_detail_quantity=on_model_detail_quantity,
        laydown_silo_type=laydown_silo_type,
        laydown_silo_quantity=laydown_silo_quantity,
        color_corrections_quantity=color_corrections_quantity,
        post_production_hours=post_production_hours,
        model_type=model_type,
        model_hours=model_hours,
        model_fitting_enabled=model_fitting_enabled,
        ai_generation_quantity=ai_generation_quantity,
    )


def render_photography_pricing() -> None:
    st.title("Photography Pricing")

    if st.button("Back to Hub", key="photo_pricing_back_to_hub"):
        st.session_state["hub_view"] = "home"
        st.rerun()

    st.markdown("#### Job type")
    job_col, disabled_col = st.columns([1, 1])
    with job_col:
        st.selectbox(
            "Active job type",
            ["Apparel"],
            key="photo_pricing_job_type",
        )
    with disabled_col:
        st.button("Misc", disabled=True, use_container_width=True, key="photo_pricing_misc_disabled")
        st.caption("Unavailable in this first phase.")

    inputs = _render_apparel_inputs()
    quote = build_apparel_quote(inputs)
    quote_payload = quote.to_payload()

    line_col, summary_col = st.columns([2, 1], gap="large")
    with line_col:
        st.subheader("Pricing Rows")
        st.dataframe(
            pd.DataFrame(_line_table_rows(quote_payload)),
            hide_index=True,
            use_container_width=True,
        )

    with summary_col:
        st.subheader("Summary")
        st.metric("Image count for account management", quote.derived_total_image_count)
        st.write(f"Account management tier: **{account_management_tier_label(quote.derived_total_image_count)}**")
        st.write(f"Account management fee: **{_money(quote.derived_account_management_fee)}**")
        st.divider()
        st.metric("Running subtotal", _money(quote.subtotal))
        st.metric("Final total", _money(quote.total))

    with st.expander("Normalized quote payload"):
        st.json(quote_payload)
