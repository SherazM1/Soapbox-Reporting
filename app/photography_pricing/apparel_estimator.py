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
    return int(
        st.number_input(
            label,
            min_value=0,
            step=1,
            value=0,
            key=key,
            help=help_text,
            label_visibility="collapsed",
        )
    )


def _hours_input(label: str, key: str) -> float:
    return float(
        st.number_input(
            label,
            min_value=0.0,
            step=0.25,
            value=0.0,
            key=key,
            format="%.2f",
            label_visibility="collapsed",
        )
    )


def _rate_note(value: float, unit_label: str) -> None:
    st.caption(f"Locked rate: {_money(value)} {unit_label}")


def _field_label(text: str) -> None:
    st.markdown(f"**{text}**")


def _line_table_rows(quote_payload: dict[str, Any]) -> list[dict[str, Any]]:
    display_labels = {
        "On-model image": "On-Model Image",
        "On-model detail": "On-Model Detail",
        "Laydown silo": "Laydown Silo",
        "Color corrections from existing images": "Color Corrections From Existing Images",
        "Post production hourly time": "Post Production Hourly Time",
        "Model hours": "Model Hours",
        "Model fitting": "Model Fitting",
        "AI generation markup": "AI Generation Markup",
        "Account management": "Account Management",
    }
    rows = []
    for line in quote_payload["line_items"]:
        rows.append(
            {
                "Item": display_labels.get(line["label"], line["label"]),
                "Quantity": line["quantity"],
                "Locked Base Rate": _money(line["unit_price"]),
                "Row Total": _money(line["total"]),
            }
        )
    return rows


def _render_apparel_inputs() -> tuple[ApparelInputs, Any]:
    image_col, production_col = st.columns(2)

    with image_col:
        st.markdown("#### Image Services")
        _field_label("On-Model Image")
        on_model_image_quantity = _quantity_input(
            "On-Model Image",
            "photo_pricing_on_model_image_quantity",
        )
        _rate_note(ON_MODEL_IMAGE_RATE, "per image")

        _field_label("On-Model Detail")
        on_model_detail_quantity = _quantity_input(
            "On-Model Detail",
            "photo_pricing_on_model_detail_quantity",
        )
        _rate_note(ON_MODEL_DETAIL_RATE, "per image")

        _field_label("Laydown Silo Type")
        laydown_silo_type = st.selectbox(
            "Laydown Silo Type",
            ["else/default", "shoes"],
            format_func=lambda value: "Shoes" if value == "shoes" else "Else / Default",
            key="photo_pricing_laydown_silo_type",
            label_visibility="collapsed",
        )
        _field_label("Laydown Silo")
        laydown_silo_quantity = _quantity_input(
            "Laydown Silo",
            "photo_pricing_laydown_silo_quantity",
        )
        _rate_note(laydown_silo_rate(laydown_silo_type), "per image")

        _field_label("Color Corrections From Existing Images")
        color_corrections_quantity = _quantity_input(
            "Color Corrections From Existing Images",
            "photo_pricing_color_corrections_quantity",
        )
        _rate_note(COLOR_CORRECTIONS_RATE, "per image")

        _field_label("AI Generation Markup")
        ai_generation_quantity = _quantity_input(
            "AI Generation Markup",
            "photo_pricing_ai_generation_quantity",
        )
        _rate_note(AI_GENERATION_MARKUP_RATE, "per image")

    with production_col:
        st.markdown("#### Production Services")
        _field_label("Post Production Hourly Time")
        post_production_hours = _hours_input(
            "Post Production Hourly Time",
            "photo_pricing_post_production_hours",
        )
        _rate_note(POST_PRODUCTION_HOURLY_RATE, "per hour")

        _field_label("Model Hours Type")
        model_type = st.radio(
            "Model Hours Type",
            ["adult", "kid"],
            format_func=lambda value: "Adult" if value == "adult" else "Kid",
            horizontal=True,
            key="photo_pricing_model_type",
            label_visibility="collapsed",
        )
        _field_label("Model Hours")
        model_hours = _hours_input("Model Hours", "photo_pricing_model_hours")
        _rate_note(model_hourly_rate(model_type), "per hour")

        model_fitting_enabled = st.checkbox(
            "Model Fitting",
            key="photo_pricing_model_fitting_enabled",
        )
        st.caption(f"Locked flat fee: {_money(MODEL_FITTING_FLAT_FEE)}")

    return (
        ApparelInputs(
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
        ),
        production_col,
    )


def _render_summary(quote_payload: Any) -> None:
    st.subheader("Summary")
    st.metric("Image Count For Account Management", quote_payload.derived_total_image_count)
    st.write(f"Account Management Tier: **{account_management_tier_label(quote_payload.derived_total_image_count)}**")
    st.write(f"Account Management Fee: **{_money(quote_payload.derived_account_management_fee)}**")
    st.divider()
    subtotal_col, total_col = st.columns(2)
    subtotal_col.metric("Running Subtotal", _money(quote_payload.subtotal))
    total_col.metric("Final Total", _money(quote_payload.total))


def render_photography_pricing() -> None:
    st.title("Photography Pricing")

    if st.button("Back to Hub", key="photo_pricing_back_to_hub"):
        st.session_state["hub_view"] = "home"
        st.rerun()

    top_left, top_right = st.columns([1, 1.35], gap="large")
    with top_left:
        st.markdown("#### Job Type")
        job_type = st.selectbox(
            "Job Type",
            ["Apparel", "Misc"],
            key="photo_pricing_job_type",
            label_visibility="collapsed",
        )

    if job_type == "Misc":
        with top_right:
            st.info("Misc pricing is not available yet.")
        st.stop()

    inputs, summary_col = _render_apparel_inputs()
    quote = build_apparel_quote(inputs)
    quote_payload = quote.to_payload()

    with summary_col:
        _render_summary(quote)

    st.subheader("Pricing Rows")
    st.dataframe(
        pd.DataFrame(_line_table_rows(quote_payload)),
        hide_index=True,
        use_container_width=True,
    )
