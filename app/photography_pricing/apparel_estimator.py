from typing import Any

import pandas as pd
import streamlit as st

from app.photography_pricing.comments_builder import build_page1_comments_payload
from app.photography_pricing.mock_contacts import contact_options
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


def _init_comments_state() -> None:
    if "photo_pricing_project_rows" not in st.session_state:
        st.session_state["photo_pricing_project_rows"] = [{}]


def _project_number_input(label: str, key: str) -> float:
    return float(
        st.number_input(
            label,
            min_value=0.0,
            step=1.0,
            value=0.0,
            key=key,
            label_visibility="collapsed",
        )
    )


def _render_comments_composer() -> dict[str, Any]:
    _init_comments_state()
    contacts = contact_options()

    st.subheader("Page 1 Comments")
    contact_col, subject_col, subtitle_col = st.columns([1, 1.2, 1.2])
    with contact_col:
        _field_label("Internal Contact")
        selected_contact_id = st.selectbox(
            "Internal Contact",
            [contact["id"] for contact in contacts],
            format_func=lambda contact_id: next(
                contact["name"] for contact in contacts if contact["id"] == contact_id
            ),
            key="photo_pricing_comments_contact_id",
            label_visibility="collapsed",
        )
    with subject_col:
        _field_label("Estimate Subject")
        estimate_subject = st.text_input(
            "Estimate Subject",
            key="photo_pricing_comments_estimate_subject",
            placeholder="Sam's Club Kids Apparel Project",
            label_visibility="collapsed",
        )
    with subtitle_col:
        _field_label("Subtitle Line")
        subtitle_line = st.text_input(
            "Subtitle Line",
            key="photo_pricing_comments_subtitle_line",
            placeholder="Spring27 - Bangladesh",
            label_visibility="collapsed",
        )

    st.markdown("#### Project Entries")
    project_rows = st.session_state["photo_pricing_project_rows"]
    if st.button("Add Project", key="photo_pricing_comments_add_project"):
        project_rows.append({})
        st.rerun()

    rendered_projects: list[dict[str, Any]] = []
    for index, _row in enumerate(project_rows):
        row_cols = st.columns([1.5, 0.7, 0.8, 0.8, 0.7, 0.8, 0.45])
        with row_cols[0]:
            _field_label("Project Name")
            project_name = st.text_input(
                "Project Name",
                key=f"photo_pricing_comments_project_name_{index}",
                label_visibility="collapsed",
            )
        with row_cols[1]:
            _field_label("On Model")
            on_model = _project_number_input("On Model", f"photo_pricing_comments_on_model_{index}")
        with row_cols[2]:
            _field_label("Laydown/Detail")
            laydown_detail = _project_number_input(
                "Laydown/Detail",
                f"photo_pricing_comments_laydown_detail_{index}",
            )
        with row_cols[3]:
            _field_label("Color Correct")
            color_correct = _project_number_input(
                "Color Correct",
                f"photo_pricing_comments_color_correct_{index}",
            )
        with row_cols[4]:
            _field_label("Post")
            post = _project_number_input("Post", f"photo_pricing_comments_post_{index}")
        with row_cols[5]:
            _field_label("Model Hours")
            model_hours = _project_number_input("Model Hours", f"photo_pricing_comments_model_hours_{index}")
        with row_cols[6]:
            st.write("")
            st.write("")
            if len(project_rows) > 1 and st.button("Remove", key=f"photo_pricing_comments_remove_{index}"):
                project_rows.pop(index)
                st.rerun()

        rendered_projects.append(
            {
                "project_name": project_name,
                "on_model": on_model,
                "laydown_detail": laydown_detail,
                "color_correct": color_correct,
                "post": post,
                "model_hours": model_hours,
            }
        )

    _field_label("Custom Notes")
    custom_notes = st.text_area(
        "Custom Notes",
        key="photo_pricing_comments_custom_notes",
        height=90,
        label_visibility="collapsed",
    )

    selected_contact = next(contact for contact in contacts if contact["id"] == selected_contact_id)
    payload = build_page1_comments_payload(
        selected_internal_contact=selected_contact,
        estimate_subject=estimate_subject,
        subtitle_line=subtitle_line,
        project_entries=rendered_projects,
        custom_notes=custom_notes,
    )
    payload_dict = payload.to_payload()
    st.session_state["photo_pricing_page1_comments_payload"] = payload_dict

    with st.expander("Page 1 Comments Preview"):
        st.text(payload.rendered_comments_block)

    return payload_dict


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

    _render_comments_composer()

    if st.button("Generate PDF", key="photo_pricing_generate_pdf"):
        from app.photography_pricing.pdf_generator import generate_page2_pricing_pdf

        st.session_state["photo_pricing_generated_pdf"] = generate_page2_pricing_pdf(quote)

    generated_pdf = st.session_state.get("photo_pricing_generated_pdf")
    if generated_pdf:
        st.download_button(
            "Download Generated PDF",
            data=generated_pdf,
            file_name="photography_pricing_quote.pdf",
            mime="application/pdf",
            key="photo_pricing_download_pdf",
        )
