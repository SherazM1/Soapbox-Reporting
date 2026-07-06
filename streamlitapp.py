import os
import re
import sys
import unicodedata
import base64
from datetime import date
import io
from typing import Any

import pandas as pd
import streamlit as st

from app.audit_helpers.combined_audit_extract import (
    combined_pdp_to_dataframe,
    map_schema2_pdp_to_cached_record,
    parse_combined_audit_html,
    reset_combined_audit_state,
)
from app.audit_helpers.image_analysis import analyze_pdp_records
from audit_analyze import analyze_primary_record
from audit_export import build_audit_export_plan
from audit_generate import generate_mvp_outputs_for_primary_entry, is_output_shell_empty
from audit_models import create_audit_result_record
from audit_powerpoint_new import build_slide4_pdp_benchmark_payload, generate_new_audit_powerpoint_from_template
from audit_style_guides import load_style_guides, match_style_guide_rule
from audit_helpers import (
    build_competitor_assignments,
    create_product_audit_entry_from_record,
    initialize_auditing_session_state,
    parse_audit_extract_upload_to_dataframe,
    process_competitor_audit_extract_sheet,
    process_competitor_pdp_urls_real,
    process_primary_audit_extract_sheet,
    process_primary_pdp_urls_real,
    update_record_tier1_derived_fields,
    urls_from_uploaded_dataframe,
)

from db import (
    init_db,
    add_client,
    get_clients,
    delete_client,
    add_preview,
    get_previews_for_client,
    delete_preview,
)

from main import (
    load_batches as load_groups,
    save_batches as save_groups,
    load_dataframe,
    compute_metrics,
    get_top_skus,
    get_skus_below,
    make_pie_bytes,
    generate_full_report,
    load_search_insights,
    load_inventory,
    load_item_sales,
    load_managed_keys,
    filter_by_managed,
)

try:
    from main import generate_3p_report
except Exception:
    from main import generate_full_report_3p as generate_3p_report


def fmt_mdy(d: date) -> str:
    return d.strftime("%#m/%#d/%Y") if sys.platform.startswith("win") else d.strftime("%-m/%-d/%Y")


VIEW_HOME = "home"
VIEW_CONTENT_REPORTING = "content_reporting"
VIEW_CONTENT_AUDITING = "content_auditing"

HUB_QUERY_TO_VIEW = {
    "reporting": VIEW_CONTENT_REPORTING,
    "auditing": VIEW_CONTENT_AUDITING,
    VIEW_CONTENT_REPORTING: VIEW_CONTENT_REPORTING,
    VIEW_CONTENT_AUDITING: VIEW_CONTENT_AUDITING,
}

VIEW_TO_HUB_QUERY = {
    VIEW_CONTENT_REPORTING: VIEW_CONTENT_REPORTING,
    VIEW_CONTENT_AUDITING: VIEW_CONTENT_AUDITING,
}

STYLE_GUIDE_AUTO_LABEL = "Auto / Detected"
STYLE_GUIDE_NONE_LABEL = "No Style Guide"


def set_hub_view(view_name: str) -> None:
    st.session_state["hub_view"] = view_name
    st.rerun()


def go_home() -> None:
    set_hub_view(VIEW_HOME)


def go_reporting() -> None:
    set_hub_view(VIEW_CONTENT_REPORTING)


def go_auditing() -> None:
    set_hub_view(VIEW_CONTENT_AUDITING)


def render_branding() -> None:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=180)


def render_home() -> None:
    st.markdown(
        """
        <style>
        .hub-subtitle {
            color: #5f6b7b;
            margin-top: -0.42rem;
            margin-bottom: 0.95rem;
            font-size: 0.96rem;
        }
        .hub-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.2rem;
        }
        .hub-link-card {
            display: block;
            text-decoration: none;
            border: 1px solid #c7d3e4;
            border-radius: 12px;
            padding: 0.85rem 0.9rem 0.8rem 0.9rem;
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            min-height: 128px;
            transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease, background 140ms ease;
            position: relative;
        }
        .hub-link-card:link,
        .hub-link-card:visited,
        .hub-link-card:hover,
        .hub-link-card:active,
        .hub-link-card:focus,
        .hub-link-card:focus-visible {
            text-decoration: none !important;
        }
        .hub-link-card:hover {
            border-color: #8ea7c7;
            box-shadow: 0 8px 22px rgba(18, 32, 58, 0.09);
            transform: translateY(-1px);
            background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%);
        }
        .hub-link-card:focus,
        .hub-link-card:focus-visible {
            outline: none;
            border-color: #6f8db5;
            box-shadow: 0 0 0 3px rgba(66, 98, 142, 0.2);
        }
        .hub-card-arrow {
            position: absolute;
            top: 0.62rem;
            right: 0.74rem;
            color: #2e4f7d;
            font-size: 0.84rem;
            line-height: 1;
        }
        .hub-link-card h3 {
            margin: 0.05rem 0 0.34rem 0;
            color: #17345d;
            font-size: 1.16rem;
            font-weight: 650;
            letter-spacing: 0.01em;
            text-decoration: none;
        }
        .hub-link-card p {
            margin: 0;
            color: #667489;
            font-size: 0.89rem;
            line-height: 1.36;
            text-decoration: none;
        }
        .hub-card-footer {
            margin-top: 0.7rem;
        }
        .hub-open-btn {
            display: inline-block;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            border: 1px solid #224a7e;
            color: #17345d;
            background: #edf3fc;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            text-decoration: none;
        }
        .hub-link-card:hover .hub-open-btn {
            background: #e4eefb;
            border-color: #1b406f;
            color: #122d50;
        }
        .hub-link-card,
        .hub-link-card * {
            text-decoration: none !important;
        }
        @media (max-width: 900px) {
            .hub-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Soapbox eCommerce and Content Hub")
    st.markdown('<div class="hub-subtitle">Choose a workflow to continue.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hub-grid">
          <a class="hub-link-card" href="?hub=content_reporting">
            <span class="hub-card-arrow">↗</span>
            <h3>Content Reporting</h3>
            <p>Run weekly 1P and 3P reporting, exports, and saved work.</p>
            <div class="hub-card-footer"><span class="hub-open-btn">Open</span></div>
          </a>
          <a class="hub-link-card" href="?hub=content_auditing">
            <span class="hub-card-arrow">↗</span>
            <h3>Content Auditing</h3>
            <p>Open the audit workspace for intake, findings, and recommendations.</p>
            <div class="hub-card-footer"><span class="hub-open-btn">Open</span></div>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_audit_state() -> None:
    initialize_auditing_session_state(st.session_state)
    # Legacy keys retained for compatibility with earlier scaffold helpers.
    if "audit_extracted_loaded" not in st.session_state:
        st.session_state["audit_extracted_loaded"] = False
    if "audit_source_method" not in st.session_state:
        st.session_state["audit_source_method"] = "Single PDP URL"
    if st.session_state.get("audit_primary_source_method") not in {
        "Audit Extract Sheet (Recommended)",
        "Fallback URL Mode",
    }:
        st.session_state["audit_primary_source_method"] = "Audit Extract Sheet (Recommended)"
    if st.session_state.get("audit_competitor_source_method") not in {
        "Audit Extract Sheet (Recommended)",
        "Fallback URL Mode",
    }:
        st.session_state["audit_competitor_source_method"] = "Audit Extract Sheet (Recommended)"
    if "audit_v2_primary_fallback_method" not in st.session_state:
        st.session_state["audit_v2_primary_fallback_method"] = "Single PDP URL"
    if "audit_v2_competitor_fallback_method" not in st.session_state:
        st.session_state["audit_v2_competitor_fallback_method"] = "Single PDP URL"
    if "audit_competitor_has_multiple_pdps" not in st.session_state:
        st.session_state["audit_competitor_has_multiple_pdps"] = False
    if "audit_competitor_pdp_group_count" not in st.session_state:
        st.session_state["audit_competitor_pdp_group_count"] = 0
    if "audit_competitor_make_multiple_slides" not in st.session_state:
        st.session_state["audit_competitor_make_multiple_slides"] = False
    if "audit_competitor_slide_mode" not in st.session_state:
        st.session_state["audit_competitor_slide_mode"] = "single_pdp"
    if "audit_competitor_slide_mode_selector" not in st.session_state:
        st.session_state["audit_competitor_slide_mode_selector"] = "Generate One Combined PDP Slide"


def _set_extracted_fields(data: dict) -> None:
    st.session_state["audit_product_title"] = data.get("product_title", "")
    st.session_state["audit_item_id"] = data.get("item_id", "")
    st.session_state["audit_brand"] = data.get("brand", "")
    st.session_state["audit_category"] = data.get("category", "")
    st.session_state["audit_subcategory"] = data.get("subcategory", "")
    st.session_state["audit_current_title"] = data.get("current_title", "")
    st.session_state["audit_current_description"] = data.get("current_description", "")
    st.session_state["audit_current_key_features"] = data.get("current_key_features", "")
    st.session_state["audit_current_image_count"] = data.get("current_image_count", 0)
    st.session_state["audit_current_specs"] = data.get("current_specs", "")
    st.session_state["audit_selected_style_guide"] = data.get("selected_style_guide", "Soapbox Standard")


def _mock_extracted_from_url(pdp_url: str) -> dict:
    clean_url = (pdp_url or "").strip()
    token = clean_url.rstrip("/").split("/")[-1][:20] if clean_url else "sample-item"
    title = "Country Fresh Greek Yogurt Cups, Strawberry Variety Pack, 12 Count"
    return {
        "product_title": title,
        "item_id": f"ITM-{abs(hash(token)) % 900000 + 100000}",
        "brand": "Country Fresh",
        "category": "Dairy",
        "subcategory": "Yogurt",
        "current_title": title,
        "current_description": "Strawberry Greek yogurt cups with creamy texture and convenient single-serve packaging for breakfast or snacks.",
        "current_key_features": "- 12 single-serve cups\n- Greek yogurt texture\n- Strawberry flavor variety\n- Convenient grab-and-go pack",
        "current_image_count": 4,
        "current_specs": "- Net weight: 12 x 5.3 oz\n- Refrigerated item\n- Contains milk",
        "selected_style_guide": "Soapbox Standard",
    }


def _build_mock_batch_preview(urls: list[str]) -> pd.DataFrame:
    rows = []
    for i, u in enumerate(urls):
        rows.append(
            {
                "PDP URL": u,
                "Detected Product Title": f"Detected Product {i + 1}",
                "Detected Item ID": f"ITM-{200100 + i}",
                "Status": "Extracted",
                "Include": i == 0,
            }
        )
    return pd.DataFrame(rows)


def _read_uploaded_table(uploaded) -> pd.DataFrame:
    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def render_audit_setup() -> None:
    with st.container(border=True):
        st.markdown("### Audit Setup")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Client Name", key="audit_client_name")
        with c2:
            st.selectbox(
                "Retailer",
                ["Walmart", "Amazon", "Target", "Instacart", "Other"],
                key="audit_retailer",
            )
        with c3:
            st.date_input("Audit Date", value=date.today(), key="audit_date")


def render_product_source() -> None:
    with st.container(border=True):
        st.markdown("### Product Source")
        method = st.radio(
            "Input Method",
            ["Single PDP URL", "Excel / CSV Upload"],
            horizontal=True,
            key="audit_source_method",
        )

        if method == "Single PDP URL":
            pdp_url = st.text_input("PDP URL", key="audit_single_pdp_url")
            if st.button("Load PDP Data", key="audit_load_single"):
                if not pdp_url.strip():
                    st.warning("Enter a PDP URL to continue.")
                else:
                    st.session_state["audit_pdp_url"] = pdp_url.strip()
                    _set_extracted_fields(_mock_extracted_from_url(pdp_url))
                    st.session_state["audit_extracted_loaded"] = True
                    st.session_state["audit_generated"] = False
                    st.success("Mock PDP extraction complete. Review and edit extracted fields below.")
        else:
            uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"], key="audit_batch_upload")
            df_uploaded = pd.DataFrame()
            if uploaded is not None:
                try:
                    df_uploaded = _read_uploaded_table(uploaded)
                except Exception as e:
                    st.error(f"Could not read upload: {e}")

            if not df_uploaded.empty:
                url_columns = [c for c in df_uploaded.columns if "url" in str(c).lower()]
                default_col = url_columns[0] if url_columns else df_uploaded.columns[0]
                st.selectbox(
                    "PDP URL Column",
                    list(df_uploaded.columns),
                    index=list(df_uploaded.columns).index(default_col),
                    key="audit_batch_url_col",
                )
            else:
                st.selectbox(
                    "PDP URL Column",
                    ["PDP URL"],
                    key="audit_batch_url_col",
                )

            if st.button("Process URLs", key="audit_process_batch"):
                urls = []
                if not df_uploaded.empty and st.session_state.get("audit_batch_url_col") in df_uploaded.columns:
                    col = st.session_state["audit_batch_url_col"]
                    urls = [str(v).strip() for v in df_uploaded[col].dropna().tolist() if str(v).strip()]
                if not urls:
                    urls = [
                        "https://www.example.com/pdp/item-1",
                        "https://www.example.com/pdp/item-2",
                        "https://www.example.com/pdp/item-3",
                    ]
                preview = _build_mock_batch_preview(urls[:15])
                st.session_state["audit_batch_preview"] = preview
                st.session_state["audit_extracted_loaded"] = True
                st.session_state["audit_generated"] = False
                first_url = preview.iloc[0]["PDP URL"]
                st.session_state["audit_pdp_url"] = first_url
                _set_extracted_fields(_mock_extracted_from_url(first_url))
                st.success("Mock batch processing complete. Review batch preview and extracted product data.")

            preview_df = st.session_state.get("audit_batch_preview", pd.DataFrame())
            if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
                st.caption("Batch Preview")
                edited = st.data_editor(
                    preview_df,
                    use_container_width=True,
                    hide_index=True,
                    disabled=["PDP URL", "Detected Product Title", "Detected Item ID", "Status"],
                    key="audit_batch_preview_editor",
                )
                st.session_state["audit_batch_preview"] = edited
                if st.button("Load Included Product", key="audit_load_included"):
                    selected = edited[edited["Include"] == True]
                    if selected.empty:
                        st.warning("Select at least one row in Include.")
                    else:
                        row = selected.iloc[0]
                        st.session_state["audit_pdp_url"] = row["PDP URL"]
                        _set_extracted_fields(_mock_extracted_from_url(str(row["PDP URL"])))
                        st.session_state["audit_extracted_loaded"] = True
                        st.session_state["audit_generated"] = False
                        st.success("Loaded first included product into extracted review fields.")


def render_extracted_product_review() -> None:
    st.markdown("### Extracted Product Data")
    if not st.session_state.get("audit_extracted_loaded"):
        st.info("Load PDP data or process uploaded URLs to review extracted product fields.")
        return

    with st.container(border=True):
        r1, r2, r3 = st.columns(3)
        with r1:
            st.text_input("Product Title", key="audit_product_title")
        with r2:
            st.text_input("Item ID", key="audit_item_id")
        with r3:
            st.text_input("Brand", key="audit_brand")

        r4, r5 = st.columns(2)
        with r4:
            st.text_input("Category", key="audit_category")
        with r5:
            st.text_input("Subcategory", key="audit_subcategory")

        st.text_area("Current Title", key="audit_current_title", height=85)
        st.text_area("Current Description", key="audit_current_description", height=130)
        st.text_area("Current Key Features", key="audit_current_key_features", height=120)

        r6, r7 = st.columns([1, 2])
        with r6:
            st.number_input("Current Image Count", min_value=0, step=1, key="audit_current_image_count")
            st.file_uploader(
                "PDP Hero Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="audit_review_pdp_hero_image",
            )
        with r7:
            st.text_area("Current Specs / Attributes", key="audit_current_specs", height=95)
            st.selectbox(
                "Selected Style Guide",
                ["Soapbox Standard", "Retailer Standard", "Brand Voice v1", "Custom"],
                key="audit_selected_style_guide",
            )


def _build_mock_audit_payload() -> dict:
    base_title = st.session_state.get("audit_current_title", "").strip() or st.session_state.get("audit_product_title", "").strip() or "Brand Product Name, Key Benefit, Size"
    return {
        "image_recommendations": (
            "- Add finished product hero image\n"
            "- Add step-by-step usage visual\n"
            "- Add lifestyle image showing product in use\n"
            "- Add \"what's included\" image\n"
            "- Add feature/benefit infographic"
        ),
        "recommended_title": f"{base_title} | Clear Use Case and Key Benefit",
        "description_recommendations": (
            "- Opening copy does not front-load the primary shopper intent; add a direct use-case lead.\n"
            "- Description lacks concrete product proof points; include measurable attributes and compatibility.\n"
            "- Current flow is paragraph-dense; improve scanability with shorter, structured blocks.\n"
            "- Value statement is generic; align messaging to audience and purchase triggers."
        ),
        "key_features_recommendations": (
            "- Feature bullets overlap in meaning; map each bullet to a distinct shopper benefit.\n"
            "- Search-intent terms are underused in top bullets; add category and use-case keywords.\n"
            "- Several bullets are non-specific; add dimensions, performance claims, or constraints.\n"
            "- Bullet sequence does not prioritize decision blockers; surface trust and fit details earlier."
        ),
        "top_priority_fixes": (
            "- Upgrade image stack with lifestyle and how-to visuals.\n"
            "- Rewrite title for stronger SEO and conversion.\n"
            "- Add clearer use cases and audience language.\n"
            "- Improve feature hierarchy and reduce repetition."
        ),
    }


def _seed_audit_output_state(payload: dict) -> None:
    st.session_state["audit_image_recommendations"] = payload["image_recommendations"]
    st.session_state["audit_recommended_title"] = payload["recommended_title"]
    st.session_state["audit_description_recommendations"] = payload["description_recommendations"]
    st.session_state["audit_key_features_recommendations"] = payload["key_features_recommendations"]
    st.session_state["audit_top_priority_fixes"] = payload["top_priority_fixes"]


def render_audit_results() -> None:
    st.divider()
    st.subheader("Product Summary")
    with st.container(border=True):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.caption("Product Title")
            st.write(st.session_state.get("audit_product_title") or "-")
        with s2:
            st.caption("Item ID")
            st.write(st.session_state.get("audit_item_id") or "-")
        with s3:
            st.caption("Retailer")
            st.write(st.session_state.get("audit_retailer") or "-")
        with s4:
            st.caption("Audit Date")
            st.write(fmt_mdy(st.session_state.get("audit_date", date.today())))

        hero = st.session_state.get("audit_review_pdp_hero_image")
        if hero is not None:
            st.caption("Hero Image Preview")
            st.image(hero, width=220)
        else:
            st.caption("Hero Image Preview: Not uploaded")

    st.subheader("Image Recommendations")
    st.text_area("Image Recommendations", key="audit_image_recommendations", height=140)

    st.subheader("Content Optimizations")
    st.text_input("Recommended Title", key="audit_recommended_title")
    st.text_area("Description Recommendations", key="audit_description_recommendations", height=140)
    st.text_area("Key Features Recommendations", key="audit_key_features_recommendations", height=140)

    st.subheader("Top Priority Fixes")
    st.text_area("Top Priority Fixes", key="audit_top_priority_fixes", height=120)

def render_content_auditing_legacy_prompt1() -> None:
    _init_audit_state()

    top_l, top_r = st.columns([1, 6])
    with top_l:
        st.button("Back to Hub", key="audit_back_home", on_click=go_home)
    with top_r:
        st.title("Content Auditing")
        st.caption("Workspace for building first-pass PDP audits and recommendations.")

    render_audit_setup()
    st.divider()
    render_product_source()
    st.divider()
    render_extracted_product_review()
    st.divider()

    with st.container(border=True):
        st.markdown("### Generate Audit")
        st.caption("Generate a mock first-pass audit draft from extracted product data.")
        generate = st.button(
            "Generate Audit",
            key="audit_generate",
            type="primary",
            disabled=not st.session_state.get("audit_extracted_loaded"),
        )
        if not st.session_state.get("audit_extracted_loaded"):
            st.info("Load PDP data or process uploaded URLs before generating an audit.")

    if generate:
        _seed_audit_output_state(_build_mock_audit_payload())
        st.session_state["audit_generated"] = True

    if st.session_state.get("audit_generated"):
        render_audit_results()


def _reset_generated_audit_state_v2() -> None:
    st.session_state["audit_generated"] = False
    st.session_state["audit_results_seeded_for"] = []
    st.session_state["audit_export_plan"] = {}


def _reset_competitor_graphics_mode_state_v2() -> None:
    st.session_state["audit_competitor_has_multiple_pdps"] = False
    st.session_state["audit_competitor_pdp_group_count"] = 0
    st.session_state["audit_competitor_make_multiple_slides"] = False
    st.session_state["audit_competitor_slide_mode"] = "single_pdp"
    st.session_state["audit_competitor_slide_mode_selector"] = "Generate One Combined PDP Slide"
    st.session_state["audit_competitor_combined_image_orders"] = {}
    st.session_state["audit_competitor_multi_image_orders_by_record"] = {}
    st.session_state["audit_competitor_selected_image_ids_by_record"] = {}
    st.session_state["audit_competitor_group_summary"] = []
    st.session_state["audit_v2_comp_selection_signature_combined"] = ()
    st.session_state["audit_v2_comp_selection_signature_multi"] = ()


def _extract_urls_from_df_v2(df_uploaded: pd.DataFrame, selected_col: str) -> list[str]:
    return urls_from_uploaded_dataframe(df_uploaded, selected_col)


def _analyze_sheet_records_with_ui(records: list[dict[str, Any]]) -> dict[str, Any]:
    progress = st.progress(0.0)
    status = st.empty()

    def update_progress(
        pdp_index: int,
        total_pdps: int,
        image_index: int,
        total_images: int,
    ) -> None:
        status.info(
            f"Analyzing PDP {pdp_index} of {total_pdps} — "
            f"image {image_index} of {total_images}"
        )
        completed_before = pdp_index - 1
        pdp_fraction = (image_index / max(total_images, 1)) if total_images else 1.0
        progress.progress(min(1.0, (completed_before + pdp_fraction) / max(total_pdps, 1)))

    summary = analyze_pdp_records(records, progress_callback=update_progress)
    progress.progress(1.0)
    status.success(
        f"{summary['pdp_count']} PDPs processed — "
        f"{summary['analyzed_image_count']} images analyzed, "
        f"{summary['failed_image_count']} could not be analyzed"
    )
    for warning in summary.get("warnings", []):
        st.warning(warning)
    return summary


def _combined_message_text(message: Any) -> str:
    if isinstance(message, dict):
        prefix = " | ".join(
            part
            for part in (
                str(message.get("source", "") or "").strip(),
                f"Row {message.get('row')}" if message.get("row") not in (None, "") else "",
            )
            if part
        )
        body = str(message.get("message", "") or message).strip()
        return f"{prefix}: {body}" if prefix else body
    return str(message or "").strip()


def _process_combined_pdp_records_v2(
    records: list[dict[str, Any]],
    *,
    role: str,
    schema_version: str = "2.0",
    client_name: str | None = None,
    retailer: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    processed: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    messages: list[str] = []
    active_client_name = (
        st.session_state.get("audit_client_name", "")
        if client_name is None
        else client_name
    )
    active_retailer = (
        st.session_state.get("audit_retailer", "")
        if retailer is None
        else retailer
    )
    for source_index, source_record in enumerate(records):
        source_row = source_record.get("sourceRow", source_index + 1)
        is_schema2 = schema_version == "2.0" or isinstance(source_record.get("data"), dict)
        if is_schema2:
            try:
                cached_record, row_messages = map_schema2_pdp_to_cached_record(
                    source_record,
                    role=role,
                    client_name=active_client_name,
                    retailer=active_retailer,
                )
            except Exception as exc:
                cached_record = None
                row_messages = [
                    f"Row {source_row}: failed to map schema 2.0 PDP record ({exc})."
                ]
            messages.extend(row_messages)
            if cached_record is None:
                continue
            records_by_id[cached_record["record_id"]] = cached_record
            if role == "Client":
                entry = create_product_audit_entry_from_record(cached_record)
                entry["entry_id"] = (
                    f"primary-combined-{source_index + 1}-"
                    f"{entry.get('record_id', 'unknown')}"
                )
                processed.append(entry)
            else:
                processed.append(cached_record)
            continue

        frame = combined_pdp_to_dataframe(source_record)
        if role == "Client":
            entries, mapped_records, row_messages = process_primary_audit_extract_sheet(
                df_uploaded=frame,
                client_name=active_client_name,
                retailer=active_retailer,
            )
            processed.extend(entries)
        else:
            entries, mapped_records, row_messages = process_competitor_audit_extract_sheet(
                df_uploaded=frame,
                client_name=active_client_name,
                retailer=active_retailer,
            )
            processed.extend(entries)
        records_by_id.update(mapped_records)
        messages.extend(
            f"{role} PDP source row {source_row}: {message}"
            for message in row_messages
        )
    return processed, records_by_id, messages


def _evidence_value(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        current: Any = record
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current not in (None, "", [], {}):
            return current
    return default


def _render_combined_evidence_preview_v2(result: dict[str, Any]) -> None:
    if not result:
        return
    with st.expander("Combined Audit Evidence Preview", expanded=False):
        pdp_rows = []
        for role, records in (
            ("Client", result.get("client_pdps", [])),
            ("Competitor", result.get("competitor_pdps", [])),
        ):
            for record in records:
                images = _evidence_value(record, "images", "imageUrls", default=[]) or []
                if not isinstance(images, list):
                    images = [images]
                pdp_rows.append(
                    {
                        "Role": role,
                        "Brand": _evidence_value(record, "inputBrandName", "brandName", "brand"),
                        "Product ID": _evidence_value(record, "productId", "itemId"),
                        "Product Title": _evidence_value(record, "productTitle", "title", "name"),
                        "Category": _evidence_value(record, "resolvedCategory", "categoryPathName", "category"),
                        "Product Type": _evidence_value(record, "resolvedProductType", "productType"),
                        "Image Count": _evidence_value(record, "imageCount", default=len(images)),
                        "Seller": _evidence_value(record, "seller", "sellerName"),
                        "Sold by Walmart": _evidence_value(record, "soldByWalmart", "fulfillment.soldByWalmart"),
                        "Shipped by Walmart": _evidence_value(record, "shippedByWalmart", "fulfillment.shippedByWalmart"),
                        "Enhanced Brand Content": _evidence_value(
                            record,
                            "enhancedBrandContentStatus",
                            "enhancedBrandContentPresent",
                            "enhancedContent.status",
                        ),
                    }
                )
        st.markdown("##### PDP Summary")
        if pdp_rows:
            st.dataframe(pd.DataFrame(pdp_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No valid PDP evidence.")

        search_rows = []
        for role, records in (
            ("Current", result.get("current_searches", [])),
            ("Benchmark", result.get("benchmark_searches", [])),
        ):
            for record in records:
                products = _evidence_value(record, "products", "capturedProducts", default=[]) or []
                search_rows.append(
                    {
                        "Role": role,
                        "Search Term": _evidence_value(record, "searchTerm", "query", "term"),
                        "Result Count": _evidence_value(record, "resultCount", "totalResults"),
                        "Products Captured": _evidence_value(
                            record,
                            "productsCaptured",
                            default=len(products) if isinstance(products, list) else 0,
                        ),
                        "Sponsored Detected": _evidence_value(
                            record,
                            "sponsoredProductsDetected",
                            "sponsoredDetected",
                        ),
                        "Screenshot Available": bool(
                            _evidence_value(record, "screenshotDataUrl", "screenshot.dataUrl")
                        ),
                        "Extraction Status": _evidence_value(record, "status", "extractionStatus"),
                    }
                )
        st.markdown("##### Search Summary")
        if search_rows:
            st.dataframe(pd.DataFrame(search_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No Search evidence was included.")

        shop_rows = []
        for role, records in (
            ("Client", result.get("client_brand_shops", [])),
            ("Competitor", result.get("competitor_brand_shops", [])),
        ):
            for record in records:
                modules = _evidence_value(record, "modules", "structuredModules", default=[]) or []
                module_types = [
                    str(_evidence_value(module, "type", "moduleType") or "")
                    for module in modules
                    if isinstance(module, dict)
                ]
                explicit_module_types = _evidence_value(record, "moduleTypes", default=[]) or []
                if isinstance(explicit_module_types, list) and explicit_module_types:
                    module_types = [str(value) for value in explicit_module_types]
                products = _evidence_value(record, "products", "productTiles", default=[]) or []
                navigation = _evidence_value(
                    record,
                    "categoryNavigation",
                    "categoryNavigationItems",
                    default=[],
                ) or []
                shop_rows.append(
                    {
                        "Role": role,
                        "Brand Name": _evidence_value(record, "inputBrandName", "brandName", "brand"),
                        "Module Count": _evidence_value(
                            record,
                            "moduleCount",
                            default=len(modules) if isinstance(modules, list) else 0,
                        ),
                        "Module Types": ", ".join(value for value in module_types if value),
                        "Category Navigation Count": len(navigation) if isinstance(navigation, list) else 0,
                        "Video Present": _evidence_value(record, "videoPresent", "hasVideo"),
                        "Product Count": _evidence_value(
                            record,
                            "productCount",
                            default=len(products) if isinstance(products, list) else 0,
                        ),
                        "Screenshot Available": bool(
                            _evidence_value(record, "screenshotDataUrl", "screenshot.dataUrl")
                        ),
                        "Extraction Status": _evidence_value(record, "status", "extractionStatus"),
                    }
                )
        st.markdown("##### Brand Shop Summary")
        if shop_rows:
            st.dataframe(pd.DataFrame(shop_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No Brand Shop evidence was included.")


def render_combined_strategic_audit_upload_v2() -> None:
    with st.container(border=True):
        st.markdown("### Combined Strategic Audit Extract")
        st.caption("Upload the HTML report generated by the Soapbox Audit Extractor.")
        uploaded = st.file_uploader(
            "Upload Audit Extraction Sheet",
            type=["html", "htm"],
            accept_multiple_files=False,
            key="audit_combined_extract_upload",
        )
        if st.button(
            "Process Audit Extraction Sheet",
            key="audit_process_combined_extract",
            type="primary",
        ):
            result = parse_combined_audit_html(uploaded)
            if result.get("is_legacy"):
                for error in result.get("errors", []):
                    st.error(_combined_message_text(error))
                st.info(
                    "This workflow supports only the combined strategic audit HTML report."
                )
            elif result.get("schema_version") != "2.0":
                for error in result.get("errors", []):
                    st.error(_combined_message_text(error))
            else:
                reset_combined_audit_state(st.session_state)
                primary_entries, primary_map, primary_messages = _process_combined_pdp_records_v2(
                    result.get("client_pdps", []) or [],
                    role="Client",
                    schema_version=result.get("schema_version", ""),
                )
                competitor_entries, competitor_map, competitor_messages = _process_combined_pdp_records_v2(
                    result.get("competitor_pdps", []) or [],
                    role="Competitor",
                    schema_version=result.get("schema_version", ""),
                )
                all_records = [
                    *(entry.get("cached_record", {}) for entry in primary_entries),
                    *competitor_entries,
                ]
                if all_records:
                    _analyze_sheet_records_with_ui(all_records)
                st.session_state["audit_primary_entries"] = primary_entries
                st.session_state["audit_competitor_entries"] = competitor_entries
                st.session_state["audit_cached_pdp_records"] = {
                    **primary_map,
                    **competitor_map,
                }
                current_searches = list(result.get("current_searches", []) or [])
                benchmark_searches = list(result.get("benchmark_searches", []) or [])
                client_shops = list(result.get("client_brand_shops", []) or [])
                competitor_shops = list(result.get("competitor_brand_shops", []) or [])
                st.session_state["audit_search_evidence"] = {
                    "current": current_searches,
                    "benchmark": benchmark_searches,
                    "all": [*current_searches, *benchmark_searches],
                }
                st.session_state["audit_brand_shop_evidence"] = {
                    "client": client_shops,
                    "competitor": competitor_shops,
                    "all": [*client_shops, *competitor_shops],
                }
                result["ingestion_messages"] = [*primary_messages, *competitor_messages]
                st.session_state["audit_combined_extract_result"] = result
                _reset_competitor_graphics_mode_state_v2()
                _reset_generated_audit_state_v2()

                if not primary_entries:
                    st.error("No valid Client PDPs were processed. Client PDPs are required.")
                if not competitor_entries:
                    st.error("No valid Competitor PDPs were processed. Competitor PDPs are required.")
                if primary_entries and competitor_entries:
                    st.success("Combined strategic audit ingestion complete.")

        result = st.session_state.get("audit_combined_extract_result", {}) or {}
        if result:
            metric_values = (
                ("Client PDPs", len(st.session_state.get("audit_primary_entries", []) or [])),
                ("Competitor PDPs", len(st.session_state.get("audit_competitor_entries", []) or [])),
                ("Current searches", len(result.get("current_searches", []) or [])),
                ("Benchmark searches", len(result.get("benchmark_searches", []) or [])),
                ("Client Brand Shops", len(result.get("client_brand_shops", []) or [])),
                ("Competitor Brand Shops", len(result.get("competitor_brand_shops", []) or [])),
                ("Warnings", len(result.get("warnings", []) or [])),
                ("Errors", len(result.get("errors", []) or [])),
            )
            for row_start in range(0, len(metric_values), 4):
                columns = st.columns(4)
                for column, (label, value) in zip(columns, metric_values[row_start : row_start + 4]):
                    column.metric(label, value)
            for warning in result.get("warnings", []) or []:
                st.warning(_combined_message_text(warning))
            for error in result.get("errors", []) or []:
                st.error(_combined_message_text(error))
            for message in result.get("ingestion_messages", []) or []:
                st.warning(message)
            _render_combined_evidence_preview_v2(result)


def _render_local_image_analysis(record: dict[str, Any]) -> None:
    analysis = record.get("image_analysis", {}) or {}
    if not analysis:
        return
    with st.expander("Local Image Analysis", expanded=False):
        st.caption(
            f"Status: {analysis.get('status', 'unknown')} | "
            f"Guide page: {analysis.get('guide_page_key') or 'No match'} | "
            f"Analyzed: {analysis.get('analyzed_image_count', 0)} | "
            f"Failed: {analysis.get('failed_image_count', 0)}"
        )
        rows = []
        for image in analysis.get("images", []) or []:
            rows.append(
                {
                    "Position": image.get("position"),
                    "Expected slot": image.get("expected_slot", ""),
                    "Probable format": image.get("probable_format", ""),
                    "OCR word count": image.get("ocr_word_count", 0),
                    "Detected signals": ", ".join(image.get("detected_signals", []) or []),
                    "Confidence": image.get("confidence", 0.0),
                    "Error status": "; ".join(image.get("errors", []) or []),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No PDP images were available for local analysis.")


def render_strategic_evidence_summary_v2() -> None:
    primary_entries = st.session_state.get("audit_primary_entries", []) or []
    competitor_entries = st.session_state.get("audit_competitor_entries", []) or []
    search_evidence = st.session_state.get("audit_search_evidence", {}) or {}
    brand_shop_evidence = st.session_state.get("audit_brand_shop_evidence", {}) or {}

    with st.expander("Strategic Evidence Summary", expanded=False):
        metric_values = (
            ("Client PDPs", len(primary_entries)),
            ("Competitor PDPs", len(competitor_entries)),
            ("Current searches", len(search_evidence.get("current", []) or [])),
            ("Benchmark searches", len(search_evidence.get("benchmark", []) or [])),
            ("Client Brand Shops", len(brand_shop_evidence.get("client", []) or [])),
            ("Competitor Brand Shops", len(brand_shop_evidence.get("competitor", []) or [])),
        )
        for row_start in range(0, len(metric_values), 3):
            columns = st.columns(3)
            for column, (label, value) in zip(columns, metric_values[row_start : row_start + 3]):
                column.metric(label, value)

        pdp_rows = []
        for role, entries in (("Client", primary_entries), ("Competitor", competitor_entries)):
            for entry in entries:
                record = entry.get("cached_record", entry) or {}
                reviews = record.get("reviews_summary", {}) or {}
                pdp_rows.append(
                    {
                        "Role": role,
                        "Product Title": entry.get("product_title") or record.get("product_title", ""),
                        "Brand": record.get("brand", ""),
                        "Item ID": entry.get("item_id") or record.get("item_id", ""),
                        "Image Count": record.get("image_count", 0),
                        "Average Rating": reviews.get("average_rating"),
                        "Review Count": reviews.get("review_count") or reviews.get("ratings_count"),
                        "Extraction Status": record.get("extraction_status", ""),
                    }
                )
        if pdp_rows:
            st.markdown("##### PDP Evidence")
            st.dataframe(pd.DataFrame(pdp_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No PDP evidence has been loaded yet.")

        search_rows = []
        for group_key, label in (("current", "Current"), ("benchmark", "Benchmark")):
            for record in search_evidence.get(group_key, []) or []:
                results = _evidence_value(record, "results", "organicResults", default=[]) or []
                search_rows.append(
                    {
                        "Group": label,
                        "Source Row": _evidence_value(record, "sourceRow", "source_row"),
                        "Query": _evidence_value(record, "query", "searchQuery"),
                        "Results": len(results) if isinstance(results, list) else 0,
                        "Screenshot Available": bool(
                            _evidence_value(record, "screenshotDataUrl", "screenshot.dataUrl")
                        ),
                    }
                )
        if search_rows:
            st.markdown("##### Search Evidence")
            st.dataframe(pd.DataFrame(search_rows), hide_index=True, use_container_width=True)

        brand_shop_rows = []
        for group_key, label in (("client", "Client"), ("competitor", "Competitor")):
            for record in brand_shop_evidence.get(group_key, []) or []:
                modules = _evidence_value(record, "modules", "structuredModules", default=[]) or []
                brand_shop_rows.append(
                    {
                        "Role": label,
                        "Source Row": _evidence_value(record, "sourceRow", "source_row"),
                        "Brand Name": _evidence_value(record, "inputBrandName", "brandName", "brand"),
                        "Module Count": _evidence_value(
                            record,
                            "moduleCount",
                            default=len(modules) if isinstance(modules, list) else 0,
                        ),
                        "Screenshot Available": bool(
                            _evidence_value(record, "screenshotDataUrl", "screenshot.dataUrl")
                        ),
                        "Extraction Status": _evidence_value(record, "status", "extractionStatus"),
                    }
                )
        if brand_shop_rows:
            st.markdown("##### Brand Shop Evidence")
            st.dataframe(pd.DataFrame(brand_shop_rows), hide_index=True, use_container_width=True)


def _render_slide4_finding_preview_v2(plan: dict[str, Any]) -> None:
    findings_by_group = (plan or {}).get("slide4_findings", {}) or {}
    if not findings_by_group:
        return

    def _format_hidden_column(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if label and reason:
                return f"{label} ({reason})"
            return label
        return ""

    with st.expander("Slide 4 Finding Preview", expanded=False):
        slide4_payload = build_slide4_pdp_benchmark_payload(
            plan or {},
            competitor_records=st.session_state.get("audit_competitor_entries", []) or [],
        )
        for warning in slide4_payload.get("warnings", []) or []:
            st.warning(warning)
        hidden = slide4_payload.get("hidden_columns", []) or []
        if hidden:
            hidden_labels = [
                formatted
                for item in hidden
                if (formatted := _format_hidden_column(item))
            ]
            if hidden_labels:
                st.caption(f"Hidden competitor sections: {', '.join(hidden_labels)}")
        for index, column in enumerate(slide4_payload.get("columns", []) or [], start=1):
            side_label = "Client" if index == 1 else f"Competitor {index - 1}"
            st.markdown(f"##### {side_label}: {column.get('label') or '-'}")
            if not column.get("active", True):
                st.caption("This section will be cleared because no PDP evidence was selected.")
                continue
            st.caption(
                f"Product: {column.get('product_title') or '-'} | "
                f"Product ID: {column.get('product_id') or '-'} | "
                f"Brand: {column.get('brand') or '-'} | "
                f"Product type: {column.get('product_type') or '-'}"
            )
            st.caption(
                f"Images: {column.get('image_count', 0)} | "
                f"Rating: {column.get('average_rating') or '-'} | "
                f"Reviews: {column.get('review_count') or 0} | "
                f"EBC present: {'Yes' if column.get('ebc_present') else 'No'} | "
                f"Sold by Walmart: {'Yes' if column.get('sold_by_walmart') else 'No'} | "
                f"Shipped by Walmart: {'Yes' if column.get('shipped_by_walmart') else 'No'}"
            )
            bullet_rows = [
                {
                    "Bullet": item.get("text", ""),
                    "Type": item.get("type", ""),
                    "Dimension": item.get("dimension", ""),
                    "Signals": ", ".join(item.get("signals", []) or []),
                    "Reason": item.get("reason", ""),
                }
                for item in column.get("bullet_debug", []) or []
            ]
            if bullet_rows:
                st.dataframe(pd.DataFrame(bullet_rows), hide_index=True, use_container_width=True)
            for warning in column.get("warnings", []) or []:
                st.warning(warning)
        st.markdown("##### Majority image-analysis findings")
        for key, findings in findings_by_group.items():
            if not isinstance(findings, dict):
                continue
            st.markdown(f"##### {findings.get('group_label') or key.replace('_', ' ').title()}")
            st.caption(
                f"Analyzed PDPs: {findings.get('analyzed_pdp_count', 0)} | "
                f"Majority threshold: {findings.get('majority_threshold', 0)} | "
                f"Category: {findings.get('category') or '-'} | "
                f"Product type: {findings.get('product_type') or '-'}"
            )
            selected = []
            all_findings = list(findings.get("strengths", []) or []) + list(
                findings.get("opportunities", []) or []
            )
            for bullet in findings.get("slide4_bullets", []) or []:
                matched = next(
                    (
                        item
                        for item in all_findings
                        if isinstance(item, dict) and item.get("text") == bullet
                    ),
                    {},
                )
                selected.append(
                    {
                        "Bullet": bullet,
                        "Signal": matched.get("signal", ""),
                        "Supporting PDPs": matched.get("supporting_pdps", 0),
                        "Analyzed PDPs": matched.get("analyzed_pdps", findings.get("analyzed_pdp_count", 0)),
                    }
                )
            if selected:
                st.dataframe(pd.DataFrame(selected), hide_index=True, use_container_width=True)
            else:
                st.caption("No local image-analysis majority findings yet. Slide 4 will use fallback bullets.")


def _render_slide2_summary_preview_v2(plan: dict[str, Any]) -> None:
    summary = (plan or {}).get("slide2_summary", {}) or {}
    if not summary:
        return
    phrases = summary.get("phrases", {}) or {}
    sections = summary.get("sections", {}) or {}
    with st.expander("Slide 2 Summary Preview", expanded=False):
        st.caption(
            f"Category: {phrases.get('category_phrase') or '-'} | "
            f"Benefit: {phrases.get('benefit_phrase') or '-'} | "
            f"Visual: {phrases.get('visual_phrase') or '-'}"
        )
        for key in ("consumer_demand", "walmart_opportunity", "competitive_benchmark"):
            section = sections.get(key, {}) or {}
            st.markdown(f"##### {section.get('label', key.replace('_', ' ').title())}")
            st.caption(
                f"Selected rating: {section.get('rating', '')} | "
                f"Allowed scale: {', '.join(section.get('allowed_ratings', []) or [])} | "
                f"Signals: {', '.join(section.get('signals', []) or [])}"
            )
            warnings = section.get("debug_warnings", []) or []
            for warning in warnings:
                st.warning(warning)
            rows = []
            for item in section.get("bullet_debug", []) or []:
                rows.append(
                    {
                        "Bullet": item.get("text", ""),
                        "Template ID": item.get("template_id", ""),
                        "Reason": item.get("reason", ""),
                        "Signals": ", ".join(item.get("signals", []) or []),
                        "Support": (
                            f"{item.get('supporting_count', 0)}/{item.get('analyzed_count', 0)}"
                            if item.get("analyzed_count", 0)
                            else ""
                        ),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No Slide 2 bullet debug metadata available.")


def _render_slide3_search_benchmark_preview_v2(plan: dict[str, Any]) -> None:
    payload = (plan or {}).get("slide3_search_benchmark", {}) or {}
    if not payload:
        return
    with st.expander("Slide 3 Search Benchmark Preview", expanded=False):
        for side in ("current", "benchmark"):
            side_payload = (payload.get(side) or {})
            st.markdown(f"##### {side.title()}")
            st.caption(
                f"Selected source row: {side_payload.get('source_row') or '-'} | "
                f"Resolved search term: {side_payload.get('search_term') or '-'} | "
                f"Category phrase: {side_payload.get('category_phrase') or '-'}"
            )
            st.caption(
                f"Screenshot available: {'Yes' if side_payload.get('screenshot') else 'No'} | "
                f"Products captured: {side_payload.get('product_count', 0) or 0} | "
                f"Client brand used for matching: {side_payload.get('client_brand') or '-'} | "
                f"Bullets: {len(side_payload.get('bullets', []) or [])}"
            )
            st.caption(
                "Client products found: "
                + (
                    ", ".join(
                        f"#{item.get('position')}: {item.get('title') or item.get('brand')}"
                        for item in (side_payload.get("client_products", []) or [])
                    )
                    or "-"
                )
            )
            st.caption(
                f"Top brands: {', '.join(side_payload.get('top_brands', []) or []) or '-'} | "
                f"Badges detected: {', '.join(side_payload.get('badges', []) or []) or '-'} | "
                f"Review counts: {', '.join(str(value) for value in (side_payload.get('review_counts', []) or [])) or '-'}"
            )
            dimension_rows = []
            for dimension, score in (side_payload.get("dimension_scores", {}) or {}).items():
                dimension_rows.append({"Dimension": dimension.replace("_", " ").title(), "Score": score})
            if dimension_rows:
                st.dataframe(pd.DataFrame(dimension_rows), hide_index=True, use_container_width=True)
            for bullet in side_payload.get("bullet_debug", []) or []:
                st.caption(
                    f"- {bullet.get('text', '')} [{bullet.get('dimension', '')} / {bullet.get('score', '')}]"
                )
            for warning in side_payload.get("warnings", []) or []:
                st.warning(warning)
        for warning in payload.get("warnings", []) or []:
            st.warning(warning)


def _render_slide6_visibility_preview_v2(plan: dict[str, Any]) -> None:
    visibility = (plan or {}).get("slide6_visibility", {}) or {}
    if not visibility:
        return
    with st.expander("Slide 6 Visibility Preview", expanded=False):
        st.caption(
            f"Pack: {visibility.get('pack_id') or '-'} | "
            f"Category phrase: {visibility.get('category_phrase') or '-'} | "
            f"Client: {visibility.get('client_label') or 'Client'}"
        )
        st.write(visibility.get("intro", ""))
        rows = []
        for item in visibility.get("segments", []) or []:
            debug = item.get("debug", {}) or {}
            fields = debug.get("matched_fields", {}) or {}
            terms = debug.get("matched_terms", {}) or {}
            rows.append(
                {
                    "Search Segment": item.get("segment", ""),
                    "Competitor": item.get("competitor_visibility", ""),
                    "Competitor Support": (
                        f"{item.get('competitor_fraction', '0/0')} "
                        f"({item.get('competitor_percentage', 0)}%)"
                    ),
                    "Client": item.get("client_visibility", ""),
                    "Client Support": (
                        f"{item.get('client_fraction', '0/0')} "
                        f"({item.get('client_percentage', 0)}%)"
                    ),
                    "Segment ID": debug.get("segment_id", ""),
                    "Selected Pack": debug.get("selected_pack", visibility.get("pack_id", "")),
                    "Matched Fields": (
                        f"Competitor: {', '.join(fields.get('competitor', []) or [])}; "
                        f"Client: {', '.join(fields.get('client', []) or [])}"
                    ),
                    "Matched Terms": (
                        f"Competitor: {', '.join(terms.get('competitor', []) or [])}; "
                        f"Client: {', '.join(terms.get('client', []) or [])}"
                    ),
                }
            )
            for warning in item.get("warnings", []) or []:
                st.warning(f"{item.get('segment', 'Segment')}: {warning}")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for warning in visibility.get("warnings", []) or []:
            st.warning(warning)
        st.write(visibility.get("takeaway", ""))


def _render_slide5_brand_shop_preview_v2(plan: dict[str, Any]) -> None:
    payload = (plan or {}).get("slide5_brand_shop", {}) or {}
    if not payload:
        return

    def _screenshot_status(value: Any) -> str:
        screenshot = str(value or "").strip()
        if not screenshot:
            return "Missing"
        if not screenshot.lower().startswith("data:image/") or "," not in screenshot:
            return "Invalid data URL"
        header, encoded = screenshot.split(",", 1)
        if ";base64" not in header.lower():
            return "Invalid data URL"
        try:
            base64.b64decode(encoded, validate=True)
        except Exception:
            return "Decode failed"
        return "Decodable"

    with st.expander("Slide 5 Brand Shop Preview", expanded=False):
        mode = payload.get("mode", "standard")
        st.caption(
            f"Selected mode: {'No Brand Shop' if mode == 'no_brand_shop' else 'Standard'} | "
            f"Client has a Walmart Brand Shop: "
            f"{'Yes' if payload.get('client_has_brand_shop', True) else 'No'}"
        )
        for warning in payload.get("warnings", []) or []:
            st.warning(warning)
        if mode == "no_brand_shop":
            competitor = payload.get("competitor")
            if not isinstance(competitor, dict):
                st.warning(
                    "No valid Competitor Brand Shop evidence is available; Slide 5 will remain unchanged."
                )
                return
            st.markdown("##### Competitor-led Opportunity Layout")
            st.caption(
                f"Selected Competitor source row: {competitor.get('source_row')} | "
                f"Brand: {competitor.get('brand_name') or '-'} | "
                f"Screenshot available: {'Yes' if competitor.get('screenshot') else 'No'} | "
                f"Decoded image status: {_screenshot_status(competitor.get('screenshot'))} | "
                "Placement: centered No Brand Shop competitor image area"
            )
            dimension_rows = []
            for dimension, score in (competitor.get("dimension_scores", {}) or {}).items():
                debug = (competitor.get("dimension_debug", {}) or {}).get(dimension, {}) or {}
                dimension_rows.append(
                    {
                        "Dimension": dimension.replace("_", " ").title(),
                        "Score": score,
                        "Supporting Signals": ", ".join(debug.get("signals", []) or []),
                        "Reason": debug.get("reason", ""),
                    }
                )
            st.dataframe(pd.DataFrame(dimension_rows), hide_index=True, use_container_width=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Bullet": item.get("text", ""),
                            "Type": item.get("type", ""),
                            "Dimension": item.get("dimension", ""),
                            "Template ID": item.get("template_id", ""),
                            "Reason": item.get("reason", ""),
                        }
                        for item in competitor.get("bullet_debug", []) or []
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            return
        for side_key, side_label in (
            ("client", "Client"),
            ("competitor", "Competitor"),
        ):
            side = payload.get(side_key)
            st.markdown(f"##### {side_label}")
            if not isinstance(side, dict):
                st.warning(
                    f"No valid {side_label} Brand Shop capture was selected; "
                    "that template side will remain unchanged."
                )
                continue
            st.caption(
                f"Source row: {side.get('source_row')} | "
                f"Brand: {side.get('brand_name') or '-'} | "
                f"Screenshot available: {'Yes' if side.get('screenshot') else 'No'} | "
                f"Decoded image status: {_screenshot_status(side.get('screenshot'))} | "
                f"Placement: {'left Current Structure' if side_key == 'client' else 'right Competitive Benchmark'}"
            )
            dimensions = side.get("dimension_scores", {}) or {}
            dimension_debug = side.get("dimension_debug", {}) or {}
            dimension_rows = []
            for dimension in (
                "brand_presentation",
                "lifestyle_merchandising",
                "category_segmentation",
                "product_discovery",
                "educational_storytelling",
                "video_rich_media",
                "cross_category_navigation",
            ):
                debug = dimension_debug.get(dimension, {}) or {}
                dimension_rows.append(
                    {
                        "Dimension": dimension.replace("_", " ").title(),
                        "Score": dimensions.get(dimension, ""),
                        "Supporting Signals": ", ".join(debug.get("signals", []) or []),
                        "Reason": debug.get("reason", ""),
                    }
                )
            st.dataframe(
                pd.DataFrame(dimension_rows),
                hide_index=True,
                use_container_width=True,
            )
            bullet_rows = [
                {
                    "Bullet": item.get("text", ""),
                    "Type": item.get("type", ""),
                    "Dimension": item.get("dimension", ""),
                    "Template ID": item.get("template_id", ""),
                    "Reason": item.get("reason", ""),
                }
                for item in side.get("bullet_debug", []) or []
            ]
            st.dataframe(
                pd.DataFrame(bullet_rows),
                hide_index=True,
                use_container_width=True,
            )
            for warning in side.get("warnings", []) or []:
                st.warning(warning)
        debug = payload.get("debug", {}) or {}
        unused_client = debug.get("unused_client_source_rows", []) or []
        unused_competitor = debug.get("unused_competitor_source_rows", []) or []
        if unused_client or unused_competitor:
            st.caption(
                f"Unused Client source rows: {unused_client or '-'} | "
                f"Unused Competitor source rows: {unused_competitor or '-'}"
            )


def render_primary_pdp_upload_v2() -> None:
    with st.container(border=True):
        st.markdown("### Primary Audit Extract Upload")
        st.caption("One row in the primary Audit Extract Sheet becomes one product audit entry.")
        method = st.radio(
            "Input Method",
            ["Audit Extract Sheet (Recommended)", "Fallback URL Mode"],
            horizontal=True,
            key="audit_primary_source_method",
        )

        if method == "Audit Extract Sheet (Recommended)":
            uploaded = st.file_uploader(
                "Upload Primary Audit Extract Sheet",
                type=["html", "htm", "xlsx", "xls", "csv"],
                key="audit_v2_primary_sheet_upload",
                help="Supported columns include Product URL, Product ID, Product Title, Image N, Description Bullet N, and optional notes/scores.",
            )
            if st.button("Process Primary Audit Sheet", key="audit_v2_process_primary_sheet"):
                if uploaded is None:
                    st.warning("Upload a primary Audit Extract Sheet to continue.")
                else:
                    uploaded_name = str(getattr(uploaded, "name", "") or "").strip()
                    uploaded_dir = os.path.dirname(uploaded_name)
                    if uploaded_dir and os.path.isdir(uploaded_dir):
                        st.session_state["audit_primary_sheet_dir"] = uploaded_dir
                    df_uploaded, parse_messages = parse_audit_extract_upload_to_dataframe(uploaded)
                    if parse_messages:
                        st.info("\n".join(parse_messages))
                    if df_uploaded.empty:
                        st.error("The uploaded sheet is empty or could not be parsed.")
                    else:
                        entries, records_map, extract_errors = process_primary_audit_extract_sheet(
                            df_uploaded=df_uploaded,
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                        )
                        _analyze_sheet_records_with_ui(
                            [entry.get("cached_record", {}) for entry in entries]
                        )
                        st.session_state["audit_primary_entries"] = entries
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        _reset_generated_audit_state_v2()
                        if entries:
                            st.success(f"Primary sheet ingestion complete for {len(entries)} row(s).")
                        else:
                            st.error("No usable primary product rows were ingested from the uploaded sheet.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:12]))
        else:
            st.caption("Fallback mode for live PDP URL extraction.")
            fallback_method = st.radio(
                "Fallback Input",
                ["Single PDP URL", "Excel / CSV of PDP URLs"],
                horizontal=True,
                key="audit_v2_primary_fallback_method",
            )
            if fallback_method == "Single PDP URL":
                pdp_url = st.text_input("PDP URL", key="audit_primary_single_pdp_url")
                if st.button("Load PDP Data (Fallback)", key="audit_v2_load_primary_single"):
                    if not pdp_url.strip():
                        st.warning("Enter a PDP URL to continue.")
                    else:
                        entries, records_map, extract_errors = process_primary_pdp_urls_real(
                            urls=[pdp_url.strip()],
                            cached_records_by_id=st.session_state.get("audit_cached_pdp_records", {}),
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                            max_count=5,
                        )
                        st.session_state["audit_primary_entries"] = entries
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        _reset_generated_audit_state_v2()
                        if not entries:
                            st.error("No usable primary PDP entries were extracted from the provided URL.")
                        else:
                            st.success(f"Primary PDP extraction complete for {len(entries)} URL.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:6]))
            else:
                uploaded = st.file_uploader(
                    "Upload Excel / CSV of PDP URLs",
                    type=["xlsx", "xls", "csv"],
                    key="audit_v2_primary_batch_upload",
                )
                df_uploaded = pd.DataFrame()
                if uploaded is not None:
                    try:
                        df_uploaded = _read_uploaded_table(uploaded)
                    except Exception as e:
                        st.error(f"Could not read upload: {e}")

                if not df_uploaded.empty:
                    url_columns = [c for c in df_uploaded.columns if "url" in str(c).lower()]
                    default_col = url_columns[0] if url_columns else df_uploaded.columns[0]
                    st.selectbox(
                        "PDP URL Column",
                        list(df_uploaded.columns),
                        index=list(df_uploaded.columns).index(default_col),
                        key="audit_v2_primary_url_col",
                    )
                else:
                    st.selectbox("PDP URL Column", ["PDP URL"], key="audit_v2_primary_url_col")

                if st.button("Process URLs (Fallback)", key="audit_v2_process_primary_batch"):
                    urls = []
                    if not df_uploaded.empty:
                        col = st.session_state.get("audit_v2_primary_url_col", "")
                        urls = _extract_urls_from_df_v2(df_uploaded, col)
                    if not urls:
                        st.error("No PDP URLs found in the selected column.")
                    else:
                        urls = urls[:5]
                        entries, records_map, extract_errors = process_primary_pdp_urls_real(
                            urls=urls,
                            cached_records_by_id=st.session_state.get("audit_cached_pdp_records", {}),
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                            max_count=5,
                        )
                        st.session_state["audit_primary_entries"] = entries
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        _reset_generated_audit_state_v2()
                        if entries:
                            st.success(f"Primary PDP extraction complete for {len(entries)} URL(s).")
                        else:
                            st.error("No usable primary PDP entries were extracted from the provided URLs.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:8]))

        entries = st.session_state.get("audit_primary_entries", [])
        if entries:
            preview_df = pd.DataFrame(
                [
                    {
                        "Product Title": e.get("product_title", ""),
                        "Item ID": e.get("item_id", ""),
                        "PDP URL": e.get("cached_record", {}).get("source_url", ""),
                        "Image Count": e.get("cached_record", {}).get("image_count", 0),
                        "Extraction Status": e.get("cached_record", {}).get("extraction_status", ""),
                    }
                    for e in entries
                ]
            )
            st.caption("Primary product entries queued for audit review")
            with st.expander("Primary Upload Preview", expanded=False):
                st.dataframe(preview_df, hide_index=True, use_container_width=True)


def _sync_primary_entry_edits_v2(entry: dict) -> None:
    entry_id = entry.get("entry_id") or f"entry-{entry.get('record_id', 'unknown')}"
    entry["entry_id"] = entry_id
    record = entry.get("cached_record", {})
    title_key = f"audit_v2_primary_current_title_{entry_id}"
    desc_key = f"audit_v2_primary_current_description_{entry_id}"
    feat_key = f"audit_v2_primary_current_features_{entry_id}"
    selected_key = f"audit_v2_primary_selected_image_{entry_id}"

    if title_key not in st.session_state:
        st.session_state[title_key] = record.get("current_title", "")
    if desc_key not in st.session_state:
        st.session_state[desc_key] = record.get("current_description_body", "")
    if feat_key not in st.session_state:
        features = record.get("current_key_features", [])
        st.session_state[feat_key] = "\n".join(
            [f"- {f.get('text', '').strip()}" for f in features if f.get("text", "").strip()]
        )

    images = record.get("images", [])
    image_labels = [f"Image {i + 1}" for i in range(len(images))]
    raw_selected_index = entry.get("selected_primary_image", {}).get("image_index", None)
    default_index = int(raw_selected_index) if isinstance(raw_selected_index, int) else 0
    if default_index < 0 or default_index >= len(image_labels):
        default_index = 0
    default_label = image_labels[default_index] if image_labels else ""
    if selected_key not in st.session_state and default_label:
        st.session_state[selected_key] = default_label

    record["current_title"] = st.session_state.get(title_key, "")
    record["current_description_body"] = st.session_state.get(desc_key, "")
    feature_lines = [ln.strip().lstrip("-").strip() for ln in st.session_state.get(feat_key, "").splitlines() if ln.strip()]
    record["current_key_features"] = [{"index": i + 1, "text": text} for i, text in enumerate(feature_lines)]
    record["current_description_bullets"] = []
    update_record_tier1_derived_fields(record)
    entry["rule_findings"] = analyze_primary_record(record)

    selected_multi = entry.get("selected_primary_images", []) or []
    if selected_multi and images:
        selected_idx_set: set[int] = set()
        for picked in selected_multi:
            try:
                selected_idx_set.add(int(picked.get("image_index", -1) or -1))
            except Exception:
                continue
        ordered_multi: list[dict[str, Any]] = []
        for image in images:
            try:
                picked_idx = int(image.get("index", -1) or -1)
            except Exception:
                continue
            if picked_idx not in selected_idx_set:
                continue
            ordered_multi.append(
                {
                    "record_id": record.get("record_id", ""),
                    "image_index": picked_idx,
                    "url": image.get("url", ""),
                }
            )
        ordered_multi = ordered_multi[:4]
        if ordered_multi:
            entry["selected_primary_images"] = ordered_multi
            entry["selected_primary_image"] = dict(ordered_multi[0])
            legacy_index = int(ordered_multi[0].get("image_index", 0) or 0)
            if 0 <= legacy_index < len(image_labels):
                st.session_state[selected_key] = image_labels[legacy_index]
            return

    selected_label = st.session_state.get(selected_key, default_label)
    if selected_label in image_labels:
        selected_index = image_labels.index(selected_label)
        selected_url = images[selected_index]["url"] if selected_index < len(images) else ""
        entry["selected_primary_image"] = {
            "record_id": record.get("record_id", ""),
            "image_index": selected_index,
            "url": selected_url,
        }
        entry["selected_primary_images"] = [dict(entry["selected_primary_image"])]


def _extract_style_guide_key_features(record: dict[str, Any]) -> list[str]:
    features = record.get("current_key_features", [])
    if isinstance(features, str):
        return [line.strip().lstrip("-").strip() for line in features.splitlines() if line.strip()]
    if not isinstance(features, list):
        return []

    values: list[str] = []
    for feature in features:
        if isinstance(feature, dict):
            text = str(feature.get("text", "") or "").strip()
        else:
            text = str(feature or "").strip()
        if text:
            values.append(text)
    return values


def _build_style_guide_product_payload(entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": record.get("current_title") or entry.get("product_title", ""),
        "current_title": record.get("current_title", ""),
        "product_title": entry.get("product_title", ""),
        "category": record.get("category", ""),
        "product_category": record.get("product_category", ""),
        "subcategory": record.get("subcategory", ""),
        "product_type": record.get("product_type", ""),
        "description": record.get("current_description_body", ""),
        "key_features": _extract_style_guide_key_features(record),
    }


def _style_guide_no_match(match_reason: str = "user_disabled", selection_source: str = "manual_none") -> dict[str, Any]:
    return {
        "matched": False,
        "category": "",
        "family_id": "",
        "family_name": "",
        "product_type_id": "",
        "product_type_name": "",
        "formula": [],
        "attributes": [],
        "match_reason": match_reason,
        "confidence": "none",
        "selection_source": selection_source,
    }


def _style_guide_formula_text(formula: list[Any]) -> str:
    return " + ".join(str(part) for part in formula if str(part).strip())


def _style_guide_option_catalog(guides: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for guide in guides:
        category = str(guide.get("category", "") or "").strip()
        if not category:
            continue
        category_item = catalog.setdefault(category, {"guide": guide, "families": {}})
        for family_id, family in guide.get("families", {}).items():
            family_name = str(family.get("display_name", family_id) or family_id)
            family_item = category_item["families"].setdefault(
                family_id,
                {"family_id": family_id, "family_name": family_name, "product_types": {}},
            )
            for product_type_id, rule in family.get("product_types", {}).items():
                product_type_name = str(rule.get("display_name", product_type_id) or product_type_id)
                formula = list(rule.get("formula", []))
                formula_text = _style_guide_formula_text(formula)
                label = f"{product_type_name} - {formula_text}" if formula_text else product_type_name
                family_item["product_types"][label] = {
                    "category": category,
                    "family_id": family_id,
                    "family_name": family_name,
                    "product_type_id": product_type_id,
                    "product_type_name": product_type_name,
                    "formula": formula,
                    "attributes": list(rule.get("attributes", [])),
                }
    return catalog


def _ensure_select_value(key: str, options: list[str]) -> None:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _render_style_guide_match_v2(entry: dict[str, Any], record: dict[str, Any]) -> None:
    entry_id = entry.get("entry_id") or f"entry-{entry.get('record_id', 'unknown')}"
    product_payload = _build_style_guide_product_payload(entry, record)
    guides = load_style_guides()
    detected = match_style_guide_rule(product_payload, guides)
    detected_with_source = dict(detected)
    detected_with_source["selection_source"] = "auto"

    catalog = _style_guide_option_catalog(guides)
    category_options = [STYLE_GUIDE_AUTO_LABEL, STYLE_GUIDE_NONE_LABEL] + sorted(catalog.keys())

    category_key = f"audit_v2_style_category_{entry_id}"
    family_key = f"audit_v2_style_family_{entry_id}"
    product_type_key = f"audit_v2_style_product_type_{entry_id}"

    st.markdown("##### Style Guide Match")
    if detected.get("matched"):
        st.caption(
            "Detected: "
            f"{detected.get('category', '')} -> {detected.get('family_name', '')} -> "
            f"{detected.get('product_type_name', '')}"
        )
    else:
        st.caption("Detected: No style guide match")
    st.caption(f"Confidence: {detected.get('confidence', 'none')}")
    st.caption(f"Reason: {detected.get('match_reason', 'no_match')}")
    if detected.get("matched"):
        st.caption(f"Formula: {_style_guide_formula_text(detected.get('formula', [])) or '-'}")

    _ensure_select_value(category_key, category_options)
    selected_category = st.selectbox("Override Category", category_options, key=category_key)

    if selected_category == STYLE_GUIDE_AUTO_LABEL:
        visible_categories = sorted(catalog.keys())
    elif selected_category == STYLE_GUIDE_NONE_LABEL:
        visible_categories = []
    else:
        visible_categories = [selected_category]

    family_label_map: dict[str, dict[str, Any]] = {}
    for category in visible_categories:
        for family_item in catalog.get(category, {}).get("families", {}).values():
            family_label_map[family_item["family_name"]] = {"category": category, **family_item}
    family_options = [STYLE_GUIDE_AUTO_LABEL, STYLE_GUIDE_NONE_LABEL] + sorted(family_label_map.keys())
    _ensure_select_value(family_key, family_options)
    selected_family = st.selectbox("Override Product Family", family_options, key=family_key)

    if selected_family == STYLE_GUIDE_AUTO_LABEL:
        visible_families = list(family_label_map.values())
    elif selected_family == STYLE_GUIDE_NONE_LABEL:
        visible_families = []
    else:
        visible_families = [family_label_map[selected_family]]

    product_type_label_map: dict[str, dict[str, Any]] = {}
    for family_item in visible_families:
        for label, product_type in family_item.get("product_types", {}).items():
            product_type_label_map[label] = product_type
    product_type_options = [STYLE_GUIDE_AUTO_LABEL, STYLE_GUIDE_NONE_LABEL] + sorted(product_type_label_map.keys())
    _ensure_select_value(product_type_key, product_type_options)
    selected_product_type = st.selectbox("Override Product Type Formula", product_type_options, key=product_type_key)

    if STYLE_GUIDE_NONE_LABEL in {selected_category, selected_family, selected_product_type}:
        entry["style_guide_match"] = _style_guide_no_match()
        return

    if selected_product_type != STYLE_GUIDE_AUTO_LABEL:
        override = product_type_label_map[selected_product_type]
        entry["style_guide_match"] = {
            "matched": True,
            "category": override["category"],
            "family_id": override["family_id"],
            "family_name": override["family_name"],
            "product_type_id": override["product_type_id"],
            "product_type_name": override["product_type_name"],
            "formula": list(override["formula"]),
            "attributes": list(override["attributes"]),
            "match_reason": "user_override",
            "confidence": "exact",
            "selection_source": "manual",
        }
        return

    entry["style_guide_match"] = detected_with_source if detected.get("matched") else _style_guide_no_match("no_match", "auto")


def _format_image_dimensions_for_preview(image: dict[str, Any]) -> str:
    width = image.get("width")
    height = image.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        w = int(width)
        h = int(height)
        if w > 0 and h > 0:
            return f"{w} W x {h} H"
    raw_dims = str(image.get("dimensions", "") or "").strip()
    match = re.search(r"(?P<w>\d{2,5})\s*[x×]\s*(?P<h>\d{2,5})", raw_dims, flags=re.IGNORECASE)
    if match:
        return f"{int(match.group('w'))} W x {int(match.group('h'))} H"
    return "Dimensions: -"


def _resolve_local_image_source(src: str) -> str:
    raw = str(src or "").strip()
    if not raw:
        return ""
    if re.match(r"(?is)^https?://", raw) or raw.startswith("data:image"):
        return raw

    candidate = raw
    if raw.lower().startswith("file://"):
        candidate = raw[7:]

    candidate = os.path.expanduser(candidate)
    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return candidate

    base_dirs: list[str] = []
    primary_dir = str(st.session_state.get("audit_primary_sheet_dir", "") or "").strip()
    if primary_dir:
        base_dirs.append(primary_dir)
    competitor_dir = str(st.session_state.get("audit_competitor_sheet_dir", "") or "").strip()
    if competitor_dir:
        base_dirs.append(competitor_dir)
    base_dirs.append(os.getcwd())

    for base_dir in base_dirs:
        try_path = os.path.normpath(os.path.join(base_dir, candidate))
        if os.path.isfile(try_path):
            return try_path

    if os.path.isfile(candidate):
        return candidate
    return ""


def safe_render_image(src: str, width: int = 120, unavailable_label: str = "Image unavailable") -> bool:
    raw = str(src or "").strip()
    if not raw:
        return False

    try:
        if re.match(r"(?is)^https?://", raw) or raw.startswith("data:image"):
            st.image(raw, width=width)
            return True

        resolved = _resolve_local_image_source(raw)
        if not resolved:
            st.caption(unavailable_label)
            return False
        st.image(resolved, width=width)
        return True
    except Exception:
        st.caption(unavailable_label)
        return False


def render_extracted_primary_product_entries_v2() -> None:
    st.caption(
        "Each primary product entry supports exactly one primary image selection for later slide mapping."
    )
    entries = st.session_state.get("audit_primary_entries", [])
    if not entries:
        st.info("Load primary PDP data first to create product audit entries.")
        return

    if "audit_select_all_primary_for_export" not in st.session_state:
        st.session_state["audit_select_all_primary_for_export"] = True
    select_all = st.checkbox(
        "Select All for Export",
        key="audit_select_all_primary_for_export",
        help="When enabled, all primary entries are included in the PowerPoint export.",
    )

    included_count = 0
    for entry in entries:
        if "include_in_export" not in entry:
            entry["include_in_export"] = True
        entry_id = entry.get("entry_id") or f"entry-{entry.get('record_id', 'unknown')}"
        include_key = f"audit_v2_primary_include_export_{entry_id}"
        if select_all:
            included_count += 1
        elif bool(st.session_state.get(include_key, entry.get("include_in_export", True))):
            included_count += 1

    summary_cols = st.columns(3)
    summary_cols[0].metric("PDPs extracted", len(entries))
    summary_cols[1].metric("Included in export", included_count)
    summary_cols[2].metric("Select All for Export", "ON" if select_all else "OFF")

    for idx, entry in enumerate(entries, start=1):
        _sync_primary_entry_edits_v2(entry)
        if "include_in_export" not in entry:
            entry["include_in_export"] = True
        if select_all:
            entry["include_in_export"] = True
        record = entry.get("cached_record", {})
        title = str(entry.get("product_title", "") or "Untitled Product").strip()
        item_id = str(entry.get("item_id", "") or "-").strip()
        with st.expander(f"Product {idx}: {title} | Item ID: {item_id}", expanded=False):
            st.markdown(f"#### Primary Product Entry {idx}")
            t1, t2, t3 = st.columns([2, 1, 1])
            with t1:
                st.write(f"**Product Title:** {entry.get('product_title', '-')}")
                st.caption(record.get("source_url", "-"))
            with t2:
                st.write(f"**Item ID:** {entry.get('item_id', '-')}")
                st.caption(f"Brand: {record.get('brand', '-')}")
            with t3:
                st.write(f"**Image Count:** {record.get('image_count', 0)}")
                reviews = record.get("reviews_summary", {})
                avg = reviews.get("average_rating")
                cnt = reviews.get("ratings_count")
                summary = f"{avg} avg rating from {cnt} ratings" if avg is not None and cnt is not None else "No review summary"
                st.caption(summary)
                st.caption(f"Extraction: {record.get('extraction_status', 'unknown')}")
            st.caption(
                f"Category: {record.get('category', '-') or '-'} | "
                f"Product Type/Subcategory: {record.get('subcategory', '-') or '-'}"
            )
            reviews = record.get("reviews_summary", {})
            review_count = reviews.get("review_count")
            if review_count is not None:
                st.caption(f"Review Count: {review_count}")

            extraction_errors = record.get("extraction_errors", [])
            if extraction_errors:
                st.caption("Extraction notes: " + "; ".join(extraction_errors[:3]))

            _render_style_guide_match_v2(entry, record)
            _render_local_image_analysis(record)

            images = record.get("images", [])
            image_models = [img for img in images if isinstance(img, dict) and str(img.get("url", "") or "").strip()]
            image_urls = [img.get("url", "") for img in image_models]
            if image_models:
                selection_signature_key = f"audit_v2_primary_selection_signature_{entry['entry_id']}"
                selection_signature = tuple(
                    f"{int(img.get('index', i) or i)}|{str(img.get('url', '') or '').strip()}"
                    for i, img in enumerate(image_models)
                )
                selected_multi = entry.get("selected_primary_images", []) or []
                default_selected_ids: set[int] = set()
                if selected_multi:
                    for sel in selected_multi:
                        try:
                            default_selected_ids.add(int(sel.get("image_index", -1) or -1))
                        except Exception:
                            continue
                if not default_selected_ids:
                    raw_selected_index = entry.get("selected_primary_image", {}).get("image_index", None)
                    try:
                        if isinstance(raw_selected_index, int):
                            default_selected_ids.add(int(raw_selected_index))
                    except Exception:
                        pass
                if not default_selected_ids:
                    default_selected_ids.add(int(image_models[0].get("index", 0) or 0))

                if st.session_state.get(selection_signature_key) != selection_signature:
                    for i, image in enumerate(image_models):
                        image_index = int(image.get("index", i) or i)
                        sel_key = f"audit_v2_primary_select_for_pdp_{entry['entry_id']}_{image_index}"
                        st.session_state[sel_key] = image_index in default_selected_ids
                    st.session_state[selection_signature_key] = selection_signature
                fallback_apply_key = f"audit_v2_primary_fallback_apply_{entry['entry_id']}"
                if st.session_state.pop(fallback_apply_key, False):
                    first_index = int(image_models[0].get("index", 0) or 0)
                    first_sel_key = f"audit_v2_primary_select_for_pdp_{entry['entry_id']}_{first_index}"
                    st.session_state[first_sel_key] = True

                thumbnails_per_row = min(6, len(image_models))
                img_cols = st.columns(thumbnails_per_row)
                selected_primary_images: list[dict[str, Any]] = []
                limit_blocked = False
                for i, image in enumerate(image_models):
                    image_url = image.get("url", "")
                    image_index = int(image.get("index", i) or i)
                    dims_key = f"audit_v2_primary_show_dims_ppt_{entry['entry_id']}_{image_index}"
                    sel_key = f"audit_v2_primary_select_for_pdp_{entry['entry_id']}_{image_index}"
                    if dims_key not in st.session_state:
                        st.session_state[dims_key] = bool(image.get("show_dimensions_in_powerpoint", False))
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = image_index in default_selected_ids
                    with img_cols[i % thumbnails_per_row]:
                        safe_render_image(image_url, width=120)
                        st.caption(f"Image {i + 1}")
                        st.caption(_format_image_dimensions_for_preview(image))
                        show_dims = st.checkbox("Show dimensions in PowerPoint", key=dims_key)
                        image["show_dimensions_in_powerpoint"] = bool(show_dims)
                        for raw_image in images:
                            if int(raw_image.get("index", -1) or -1) == image_index:
                                raw_image["show_dimensions_in_powerpoint"] = bool(show_dims)
                                break
                        include_primary = st.checkbox("Selected for PDP Slide", key=sel_key)
                        if include_primary:
                            if len(selected_primary_images) < 4:
                                selected_primary_images.append(
                                    {
                                        "record_id": record.get("record_id", ""),
                                        "image_index": image_index,
                                        "url": image_url,
                                    }
                                )
                            else:
                                st.session_state[sel_key] = False
                                limit_blocked = True

                if limit_blocked:
                    st.warning("You can select up to 4 primary images.")

                if not selected_primary_images:
                    first_image = image_models[0]
                    first_index = int(first_image.get("index", 0) or 0)
                    st.session_state[fallback_apply_key] = True
                    selected_primary_images = [
                        {
                            "record_id": record.get("record_id", ""),
                            "image_index": first_index,
                            "url": first_image.get("url", ""),
                        }
                    ]

                entry["selected_primary_images"] = selected_primary_images[:4]
                entry["selected_primary_image"] = dict(entry["selected_primary_images"][0])
                selected_url = entry["selected_primary_image"].get("url", "")
                selected_preview = next(
                    (
                        img
                        for img in image_models
                        if int(img.get("index", -1) or -1) == int(entry["selected_primary_image"].get("image_index", -1) or -1)
                    ),
                    image_models[0],
                )

                st.caption(f"Selected primary images: {len(entry['selected_primary_images'])} / 4")
                st.caption("Selected primary image preview")
                safe_render_image(selected_url, width=180)
                st.caption(_format_image_dimensions_for_preview(selected_preview))

            st.text_area("Current Title", key=f"audit_v2_primary_current_title_{entry['entry_id']}", height=85)
            st.text_area("Current Description", key=f"audit_v2_primary_current_description_{entry['entry_id']}", height=120)
            st.text_area("Current Key Features", key=f"audit_v2_primary_current_features_{entry['entry_id']}", height=110)
            include_key = f"audit_v2_primary_include_export_{entry['entry_id']}"
            if include_key not in st.session_state:
                st.session_state[include_key] = bool(entry.get("include_in_export", True))
            if select_all:
                st.session_state[include_key] = True
            include_value = st.checkbox(
                "Include in Export",
                key=include_key,
                disabled=select_all,
            )
            entry["include_in_export"] = bool(include_value)
            ingest_metadata = record.get("ingest_metadata", {}) or {}
            if ingest_metadata:
                with st.expander("Sheet Metadata", expanded=False):
                    st.json(ingest_metadata)

            findings = entry.get("rule_findings", [])
            with st.expander(f"Findings Preview ({len(findings)})", expanded=False):
                if not findings:
                    st.caption("No deterministic findings currently flagged.")
                else:
                    findings_df = pd.DataFrame(
                        [
                            {
                                "Section": f.get("section"),
                                "Severity": f.get("severity"),
                                "Issue": f.get("issue_type"),
                                "Message": f.get("message"),
                            }
                            for f in findings
                        ]
                    )
                    st.dataframe(findings_df, use_container_width=True, hide_index=True)
                    st.caption("Developer view: structured findings with evidence attached on entry `rule_findings`.")


def render_competitor_pdp_upload_v2() -> None:
    with st.container(border=True):
        st.markdown("### Competitor Audit Extract Upload")
        st.caption("Competitor extract rows feed shared competitor graphics content.")
        method = st.radio(
            "Input Method",
            ["Audit Extract Sheet (Recommended)", "Fallback URL Mode"],
            horizontal=True,
            key="audit_competitor_source_method",
        )

        if method == "Audit Extract Sheet (Recommended)":
            uploaded = st.file_uploader(
                "Upload Competitor Audit Extract Sheet",
                type=["html", "htm", "xlsx", "xls", "csv"],
                key="audit_v2_competitor_sheet_upload",
            )
            if st.button("Process Competitor Audit Sheet", key="audit_v2_process_competitor_sheet"):
                if uploaded is None:
                    st.warning("Upload a competitor Audit Extract Sheet to continue.")
                else:
                    uploaded_name = str(getattr(uploaded, "name", "") or "").strip()
                    uploaded_dir = os.path.dirname(uploaded_name)
                    if uploaded_dir and os.path.isdir(uploaded_dir):
                        st.session_state["audit_competitor_sheet_dir"] = uploaded_dir
                    df_uploaded, parse_messages = parse_audit_extract_upload_to_dataframe(uploaded)
                    if parse_messages:
                        st.info("\n".join(parse_messages))
                    if df_uploaded.empty:
                        st.error("The uploaded sheet is empty or could not be parsed.")
                    else:
                        records, records_map, extract_errors = process_competitor_audit_extract_sheet(
                            df_uploaded=df_uploaded,
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                        )
                        _analyze_sheet_records_with_ui(records)
                        st.session_state["audit_competitor_entries"] = records
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        st.session_state["audit_competitor_image_orders"] = {}
                        _reset_competitor_graphics_mode_state_v2()
                        _reset_generated_audit_state_v2()
                        if records:
                            st.success(f"Competitor sheet ingestion complete for {len(records)} row(s).")
                        else:
                            st.error("No usable competitor rows were ingested from the uploaded sheet.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:12]))
        else:
            st.caption("Fallback mode for live competitor PDP URL extraction.")
            fallback_method = st.radio(
                "Fallback Input",
                ["Single PDP URL", "Excel / CSV of PDP URLs"],
                horizontal=True,
                key="audit_v2_competitor_fallback_method",
            )
            if fallback_method == "Single PDP URL":
                pdp_url = st.text_input("Competitor PDP URL", key="audit_v2_competitor_single_pdp_url")
                if st.button("Load Competitor PDP (Fallback)", key="audit_v2_load_competitor_single"):
                    if not pdp_url.strip():
                        st.warning("Enter a competitor PDP URL to continue.")
                    else:
                        records, records_map, extract_errors = process_competitor_pdp_urls_real(
                            urls=[pdp_url.strip()],
                            cached_records_by_id=st.session_state.get("audit_cached_pdp_records", {}),
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                            max_count=5,
                        )
                        st.session_state["audit_competitor_entries"] = records
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        st.session_state["audit_competitor_image_orders"] = {}
                        _reset_competitor_graphics_mode_state_v2()
                        _reset_generated_audit_state_v2()
                        if records:
                            st.success(f"Competitor PDP extraction complete for {len(records)} URL.")
                        else:
                            st.error("No usable competitor PDP entries were extracted from the provided URL.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:6]))
            else:
                uploaded = st.file_uploader(
                    "Upload Excel / CSV of Competitor PDP URLs",
                    type=["xlsx", "xls", "csv"],
                    key="audit_v2_competitor_batch_upload",
                )
                df_uploaded = pd.DataFrame()
                if uploaded is not None:
                    try:
                        df_uploaded = _read_uploaded_table(uploaded)
                    except Exception as e:
                        st.error(f"Could not read upload: {e}")

                if not df_uploaded.empty:
                    url_columns = [c for c in df_uploaded.columns if "url" in str(c).lower()]
                    default_col = url_columns[0] if url_columns else df_uploaded.columns[0]
                    st.selectbox(
                        "PDP URL Column",
                        list(df_uploaded.columns),
                        index=list(df_uploaded.columns).index(default_col),
                        key="audit_v2_competitor_url_col",
                    )
                else:
                    st.selectbox("PDP URL Column", ["PDP URL"], key="audit_v2_competitor_url_col")

                if st.button("Process Competitor URLs (Fallback)", key="audit_v2_process_competitor_batch"):
                    urls = []
                    if not df_uploaded.empty:
                        col = st.session_state.get("audit_v2_competitor_url_col", "")
                        urls = _extract_urls_from_df_v2(df_uploaded, col)
                    if not urls:
                        st.error("No competitor PDP URLs found in the selected column.")
                    else:
                        urls = urls[:5]
                        records, records_map, extract_errors = process_competitor_pdp_urls_real(
                            urls=urls,
                            cached_records_by_id=st.session_state.get("audit_cached_pdp_records", {}),
                            client_name=st.session_state.get("audit_client_name", ""),
                            retailer=st.session_state.get("audit_retailer", ""),
                            max_count=5,
                        )
                        st.session_state["audit_competitor_entries"] = records
                        st.session_state.setdefault("audit_cached_pdp_records", {}).update(records_map)
                        st.session_state["audit_competitor_image_orders"] = {}
                        _reset_competitor_graphics_mode_state_v2()
                        _reset_generated_audit_state_v2()
                        if records:
                            st.success(f"Competitor PDP extraction complete for {len(records)} URL(s).")
                        else:
                            st.error("No usable competitor PDP entries were extracted from the provided URLs.")
                        if extract_errors:
                            st.warning("\n".join(extract_errors[:8]))


def render_extracted_competitor_entries_v2() -> None:
    entries = st.session_state.get("audit_competitor_entries", [])
    if not entries:
        st.info("Competitor upload is optional. Add competitor PDPs when needed.")
        st.session_state["audit_competitor_has_multiple_pdps"] = False
        st.session_state["audit_competitor_pdp_group_count"] = 0
        st.session_state["audit_competitor_make_multiple_slides"] = False
        st.session_state["audit_competitor_slide_mode"] = "single_pdp"
        st.session_state["audit_competitor_slide_mode_selector"] = "Generate One Combined PDP Slide"
        return

    max_slots = 10

    def _normalize_image_models(raw_images: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, img in enumerate(raw_images or []):
            if isinstance(img, dict):
                url = str(img.get("url", "") or "").strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "index": int(img.get("index", idx) or idx),
                        "url": url,
                        "dimensions": str(img.get("dimensions", "") or ""),
                        "width": img.get("width"),
                        "height": img.get("height"),
                        "show_dimensions_in_powerpoint": bool(img.get("show_dimensions_in_powerpoint", False)),
                    }
                )
            else:
                url = str(img or "").strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "index": idx,
                        "url": url,
                        "dimensions": "",
                        "width": None,
                        "height": None,
                        "show_dimensions_in_powerpoint": False,
                    }
                )
        return normalized

    grouped_entries: list[dict[str, Any]] = []
    all_image_ids: list[str] = []
    image_meta: dict[str, dict[str, Any]] = {}

    for idx, record in enumerate(entries, start=1):
        image_models = _normalize_image_models(record.get("images", []))
        record_id = str(record.get("record_id", "") or f"record-{idx}")
        for image_pos, image in enumerate(image_models):
            image_index = int(image.get("index", image_pos))
            image_id = f"{record_id}|{image_index}"
            all_image_ids.append(image_id)
            image_meta[image_id] = {
                "record": record,
                "entry_idx": idx,
                "image_pos": image_pos,
                "image_label": f"Image {image_pos + 1}",
                "url": image.get("url", ""),
            }
        grouped_entries.append(
            {
                "entry_idx": idx,
                "record": record,
                "record_id": record_id,
                "images": image_models,
            }
        )
        if int(record.get("image_count", 0) or 0) != len(image_models):
            record["image_count"] = len(image_models)

    group_count = len(grouped_entries)
    has_multiple_pdps = group_count > 1
    st.session_state["audit_competitor_has_multiple_pdps"] = has_multiple_pdps
    st.session_state["audit_competitor_pdp_group_count"] = group_count

    if not has_multiple_pdps:
        st.session_state["audit_competitor_make_multiple_slides"] = False
        slide_mode = "single_pdp"
        st.session_state["audit_competitor_slide_mode_selector"] = "Generate One Combined PDP Slide"
        st.caption("Competitor images are grouped by PDP. Select up to 10 images for the competitor graphics slide.")
    else:
        st.caption(
            "Competitor images are grouped by PDP. Choose one combined competitor slide "
            "or one competitor slide per PDP group."
        )
        selector_default = (
            "Generate Multiple PDP Slides"
            if bool(st.session_state.get("audit_competitor_make_multiple_slides", False))
            else "Generate One Combined PDP Slide"
        )
        if st.session_state.get("audit_competitor_slide_mode_selector") not in {
            "Generate Multiple PDP Slides",
            "Generate One Combined PDP Slide",
        }:
            st.session_state["audit_competitor_slide_mode_selector"] = selector_default
        selected_mode_label = st.radio(
            "Competitor Graphics Generation Mode",
            options=[
                "Generate Multiple PDP Slides",
                "Generate One Combined PDP Slide",
            ],
            key="audit_competitor_slide_mode_selector",
            horizontal=True,
        )
        st.session_state["audit_competitor_make_multiple_slides"] = selected_mode_label == "Generate Multiple PDP Slides"
        slide_mode = "per_pdp" if selected_mode_label == "Generate Multiple PDP Slides" else "combined"
        if slide_mode == "per_pdp":
            st.caption("Multi-slide mode: select up to 10 images per competitor PDP group.")
        else:
            st.caption("Combined mode: select up to 10 total images across all competitor PDP groups.")
    st.session_state["audit_competitor_slide_mode"] = slide_mode

    def _default_selected_image_ids_round_robin() -> list[str]:
        if len(grouped_entries) <= 1:
            return all_image_ids[:max_slots]

        selected: list[str] = []
        pass_index = 0
        while len(selected) < max_slots:
            added_in_pass = False
            for group in grouped_entries:
                group_images = group.get("images", [])
                if pass_index >= len(group_images):
                    continue
                image = group_images[pass_index]
                image_index = int(image.get("index", pass_index))
                image_id = f"{group['record_id']}|{image_index}"
                selected.append(image_id)
                added_in_pass = True
                if len(selected) >= max_slots:
                    break
            if not added_in_pass:
                break
            pass_index += 1
        return selected

    def _default_selected_image_ids_multi() -> set[str]:
        selected_multi: set[str] = set()
        for group in grouped_entries:
            group_images = group.get("images", [])
            for image in group_images[:max_slots]:
                image_index = int(image.get("index", 0))
                image_id = f"{group['record_id']}|{image_index}"
                selected_multi.add(image_id)
        return selected_multi

    selection_signature = tuple(all_image_ids)
    selection_prefix = "audit_v2_comp_include_combined_" if slide_mode in {"single_pdp", "combined"} else "audit_v2_comp_include_multi_"
    signature_key = (
        "audit_v2_comp_selection_signature_combined"
        if slide_mode in {"single_pdp", "combined"}
        else "audit_v2_comp_selection_signature_multi"
    )
    if st.session_state.get(signature_key) != selection_signature:
        default_ids = (
            set(_default_selected_image_ids_round_robin())
            if slide_mode in {"single_pdp", "combined"}
            else _default_selected_image_ids_multi()
        )
        for image_id in all_image_ids:
            st.session_state[f"{selection_prefix}{image_id}"] = image_id in default_ids
        st.session_state[signature_key] = selection_signature

    control_col_1, control_col_2, control_col_3 = st.columns([1, 1, 2])
    with control_col_1:
        if st.button("Select Default 10", key="audit_v2_comp_select_default_10"):
            default_ids = (
                set(_default_selected_image_ids_round_robin())
                if slide_mode in {"single_pdp", "combined"}
                else _default_selected_image_ids_multi()
            )
            for image_id in all_image_ids:
                st.session_state[f"{selection_prefix}{image_id}"] = image_id in default_ids
    with control_col_2:
        if st.button("Clear All", key="audit_v2_comp_clear_all"):
            for image_id in all_image_ids:
                st.session_state[f"{selection_prefix}{image_id}"] = False
    with control_col_3:
        if slide_mode in {"single_pdp", "combined"}:
            current_selected = sum(
                1 for image_id in all_image_ids if bool(st.session_state.get(f"{selection_prefix}{image_id}", False))
            )
            st.caption(f"Selected: {current_selected} / {max_slots}")
        else:
            per_group_selected: list[str] = []
            for group in grouped_entries:
                group_images = group.get("images", [])
                group_selected_count = 0
                for i, image in enumerate(group_images):
                    image_index = int(image.get("index", i))
                    image_id = f"{group.get('record_id', '')}|{image_index}"
                    if bool(st.session_state.get(f"{selection_prefix}{image_id}", False)):
                        group_selected_count += 1
                per_group_selected.append(f"{group.get('entry_idx', 0)}: {group_selected_count}/{max_slots}")
            st.caption("Per-group selected: " + " | ".join(per_group_selected))

    current_selected_total = sum(
        1 for image_id in all_image_ids if bool(st.session_state.get(f"{selection_prefix}{image_id}", False))
    )
    summary_cols = st.columns(3)
    summary_cols[0].metric("Competitor PDPs extracted", len(entries))
    summary_cols[1].metric("Images found", len(all_image_ids))
    summary_cols[2].metric("Selected for deck", current_selected_total)

    selected_image_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    selected_ids_by_record: dict[str, list[str]] = {}
    group_image_orders: dict[str, dict[str, int]] = {}
    limit_blocked = False

    for group in grouped_entries:
        idx = int(group.get("entry_idx", 0))
        record = group.get("record", {})
        image_models = group.get("images", [])
        record_id = str(group.get("record_id", ""))
        group_selected_ids: list[str] = []
        competitor_title = str(record.get("product_title", "") or "Untitled Competitor").strip()
        with st.expander(f"Competitor {idx}: {competitor_title} | {len(image_models)} images", expanded=False):
            st.markdown(f"#### Competitor Entry {idx}")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.write(f"**Title:** {record.get('product_title', '-')}")
                st.caption(record.get("source_url", "-"))
            with c2:
                st.write(f"**Brand:** {record.get('brand', '-')}")
                item_id = record.get("item_id", "")
                st.caption(f"Item ID: {item_id if item_id else '-'}")
            with c3:
                st.write(f"**Image Count:** {record.get('image_count', 0)}")
                st.caption(f"Extraction: {record.get('extraction_status', 'unknown')}")

            desc_present = bool(
                (record.get("current_description_body") or "").strip()
                or record.get("current_description_bullets", [])
            )
            key_feature_count = len(record.get("current_key_features", []))
            reviews = record.get("reviews_summary", {})
            avg = reviews.get("average_rating")
            ratings = reviews.get("ratings_count")
            review_summary = f"{avg} avg / {ratings} ratings" if avg is not None and ratings is not None else "No review summary"
            st.caption(
                f"Description: {'Yes' if desc_present else 'No'} | "
                f"Key Features: {key_feature_count} | "
                f"Reviews: {review_summary}"
            )
            _render_local_image_analysis(record)

            img_cols = st.columns(min(6, max(1, len(image_models))))
            for i, image in enumerate(image_models):
                image_url = image.get("url", "")
                image_index = image.get("index", i)
                image_id = f"{group.get('record_id', '')}|{image_index}"
                include_key = f"{selection_prefix}{image_id}"
                dims_key = f"audit_v2_comp_show_dims_ppt_{image_id}"
                if include_key not in st.session_state:
                    st.session_state[include_key] = False
                if dims_key not in st.session_state:
                    st.session_state[dims_key] = bool(image.get("show_dimensions_in_powerpoint", False))
                with img_cols[i % len(img_cols)]:
                    safe_render_image(image_url, width=95)
                    st.caption(f"Image {i + 1}")
                    st.caption(_format_image_dimensions_for_preview(image))
                    show_dims = st.checkbox("Show dimensions in PowerPoint", key=dims_key)
                    image["show_dimensions_in_powerpoint"] = bool(show_dims)
                    for raw_image in record.get("images", []):
                        if int(raw_image.get("index", -1) or -1) == int(image_index):
                            raw_image["show_dimensions_in_powerpoint"] = bool(show_dims)
                            break
                    include = st.checkbox("Selected", key=include_key)
                    if include:
                        if slide_mode in {"single_pdp", "combined"}:
                            if len(selected_image_ids) < max_slots:
                                selected_image_ids.append(image_id)
                                group_selected_ids.append(image_id)
                            else:
                                st.session_state[include_key] = False
                                limit_blocked = True
                        else:
                            if len(group_selected_ids) < max_slots:
                                selected_image_ids.append(image_id)
                                group_selected_ids.append(image_id)
                            else:
                                st.session_state[include_key] = False
                                limit_blocked = True
                    st.caption(f"Source URL: {image_url}")
        selected_ids_by_record[record_id] = list(group_selected_ids)
        group_image_orders[record_id] = {image_id: order for order, image_id in enumerate(group_selected_ids, start=1)}

    image_orders: dict[str, int] = {image_id: 0 for image_id in all_image_ids}
    for display_order, image_id in enumerate(selected_image_ids, start=1):
        image_orders[image_id] = display_order
        meta = image_meta.get(image_id, {})
        record = meta.get("record", {})
        selected_rows.append(
            {
                "Display Order": display_order,
                "Competitor Title": record.get("product_title", "-"),
                "Brand": record.get("brand", "-"),
                "Item ID": record.get("item_id", "-") or "-",
                "Image Label": meta.get("image_label", "-"),
                "Image URL": meta.get("url", ""),
                "Source URL": record.get("source_url", ""),
                "_image_id": image_id,
            }
        )

    if limit_blocked:
        if slide_mode in {"single_pdp", "combined"}:
            st.warning("You can select up to 10 competitor images in combined mode.")
        else:
            st.warning("You can select up to 10 competitor images per PDP group in multi-slide mode.")

    selected_count = len(selected_image_ids)
    st.session_state["audit_competitor_image_orders"] = image_orders
    st.session_state["audit_competitor_combined_image_orders"] = dict(image_orders)
    st.session_state["audit_competitor_multi_image_orders_by_record"] = dict(group_image_orders)
    st.session_state["audit_competitor_selected_image_ids_by_record"] = dict(selected_ids_by_record)
    st.session_state["audit_competitor_group_summary"] = [
        {
            "record_id": str(group.get("record_id", "")),
            "entry_idx": int(group.get("entry_idx", 0)),
            "source_url": str((group.get("record", {}) or {}).get("source_url", "")),
            "product_title": str((group.get("record", {}) or {}).get("product_title", "")),
            "image_ids": [
                f"{group.get('record_id', '')}|{int(image.get('index', i))}"
                for i, image in enumerate(group.get("images", []))
            ],
        }
        for group in grouped_entries
    ]
    st.session_state["audit_competitor_assignments"] = build_competitor_assignments(entries, image_orders)

    if slide_mode in {"single_pdp", "combined"}:
        st.caption(f"Selected: {selected_count} / {max_slots}")
    else:
        st.caption(f"Selected Total Across Groups: {selected_count}")

    st.session_state["audit_competitor_mode_payload"] = {
        "mode": slide_mode,
        "has_multiple_pdps": has_multiple_pdps,
        "max_images_per_slide": max_slots,
        "combined_image_orders": st.session_state.get("audit_competitor_combined_image_orders", {}),
        "selected_image_ids_by_record": st.session_state.get("audit_competitor_selected_image_ids_by_record", {}),
        "multi_image_orders_by_record": st.session_state.get("audit_competitor_multi_image_orders_by_record", {}),
        "group_summary": st.session_state.get("audit_competitor_group_summary", []),
    }

    if selected_rows:
        ordered_df = (
            pd.DataFrame(selected_rows)
            .sort_values(by=["Display Order", "Competitor Title", "Image Label"])
            .reset_index(drop=True)
        )
        preview_df = ordered_df[
            ["Display Order", "Competitor Title", "Brand", "Item ID", "Image Label", "Image URL"]
        ]
        if slide_mode in {"single_pdp", "combined"}:
            st.caption("Ordered competitor image preview for shared template slots")
        else:
            st.caption("Ordered competitor image preview for active multi-slide selection state")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
    else:
        st.info("No competitor images selected yet.")

def _seed_mock_results_for_products_v2(entries: list[dict]) -> None:
    seeded_ids = []
    for entry in entries:
        entry_id = entry["entry_id"]
        seeded_ids.append(entry_id)
        generated_payload = generate_mvp_outputs_for_primary_entry(entry)
        entry["generated_outputs"] = generated_payload
        if is_output_shell_empty(entry.get("edited_outputs")):
            entry["edited_outputs"] = {
                "image_recommendations": list(generated_payload.get("image_recommendations", [])),
                "recommended_title": generated_payload.get("recommended_title", ""),
                "description_recommendations": list(generated_payload.get("description_recommendations", [])),
                "key_features_recommendations": list(generated_payload.get("key_features_recommendations", [])),
                "top_priority_fixes": list(generated_payload.get("top_priority_fixes", [])),
            }
        entry["status"] = "generated_mvp"

    st.session_state["audit_results_seeded_for"] = seeded_ids


def _refresh_audit_export_plan_v2() -> None:
    entries = st.session_state.get("audit_primary_entries", []) or []
    competitor_assignments: list[dict[str, Any]] = []
    competitor_records = st.session_state.get("audit_competitor_entries", []) or []
    audit_record = st.session_state.get("audit_result_record", {}) or {}

    if audit_record:
        audit_record["product_audit_entries"] = entries
        audit_record["competitor_graphics_assignments"] = competitor_assignments
        audit_record["competitor_graphics_mode"] = st.session_state.get("audit_competitor_slide_mode", "single_pdp")
        audit_record["competitor_graphics_has_multiple_pdps"] = bool(
            st.session_state.get("audit_competitor_has_multiple_pdps", False)
        )
        audit_record["competitor_graphics_make_multiple_slides"] = bool(
            st.session_state.get("audit_competitor_make_multiple_slides", False)
        )
        audit_record["competitor_graphics_mode_payload"] = st.session_state.get("audit_competitor_mode_payload", {})
        audit_record["client_has_brand_shop"] = bool(
            st.session_state.get("audit_client_has_brand_shop", True)
        )

    st.session_state["audit_export_plan"] = build_audit_export_plan(
        audit_record=audit_record,
        primary_entries=entries,
        competitor_assignments=competitor_assignments,
        competitor_records=competitor_records,
        search_evidence=st.session_state.get("audit_search_evidence", {}),
        brand_shop_evidence=st.session_state.get("audit_brand_shop_evidence", {}),
    )


def render_generate_audit_v2() -> None:
    entries = st.session_state.get("audit_primary_entries", [])
    with st.container(border=True):
        st.markdown("### Generate Audit")
        st.caption("Build the strategic audit export plan from the combined evidence.")
        generate = st.button(
            "Generate Audit",
            key="audit_v2_generate",
            type="primary",
            disabled=not entries,
        )
        if not entries:
            st.info("Add at least one primary product entry before generating the audit.")
        if generate:
            _seed_mock_results_for_products_v2(entries)
            st.session_state["audit_result_record"] = create_audit_result_record(
                client_name=st.session_state.get("audit_client_name", ""),
                retailer=st.session_state.get("audit_retailer", ""),
                audit_date=str(st.session_state.get("audit_date", date.today())),
                status="generated_mvp",
                product_audit_entries=entries,
                competitor_graphics_assignments=[],
            )
            _refresh_audit_export_plan_v2()
            st.session_state["audit_generated"] = True


def render_mocked_audit_results_v2() -> None:
    if not st.session_state.get("audit_generated"):
        return

    entries = st.session_state.get("audit_primary_entries", [])
    if not entries:
        return

    st.markdown("### Audit Results")
    for idx, entry in enumerate(entries, start=1):
        entry_id = entry["entry_id"]
        record = entry.get("cached_record", {})
        image_urls = [img.get("url", "") for img in record.get("images", []) if img.get("url")]
        raw_selected_index = entry.get("selected_primary_image", {}).get("image_index", None)
        selected_index = int(raw_selected_index) if isinstance(raw_selected_index, int) else 0
        selected_index = min(selected_index, max(0, len(image_urls) - 1))
        selected_url = image_urls[selected_index] if image_urls else None

        title = str(entry.get("product_title", "") or "Untitled Product").strip()
        item_id = str(entry.get("item_id", "") or "-").strip()
        with st.expander(f"Product {idx} Outputs: {title} | Item ID: {item_id}", expanded=False):
            st.markdown(f"#### Product Audit Entry {idx}")
            st.markdown("##### Product Summary")
            s1, s2, s3 = st.columns([2, 1, 1])
            with s1:
                st.write(f"**Product Title:** {entry.get('product_title', '-')}")
            with s2:
                st.write(f"**Item ID:** {entry.get('item_id', '-')}")
            with s3:
                st.write(f"**Selected Primary Image:** Image {selected_index + 1 if image_urls else 0}")
            if selected_url:
                st.image(selected_url, width=220)

            generated = entry.get("generated_outputs", {}) or {}
            edited = entry.get("edited_outputs", {}) or {}
            active = edited if not is_output_shell_empty(edited) else generated

            key_image = f"audit_v2_result_image_recommendations_{entry_id}"
            key_title = f"audit_v2_result_recommended_title_{entry_id}"
            key_desc = f"audit_v2_result_description_recommendations_{entry_id}"
            key_feat = f"audit_v2_result_key_features_recommendations_{entry_id}"
            key_fix = f"audit_v2_result_top_priority_fixes_{entry_id}"

            if key_image not in st.session_state:
                st.session_state[key_image] = "\n".join(active.get("image_recommendations", []))
            if key_title not in st.session_state:
                st.session_state[key_title] = active.get("recommended_title", "")
            if key_desc not in st.session_state:
                st.session_state[key_desc] = "\n".join(active.get("description_recommendations", []))
            if key_feat not in st.session_state:
                st.session_state[key_feat] = "\n".join(active.get("key_features_recommendations", []))
            if key_fix not in st.session_state:
                st.session_state[key_fix] = "\n".join(active.get("top_priority_fixes", []))

            st.markdown("##### Image Recommendations")
            st.text_area("Image Recommendations", key=key_image, height=120)

            st.markdown("##### Content Optimizations")
            st.text_input("Recommended Title", key=key_title)
            st.text_area("Description Recommendations", key=key_desc, height=120)
            st.text_area("Key Features Recommendations", key=key_feat, height=120)

            st.markdown("##### Top Priority Fixes")
            st.text_area("Top Priority Fixes", key=key_fix, height=100)

            entry["edited_outputs"] = {
                "image_recommendations": [ln.strip() for ln in st.session_state[key_image].splitlines() if ln.strip()],
                "recommended_title": st.session_state[key_title].strip(),
                "description_recommendations": [ln.strip() for ln in st.session_state[key_desc].splitlines() if ln.strip()],
                "key_features_recommendations": [ln.strip() for ln in st.session_state[key_feat].splitlines() if ln.strip()],
                "top_priority_fixes": [ln.strip() for ln in st.session_state[key_fix].splitlines() if ln.strip()],
            }

def render_audit_powerpoint_export_v2() -> None:
    if not st.session_state.get("audit_generated"):
        st.info("Generate audit outputs before creating the audit PowerPoint.")
        return

    _refresh_audit_export_plan_v2()
    plan = st.session_state.get("audit_export_plan", {}) or {}
    with st.expander("Export Mapping Preview", expanded=False):
        if not plan:
            st.caption("No export mapping plan available yet.")
        else:
            summary = plan.get("summary", {})
            st.write(
                f"Included primary entries: {summary.get('included_primary_entry_count', 0)} | "
                f"Client PDPs: {len(st.session_state.get('audit_primary_entries', []) or [])} | "
                f"Competitor PDPs: {len(st.session_state.get('audit_competitor_entries', []) or [])}"
            )
            st.caption(
                "The strategic deck uses combined HTML evidence for Slides 2, 3, 4, 5, and 6."
            )
            compact = {
                "audit_metadata": plan.get("audit_metadata", {}),
                "summary": {
                    "included_primary_entry_count": summary.get("included_primary_entry_count", 0),
                    "client_pdp_count": len(st.session_state.get("audit_primary_entries", []) or []),
                    "competitor_pdp_count": len(st.session_state.get("audit_competitor_entries", []) or []),
                },
                "search_evidence_counts": {
                    "current": len((plan.get("search_evidence", {}) or {}).get("current", []) or []),
                    "benchmark": len((plan.get("search_evidence", {}) or {}).get("benchmark", []) or []),
                },
                "brand_shop_evidence_counts": {
                    "client": len((plan.get("brand_shop_evidence", {}) or {}).get("client", []) or []),
                    "competitor": len((plan.get("brand_shop_evidence", {}) or {}).get("competitor", []) or []),
                },
            }
            st.json(compact)

    _render_slide2_summary_preview_v2(plan)
    _render_slide3_search_benchmark_preview_v2(plan)
    _render_slide4_finding_preview_v2(plan)
    _render_slide5_brand_shop_preview_v2(plan)
    _render_slide6_visibility_preview_v2(plan)

    included_count = int((plan.get("summary", {}) or {}).get("included_primary_entry_count", 0))
    if included_count <= 0:
        st.info("Include at least one primary product entry to generate the audit PowerPoint.")
        return
    include_slide_9 = st.checkbox(
        "Include Slide 9 (Walmart Cash Program Visibility)",
        value=False,
        key="audit_include_slide_9",
    )
    
    try:
        template_path = os.path.join("templates", "Audit_Template_New.pptx")
        if not os.path.exists(template_path):
            raise FileNotFoundError(template_path)
    except Exception as exc:
        st.error(f"Audit template not found: {exc}")
        return

    if st.button("Generate Audit PowerPoint", key="audit_v2_generate_ppt", type="primary"):
        try:
            competitor_records = st.session_state.get("audit_competitor_entries", []) or []
            ppt_bytes = generate_new_audit_powerpoint_from_template(
                export_plan=plan,
                template_path=template_path,
                include_slide_9=include_slide_9,
                competitor_records=competitor_records,
            )
            st.session_state["audit_ppt_bytes"] = ppt_bytes
            st.session_state["audit_ppt_filename"] = (
                f"audit_{st.session_state.get('audit_client_name', 'client').strip() or 'client'}"
                f"_{st.session_state.get('audit_date', date.today())}.pptx"
            ).replace(" ", "_")
            st.success("Audit PowerPoint generated.")
        except Exception as exc:
            st.error(f"Failed to generate audit PowerPoint: {exc}")

    ppt_bytes = st.session_state.get("audit_ppt_bytes")
    if ppt_bytes:
        st.download_button(
            "Download Audit PowerPoint",
            data=ppt_bytes,
            file_name=st.session_state.get("audit_ppt_filename", "audit_export.pptx"),
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key="audit_v2_download_ppt",
        )


def render_content_auditing() -> None:
    _init_audit_state()
    st.session_state["audit_template_version"] = "New Strategic Template"

    top_l, top_r = st.columns([1, 6])
    with top_l:
        st.button("Back to Hub", key="audit_back_home", on_click=go_home)
    with top_r:
        st.title("Content Auditing")
        st.caption("Workspace for generating the strategic audit deck from the combined HTML report.")

    st.header("Audit Setup")
    render_audit_setup()
    st.checkbox(
        "Client has a Walmart Brand Shop",
        value=True,
        key="audit_client_has_brand_shop",
    )
    st.divider()

    st.header("Audit Evidence Sheet Upload")
    render_combined_strategic_audit_upload_v2()
    render_strategic_evidence_summary_v2()
    st.divider()

    render_generate_audit_v2()
    st.divider()

    st.header("PowerPoint Export")
    render_audit_powerpoint_export_v2()


def _coerce_conversion_pp(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    nonnull = s[s.notna()]
    if len(nonnull):
        share_gt1 = (nonnull > 1.0).mean()
        if share_gt1 < 0.8:
            s = s * 100.0
    return s.clip(0.0, 100.0).astype(float)


def _excel_mean_conversion_pp(series: pd.Series) -> float:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    s = pd.to_numeric(s, errors="coerce")
    nonnull = s.dropna()
    if nonnull.empty:
        return 0.0
    if (nonnull <= 1.0).mean() >= 0.8:
        nonnull = nonnull * 100.0
    return float(nonnull.mean())


def _norm_hdr_front(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return re.sub(r"[^0-9a-z]+", "", s)


AD_ALIASES_FRONT = {
    "Ad Spend": {
        "ad spend", "ad_spend", "spend", "ad spend ($)", "adspend",
        "total spend", "total ad spend"
    },
    "Conversion Rate": {
        "conversion rate", "conversion_rate", "conv rate", "conv_rate", "conversionrate",
        "conversion rate - 14 day", "conversion rate - 14 day", "conversion rate 14 day", "conversionrate14day"
    },
    "ROAS": {
        "roas", "return on ad spend", "returnonadspend",
        "roas - 14 day", "roas - 14 day", "roas 14 day", "roas14day"
    },
}


def resolve_ad_columns_front(cols):
    lc_to_actual = {_norm_hdr_front(c): c for c in cols}
    mapping = {"Ad Spend": None, "Conversion Rate": None, "ROAS": None}
    for canon, aliases in AD_ALIASES_FRONT.items():
        key = _norm_hdr_front(canon)
        if key in lc_to_actual:
            mapping[canon] = lc_to_actual[key]
            continue
        for a in aliases:
            k = _norm_hdr_front(a)
            if k in lc_to_actual:
                mapping[canon] = lc_to_actual[k]
                break
    return mapping


def render_content_reporting() -> None:
    top_l, top_r = st.columns([1, 6])
    with top_l:
        st.button("Back to Hub", key="reporting_back_home", on_click=go_home)
    with top_r:
        st.title("Weekly Content Reporting")
        st.caption("Upload your data, manage clients, and export reports (1P & 3P).")

    _1p = {
        "has_data": False,
        "df": None,
        "metrics": None,
        "top5": None,
        "below": None,
        "client_name": None,
        "rpt_date": None,
        "notes": None,
    }

    st.markdown("## 1P - Weekly Content Reporting")
    col_left, col_right = st.columns([1.3, 1.7], gap="large")
    with col_left:
        uploaded_1p = st.file_uploader("Upload excel or csv", type=["xlsx", "csv"], key="uploader_1p")
    with col_right:
        client_name_1p = st.text_input("Client name", value="Country Fresh", key="client_name_1p")
        rpt_date_obj_1p = st.date_input("Report Date", value=date.today(), key="rpt_date_1p")
        rpt_date_str_1p = fmt_mdy(rpt_date_obj_1p)
        notes_1p = st.text_area("Content Notes", value="", height=100, key="notes_1p")

    st.divider()

    if uploaded_1p:
        df_1p = load_dataframe(uploaded_1p)
        _1p.update({"has_data": True, "df": df_1p, "client_name": client_name_1p, "rpt_date": rpt_date_str_1p, "notes": notes_1p})

        st.subheader("Data Preview (1P)")
        st.dataframe(df_1p.head(10), height=200, use_container_width=True)

        m = compute_metrics(df_1p)
        _1p["metrics"] = m

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{m['above']}/{m['total']} ({m['pct_above']:.1f}%) products >= {int(m['threshold'])}%")
            pie_buf = make_pie_bytes(m)
            st.image(pie_buf, caption="Score Distribution", use_column_width=False)
        with col2:
            st.subheader("Key Metrics (1P)")
            st.write(f"- **Average CQS:** {m['avg_cqs']:.1f}%")
            st.write(f"- **SKUs >= {int(m['threshold'])}%:** {m['above']}")
            st.write(f"- **SKUs < {int(m['threshold'])}%:** {m['below']}")
            st.write(f"- **Buybox Ownership:** {m['buybox']:.1f}%")

        st.markdown("---")

        st.subheader("Top 5 SKUs by Content Quality Score (1P)")
        top5_1p = get_top_skus(df_1p)
        _1p["top5"] = top5_1p
        st.dataframe(top5_1p, height=200, use_container_width=True)

        st.subheader(f"SKUs Below {int(m['threshold'])}% (1P)")
        below_1p = get_skus_below(df_1p)
        _1p["below"] = below_1p
        st.dataframe(below_1p, height=300, use_container_width=True)

        csv_data = below_1p.to_csv(index=False).encode("utf-8")
        st.download_button("Download SKUs Below CSV (1P)", data=csv_data, file_name="skus_below_1p.csv", mime="text/csv")

        st.markdown("### Export 1P PDF")
        if st.button("Generate 1P PDF", key="export_pdf_1p"):
            pdf_bytes = generate_full_report(uploaded_1p, client_name_1p, rpt_date_str_1p, notes_1p)
            st.success("1P PDF ready!")
            st.download_button("Download 1P PDF", data=pdf_bytes, file_name="1p.pdf", mime="application/pdf")
    else:
        st.info("Upload a 1P data file to see preview and metrics.")

    st.divider()

    st.markdown("## 3P - Weekly Content Reporting")
    mode_3p = st.radio("Mode", ["Catalog", "Managed"], horizontal=True, key="mode_3p")

    c_hdr1, c_hdr2 = st.columns([2, 1])
    with c_hdr1:
        client_name_3p = st.text_input("Client Name (3P)", value="", key="client_name_3p")
    with c_hdr2:
        report_date_3p = st.date_input("Report Date (3P)", value=date.today(), key="report_date_3p")

    managed_file = None
    if mode_3p == "Managed":
        managed_file = st.file_uploader(
            "Managed SKUs (IDs/Names list)",
            type=["xlsx", "csv"],
            key="uploader_3p_managed",
            help="Upload a file containing the SKU/Item ID list to limit calculations to managed items."
        )

    st.markdown("**Upload excel(s) or csv - these will each be labeled**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        file_x = st.file_uploader("Upload Item Sales Report", type=["xlsx", "csv"], key="uploader_3p_x")
    with c2:
        file_y = st.file_uploader("Upload Inventory Report", type=["xlsx", "csv"], key="uploader_3p_y")
    with c3:
        file_z = st.file_uploader("Upload Search Insights", type=["xlsx", "csv"], key="uploader_3p_z")
    with c4:
        file_adv = st.file_uploader("Upload Advertising Report", type=["xlsx", "csv"], key="uploader_3p_adv")

    st.markdown("### Manual Content (renders in the PDF)")
    c_txt1, c_txt2, c_txt3, c_txt4 = st.columns(4)
    with c_txt1:
        manual_top_skus = st.text_area("Top SKUs (Item Sales) - manual", height=150, key="manual_top_skus")
    with c_txt2:
        manual_inventory_callouts = st.text_area("Key Callouts (Inventory) - manual", height=150, key="manual_inventory_callouts")
    with c_txt3:
        manual_search_highlights = st.text_area("Highlights (Search Insights) - manual", height=150, key="manual_search_highlights")
    with c_txt4:
        manual_adv_notes = st.text_area("Notes (Advertising) - manual", height=150, key="manual_adv_notes")

    managed_ids, managed_names = set(), set()
    if mode_3p == "Managed" and managed_file is not None:
        try:
            managed_ids, managed_names = load_managed_keys(managed_file)
            st.caption(f"Mode: **Managed** - Loaded {len(managed_ids) or len(managed_names)} SKUs")
        except Exception as e:
            st.error(f"Managed SKUs file error: {e}")
    elif mode_3p == "Managed":
        st.info("Mode: **Managed** - Please upload the Managed SKUs (IDs/Names list).")

    preview_cols = st.columns(4)

    with preview_cols[0]:
        st.caption("Data Preview (Item Sales)")
        if file_x:
            try:
                df_x = load_item_sales(file_x)
                df_view = df_x
                if mode_3p == "Managed" and (managed_ids or managed_names):
                    df_view, stats_x = filter_by_managed(df_x, managed_ids, managed_names)
                    st.write(f"Showing {stats_x['matched']} of {stats_x['total']} (managed subset)")
                    if stats_x["matched"] == 0 and stats_x["unmatched_sample"]:
                        st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_x['unmatched_sample'])}")
                st.dataframe(df_view.head(15), height=260, use_container_width=True)

                units_total = int(pd.to_numeric(df_view.get("Units Sold"), errors="coerce").fillna(0).sum()) if not df_view.empty else 0
                sales_total = float(pd.to_numeric(df_view.get("Auth Sales"), errors="coerce").sum()) if not df_view.empty else 0.0
                avg_conv_pct = _excel_mean_conversion_pp(df_view.get("Item conversion", pd.Series(dtype=float))) if not df_view.empty else 0.0

                st.markdown("**Calculated (Item Sales):**")
                st.write(f"- Units Sold: {units_total:,}")
                st.write(f"- Auth Sales: ${sales_total:,.2f}")
                st.write(f"- Avg Conversion: {avg_conv_pct:.2f}%")
            except Exception as e:
                st.error(
                    "Item Sales file does not match required columns. "
                    "Expecting (canonical): Item ID, Item Name, Orders, Units Sold, "
                    "Auth Sales, Item pageviews, Item conversion.\n\n"
                    f"Details: {e}"
                )
        else:
            st.info("Upload Item Sales Report to preview.")

    with preview_cols[1]:
        st.caption("Data Preview (Inventory)")
        if file_y:
            try:
                df_y = load_inventory(file_y)
                df_view = df_y
                if mode_3p == "Managed" and (managed_ids or managed_names):
                    df_view, stats_y = filter_by_managed(df_y, managed_ids, managed_names)
                    st.write(f"Showing {stats_y['matched']} of {stats_y['total']} (managed subset)")
                    if stats_y["matched"] == 0 and stats_y["unmatched_sample"]:
                        st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_y['unmatched_sample'])}")
                st.dataframe(df_view.head(15), height=260, use_container_width=True)

                status_col = "Status" if "Status" in df_view.columns else ("Stock status" if "Stock status" in df_view.columns else None)
                in_stock_rate_pct = 0
                raw_in = raw_total = raw_oos = raw_risk = 0
                if status_col and not df_view.empty:
                    status_norm = df_view[status_col].astype(str).str.strip().str.lower()
                    raw_total = len(status_norm)
                    raw_in = int((status_norm == "in stock").sum())
                    raw_oos = int((status_norm == "out of stock").sum())
                    raw_risk = int((status_norm == "at risk").sum())
                    in_stock_rate_pct = int(round((raw_in / raw_total) * 100)) if raw_total else 0

                st.markdown("**Calculated (Inventory):**")
                st.write(f"- In-stock rate: {in_stock_rate_pct}%  _(preview: {raw_in}/{raw_total} in stock)_")
                st.write(f"- OOS SKUs: {raw_oos}")
                st.write(f"- At-risk SKUs: {raw_risk}")
            except Exception as e:
                st.error(
                    "Inventory file does not match required columns. "
                    "Expecting (canonical): Item ID, Item Name, Daily sales, Daily units sold, Stock status.\n\n"
                    f"Details: {e}"
                )
        else:
            st.info("Upload Inventory Report to preview.")

    with preview_cols[2]:
        st.caption("Data Preview (Search Insights)")
        if file_z:
            try:
                df_z = load_search_insights(file_z)
                df_view = df_z
                if mode_3p == "Managed" and (managed_ids or managed_names):
                    df_view, stats_z = filter_by_managed(df_z, managed_ids, managed_names)
                    st.write(f"Showing {stats_z['matched']} of {stats_z['total']} (managed subset)")
                    if stats_z["matched"] == 0 and stats_z["unmatched_sample"]:
                        st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_z['unmatched_sample'])}")
                st.dataframe(df_view.head(15), height=260, use_container_width=True)

                impr = pd.to_numeric(df_view.get("Impressions Rank"), errors="coerce").dropna()
                avg_impr_rank = int(round(float(impr.mean()))) if len(impr) else 0
                top10_impr = int((impr <= 10).sum())
                sales_rank = pd.to_numeric(df_view.get("Sales Rank"), errors="coerce")
                top10_sales = int((sales_rank <= 10).sum()) if "Sales Rank" in df_view.columns else 0

                st.markdown("**Calculated (Search Insights):**")
                st.write(f"- Avg Impressions Rank: {avg_impr_rank}")
                st.write(f"- SKUs in Top 10 Impressions: {top10_impr}")
                st.write(f"- SKUs in Top 10 Sales Rank: {top10_sales}")
            except Exception as e:
                st.error(
                    "Search Insights file does not match required columns. "
                    "Expecting: Item ID, Item Name, Impressions Rank, Clicks Rank, "
                    "Added to Cart Rank, Sales Rank.\n\n"
                    f"Details: {e}"
                )
        else:
            st.info("Upload Search Insights to preview.")

    with preview_cols[3]:
        st.caption("Data Preview (Advertising)")
        if file_adv:
            try:
                def _norm_hdr_detect(s: str) -> str:
                    s = unicodedata.normalize("NFKC", str(s or "")).lower()
                    return re.sub(r"[^0-9a-z]+", "", s)

                expected_hdrs = {"adspend", "conversionrate14day", "roas14day"}
                _data = file_adv.getvalue()
                _ext = os.path.splitext(file_adv.name)[1].lower()
                hdr_idx = None
                try:
                    if _ext in (".xls", ".xlsx"):
                        tmp = pd.read_excel(io.BytesIO(_data), header=None, nrows=25)
                    else:
                        tmp = pd.read_csv(io.BytesIO(_data), header=None, nrows=25, engine="python", encoding="utf-8", on_bad_lines="skip")
                    for i in range(min(20, len(tmp))):
                        row_norm = {_norm_hdr_detect(x) for x in tmp.iloc[i].tolist()}
                        if expected_hdrs.issubset(row_norm):
                            hdr_idx = i
                            break
                except Exception:
                    hdr_idx = None

                if hdr_idx is not None:
                    if _ext in (".xls", ".xlsx"):
                        df_adv_raw = pd.read_excel(io.BytesIO(_data), header=hdr_idx)
                    else:
                        df_adv_raw = pd.read_csv(io.BytesIO(_data), header=hdr_idx, engine="python", encoding="utf-8", on_bad_lines="skip")
                else:
                    df_adv_raw = load_dataframe(file_adv)

                cols_map = resolve_ad_columns_front(df_adv_raw.columns)

                def to_currency(s: pd.Series) -> pd.Series:
                    return pd.to_numeric(
                        s.astype(str)
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.replace(" ", "", regex=False)
                        .str.strip(),
                        errors="coerce"
                    ).fillna(0.0)

                def to_roas(s: pd.Series) -> pd.Series:
                    cleaned = (
                        s.astype(str)
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.replace("x", "", case=False, regex=False)
                        .str.strip()
                    )
                    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

                def to_percent_vals(s: pd.Series) -> pd.Series:
                    vals = (
                        s.astype(str)
                        .str.replace("%", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.strip()
                    )
                    vals = pd.to_numeric(vals, errors="coerce")
                    nonnull = vals.dropna()
                    if not nonnull.empty and (nonnull <= 1.0).mean() >= 0.8:
                        vals = vals * 100.0
                    return vals

                spend = to_currency(df_adv_raw[cols_map["Ad Spend"]]) if cols_map["Ad Spend"] else pd.Series([], dtype=float)
                roas = to_roas(df_adv_raw[cols_map["ROAS"]]) if cols_map["ROAS"] else pd.Series([], dtype=float)
                conv = to_percent_vals(df_adv_raw[cols_map["Conversion Rate"]]) if cols_map["Conversion Rate"] else pd.Series([], dtype=float)

                adv_spend_total = float(spend.sum()) if len(spend) else 0.0
                adv_roas_total = float(roas.sum()) if len(roas) else 0.0
                conv_nonnull = conv.dropna()
                adv_conv_avg = float(conv_nonnull.mean()) if len(conv_nonnull) else 0.0

                keep_cols = [c for c in [cols_map["Ad Spend"], cols_map["Conversion Rate"], cols_map["ROAS"]] if c]
                if keep_cols:
                    st.dataframe(df_adv_raw[keep_cols].head(15), height=260, use_container_width=True)
                else:
                    st.info("Could not find expected Advertising columns; showing calculated zeros.")

                st.markdown("**Calculated (Advertising):**")
                st.write(f"- Total Ad Spend: ${adv_spend_total:,.2f}")
                st.write(f"- Total ROAS: ${adv_roas_total:,.2f}")
                st.write(f"- Avg Conversion Rate: {adv_conv_avg:.2f}%")
            except Exception as e:
                st.error(f"Advertising file error: {e}")
        else:
            st.info("Upload Advertising Report to preview.")

    st.markdown("### Export 3P PDF")
    export_disabled = (
        any(v is None for v in [file_x, file_y, file_z, file_adv]) or
        (mode_3p == "Managed" and managed_file is None)
    )
    if export_disabled:
        st.info("Upload Item Sales, Inventory, Search Insights, and Advertising (and Managed list if Managed mode) to enable export.")

    if st.button("Generate 3P Dashboard PDF", key="export_pdf_3p", disabled=export_disabled):
        try:
            pdf3 = generate_3p_report(
                item_sales_src=file_x,
                inventory_src=file_y,
                search_insights_src=file_z,
                managed_src=managed_file,
                mode=mode_3p.lower(),
                client_name=client_name_3p or "Client",
                report_date=fmt_mdy(report_date_3p) if "fmt_mdy" in globals() else report_date_3p.strftime("%B %d, %Y"),
                top_skus_text=manual_top_skus or "",
                inventory_callouts_text=manual_inventory_callouts or "",
                search_highlights_text=manual_search_highlights or "",
                advertising_src=file_adv,
                advertising_notes_text=manual_adv_notes,
                logo_path=None,
            )
        except TypeError:
            pdf3 = generate_3p_report(
                data_src=None,
                client_name=client_name_3p or "Client",
                report_date=fmt_mdy(report_date_3p) if "fmt_mdy" in globals() else report_date_3p.strftime("%B %d, %Y"),
                logo_path=None,
            )

        if not pdf3:
            st.error("3P generator returned no bytes. Ensure main.py defines the updated generate_3p_report(...).")
        else:
            st.success("3P Dashboard PDF ready!")
            st.download_button("Download 3P PDF", data=pdf3, file_name="dashboard_3p.pdf", mime="application/pdf")

    st.divider()

    st.header("Save This Preview (Snapshot)")
    db_clients = get_clients()
    db_client_names = [c["client_name"] for c in db_clients]
    db_client_ids = {c["client_name"]: c["client_id"] for c in db_clients}

    if not db_clients:
        st.warning("Add at least one client group below before saving previews.")
    elif not _1p["has_data"]:
        st.info("Load 1P data to enable saving a snapshot.")
    else:
        db_selected_client_name = st.selectbox(
            "Assign this snapshot to client group:",
            db_client_names,
            key="db_client_dropdown_save",
        )
        db_selected_client_id = db_client_ids[db_selected_client_name]
        preview_name = st.text_input(
            "Preview Name (for saving this report snapshot)",
            value=f"{db_selected_client_name} - {_1p['rpt_date']}",
            key="preview_name_1p",
        )
        if st.button("Save Preview", key="save_preview_btn"):
            m = _1p["metrics"]
            top5 = _1p["top5"]
            below = _1p["below"]
            data_json = {
                "metrics": m,
                "top5": top5.to_dict(orient="records") if top5 is not None else [],
                "skus_below": below.to_dict(orient="records") if below is not None else [],
                "client_notes": _1p["notes"],
                "pdf_client_name": _1p["client_name"],
                "pdf_report_date": _1p["rpt_date"],
            }
            add_preview(
                client_id=db_selected_client_id,
                preview_name=preview_name,
                report_date=_1p["rpt_date"],
                notes=_1p["notes"] or "",
                data_json=data_json,
            )
            st.success("Preview saved! See saved previews below.")

    st.divider()

    st.header("View Saved Previews")
    view_clients = get_clients()
    view_client_names = [c["client_name"] for c in view_clients]
    view_client_ids = {c["client_name"]: c["client_id"] for c in view_clients}

    if not view_client_names:
        st.info("No client groups exist. Add one below.")
    else:
        view_selected_client_name = st.selectbox(
            "Select Client Group to View Previews",
            view_client_names,
            key="view_client_dropdown",
        )
        view_selected_client_id = view_client_ids[view_selected_client_name]
        previews = get_previews_for_client(view_selected_client_id)

        if not previews:
            st.info(f"No saved previews yet for {view_selected_client_name}.")
        else:
            options = [f"{p['preview_name']} ({p['report_date']})" for p in previews]
            option_map = {f"{p['preview_name']} ({p['report_date']})": p for p in previews}
            selected_option = st.selectbox("Select a saved preview to view", options, key="saved_preview_select")
            selected_preview = option_map[selected_option]

            st.markdown(f"**Report Date:** {selected_preview['report_date']}")
            st.markdown(f"**Preview Name:** {selected_preview['preview_name']}")
            st.markdown(f"**Notes:** {selected_preview['notes']}")
            st.markdown(f"**Date Saved:** {selected_preview['date_created']}")

            m2 = selected_preview["data_json"]["metrics"]
            st.subheader("Saved Metrics")
            st.write(f"- **Average CQS:** {m2['avg_cqs']}%")
            st.write(f"- **SKUs >= {int(m2['threshold'])}%:** {m2['above']}")
            st.write(f"- **SKUs < {int(m2['threshold'])}%:** {m2['below']}")
            st.write(f"- **Buybox Ownership:** {m2['buybox']}%")

            st.subheader("Top 5 SKUs by Content Quality Score (Saved)")
            top5_df = pd.DataFrame(selected_preview["data_json"]["top5"])
            st.dataframe(top5_df, height=200, use_container_width=True)

            st.subheader(f"SKUs Below {int(m2['threshold'])}% (Saved)")
            below_df = pd.DataFrame(selected_preview["data_json"]["skus_below"])
            st.dataframe(below_df, height=300, use_container_width=True)

            if st.button("Delete This Preview", key="delete_preview_btn"):
                delete_preview(selected_preview["preview_id"])
                st.success("Preview deleted.")
                st.rerun()

    st.divider()

    st.header("Manage Clients")
    clients_bottom = get_clients()
    if clients_bottom:
        st.write("Existing Clients:", ", ".join(client["client_name"] for client in clients_bottom))
    else:
        st.write("_No clients defined yet._")

    new_client_bottom = st.text_input("New Client name", key="new_client_bottom")
    if st.button("Add Client", key="add_client_bottom"):
        if new_client_bottom and new_client_bottom not in [c["client_name"] for c in clients_bottom]:
            add_client(new_client_bottom)
            st.rerun()
        else:
            st.warning("Enter a unique client name.")

    if clients_bottom:
        client_names_bottom = [c["client_name"] for c in clients_bottom]
        name_to_id_bottom = {c["client_name"]: c["client_id"] for c in clients_bottom}
        to_delete_bottom = st.selectbox("Client to delete", client_names_bottom, key="del_client_bottom")
        if st.button("Delete Client", key="delete_client_bottom"):
            delete_client(name_to_id_bottom[to_delete_bottom])
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Soapbox eCommerce and Content Hub", layout="wide")
    init_db()
    render_branding()

    hub_from_query = st.query_params.get("hub")
    normalized_from_query = HUB_QUERY_TO_VIEW.get(hub_from_query)
    if normalized_from_query is not None:
        st.session_state["hub_view"] = normalized_from_query
        st.query_params.clear()
        st.rerun()

    if "hub_view" not in st.session_state:
        st.session_state["hub_view"] = VIEW_HOME
    else:
        st.session_state["hub_view"] = HUB_QUERY_TO_VIEW.get(
            st.session_state["hub_view"],
            st.session_state["hub_view"],
        )

    current = st.session_state["hub_view"]
    if current == VIEW_CONTENT_REPORTING:
        render_content_reporting()
    elif current == VIEW_CONTENT_AUDITING:
        render_content_auditing()
    else:
        st.session_state["hub_view"] = VIEW_HOME
        render_home()


if __name__ == "__main__":
    main()
