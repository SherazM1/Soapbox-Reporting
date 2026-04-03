import os
import re
import sys
import unicodedata
from datetime import date
import io

import pandas as pd
import streamlit as st

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


def go_home() -> None:
    st.session_state["hub_view"] = "home"
    st.rerun()


def go_reporting() -> None:
    st.session_state["hub_view"] = "reporting"
    st.rerun()


def go_auditing() -> None:
    st.session_state["hub_view"] = "auditing"
    st.rerun()


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
          <a class="hub-link-card" href="?hub=reporting">
            <span class="hub-card-arrow">↗</span>
            <h3>Content Reporting</h3>
            <p>Run weekly 1P and 3P reporting, exports, and saved work.</p>
            <div class="hub-card-footer"><span class="hub-open-btn">Open</span></div>
          </a>
          <a class="hub-link-card" href="?hub=auditing">
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
    defaults = {
        "audit_generated": False,
        "audit_extracted_loaded": False,
        "audit_source_method": "Single PDP URL",
        "audit_batch_preview": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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


def render_audit_preferences() -> None:
    with st.container(border=True):
        st.markdown("### Audit Preferences")
        p1, p2 = st.columns(2)
        with p1:
            st.selectbox("Tone", ["Standard", "Aggressive"], key="audit_tone")
        with p2:
            st.selectbox(
                "Audit Goal",
                ["SEO", "Conversion", "Compliance", "Image Improvement"],
                key="audit_goal",
            )
        st.text_area("Notes", key="audit_notes", height=80)


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
    if "audit_competitor_graphics_notes" not in st.session_state:
        st.session_state["audit_competitor_graphics_notes"] = ""
    if "audit_retail_media_optimizations" not in st.session_state:
        st.session_state["audit_retail_media_optimizations"] = ""
    if "audit_competitor_ad_graphics_notes" not in st.session_state:
        st.session_state["audit_competitor_ad_graphics_notes"] = ""


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

    st.divider()
    st.subheader("Competitor Graphics Notes")
    st.caption("Manual/future section for competitor graphics notes.")
    st.text_area("Competitor Graphics Notes", key="audit_competitor_graphics_notes", height=110)

    st.subheader("Retail Media Optimizations")
    st.caption("Manual/future section for retail media optimization ideas.")
    st.text_area("Retail Media Optimizations", key="audit_retail_media_optimizations", height=110)

    st.subheader("Competitor Ad Graphics Notes")
    st.caption("Manual/future section for competitor ad creative observations.")
    st.text_area("Competitor Ad Graphics Notes", key="audit_competitor_ad_graphics_notes", height=110)


def render_content_auditing() -> None:
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
    render_audit_preferences()
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
    if hub_from_query in {"reporting", "auditing"}:
        st.session_state["hub_view"] = hub_from_query
        st.query_params.clear()
        st.rerun()

    if "hub_view" not in st.session_state:
        st.session_state["hub_view"] = "home"

    current = st.session_state["hub_view"]
    if current == "reporting":
        render_content_reporting()
    elif current == "auditing":
        render_content_auditing()
    else:
        render_home()


if __name__ == "__main__":
    main()
