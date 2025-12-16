# streamlit_app.py  — UI reorg per your spec, minimal logic changes

import os
import sys
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# at the top with imports



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

# 3P export hook (why: keep UI now, wire backend later)
# 3P export hook: prefer new generator, fallback to legacy name
try:
    from main import generate_3p_report  # returns PDF bytes
except Exception:
    from main import generate_full_report_3p as generate_3p_report  # legacy alias


# why: Windows-safe date formatting (no %-m/%-d)
def fmt_mdy(d: date) -> str:
    return d.strftime("%#m/%#d/%Y") if sys.platform.startswith("win") else d.strftime("%-m/%-d/%Y")

# init
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Page config & branding
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SOAPBOX Reporting", layout="wide")

if os.path.exists("logo.png"):
    st.image("logo.png", width=180)

st.title("Weekly Content Reporting")
st.caption("Upload your data, manage clients, and export reports (1P & 3P).")

# Keep some state handles (why: enable Save Preview later)
_1p = {"has_data": False, "df": None, "metrics": None, "top5": None, "below": None, "client_name": None, "rpt_date": None, "notes": None}

# ─────────────────────────────────────────────────────────────────────────────
# 1) 1P — Inputs → Preview/Metrics/Tables → Export 1P PDF
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 1P — Weekly Content Reporting")

# Inputs (1P)
col_left, col_right = st.columns([1.3, 1.7], gap="large")
with col_left:
    uploaded_1p = st.file_uploader("Upload excel or csv", type=["xlsx", "csv"], key="uploader_1p")
with col_right:
    client_name_1p = st.text_input("Client name", value="Country Fresh", key="client_name_1p")
    rpt_date_obj_1p = st.date_input("Report Date", value=date.today(), key="rpt_date_1p")
    rpt_date_str_1p = fmt_mdy(rpt_date_obj_1p)
    notes_1p = st.text_area("Content Notes", value="", height=100, key="notes_1p")

st.divider()

# Load & preview (1P)
if uploaded_1p:
    df_1p = load_dataframe(uploaded_1p)
    _1p.update({"has_data": True, "df": df_1p, "client_name": client_name_1p, "rpt_date": rpt_date_str_1p, "notes": notes_1p})

    st.subheader("Data Preview (1P)")
    st.dataframe(df_1p.head(10), height=200, use_container_width=True)

    # Metrics (1P)
    m = compute_metrics(df_1p)
    _1p["metrics"] = m

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"{m['above']}/{m['total']} ({m['pct_above']:.1f}%) products ≥ {int(m['threshold'])}%")
        pie_buf = make_pie_bytes(m)
        st.image(pie_buf, caption="Score Distribution", use_column_width=False)
    with col2:
        st.subheader("Key Metrics (1P)")
        st.write(f"- **Average CQS:** {m['avg_cqs']:.1f}%")
        st.write(f"- **SKUs ≥ {int(m['threshold'])}%:** {m['above']}")
        st.write(f"- **SKUs < {int(m['threshold'])}%:** {m['below']}")
        st.write(f"- **Buybox Ownership:** {m['buybox']:.1f}%")

    st.markdown("---")

    # Tables (1P)
    st.subheader("Top 5 SKUs by Content Quality Score (1P)")
    top5_1p = get_top_skus(df_1p)
    _1p["top5"] = top5_1p
    st.dataframe(top5_1p, height=200, use_container_width=True)

    st.subheader(f"SKUs Below {int(m['threshold'])}% (1P)")
    below_1p = get_skus_below(df_1p)
    _1p["below"] = below_1p
    st.dataframe(below_1p, height=300, use_container_width=True)

    csv_data = below_1p.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download SKUs Below CSV (1P)", data=csv_data, file_name="skus_below_1p.csv", mime="text/csv")

    # Export 1P PDF (right under tables)
    st.markdown("### Export 1P PDF")
    if st.button("📄 Generate 1P PDF", key="export_pdf_1p"):
        pdf_bytes = generate_full_report(uploaded_1p, client_name_1p, rpt_date_str_1p, notes_1p)
        st.success("✅ 1P PDF ready!")
        st.download_button("⬇️ Download 1P PDF", data=pdf_bytes, file_name="1p.pdf", mime="application/pdf")
else:
    st.info("Upload a 1P data file to see preview and metrics.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2) 3P — Inputs (3 uploads + manual metrics) → Previews → Export 3P PDF
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 3P — Weekly Content Reporting")

# Mode selector: Catalog vs Managed
mode_3p = st.radio("Mode", ["Catalog", "Managed"], horizontal=True, key="mode_3p")

# 3P header inputs (separate from 1P)
c_hdr1, c_hdr2 = st.columns([2, 1])
with c_hdr1:
    client_name_3p = st.text_input("Client Name (3P)", value="", key="client_name_3p")
with c_hdr2:
    report_date_3p = st.date_input("Report Date (3P)", value=date.today(), key="report_date_3p")

# If Managed, show an extra uploader for the managed SKUs list
managed_file = None
if mode_3p == "Managed":
    managed_file = st.file_uploader(
        "Managed SKUs (IDs/Names list)",
        type=["xlsx", "csv"],
        key="uploader_3p_managed",
        help="Upload a file containing the SKU/Item ID list to limit calculations to managed items."
    )

st.markdown("**Upload excel(s) or csv — these will each be labeled**")
c1, c2, c3 = st.columns(3)
with c1:
    file_x = st.file_uploader("Upload Item Sales Report", type=["xlsx", "csv"], key="uploader_3p_x")
with c2:
    file_y = st.file_uploader("Upload Inventory Report", type=["xlsx", "csv"], key="uploader_3p_y")
with c3:
    file_z = st.file_uploader("Upload Search Insights", type=["xlsx", "csv"], key="uploader_3p_z")

# Manual textareas (always visible; non-persistent)
st.markdown("### Manual Content (renders in dashed areas on the PDF)")
c_txt1, c_txt2, c_txt3 = st.columns(3)
with c_txt1:
    manual_top_skus = st.text_area("Top SKUs (Item Sales) — manual", height=150, key="manual_top_skus")
with c_txt2:
    manual_inventory_callouts = st.text_area("Key Callouts (Inventory) — manual", height=150, key="manual_inventory_callouts")
with c_txt3:
    manual_search_highlights = st.text_area("Highlights (Search Insights) — manual", height=150, key="manual_search_highlights")

# Managed context (IDs/Names sets)
managed_ids, managed_names = set(), set()
if mode_3p == "Managed" and managed_file is not None:
    try:
        managed_ids, managed_names = load_managed_keys(managed_file)
        st.caption(f"Mode: **Managed** • Loaded {len(managed_ids) or len(managed_names)} SKUs")
    except Exception as e:
        st.error(f"Managed SKUs file error: {e}")
elif mode_3p == "Managed":
    st.info("Mode: **Managed** — Please upload the Managed SKUs (IDs/Names list).")

# Previews (3P) — Managed-aware filtering
preview_cols = st.columns(3)

# Helper A: normalize Item conversion to 0..100 (blanks→0) — retained for other uses
def _coerce_conversion_pp(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    nonnull = s[s.notna()]
    if len(nonnull):
        share_gt1 = (nonnull > 1.0).mean()
        if share_gt1 < 0.8:
            s = s * 100.0
    return s.clip(0.0, 100.0).astype(float)

# Helper B: Excel-style mean for conversion (🟨 used for preview Avg Conversion)
def _excel_mean_conversion_pp(series: pd.Series) -> float:
    # strip symbols -> numeric; blanks stay NaN and are excluded from mean
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    s = pd.to_numeric(s, errors="coerce")  # blanks -> NaN
    nonnull = s.dropna()
    if nonnull.empty:
        return 0.0
    # scale detection: if most values ≤ 1, treat as fractions
    if (nonnull <= 1.0).mean() >= 0.8:
        nonnull = nonnull * 100.0
    return float(nonnull.mean())

# ---------- X — Item Sales ----------
with preview_cols[0]:
    st.caption("Data Preview (Item Sales)")
    df_x = None
    if file_x:
        try:
            df_x = load_item_sales(file_x)
            df_view = df_x
            stats_x = None
            if mode_3p == "Managed" and (managed_ids or managed_names):
                df_view, stats_x = filter_by_managed(df_x, managed_ids, managed_names)
                st.write(f"Showing {stats_x['matched']} of {stats_x['total']} (managed subset)")
                if stats_x["matched"] == 0 and stats_x["unmatched_sample"]:
                    st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_x['unmatched_sample'])}")
            st.dataframe(df_view.head(15), height=260, use_container_width=True)

            # Calculated preview: Units, Auth Sales (with cents), Avg Conversion (Excel-style mean, 2 decimals)
            units_total = int(pd.to_numeric(df_view.get("Units Sold"), errors="coerce").fillna(0).sum()) if not df_view.empty else 0
            sales_total = float(pd.to_numeric(df_view.get("Auth Sales"), errors="coerce").sum()) if not df_view.empty else 0.0
            avg_conv_pct = _excel_mean_conversion_pp(df_view.get("Item conversion", pd.Series(dtype=float))) if not df_view.empty else 0.0

            st.markdown("**Calculated (Item Sales):**")
            st.write(f"- Units Sold: {units_total:,}")
            st.write(f"- Auth Sales: ${sales_total:,.2f}")          # <-- show cents
            st.write(f"- Avg Conversion: {avg_conv_pct:.2f}%")      # <-- Excel-style mean, 2 decimals
        except Exception as e:
            st.error(
                "Item Sales file doesn’t match required columns. "
                "Expecting (canonical): Item ID, Item Name, Orders, Units Sold, "
                "Auth Sales, Item pageviews, Item conversion.\n\n"
                f"Details: {e}"
            )
    else:
        st.info("Upload Item Sales Report to preview.")

# ---------- Y — Inventory ----------
with preview_cols[1]:
    st.caption("Data Preview (Inventory)")
    df_y = None
    if file_y:
        try:
            df_y = load_inventory(file_y)
            df_view = df_y
            stats_y = None
            if mode_3p == "Managed" and (managed_ids or managed_names):
                df_view, stats_y = filter_by_managed(df_y, managed_ids, managed_names)
                st.write(f"Showing {stats_y['matched']} of {stats_y['total']} (managed subset)")
                if stats_y["matched"] == 0 and stats_y["unmatched_sample"]:
                    st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_y['unmatched_sample'])}")
            st.dataframe(df_view.head(15), height=260, use_container_width=True)

            # Calculated preview: In-stock rate (whole %), raw counts (preview only)
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
                "Inventory file doesn’t match required columns. "
                "Expecting (canonical): Item ID, Item Name, Daily sales, Daily units sold, Stock status.\n\n"
                f"Details: {e}"
            )
    else:
        st.info("Upload Inventory Report to preview.")

# ---------- Z — Search Insights ----------
with preview_cols[2]:
    st.caption("Data Preview (Search Insights)")
    df_z = None
    if file_z:
        try:
            df_z = load_search_insights(file_z)
            df_view = df_z
            stats_z = None
            if mode_3p == "Managed" and (managed_ids or managed_names):
                df_view, stats_z = filter_by_managed(df_z, managed_ids, managed_names)
                st.write(f"Showing {stats_z['matched']} of {stats_z['total']} (managed subset)")
                if stats_z["matched"] == 0 and stats_z["unmatched_sample"]:
                    st.warning(f"No matches. Sample unmatched IDs: {', '.join(stats_z['unmatched_sample'])}")
            st.dataframe(df_view.head(15), height=260, use_container_width=True)

            # Calculated preview: Avg Impressions Rank (whole), Top-10 counts
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
                "Search Insights file doesn’t match required columns. "
                "Expecting: Item ID, Item Name, Impressions Rank, Clicks Rank, "
                "Added to Cart Rank, Sales Rank.\n\n"
                f"Details: {e}"
            )
    else:
        st.info("Upload Search Insights to preview.")

# Export 3P PDF
st.markdown("### Export 3P PDF")
export_disabled = any(v is None for v in [file_x, file_y, file_z]) or (mode_3p == "Managed" and managed_file is None)
if export_disabled:
    st.info("Upload Item Sales, Inventory, and Search Insights (and Managed list if Managed mode) to enable export.")

if st.button("📄 Generate 3P Dashboard PDF", key="export_pdf_3p", disabled=export_disabled):
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
            logo_path=None,
        )
    except TypeError:
        pdf3 = generate_3p_report(
            data_src=None,
            client_name=client_name_3p or "Client",
            report_date=fmt_mdy(report_date_3p) if "fmt_mdy" in globals() else report_date_3p.strftime("%B %d, %Y"),
            logo_path=None,
        )  # type: ignore

    if not pdf3:
        st.error("3P generator returned no bytes. Ensure main.py defines the updated generate_3p_report(...).")
    else:
        st.success("✅ 3P Dashboard PDF ready!")
        st.download_button("⬇️ Download 3P PDF", data=pdf3, file_name="dashboard_3p.pdf", mime="application/pdf")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# 3) Save Preview (Snapshot) — bottom, enabled only if 1P metrics exist
# ─────────────────────────────────────────────────────────────────────────────
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
        value=f"{db_selected_client_name} – {_1p['rpt_date']}",
        key="preview_name_1p",
    )
    if st.button("💾 Save Preview", key="save_preview_btn"):
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

# ─────────────────────────────────────────────────────────────────────────────
# 4) View Saved Previews — bottom
# ─────────────────────────────────────────────────────────────────────────────
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
        st.write(f"- **SKUs ≥ {int(m2['threshold'])}%:** {m2['above']}")
        st.write(f"- **SKUs < {int(m2['threshold'])}%:** {m2['below']}")
        st.write(f"- **Buybox Ownership:** {m2['buybox']}%")

        st.subheader("Top 5 SKUs by Content Quality Score (Saved)")
        top5_df = pd.DataFrame(selected_preview["data_json"]["top5"])
        st.dataframe(top5_df, height=200, use_container_width=True)

        st.subheader(f"SKUs Below {int(m2['threshold'])}% (Saved)")
        below_df = pd.DataFrame(selected_preview["data_json"]["skus_below"])
        st.dataframe(below_df, height=300, use_container_width=True)

        if st.button("🗑️ Delete This Preview", key="delete_preview_btn"):
            delete_preview(selected_preview["preview_id"])
            st.success("Preview deleted.")
            st.experimental_rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 5) Manage Clients — bottom
# ─────────────────────────────────────────────────────────────────────────────
st.header("Manage Clients")
clients_bottom = get_clients()
if clients_bottom:
    st.write("Existing Clients:", ", ".join(client["client_name"] for client in clients_bottom))
else:
    st.write("_No clients defined yet._")

new_client_bottom = st.text_input("New Client name", key="new_client_bottom")
if st.button("➕ Add Client", key="add_client_bottom"):
    if new_client_bottom and new_client_bottom not in [c["client_name"] for c in clients_bottom]:
        add_client(new_client_bottom)
        st.experimental_rerun()
    else:
        st.warning("Enter a unique client name.")

if clients_bottom:
    client_names_bottom = [c["client_name"] for c in clients_bottom]
    name_to_id_bottom = {c["client_name"]: c["client_id"] for c in clients_bottom}
    to_delete_bottom = st.selectbox("Client to delete", client_names_bottom, key="del_client_bottom")
    if st.button("🗑️ Delete Client", key="delete_client_bottom"):
        delete_client(name_to_id_bottom[to_delete_bottom])
        st.experimental_rerun()
