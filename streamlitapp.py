# streamlit_app.py  — UI reorg per your spec, minimal logic changes

import os
import sys
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
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
    generate_full_report,  # 1P
)

# 3P export hook (why: keep UI now, wire backend later)
try:
    from main import generate_full_report_3p  # type: ignore[attr-defined]
except Exception:
    def generate_full_report_3p(*_args, **_kwargs):  # noqa: D401
        """Placeholder until 3P backend is ready."""
        return None

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
st.caption("Upload your data, manage clients, and export dashboards (1P & 3P).")

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
    if st.button("📄 Generate 1P Dashboard PDF", key="export_pdf_1p"):
        pdf_bytes = generate_full_report(uploaded_1p, client_name_1p, rpt_date_str_1p, notes_1p)
        st.success("✅ 1P Dashboard PDF ready!")
        st.download_button("⬇️ Download 1P PDF", data=pdf_bytes, file_name="dashboard_1p.pdf", mime="application/pdf")
else:
    st.info("Upload a 1P data file to see preview and metrics.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2) 3P — Inputs (3 uploads + manual metrics) → Previews → Export 3P PDF
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 3P — Weekly Content Reporting")

# Mode selector: Catalog vs Managed (UI-only; no filtering yet)
mode_3p = st.radio("Mode", ["Catalog", "Managed"], horizontal=True, key="mode_3p")

# If Managed, show an extra uploader for the managed SKUs list
managed_file = None
if mode_3p == "Managed":
    managed_file = st.file_uploader(
        "Managed SKUs (IDs list)",
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

# Replace textarea with dropdown + integer value (keeps downstream variable name)
metric_period_3p = st.selectbox(
    "Metric period",
    [
        "today",
        "yesterday",
        "last 7 days",
        "last 30 days",
        "month to date (MTD)",
        "year to date (YTD)",
    ],
    key="metric_period_3p",
)
metric_value_3p = st.number_input("Value", value=0, step=1, format="%d", key="metric_value_3p")
# Preserve existing export call signature by composing a string payload.
metrics_3p_text = f"{metric_period_3p}: {metric_value_3p}"

# Previews (3P) — simple head previews if files present (UI only; no filtering yet)
if mode_3p == "Managed" and managed_file is not None:
    st.caption("Mode: **Managed** • Managed SKUs file uploaded")
elif mode_3p == "Managed":
    st.info("Mode: **Managed** — Please upload the Managed SKUs (IDs list).")

preview_cols = st.columns(3)
for idx, (lbl, f, col) in enumerate(
    [
        ("Item Sales", file_x, preview_cols[0]),
        ("Inventory", file_y, preview_cols[1]),
        ("Search Insights", file_z, preview_cols[2]),
    ],
    start=1
):
    with col:
        st.caption(f"Data Preview ({lbl})")
        if f:
            try:
                df_tmp = load_dataframe(f)
                st.dataframe(df_tmp.head(10), height=200, use_container_width=True)
            except Exception:
                st.warning(f"Could not preview {lbl} — unsupported format or read error.")
        else:
            st.info(f"Upload {lbl} to preview.")

# Export 3P PDF
st.markdown("### Export 3P PDF")
if st.button("📄 Generate 3P Dashboard PDF", key="export_pdf_3p"):
    pdf3 = generate_full_report_3p(file_x, file_y, file_z, metrics_3p_text)
    if pdf3:
        st.success("✅ 3P Dashboard PDF ready!")
        st.download_button("⬇️ Download 3P PDF", data=pdf3, file_name="dashboard_3p.pdf", mime="application/pdf")
    else:
        st.info("3P export is a placeholder until backend logic is added.")

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
