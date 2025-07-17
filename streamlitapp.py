# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from main import (
    load_batches as load_groups,
    save_batches as save_groups,
    load_dataframe,
    compute_metrics,
    get_top_skus,
    get_skus_below,
    make_pie_bytes,
    generate_full_report
)
   # You know this works because st.image() works

# Pass logo_path into your PDF generator


# ─────────────────────────────────────────────────────────────────────────────
# Page config & branding
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOAPBOX Dashboard",
    page_icon="🛠",
    layout="wide",
)

if os.path.exists("logo.png"):
    st.image("logo.png", width=180)

st.title("Weekly Content Reporting")
st.markdown(
    "Upload your data, manage your groups, and export the full dashboard PDF with client & date."
)

# ─────────────────────────────────────────────────────────────────────────────
# Group Management
# ─────────────────────────────────────────────────────────────────────────────
st.header("Manage Clients")
groups = load_groups()

if groups:
    st.write("Existing Clients:", ", ".join(g["name"] for g in groups))
else:
    st.write("_No clients defined yet._")

new_group = st.text_input("New Client name")
if st.button("➕ Add Client"):
    if new_group and new_group not in [g["name"] for g in groups]:
        groups.append({
            "name": new_group,
            "created": date.today().isoformat()
        })
        save_groups(groups)
        st.experimental_rerun()
    else:
        st.warning("Enter a unique client name.")

if groups:
    to_delete = st.selectbox("Client to delete", [g["name"] for g in groups])
    if st.button("🗑️ Delete Client"):
        groups = [g for g in groups if g["name"] != to_delete]
        save_groups(groups)
        st.experimental_rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Inputs: File, Client, Date
# ─────────────────────────────────────────────────────────────────────────────
st.header("Inputs")
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
client_name = st.text_input("Client Name", value="Country Fresh")
rpt_date = st.date_input("Report Date", value=date.today()).strftime("%-m/%-d/%Y")
client_notes = st.text_area("Content Notes", value="", height=100)
if not uploaded:
    st.info("Please upload a data file to see the dashboard below.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load & Preview Data
# ─────────────────────────────────────────────────────────────────────────────
df = load_dataframe(uploaded)
st.subheader("Data Preview")
st.dataframe(df.head(10), height=200)

# ─────────────────────────────────────────────────────────────────────────────
# Compute & Display Metrics
# ─────────────────────────────────────────────────────────────────────────────
m = compute_metrics(df)

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"{m['above']}/{m['total']} ({m['pct_above']:.1f}%) products ≥ {int(m['threshold'])}%")
    pie_buf = make_pie_bytes(m)
    st.image(pie_buf, caption="Score Distribution", use_column_width=False)

with col2:
    st.subheader("Key Metrics")
    st.write(f"- **Average CQS:** {m['avg_cqs']:.1f}%")
    st.write(f"- **SKUs ≥ {int(m['threshold'])}%:** {m['above']}")
    st.write(f"- **SKUs < {int(m['threshold'])}%:** {m['below']}")
    st.write(f"- **Buybox Ownership:** {m['buybox']:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Top 5 & Below Tables
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Top 5 SKUs by Content Quality Score")
top5 = get_top_skus(df)
st.dataframe(top5, height=200)

st.subheader(f"SKUs Below {int(m['threshold'])}%")
skus_below = get_skus_below(df)
st.dataframe(skus_below, height=300)

# Export SKUs Below CSV
csv_data = skus_below.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download SKUs Below CSV",
    data=csv_data,
    file_name="skus_below.csv",
    mime="text/csv"
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Export Full Dashboard PDF
# ─────────────────────────────────────────────────────────────────────────────
st.header("Export Dashboard PDF")
if st.button("📄 Generate Dashboard PDF"):
    pdf_bytes = generate_full_report(uploaded, client_name, rpt_date, client_notes)
    st.success("✅ Dashboard PDF ready!")
    st.download_button(
        "⬇️ Download Dashboard PDF",
        data=pdf_bytes,
        file_name="dashboard.pdf",
        mime="application/pdf"
    )
