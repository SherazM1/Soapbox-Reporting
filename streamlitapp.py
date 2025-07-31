# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from db import init_db, add_client, get_clients, delete_client, add_preview, get_previews_for_client, delete_preview


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
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Page config & branding
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOAPBOX Reporting",
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
clients = get_clients()

if clients:
    st.write("Existing Clients:", ", ".join(client["client_name"] for client in clients))
else:
    st.write("_No clients defined yet._")

new_client = st.text_input("New Client name")
if st.button("➕ Add Client"):
    if new_client and new_client not in [c["client_name"] for c in clients]:
        add_client(new_client)
        st.experimental_rerun()
    else:
        st.warning("Enter a unique client name.")

if clients:
    client_names = [c["client_name"] for c in clients]
    name_to_id = {c["client_name"]: c["client_id"] for c in clients}
    to_delete = st.selectbox("Client to delete", client_names)
    if st.button("🗑️ Delete Client"):
        delete_client(name_to_id[to_delete])
        st.experimental_rerun()

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

st.markdown("---")

st.header("View Saved Previews")

# Use the same client groups from before
view_client_names = [c["client_name"] for c in clients]
view_client_ids = {c["client_name"]: c["client_id"] for c in clients}

if not view_client_names:
    st.info("No client groups exist. Add one above to begin saving and viewing previews.")
else:
    view_selected_client_name = st.selectbox(
        "Select Client Group to View Previews",
        view_client_names,
        key="view_client_dropdown"
    )
    view_selected_client_id = view_client_ids[view_selected_client_name]

    # Fetch previews for this client group
    previews = get_previews_for_client(view_selected_client_id)

    if not previews:
        st.info(f"No saved previews yet for {view_selected_client_name}.")
    else:
        # Let user pick a preview by name and date
        preview_options = [
            f"{p['preview_name']} ({p['report_date']})" for p in previews
        ]
        option_to_preview = {f"{p['preview_name']} ({p['report_date']})": p for p in previews}

        selected_option = st.selectbox(
            "Select a saved preview to view",
            preview_options,
            key="saved_preview_select"
        )
        selected_preview = option_to_preview[selected_option]

        # Show preview details
        st.markdown(f"**Report Date:** {selected_preview['report_date']}")
        st.markdown(f"**Preview Name:** {selected_preview['preview_name']}")
        st.markdown(f"**Notes:** {selected_preview['notes']}")
        st.markdown(f"**Date Saved:** {selected_preview['date_created']}")

        # Display metrics
        m2 = selected_preview["data_json"]["metrics"]
        st.subheader("Saved Metrics")
        st.write(f"- **Average CQS:** {m2['avg_cqs']}%")
        st.write(f"- **SKUs ≥ {int(m2['threshold'])}%:** {m2['above']}")
        st.write(f"- **SKUs < {int(m2['threshold'])}%:** {m2['below']}")
        st.write(f"- **Buybox Ownership:** {m2['buybox']}%")

        # Top 5 Table
        st.subheader("Top 5 SKUs by Content Quality Score (Saved)")
        top5_df = pd.DataFrame(selected_preview["data_json"]["top5"])
        st.dataframe(top5_df, height=200)

        # SKUs Below Table
        st.subheader(f"SKUs Below {int(m2['threshold'])}% (Saved)")
        below_df = pd.DataFrame(selected_preview["data_json"]["skus_below"])
        st.dataframe(below_df, height=300)

        # Notes
        st.markdown("**Content Notes (Saved):**")
        st.markdown(selected_preview["data_json"].get("client_notes", ""))
    # Add Delete Preview button (after showing the loaded preview)
    
        if st.button("🗑️ Delete This Preview"):
            delete_preview(selected_preview["preview_id"])
            st.success("Preview deleted!")
            st.experimental_rerun()  # Refresh page to update the list


# ─────────────────────────────────────────────────────────────────────────────
# Inputs: File, Client, Date
# ─────────────────────────────────────────────────────────────────────────────
st.header("Inputs")
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
# Save Preview Section
# ─────────────────────────────────────────────────────────────────────────────
st.header("Save This Preview (Snapshot)")

# Get client groups from DB for this dropdown
db_clients = get_clients()
db_client_names = [c["client_name"] for c in db_clients]
db_client_ids = {c["client_name"]: c["client_id"] for c in db_clients}

if not db_clients:
    st.warning("You must add at least one client group above before saving previews.")
else:
    db_selected_client_name = st.selectbox(
        "Assign this snapshot to client group:",
        db_client_names,
        key="db_client_dropdown_save"
    )
    db_selected_client_id = db_client_ids[db_selected_client_name]

    preview_name = st.text_input(
        "Preview Name (for saving this report snapshot)",
        value=f"{db_selected_client_name} – {rpt_date}"
    )

    if st.button("💾 Save Preview"):
        data_json = {
            "metrics": m,
            "top5": top5.to_dict(orient="records"),
            "skus_below": skus_below.to_dict(orient="records"),
            "client_notes": client_notes,
            "pdf_client_name": client_name,      # for display on PDF
            "pdf_report_date": rpt_date
        }
        add_preview(
            client_id=db_selected_client_id,
            preview_name=preview_name,
            report_date=rpt_date,
            notes=client_notes,
            data_json=data_json
        )
        st.success("Preview saved! See below to view all saved previews.")


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
