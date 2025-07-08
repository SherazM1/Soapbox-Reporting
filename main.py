# main.py

import os
import sys
import json
import pandas as pd
import subprocess
import tempfile
import io
from datetime import datetime
from io import BytesIO
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD     = 95.0
TOP_N         = 5
BATCHES_PATH  = "dashboards/batches.json"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def resource_path(rel_path: str) -> str:
    """Return absolute path to bundled resources (fonts, etc.)."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)

# ─────────────────────────────────────────────────────────────────────────────
# Group Persistence
# ─────────────────────────────────────────────────────────────────────────────
def load_batches(path: str = BATCHES_PATH) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_batches(batches: list, path: str = BATCHES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(batches, f, indent=2, default=str)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataframe(src) -> pd.DataFrame:
    """
    Load a CSV or Excel file (path or Streamlit upload) into a DataFrame.
    For uploads, reads bytes via BytesIO with a forgiving parser.
    """
    if hasattr(src, "read") and hasattr(src, "name"):
        content = src.getvalue()
        ext = os.path.splitext(src.name)[1].lower()
        if ext == ".csv":
            return pd.read_csv(
                io.BytesIO(content),
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip"
            )
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    ext = os.path.splitext(src)[1].lower()
    if ext == ".csv":
        return pd.read_csv(src)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(src)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Tables
# ─────────────────────────────────────────────────────────────────────────────
from typing import Tuple

def split_by_threshold(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    above = df[df["Content Quality Score"] >= THRESHOLD].copy()
    below = df[df["Content Quality Score"] <  THRESHOLD].copy()
    return below, above

def compute_metrics(df: pd.DataFrame) -> dict:
    below, above = split_by_threshold(df)
    total       = len(df)
    count_above = len(above)
    pct_above   = (count_above / total * 100) if total else 0.0
    avg_cqs     = df["Content Quality Score"].mean() if total else 0.0
    buybox      = df["Buybox Ownership"].mean() if "Buybox Ownership" in df else 0.0

    return {
        "total":     total,
        "above":     count_above,
        "below":     len(below),
        "pct_above": pct_above,
        "avg_cqs":   avg_cqs,
        "buybox":    buybox,
        "threshold": THRESHOLD
    }

def get_top_skus(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .sort_values("Content Quality Score", ascending=False)
        .head(TOP_N)[["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )

def get_skus_below(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["Content Quality Score"] < THRESHOLD]
        [["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )

# ─────────────────────────────────────────────────────────────────────────────
# Pie Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_pie_bytes(metrics: dict) -> BytesIO:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [metrics["below"], metrics["above"]],
        labels=[f"Below {int(THRESHOLD)}%", f"Above {int(THRESHOLD)}%"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
# HTML Report
# ─────────────────────────────────────────────────────────────────────────────
def build_html_report(
    df: pd.DataFrame,
    client_name: str,
    report_date: str
) -> str:
    metrics = compute_metrics(df)
    top5    = get_top_skus(df)
    below   = get_skus_below(df)

    font_file = resource_path(os.path.join("fonts", "Raleway-Regular.ttf"))
    css = f"""
    <style>
      @font-face {{
        font-family: 'Raleway';
        font-style: normal;
        font-weight: normal;
        src: url("file://{font_file}") format("truetype");
      }}
      body {{
        font-family: 'Raleway', sans-serif;
        font-weight: normal;
        margin: 30px;
      }}
      h1 {{ font-size: 24px; margin-bottom: 8px; }}
      h2 {{ font-size: 18px; margin-top: 24px; margin-bottom: 8px; }}
      .metrics {{
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 16px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
      }}
      th, td {{
        border: 1px solid #333;
        padding: 6px;
        font-size: 14px;
        font-weight: normal;
      }}
      th {{ background-color: #f2f2f2; }}
    </style>
    """

    header = (
        f"<div class='metrics'>"
        f"<strong>Client:</strong> {client_name}<br>"
        f"<strong>Date:</strong> {report_date}"
        f"</div>"
    )

    m = metrics
    metrics_html = (
        "<div class='metrics'>"
        f"• Total SKUs: {m['total']}<br>"
        f"• SKUs ≥ {int(m['threshold'])}%: {m['above']} ({m['pct_above']:.1f}%)<br>"
        f"• SKUs < {int(m['threshold'])}%: {m['below']} ({100-m['pct_above']:.1f}%)<br>"
        f"• Average CQS: {m['avg_cqs']:.1f}%<br>"
        f"• Buybox Ownership: {m['buybox']:.1f}%<br>"
        "</div>"
    )

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        css,
        "</head><body>",
        "<h1>Weekly Content Reporting</h1>",
        header,
        metrics_html,
        "<h2>Top 5 SKUs by Content Quality Score</h2>",
        "<table><tr><th>Product Name</th><th>Item ID</th><th>CQS</th></tr>"
    ]

    for _, row in top5.iterrows():
        html_parts.append(
            "<tr>"
            f"<td>{row['Product Name']}</td>"
            f"<td>{row['Item ID']}</td>"
            f"<td>{int(row['Content Quality Score'])}%</td>"
            "</tr>"
        )

    html_parts.extend([
        "</table>",
        "<h2>SKUs Below Threshold</h2>",
        "<table><tr><th>Product Name</th><th>Item ID</th><th>CQS</th></tr>"
    ])

    for _, row in below.iterrows():
        html_parts.append(
            "<tr>"
            f"<td>{row['Product Name']}</td>"
            f"<td>{row['Item ID']}</td>"
            f"<td>{int(row['Content Quality Score'])}%</td>"
            "</tr>"
        )

    html_parts.append("</table></body></html>")
    return "\n".join(html_parts)

# ─────────────────────────────────────────────────────────────────────────────
# HTML → PDF Conversion (system wkhtmltopdf via apt.txt)
# ─────────────────────────────────────────────────────────────────────────────
def html_to_pdf_bytes(html_str: str) -> bytes:
    # must have wkhtmltopdf installed via apt.txt
    wk = shutil.which("wkhtmltopdf")
    if not wk or not os.path.isfile(wk):
        raise FileNotFoundError(
            f"wkhtmltopdf not found in PATH. Please ensure it's installed via apt.txt."
        )

    tmp_html = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
    tmp_html.write(html_str)
    tmp_html.close()

    pdf_path = tmp_html.name.replace(".html", ".pdf")
    subprocess.run([wk, tmp_html.name, pdf_path], check=True)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    for p in (tmp_html.name, pdf_path):
        try: os.remove(p)
        except: pass

    return pdf_bytes

# ─────────────────────────────────────────────────────────────────────────────
# One-call PDF Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_full_report(
    data_src,
    client_name: str,
    report_date: str
) -> bytes:
    df = load_dataframe(data_src)
    html = build_html_report(df, client_name, report_date)
    return html_to_pdf_bytes(html)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dashboard PDF")
    parser.add_argument("input_file", help="CSV or Excel file")
    parser.add_argument("--client", required=True, help="Client name")
    parser.add_argument("--date",   required=True, help="Report date (M/D/YYYY)")
    parser.add_argument("--out",    default="dashboard.pdf", help="Output PDF")
    args = parser.parse_args()

    pdf = generate_full_report(args.input_file, args.client, args.date)
    with open(args.out, "wb") as f:
        f.write(pdf)
    print(f"Wrote {args.out}")
