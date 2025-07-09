# main.py

import os
import sys
import json
import pandas as pd
import io
from io import BytesIO
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def resource_path(rel_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)

# ─────────────────────────────────────────────────────────────────────────────
# Register Raleway font
# ─────────────────────────────────────────────────────────────────────────────
pdfmetrics.registerFont(
    TTFont(
        "Raleway",
        resource_path(os.path.join("fonts", "Raleway-Regular.ttf"))
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Persistence
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD    = 95.0
TOP_N        = 5
BATCHES_PATH = "dashboards/batches.json"

def load_batches(path: str = BATCHES_PATH) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_batches(batches: list, path: str = BATCHES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(batches, f, indent=2, default=str)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataframe(src) -> pd.DataFrame:
    # Streamlit upload
    if hasattr(src, "read") and hasattr(src, "name"):
        data = src.getvalue()
        ext  = os.path.splitext(src.name)[1].lower()
        if ext == ".csv":
            return pd.read_csv(io.BytesIO(data), encoding="utf-8",
                               engine="python", on_bad_lines="skip")
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(io.BytesIO(data))
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    # File path
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
def split_by_threshold(df: pd.DataFrame):
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
        .head(TOP_N)[["Product Name","Item ID","Content Quality Score"]]
        .copy()
    )

def get_skus_below(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["Content Quality Score"] < THRESHOLD]
        [["Product Name","Item ID","Content Quality Score"]]
        .copy()
    )

# ─────────────────────────────────────────────────────────────────────────────
# Pie Chart Helper
# ─────────────────────────────────────────────────────────────────────────────
def make_pie_bytes(metrics: dict) -> BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [metrics["below"], metrics["above"]],
        labels=[f"Below {int(THRESHOLD)}%", f"Above {int(THRESHOLD)}%"],
        autopct="%1.0f%%",
        startangle=90
    )
    ax.axis("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
# PDF Generation via ReportLab
# ─────────────────────────────────────────────────────────────────────────────
def generate_full_report(data_src, client_name: str, report_date: str) -> bytes:
    # Load and compute
    df      = load_dataframe(data_src)
    metrics = compute_metrics(df)
    top5    = get_top_skus(df)
    below   = get_skus_below(df)

    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    # Logo
    logo = ImageReader(resource_path("retaillogo.png"))
    c.drawImage(logo, inch*0.5, h - inch*1.2, width=1.5*inch, preserveAspectRatio=True, mask="auto")

    # Title & Header
    c.setFont("Raleway", 18)
    c.drawString(inch*2.2, h - inch*0.8, "Weekly Content Reporting")
    c.setFont("Raleway", 10)
    c.drawString(inch*2.2, h - inch*1.1, f"{client_name}    {report_date}")

    # Summary line
    total = metrics["total"]
    above = metrics["above"]
    summary = f"{above}/{total} ({metrics['pct_above']:.1f}%) products have Content Quality Score ≥ {int(THRESHOLD)}%."
    c.setFont("Raleway", 12)
    c.drawString(inch*0.5, h - inch*1.5, summary)

    # Pie chart
    pie_buf = make_pie_bytes(metrics)
    pie     = ImageReader(pie_buf)
    c.drawImage(pie, inch*0.5, h - inch*4.0, width=3*inch, height=3*inch)

    # Metrics panel
    box_x, box_y = inch*4.0, h - inch*1.8
    box_w, box_h = inch*3.5, inch*2.5
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=10, stroke=1, fill=0)
    c.setFont("Raleway", 10)
    y = box_y - 14
    for label, key in [
        ("Average CQS", "avg_cqs"),
        (f"SKUs ≥ {int(THRESHOLD)}%", "above"),
        (f"SKUs < {int(THRESHOLD)}%", "below"),
        ("Buybox Ownership", "buybox")
    ]:
        val = metrics[key]
        if key in ("above", "below"):
            val = int(val)
        elif isinstance(val, float):
            val = f"{val:.1f}%"
        c.drawString(box_x + 8, y, f"• {label}: {val}")
        y -= 14

    # Top 5 table
    data = [top5.columns.tolist()] + top5.astype(str).values.tolist()
    table = Table(data, colWidths=[2.5*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Raleway"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003554")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    tw, th = table.wrapOn(c, w, h)
    table.drawOn(c, inch*0.5, box_y - box_h - inch*0.2 - th)

    # Finish
    c.showPage()
    c.save()

    buf.seek(0)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dashboard PDF")
    parser.add_argument("input_file", help="CSV or Excel input")
    parser.add_argument("--client", required=True, help="Client name")
    parser.add_argument("--date",   required=True, help="Report date")
    parser.add_argument("--out",    default="dashboard.pdf", help="Output PDF")
    args = parser.parse_args()

    pdf = generate_full_report(args.input_file, args.client, args.date)
    with open(args.out, "wb") as f:
        f.write(pdf)
    print(f"Wrote {args.out}")
