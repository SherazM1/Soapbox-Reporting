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
    # Load data & compute
    df      = load_dataframe(data_src)
    metrics = compute_metrics(df)
    top5    = get_top_skus(df)
    below   = get_skus_below(df)

    # Prepare canvas
    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    # ─── Header ────────────────────────────────────────────────────────────────
    # Draw logo at fixed top-left
    logo_path = resource_path("retaillogo.png")
    if os.path.isfile(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(
            logo,
            x=inch * 0.5,
            y=h - inch * 0.5,
            width=1.5 * inch,
            preserveAspectRatio=True,
            mask="auto"
        )
    else:
        # Optional: draw a placeholder box so you can see where it should be
        c.setStrokeColor(colors.red)
        c.rect(inch * 0.5, h - inch * 0.5, 1.5 * inch, 1.5 * inch)

    # Client name in teal, larger
    teal = colors.HexColor("#4CC9C8")
    c.setFillColor(teal)
    c.setFont("Raleway", 16)
    c.drawString(inch * 2.0, h - inch * 0.6, client_name)

    # Title in dark navy, much larger
    navy = colors.HexColor("#003554")
    c.setFillColor(navy)
    c.setFont("Raleway", 24)
    c.drawString(inch * 2.0, h - inch * 1.0, "Weekly Content Reporting")

    # Date below title
    c.setFont("Raleway", 12)
    c.drawString(inch * 2.0, h - inch * 1.3, report_date)

    c.setFillColor(colors.black)  # reset

    # ─── Summary Line ─────────────────────────────────────────────────────────
    total = metrics["total"]; above = metrics["above"]
    summary = (
        f"{above}/{total} "
        f"({metrics['pct_above']:.1f}%) products have "
        f"Content Quality Score ≥ {int(metrics['threshold'])}%."
    )
    c.setFont("Raleway", 12)
    c.drawString(inch * 0.5, h - inch * 1.6, summary)

    # ─── Pie Chart ─────────────────────────────────────────────────────────────
    panel_x, panel_top = inch * 0.5, h - inch * 1.8
    panel_w, panel_h  = 3.5 * inch, 3.5 * inch  # bump height to 3.5"

    # Draw rounded border
    c.setStrokeColor(navy)
    c.setLineWidth(1)
    c.roundRect(panel_x, panel_top - panel_h, panel_w, panel_h, radius=8, stroke=1, fill=0)

    # Panel title
    c.setFont("Raleway", 12)
    c.setFillColor(navy)
    c.drawCentredString(panel_x + panel_w/2, panel_top - 14, "Score Distribution")

    # Pie chart: center in panel
    pie_buf = make_pie_bytes(metrics)
    pie     = ImageReader(pie_buf)
    pie_size = 2.2 * inch
    # calculate panel center
    panel_center_x = panel_x + panel_w/2
    panel_center_y = (panel_top - panel_h) + panel_h/2
    # draw pie centered
    c.drawImage(
        pie,
        x=panel_center_x - pie_size/2,
        y=panel_center_y - pie_size/2,
        width=pie_size,
        height=pie_size
    )

    # Legend squares + labels below pie
    legend_y = (panel_top - panel_h) + 20
    square_size = 8
    label_x = panel_x + 15
    # Below 95%
    c.setFillColor(navy)
    c.rect(label_x, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Raleway", 10)
    c.drawString(label_x + square_size + 4, legend_y + 1, f"Below {int(THRESHOLD)}%")
    # Above 95%
    x2 = label_x + 100
    c.setFillColor(teal)
    c.rect(x2, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(x2 + square_size + 4, legend_y + 1, f"Above {int(THRESHOLD)}%")

    # Reset fill color
    c.setFillColor(colors.black)

    # ─── Metrics Panel ─────────────────────────────────────────────────────────
    box_x, box_y = inch * 4.0, h - inch * 1.8
    box_w, box_h = inch * 3.5, inch * 2.5
    c.setStrokeColor(navy)
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=8, stroke=1, fill=0)

    # Draw each bullet line
    c.setFont("Raleway", 12)               # bump up font size
    y = box_y - 20
    for label, key in [
        ("Average CQS",    "avg_cqs"),
        (f"SKUs ≥ {int(metrics['threshold'])}%", "above"),
        (f"SKUs < {int(metrics['threshold'])}%", "below"),
        ("Buybox Ownership","buybox"),
    ]:
        val = metrics[key]
        if key in ("above", "below"):
            val = int(val)
        elif isinstance(val, float):
            val = f"{val:.1f}%"

        # Draw a navy bullet
        c.setFillColor(navy)
        c.drawString(box_x + 8, y, "•")  

        # Draw the text in black (or keep navy if you prefer)
        c.setFillColor(colors.black)
        c.drawString(box_x + 24, y, f"{label}: {val}")
        y -= 18  # increase line spacing

    c.setFillColor(colors.black)

    # ─── Top 5 Table ───────────────────────────────────────────────────────────
    data = [top5.columns.tolist()] + top5.astype(str).values.tolist()
    table = Table(data, colWidths=[2.5 * inch, 1 * inch, 1 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Raleway"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    tw, th = table.wrapOn(c, w, h)
    table.drawOn(c, inch * 0.5, box_y - box_h - inch * 0.2 - th)

    # Finish up
    data = [top5.columns.tolist()] + top5.astype(str).values.tolist()

    # Compute table width and column widths
    margin = inch * 0.5
    table_w = w - 2 * margin
    col_widths = [
        table_w * 0.60,  # Product Name takes 60%
        table_w * 0.20,  # Item ID 20%
        table_w * 0.20,  # CQS 20%
    ]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Raleway"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),  # a bit larger
        ("BACKGROUND",(0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("LEFTPADDING",(0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))

    # Draw the table at the bottom, just above the page margin
    tw, th = table.wrap(table_w, h)
    table.drawOn(c, margin, box_y - box_h - inch * 0.3 - th)
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
