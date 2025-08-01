# main.py

import os
import sys
import json
import pandas as pd
import io
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
import pathlib
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def resource_path(rel_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))  # <-- Use script's directory
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
pdfmetrics.registerFont(
    TTFont(
        "Raleway-Bold",
        resource_path(os.path.join("fonts", "Raleway-Bold.ttf"))
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Persistence
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD    = 0.95
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
    # ROUNDING: ensure values like 0.945 (rounded to 0.95) are included in the "above"
    rounded = df["Content Quality Score"].round(2)
    above = df[rounded >= THRESHOLD].copy()
    below = df[rounded <  THRESHOLD].copy()
    return below, above

def compute_metrics(df: pd.DataFrame) -> dict:
    below, above = split_by_threshold(df)
    total       = len(df)
    count_above = len(above)
    avg_cqs_raw = df["Content Quality Score"].mean() if total else 0.0
    avg_cqs_pct = int(round(avg_cqs_raw * 100))
    pct_above   = int(round((count_above / total * 100))) if total else 0
    buybox = compute_buybox_ownership(df)
    buybox_pct = int(round(buybox * 100))

    return {
        "total":     total,
        "above":     count_above,
        "below":     len(below),
        "pct_above": pct_above,                # int percent (e.g. 83)
        "avg_cqs":   avg_cqs_pct,              # int percent (e.g. 92)
        "buybox":    buybox_pct,               # int percent
        "threshold": int(round(THRESHOLD*100)) # for display (95)
    }

def get_top_skus(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df
        .sort_values("Content Quality Score", ascending=False)
        .head(TOP_N)[["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )
    # Convert Content Quality Score to percent and round
    table["Content Quality Score"] = (table["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"
    return table

def get_skus_below(df: pd.DataFrame) -> pd.DataFrame:
    # Apply same rounding before comparison for consistency
    rounded = df["Content Quality Score"].round(2)
    table = (
        df[rounded < THRESHOLD]
        [["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )
    table["Content Quality Score"] = (table["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"
    return table

def compute_buybox_ownership(df: pd.DataFrame) -> float:
    if "Buy Box Winner" not in df.columns:
        return 0.0
    total = len(df)
    if total == 0:
        return 0.0
    buybox_yes = df["Buy Box Winner"].str.strip().str.lower().eq("yes").sum()
    return buybox_yes / total  # decimal fraction


# ─────────────────────────────────────────────────────────────────────────────
# Pie Chart Helper
# ─────────────────────────────────────────────────────────────────────────────
from io import BytesIO
import matplotlib.pyplot as plt

def make_pie_bytes(metrics: dict) -> BytesIO:
    threshold = int(metrics.get("threshold", .95))
    below = int(metrics.get("below", 0))
    above = int(metrics.get("above", 0))
    
    # If no data, show as all "below"
    if below == 0 and above == 0:
        below = 1
        above = 0

    # Use your preferred navy/teal
    colors = ["#002c47", "#4bc3cf"]

    # Make pie chart
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=100)  # 2.2" matches your usage, but you can tweak
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Draw pie with a crisp border between slices
    ax.pie(
        [below, above],
        colors=colors,
        startangle=90,
        labels=None,
        autopct=None,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # This is the key for a crisp white border
    )
    ax.set(aspect="equal")  # Keep it a circle
    ax.axis("off")          # Hide all axes/ticks

    # Transparent figure background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.0)  # Small pad for no cropping
    buf.seek(0)
    plt.close(fig)
    return buf
# ─────────────────────────────────────────────────────────────────────────────
# PDF Generation via ReportLab
# ─────────────────────────────────────────────────────────────────────────────
# ...existing code...

def generate_full_report(
    data_src, 
    client_name: str, 
    report_date: str, 
    client_notes: str,
    logo_path: str = None
) -> bytes:
    # Load data & compute
    df      = load_dataframe(data_src)
    metrics = compute_metrics(df)
    top5    = get_top_skus(df)
    below   = get_skus_below(df)

    # Prepare canvas
    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    # Colors & Spacing
    margin      = inch * 0.75
    section_gap = 0.55 * inch
    navy        = colors.HexColor("#002c47")
    teal        = colors.HexColor("#4bc3cf")
    panel_bg    = colors.HexColor("#ffffff")
    header_bg   = navy
    row_bg      = colors.HexColor("#eaf3fa")

    # Panel Sizes and Positions
    panel_w = 3.7 * inch
    panel_h = 3.6 * inch
    panel_y = h - margin - 1.0 * inch

    panels_total_width = 2 * panel_w + 0.15 * inch
    side_margin = (w - panels_total_width) / 2
    pie_panel_x = side_margin
    summary_panel_x = pie_panel_x + panel_w + 0.15 * inch

    # Header (now left-aligned to pie_panel_x)
    base_path = pathlib.Path(__file__).parent.resolve()
    logo_path = base_path / "logo.png"

    print(f"Resolved logo path: {logo_path}")
    print(f"Logo file exists? {logo_path.is_file()}")

    if logo_path.is_file():
        logo = ImageReader(str(logo_path))
        width = 1.3 * inch       # smaller width
        height = 1.3 * inch      # smaller height
        x_margin = 0.4 * inch    # horizontal margin from right edge
        y_margin = 0.45 * inch    # vertical margin from top edge
        x = w - x_margin - width
        y = h - y_margin - height
        c.drawImage(
        logo,
        x=x,
        y=y,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto"
    )
    else:
        print("Logo file not found; skipping logo drawing.")


    c.setFillColor(teal)
    c.setFont("Raleway-Bold", 19)
    c.drawString(pie_panel_x, h - margin - 0.0 * inch, client_name)

    c.setFillColor(navy)
    c.setFont("Raleway", 22)
    c.drawString(pie_panel_x, h - margin - 0.4 * inch, "Weekly Content Reporting")

    c.setFont("Raleway", 15)
    c.setFillColor(navy)
    c.drawString(pie_panel_x, h - margin - 0.71 * inch, report_date)

    # Pie Chart Panel (LEFT)
    c.setFillColor(panel_bg)
    c.roundRect(pie_panel_x, panel_y - panel_h, panel_w, panel_h, radius=10, stroke=0, fill=1)
    c.setStrokeColor(navy)
    c.roundRect(pie_panel_x, panel_y - panel_h, panel_w, panel_h, radius=10, stroke=1, fill=0)

    # Panel title - bigger and lower
    title_fontsize = 22
    title_y_offset = 38
    c.setFont("Raleway-Bold", title_fontsize)
    c.setFillColor(navy)
    c.drawCentredString(pie_panel_x + panel_w / 2, panel_y - title_y_offset, "Score Distribution")

    # Pie chart centered
    pie_buf = make_pie_bytes(metrics)
    pie = ImageReader(pie_buf)
    pie_size = 2.2 * inch
    c.drawImage(
        pie,
        x=pie_panel_x + (panel_w - pie_size) / 2,
        y=panel_y - panel_h / 2 - pie_size / 2 - 0.1 * inch,
        width=pie_size,
        height=pie_size
    )

    # Legend
    legend_y = panel_y - panel_h + 22
    square_size = 9
    gap = 7

    below_box_x = pie_panel_x + 40
    c.setFillColor(navy)
    c.rect(below_box_x, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Raleway", 10)
    c.drawString(below_box_x + square_size + gap, legend_y + 1, f"Below {int(metrics['threshold'])}%")

    above_box_x = below_box_x + 120
    c.setFillColor(teal)
    c.rect(above_box_x, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(above_box_x + square_size + gap, legend_y + 1, f"Above {int(metrics['threshold'])}%")

    # Summary Panel (Bullets Box, RIGHT, teal background)
    box_w = panel_w
    box_h = panel_h
    box_x = summary_panel_x
    box_y = panel_y

    c.setFillColor(panel_bg)
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=10, stroke=0, fill=1)
    c.setStrokeColor(teal)
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=10, stroke=1, fill=0)

    bullet_offset_x = 16
    text_offset_x = 38
    line_height = 32

    summary_title_y = box_y - title_y_offset
    bullets_start_y = summary_title_y - 50

    summary_title_x = box_x + box_w / 2
    c.setFont("Raleway-Bold", 22)
    c.setFillColor(navy)
    c.drawCentredString(summary_title_x, summary_title_y, "Summary")

    y = bullets_start_y
    c.setFont("Raleway", 16)
    for label, key in [
        ("Average CQS",        "avg_cqs"),
        ("SKUs Above 95%",     "above"),
        ("SKUs Below 95%",     "below"),
        ("Buybox Ownership",   "buybox"),
    ]:
        val = metrics.get(key, "")
        if key in ("above", "below") and val != "":
            val = int(val)
        if key in ("avg_cqs", "buybox") and val != "":
            val = f"{val}%"
        c.setFillColor(navy)
        c.setStrokeColor(navy)
        c.circle(box_x + bullet_offset_x, y + 4, 4, fill=1)
        c.setFillColor(navy)
        c.setFont("Raleway", 16)
        c.drawString(box_x + text_offset_x, y, f"{label}: {val}")
        y -= line_height

    c.setFillColor(colors.black)

    # Top 5 Table Section (now tighter spacing)
    table_title_y = panel_y - panel_h - 32
    c.setFont("Raleway-Bold", 18)
    c.setFillColor(navy)
    c.drawString(pie_panel_x, table_title_y, "Top 5 SKUs by Content Quality Score")

    # Table data and style
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleN.fontName = "Raleway"
    styleN.fontSize = 10

    data = [top5.columns.tolist()]
    for row in top5.astype(str).values.tolist():
        row[0] = Paragraph(row[0], styleN)
        data.append(row)

    table_w = w - 2 * pie_panel_x
    col_widths = [table_w * 0.5, table_w * 0.20, table_w * 0.28]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Raleway"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), row_bg),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#002c47")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("ALIGN", (2,1), (2,-1), "CENTER"),
    ]))
    tw, th = table.wrap(table_w, h)
    table.drawOn(c, pie_panel_x, table_title_y - 14 - th)

    # --- Content Notes Section (BOTTOM) ---
    table_bottom_y = table_title_y - 14 - th
    spacing = 48
    box_y = table_bottom_y - spacing

    convention_text = client_notes.replace("\n", "<br/>")
    c.setFont("Raleway-Bold", 18)
    c.setFillColor(navy)
    title_y = box_y + 20
    c.drawString(pie_panel_x, title_y, "Content Updates")

    para_style = ParagraphStyle(
        name='ConventionBox',
        fontName="Raleway",
        fontSize=15,
        leading=20,
        textColor=navy,
        spaceAfter=0,
    )

    box_x = pie_panel_x
    box_w = table_w
    box_padding = 20
    para_width = box_w - 2 * box_padding

    para = Paragraph(convention_text, para_style)
    _, para_height = para.wrap(para_width, h)
    box_height = para_height + 2 * box_padding

    c.roundRect(box_x, box_y - box_height, box_w, box_height, radius=10, stroke=0, fill=0)
    gap = 5
    note_y = title_y - gap
    para.drawOn(c, pie_panel_x, note_y - para_height)

    c.setFont("Raleway", 8)
    c.setFillColor(colors.HexColor("#200453"))
    c.drawCentredString(w / 2, 0.45 * inch, f"Generated by Soapbox Retail • {datetime.now().strftime('%B %d, %Y')}")
    c.setFillColor(colors.black)

    c.save()
    buf.seek(0)
    return buf.getvalue()


  # <--- THIS LINE is important!

# ...existing code...
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
