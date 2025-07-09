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
from io import BytesIO
import matplotlib.pyplot as plt

def make_pie_bytes(metrics: dict) -> BytesIO:
    threshold = int(metrics.get("threshold", 95))
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

    # Colors & Spacing
    margin      = inch * 0.75
    section_gap = 0.55 * inch
    navy        = colors.HexColor("#002c47")
    teal        = colors.HexColor("#4bc3cf")
    panel_bg    = colors.HexColor("#ffffff")
    header_bg   = navy
    row_bg      = colors.HexColor("#eaf3fa")

    # Header
    logo_path = resource_path("retaillogo.png")
    if os.path.isfile(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(logo,
                    x=margin,
                    y=h - margin - 1.2 * inch,
                    width=1.5 * inch,
                    preserveAspectRatio=True,
                    mask="auto")
    # Client name
    c.setFillColor(teal)
    c.setFont("Raleway", 19)
    c.drawString(margin + 1.7 * inch, h - margin - 0.5 * inch, client_name)
    # Title
    c.setFillColor(navy)
    c.setFont("Raleway", 22)
    c.drawString(margin + 1.7 * inch, h - margin - 0.9 * inch, "Weekly Content Reporting")
    # Date
    c.setFont("Raleway", 15)
    c.setFillColor(navy)
    c.drawString(margin + 1.7 * inch, h - margin - 1.21 * inch, report_date)

    # Panels: Pie Chart & Metrics
    panel_y = h - margin - 2.0 * inch
    panel_h = 3.6 * inch
    panel_w = 3.7 * inch

    # Pie Chart Panel
    c.setFillColor(panel_bg)
    c.roundRect(margin,
                panel_y - panel_h,
                panel_w,
                panel_h,
                radius=10,
                stroke=0,
                fill=1)
    c.setStrokeColor(navy)
    c.roundRect(margin,
                panel_y - panel_h,
                panel_w,
                panel_h,
                radius=10,
                stroke=1,
                fill=0)
    # Panel title - now bigger and lower
    c.setFont("Raleway", 22)
    c.setFillColor(navy)
    c.drawCentredString(margin + panel_w / 2, panel_y - 38, "Score Distribution")
    # Pie chart centered
    pie_buf = make_pie_bytes(metrics)
    pie = ImageReader(pie_buf)
    pie_size = 2.2 * inch
    c.drawImage(pie,
                x=margin + (panel_w - pie_size) / 2,
                y=panel_y - panel_h / 2 - pie_size / 2 - 0.1 * inch,
                width=pie_size,
                height=pie_size)
    # Legend
    legend_y = panel_y - panel_h + 22
    square_size = 9
    # Below
    c.setFillColor(navy)
    c.rect(margin + 18, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Raleway", 10)
    c.drawString(margin + 18 + square_size + 6, legend_y + 1, f"Below {int(metrics['threshold'])}%")
    # Above
    c.setFillColor(teal)
    x2 = margin + 120
    c.rect(x2, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(x2 + square_size + 6, legend_y + 1, f"Above {int(metrics['threshold'])}%")

    # Metrics Panel (Summary Box with Bullets) - now same size as pie panel
    box_w = panel_w
    box_h = panel_h
    box_x = margin + panel_w + 0.15 * inch
    box_y = panel_y - 0.18 * inch

    # Background + border
    c.setFillColor(panel_bg)
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=10, stroke=0, fill=1)
    c.setStrokeColor(navy)
    c.roundRect(box_x, box_y - box_h, box_w, box_h, radius=10, stroke=1, fill=0)

    # Bullets and Title
    bullet_offset_x = 16
    text_offset_x = 38
    line_height = 32

    # Title placement (centered over the text area)
    title_y = box_y - 20
    bullets_left = box_x + text_offset_x
    bullets_width = box_w - (text_offset_x - bullet_offset_x) * 2
    title_x = bullets_left + bullets_width / 2

    c.setFont("Raleway", 18)
    c.setFillColor(navy)
    c.drawCentredString(title_x - 17, title_y, "Summary")

    # Start bullets below the title
    y = title_y - 32
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
        elif isinstance(val, float):
            val = f"{val:.1f}%"
        # Draw navy bullet
        c.setFillColor(navy)
        c.circle(box_x + bullet_offset_x, y + 4, 4, fill=1)
        # Draw text
        c.setFillColor(navy)
        c.setFont("Raleway", 16)
        c.drawString(box_x + text_offset_x, y, f"{label}: {val}")
        y -= line_height

    c.setFillColor(colors.black)

    # Top 5 Table Section (now tighter spacing)
    table_title_y = panel_y - panel_h - 32
    c.setFont("Raleway", 18)
    c.setFillColor(navy)
    c.drawString(margin, table_title_y, "Top 5 SKUs by Content Quality Score")
    c.setFillColor(colors.black)

    # Table data and style
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleN.fontName = "Raleway"
    styleN.fontSize = 10

    data = [top5.columns.tolist()]
    for row in top5.astype(str).values.tolist():
        row[0] = Paragraph(row[0], styleN)  # wrap first column
        data.append(row)

    table_w = w - 2 * margin
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
    ]))
    tw, th = table.wrap(table_w, h)
    table.drawOn(c, margin, table_title_y - 14 - th)

    # Footer
    c.setFont("Raleway", 8)
    c.setFillColor(colors.HexColor("#200453"))
    c.drawCentredString(w / 2, 0.45 * inch, f"Generated by SOAPBOX • {datetime.now().strftime('%B %d, %Y')}")
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
