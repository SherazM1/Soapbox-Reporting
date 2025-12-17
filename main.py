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
from typing import Optional, Set, Tuple



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
def _safe_register_font(name: str, rel_path: str) -> None:
    """Register a TTF font if present; never raise at import-time."""
    try:
        full_path = resource_path(os.path.join("fonts", rel_path))
        if os.path.isfile(full_path):
            pdfmetrics.registerFont(TTFont(name, full_path))
        else:
            # Optional: log to console; do not fail import
            print(f"[fonts] Skipping {name}: not found at {full_path}")
    except Exception as e:
        print(f"[fonts] Skipping {name}: {e}")

# Register Raleway variants only if available
_safe_register_font("Raleway", "Raleway-Regular.ttf")
_safe_register_font("Raleway-Bold", "Raleway-Bold.ttf")

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
    

# Search Insights schema 
SEARCH_INSIGHTS_REQUIRED = [
    "Item ID",
    "Item Name",
    "Impressions Rank",
    "Clicks Rank",
    "Added to Cart Rank",
    "Sales Rank",
]

def load_search_insights(src) -> pd.DataFrame:
    """
    """
    df = load_dataframe(src)

    # Validate required headers
    missing = [c for c in SEARCH_INSIGHTS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Search Insights file is missing required columns: "
            + ", ".join(missing)
        )

    # Preserve leading zeros / ensure strings
    df["Item ID"] = df["Item ID"].astype(str).str.strip()
    df["Item Name"] = df["Item Name"].astype(str).str.strip()

    # Coerce ranks to integers (nullable)
    rank_cols = [
        "Impressions Rank",
        "Clicks Rank",
        "Added to Cart Rank",
        "Sales Rank",
    ]
    for col in rank_cols:
        # tolerant parse; blanks become <NA>
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Return only the required columns in the canonical order
    return df[SEARCH_INSIGHTS_REQUIRED].copy()

# Inventory schema 
INVENTORY_REQUIRED = [
    "Item ID",
    "Item Name",
    "Daily sales",
    "Daily units sold",
    "Stock status",
]

# Lowercase alias map for each canonical column
_INVENTORY_ALIASES = {
    "Item ID": {"item id", "item_id", "id"},
    "Item Name": {"item name", "item_name", "product name", "name"},
    "Daily sales": {"daily sales", "net sales", "sales ($)", "sales"},
    "Daily units sold": {"daily units sold", "units sold", "daily units"},
    "Stock status": {"stock status", "status"},
}

def _resolve_inventory_columns(cols):
    """Map actual headers to canonical names using case-insensitive aliases."""
    lc_to_actual = {str(c).strip().lower(): c for c in cols}
    mapping = {}
    missing = []
    for canon, aliases in _INVENTORY_ALIASES.items():
        found = next((lc_to_actual[a] for a in aliases if a in lc_to_actual), None)
        if found is None:
            missing.append(canon)
        else:
            mapping[canon] = found
    return mapping, missing

def load_inventory(src) -> pd.DataFrame:
    """
    Read the Inventory report and return required columns with clean types.

    Returns columns (in order):
      Item ID (str), Item Name (str),
      Daily sales (float), Daily units sold (float),
      Stock status (str: In Stock | Out of Stock | At Risk | other)
    """
    df = load_dataframe(src)

    mapping, missing = _resolve_inventory_columns(df.columns)
    if missing:
        raise ValueError(
            "Inventory file is missing required columns: " + ", ".join(missing)
        )

    # Select and rename to canonical headers
    df = df.rename(columns={v: k for k, v in mapping.items()})[INVENTORY_REQUIRED].copy()

    # Types
    df["Item ID"] = df["Item ID"].astype(str).str.strip()
    df["Item Name"] = df["Item Name"].astype(str).str.strip()

    for col in ("Daily sales", "Daily units sold"):
        # currency/float tolerant parsing
        df[col] = (
            pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False),
                errors="coerce",
            )
            .astype(float)
        )

    # Normalize status to title-case, but keep original values if outside expected set
    def _norm_status(s):
        s = str(s).strip().lower()
        if s in {"in stock", "instock"}:
            return "In Stock"
        if s in {"out of stock", "oos"}:
            return "Out of Stock"
        if s in {"at risk", "risk"}:
            return "At Risk"
        return s.title() if s else ""

    df["Stock status"] = df["Stock status"].map(_norm_status)

    return df

# Item Sales schema (canonical column names)
ITEM_SALES_REQUIRED = [
    "Item ID",
    "Item Name",
    "Orders",
    "Units Sold",
    "Auth Sales",
    "Item pageviews",
    "Item conversion",
    # "SKU" is optional; included when present
]

# Case/underscore-insensitive aliases for each canonical field
_ITEM_SALES_ALIASES = {
    # Canonical Item ID comes ONLY from Item_id. We do not use Base_Item_Id.
    "Item_id": {"item_id", "item id", "itemid"},
    "Base_Item_Id": {"base_item_id", "base item id", "baseitemid"},  # parsed if present (used only for gated fallback)
    "Item Name": {"item_name", "item name"},
    "Orders": {"orders", "order"},
    "Units Sold": {"units_sold", "units sold"},
    "Auth Sales": {"auth_sales", "auth sales", "gmv", "net sales"},
    "Item pageviews": {"item_pageviews", "item pageviews"},
    "Item conversion": {"item_conversion", "item conversion", "conversion", "item_conver"},
    "SKU": {"sku", "sku #", "sku#", "sku number", "sku_no", "sku id", "skuid"},
}

def _normalize_header_map(cols):
    """Map normalized header → actual header for alias lookup."""
    def norm(s):
        return str(s).strip().lower().replace("_", "").replace(" ", "")
    return {norm(c): c for c in cols}

def load_item_sales(src) -> pd.DataFrame:
    # Force-read as strings to avoid scientific-notation/precision loss on IDs.
    if hasattr(src, "read") and hasattr(src, "name"):
        data = src.getvalue()
        ext = os.path.splitext(src.name)[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        elif ext == ".csv":
            df = pd.read_csv(
                io.BytesIO(data),
                dtype=str,
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip",
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        ext = os.path.splitext(str(src))[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(src, dtype=str)
        elif ext == ".csv":
            df = pd.read_csv(
                src,
                dtype=str,
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip",
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    norm_map = _normalize_header_map(df.columns)

    # Resolve Item_id (canonical ID source). DO NOT use Base_Item_Id for canonical ID.
    item_id_actual = None
    for candidate in _ITEM_SALES_ALIASES["Item_id"]:
        key = candidate.replace("_", "").replace(" ", "")
        if key in norm_map:
            item_id_actual = norm_map[key]
            break

    # Resolve the rest (Base_Item_Id kept only for gated fallback / debug)
    def resolve(canon):
        for alias in _ITEM_SALES_ALIASES[canon]:
            key = alias.replace("_", "").replace(" ", "")
            if key in norm_map:
                return norm_map[key]
        return None

    base_item_id_actual = resolve("Base_Item_Id")
    item_name_actual    = resolve("Item Name")
    orders_actual       = resolve("Orders")
    units_actual        = resolve("Units Sold")
    auth_sales_actual   = resolve("Auth Sales")
    pageviews_actual    = resolve("Item pageviews")
    conversion_actual   = resolve("Item conversion")
    sku_actual          = resolve("SKU")  # optional

    missing = []
    if item_name_actual is None:
        missing.append("Item Name")
    if item_id_actual is None:
        missing.append("Item ID (Item_id)")
    for canon, actual in [
        ("Orders", orders_actual),
        ("Units Sold", units_actual),
        ("Auth Sales", auth_sales_actual),
        ("Item pageviews", pageviews_actual),
        ("Item conversion", conversion_actual),
    ]:
        if actual is None:
            missing.append(canon)

    if missing:
        raise ValueError(
            "Item Sales file is missing required columns: " + ", ".join(missing)
        )

    # Rename → canonical working columns
    rename_map = {item_id_actual: "Item_id_raw"}
    if base_item_id_actual:
        rename_map[base_item_id_actual] = "Base_Item_Id"  # visible for gated fallback
    rename_map.update(
        {
            item_name_actual: "Item Name",
            orders_actual: "Orders",
            units_actual: "Units Sold",
            auth_sales_actual: "Auth Sales",
            pageviews_actual: "Item pageviews",
            conversion_actual: "Item conversion",
        }
    )
    if sku_actual:
        rename_map[sku_actual] = "SKU"

    df_work = df.rename(columns=rename_map)

    # Canonical "Item ID" comes STRICTLY from Item_id_raw
    s_item = df_work.get("Item_id_raw")
    if s_item is not None:
        s_item = s_item.astype(str).str.strip()
    df_work["Item ID"] = s_item

    # Restrict to canonical columns (+ optional SKU, Base_Item_Id for gated fallback)
    cols = [
        "Item ID",
        "Item Name",
        "Orders",
        "Units Sold",
        "Auth Sales",
        "Item pageviews",
        "Item conversion",
    ]
    if "SKU" in df_work.columns:
        cols.append("SKU")
    if "Base_Item_Id" in df_work.columns:
        cols.append("Base_Item_Id")  # used only in fallback when name confirms identity
    df_work = df_work[cols].copy()

    # Types/coercions (only non-ID fields)
    df_work["Item ID"] = df_work["Item ID"].astype(str).str.strip()
    df_work["Item Name"] = df_work["Item Name"].astype(str).str.strip()
    if "SKU" in df_work.columns:
        df_work["SKU"] = df_work["SKU"].astype(str).str.strip()

    for col in ("Orders", "Units Sold", "Item pageviews"):
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce").astype("Int64")

    df_work["Auth Sales"] = (
        df_work["Auth Sales"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    df_work["Auth Sales"] = pd.to_numeric(
        df_work["Auth Sales"],
        errors="coerce",
    ).astype(float)

    df_work["Item conversion"] = pd.to_numeric(
        df_work["Item conversion"],
        errors="coerce",
    ).astype(float)

    return df_work


# -------- Robust Item Name normalization (for ID-or-Name & gated fallback) --------
import re, unicodedata

def _norm_name_strict(s: str) -> str:
    """
    Robust canonical form for equality across files:
      - Unicode NFKC
      - remove trademark/encoding artifacts: ® ™ ¬Æ
      - unify dashes: – — → -
      - drop most punctuation/separators (non-alnum & non-hyphen) → space
      - collapse whitespace
      - casefold
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("®", "").replace("™", "").replace("¬Æ", "")
    s = s.replace("\u00AE", "").replace("\u2122", "")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^0-9A-Za-z\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().casefold()


_MANAGED_ALIASES = {
    "Item ID": {"item id", "item_id", "itemid", "base item id", "base_item_id", "baseitemid"},
    "Item Name": {"item name", "item_name", "product name", "name"},
}

def _normalize_headers_map(cols):
    def norm(s):
        return str(s).strip().lower().replace("_", "").replace(" ", "")
    return {norm(c): c for c in cols}

def _resolve_managed_columns(cols) -> Tuple[Optional[str], Optional[str]]:
    """Return actual header names for Item ID and Item Name if present."""
    norm_map = _normalize_headers_map(cols)
    item_id_actual = None
    item_name_actual = None

    for a in _MANAGED_ALIASES["Item ID"]:
        k = a.replace("_", "").replace(" ", "")
        if k in norm_map:
            item_id_actual = norm_map[k]
            break
    for a in _MANAGED_ALIASES["Item Name"]:
        k = a.replace("_", "").replace(" ", "")
        if k in norm_map:
            item_name_actual = norm_map[k]
            break
    return item_id_actual, item_name_actual

def load_managed_keys(src) -> Tuple[Set[str], Set[str]]:
    """
    Read Managed SKUs list, returning (ids_set, names_set).
    Source of truth for filtering by Item ID OR Item Name; also used for gated Base-ID fallback.
    """
    df = load_dataframe(src)
    item_id_actual, item_name_actual = _resolve_managed_columns(df.columns)
    if not item_id_actual and not item_name_actual:
        raise ValueError("Managed SKUs file must include 'Item ID' or 'Item Name'.")

    ids_set: Set[str] = set()
    names_set: Set[str] = set()

    if item_id_actual:
        ids_set = set(
            df[item_id_actual].astype(str).str.strip().replace("nan", pd.NA).dropna().tolist()
        )
    if item_name_actual:
        names_set = set(
            df[item_name_actual]
            .astype(str)
            .map(_norm_name_strict)
            .replace("nan", pd.NA)
            .dropna()
            .tolist()
        )
    return ids_set, names_set

def filter_by_managed(df: pd.DataFrame, ids_set: Set[str], names_set: Set[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Filter to managed SKUs with a controlled Base-ID fallback.

    Rule:
      1) Keep a row if (Item ID ∈ managed_ids) OR (normalized Item Name ∈ managed_names).
      2) If still not matched, allow a gated fallback:
         Keep the row if (Base_Item_Id ∈ managed_ids) AND (normalized Item Name ∈ managed_names).
         (Prevents unrelated child variants from leaking in.)

    Returns (filtered_df, stats).
    """
    total = len(df)
    if total == 0:
        return df.copy(), {
            "total": 0,
            "matched": 0,
            "unmatched_count": len(ids_set),
            "unmatched_sample": [],
        }

    # Normalize series
    id_series = df["Item ID"].astype(str).str.strip() if "Item ID" in df.columns else None
    name_series = df["Item Name"].astype(str) if "Item Name" in df.columns else None
    name_series_norm = name_series.map(_norm_name_strict) if name_series is not None else None

    base_series = df["Base_Item_Id"].astype(str).str.strip() if "Base_Item_Id" in df.columns else None

    ids_set_norm = {s.strip() for s in ids_set} if ids_set else set()
    names_set_norm = {s for s in names_set} if names_set else set()

    # 1) Primary: Item ID or Name
    id_mask   = id_series.isin(ids_set_norm) if (id_series is not None and ids_set_norm) else False
    name_mask = name_series_norm.isin(names_set_norm) if (name_series_norm is not None and names_set_norm) else False
    primary_mask = id_mask | name_mask

    # 2) Gated fallback: Base ID + Name
    base_mask = False
    if base_series is not None and ids_set_norm and name_series_norm is not None and names_set_norm:
        base_mask = base_series.isin(ids_set_norm) & name_series_norm.isin(names_set_norm)

    mask = primary_mask | base_mask
    filtered = df[mask].copy()
    matched = int(mask.sum())

    # Unmatched sample based on managed IDs not present as Item ID in result (debug aid)
    unmatched_sample: list[str] = []
    if id_series is not None and ids_set_norm and "Item ID" in filtered.columns:
        matched_ids = set(filtered["Item ID"].astype(str).str.strip())
        unmatched = ids_set_norm - matched_ids
        unmatched_sample = list(sorted(unmatched))[:10]

    return filtered, {
        "total": total,
        "matched": matched,
        "unmatched_count": len(unmatched_sample),
        "unmatched_sample": unmatched_sample,
    }

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

def generate_3p_report(
    item_sales_src,
    inventory_src,
    search_insights_src,
    managed_src,
    mode: str,                                # "catalog" | "managed"
    client_name: str,
    report_date: str,
    top_skus_text: str = "",                  # manual (Item Sales box)
    inventory_callouts_text: str = "",        # manual (Inventory box)
    search_highlights_text: str = "",         # manual (Search box)
    advertising_src=None,                     # NEW: Advertising report (CSV/XLSX)
    advertising_notes_text: str = "",         # NEW: manual notes for Advertising
    logo_path: str | None = None
) -> bytes:
    """
    3P template PDF populated from three reports after Catalog/Managed filtering,
    plus an independent Advertising report.

    Item Sales/Auth/Conversion are computed from Item Sales (managed-aware).
    Inventory metrics from Inventory (managed-aware).
    Search metrics from Search Insights (managed-aware).
    Advertising metrics (sum Ad Spend, sum ROAS, avg Conversion Rate) are independent
    of Catalog/Managed and use only the uploaded Advertising file.
    Manual text areas are rendered as plain text (no dashed boxes). Returns PDF bytes.
    """
    # ── Load data
    df_sales  = load_item_sales(item_sales_src)
    df_inv    = load_inventory(inventory_src)
    df_search = load_search_insights(search_insights_src)

    # ── Managed filter (ID OR Name + gated Base+Name as implemented elsewhere)
    if str(mode).strip().lower() == "managed":
        ids_set, names_set = load_managed_keys(managed_src)
        df_sales,  _ = filter_by_managed(df_sales,  ids_set, names_set)
        df_inv,    _ = filter_by_managed(df_inv,    ids_set, names_set)
        df_search, _ = filter_by_managed(df_search, ids_set, names_set)

    # ── Helpers
    def _mean_conversion_pp_excel(series: pd.Series) -> float:
        """
        Excel-like mean for conversion:
        - Blanks ignored (NaN); zeros included.
        - Detect scale: if most non-null values ≤ 1 → treat as fractions (×100).
        - Return mean as percent points (0..100).
        """
        if series is None or len(series) == 0:
            return 0.0
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
        if (nonnull <= 1.0).mean() >= 0.8:  # likely 0..1
            nonnull = nonnull * 100.0
        return float(nonnull.mean())

    def _avg_impressions_rank_whole(series: pd.Series) -> int:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return int(round(float(s.mean()))) if len(s) else 0

    def _count_top_n(series: pd.Series, n: int) -> int:
        s = pd.to_numeric(series, errors="coerce")
        return int((s <= n).sum())

    def _writein_text(x: float, y: float, w_: float, title: str, body_text: str, min_height: float = 80.0) -> float:
        """Title + optional body text. No box. Returns bottom Y for spacing."""
        c.setFont("Raleway-Bold", 14)
        c.setFillColor(navy)
        c.drawString(x, y, title)
        content = (body_text or "").strip()
        used_h = 0.0
        if content:
            para_style = ParagraphStyle(
                name="ManualText",
                fontName="Raleway",
                fontSize=12,
                leading=16,
                textColor=navy,
            )
            para = Paragraph(content.replace("\n", "<br/>"), para_style)
            usable_w = w_
            _, ph = para.wrap(usable_w, min_height)
            para.drawOn(c, x, y - 16 - ph)
            used_h = 16 + ph  # title gap + paragraph
        block_h = max(min_height, used_h or 16)
        return y - block_h

    # --- Advertising loader (internal, tolerant) ---
    def _load_advertising_df(src) -> pd.DataFrame:
        """
        Reads CSV/XLSX and returns a df with three canonical columns:
          - Ad Spend (currency)
          - Conversion Rate – 14 Day (percent)
          - RoAS – 14 Day (currency)
        Only these three columns are used; others are ignored.
        """
        if src is None:
            return pd.DataFrame(columns=["Ad Spend", "Conversion Rate – 14 Day", "RoAS – 14 Day"])
        df_raw = load_dataframe(src)

        # Build a case/spacing/alias-insensitive map
        def _norm(s): return str(s).strip().lower().replace("_", "").replace(" ", "")
        colmap = {_norm(c): c for c in df_raw.columns}

        def _find(*aliases):
            for a in aliases:
                k = _norm(a)
                if k in colmap:
                    return colmap[k]
            return None

        col_spend = _find("Ad Spend", "Spend", "Ad_Spend", "adspend")
        col_conv  = _find("Conversion Rate – 14 Day", "Conversion Rate - 14 Day", "Conversion Rate 14 Day", "Conversion Rate", "conv rate", "conversionrate")
        col_roas  = _find("RoAS – 14 Day", "RoAS - 14 Day", "RoAS 14 Day", "ROAS", "roas")

        # Create a working frame with only the needed columns (tolerant missing -> empty)
        df = pd.DataFrame()
        if col_spend is not None:
            df["Ad Spend"] = df_raw[col_spend]
        else:
            df["Ad Spend"] = pd.Series(dtype=float)

        if col_conv is not None:
            df["Conversion Rate – 14 Day"] = df_raw[col_conv]
        else:
            df["Conversion Rate – 14 Day"] = pd.Series(dtype=float)

        if col_roas is not None:
            df["RoAS – 14 Day"] = df_raw[col_roas]
        else:
            df["RoAS – 14 Day"] = pd.Series(dtype=float)

        # Coercions
        def _to_currency(s: pd.Series) -> pd.Series:
            return pd.to_numeric(
                s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            ).fillna(0.0)

        def _to_percent_vals(s: pd.Series) -> pd.Series:
            # keep as 0..100 numbers; blanks ignored in mean calc later
            s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
            s = pd.to_numeric(s, errors="coerce")  # NaN for blanks
            # If mostly ≤1, treat as fraction → ×100
            nonnull = s.dropna()
            if not nonnull.empty and (nonnull <= 1.0).mean() >= 0.8:
                s = s * 100.0
            return s

        df["Ad Spend"] = _to_currency(df["Ad Spend"])
        df["RoAS – 14 Day"] = _to_currency(df["RoAS – 14 Day"])
        df["Conversion Rate – 14 Day"] = _to_percent_vals(df["Conversion Rate – 14 Day"])
        return df

    # ── Compute metrics (Excel-consistent)
    units_total = int(pd.to_numeric(df_sales.get("Units Sold"), errors="coerce").fillna(0).sum()) if not df_sales.empty else 0
    # keep cents for Auth Sales; display with 2 decimals
    sales_total = float(pd.to_numeric(df_sales.get("Auth Sales"), errors="coerce").sum()) if not df_sales.empty else 0.0
    # Avg conversion: ignore blanks, include zeros; 2 decimals for display
    avg_conv_pct = _mean_conversion_pp_excel(df_sales.get("Item conversion", pd.Series(dtype=float))) if not df_sales.empty else 0.0

    # Inventory (support "Status" or "Stock status")
    status_col = "Status" if "Status" in df_inv.columns else ("Stock status" if "Stock status" in df_inv.columns else None)
    in_stock_rate_pct = 0
    oos_count = 0
    at_risk_count = 0
    if status_col and not df_inv.empty:
        status_norm = df_inv[status_col].astype(str).str.strip().str.lower()
        total_inv = len(status_norm)
        if total_inv > 0:
            in_stock_rate_pct = int(round((status_norm == "in stock").sum() / total_inv * 100))
        oos_count = int((status_norm == "out of stock").sum())
        at_risk_count = int((status_norm == "at risk").sum())

    # Search Insights
    avg_impr_rank = _avg_impressions_rank_whole(df_search.get("Impressions Rank", pd.Series(dtype=float))) if not df_search.empty else 0
    top10_impr = _count_top_n(df_search.get("Impressions Rank", pd.Series(dtype=float)), 10) if not df_search.empty else 0
    top10_sales = _count_top_n(df_search.get("Sales Rank", pd.Series(dtype=float)), 10) if not df_search.empty else 0

    # Advertising (independent)
    df_adv = _load_advertising_df(advertising_src)
    if df_adv.empty:
        adv_spend_total = 0.0
        adv_roas_total = 0.0
        adv_conv_avg = 0.0
    else:
        adv_spend_total = float(df_adv["Ad Spend"].sum())
        adv_roas_total  = float(df_adv["RoAS – 14 Day"].sum())
        nonnull_conv = df_adv["Conversion Rate – 14 Day"].dropna()
        adv_conv_avg = float(nonnull_conv.mean()) if len(nonnull_conv) else 0.0

    # ── PDF drawing
    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    margin   = inch * 0.75
    navy     = colors.HexColor("#002c47")
    teal     = colors.HexColor("#4bc3cf")
    panel_bg = colors.white

    # Header
    base_path = pathlib.Path(__file__).parent.resolve()
    auto_logo = base_path / "logo.png"
    chosen_logo = (pathlib.Path(logo_path) if logo_path else auto_logo) if (logo_path or auto_logo.is_file()) else None
    if chosen_logo and chosen_logo.is_file():
        logo   = ImageReader(str(chosen_logo))
        width  = 1.3 * inch; height = 1.3 * inch
        x_margin, y_margin = 0.4 * inch, 0.45 * inch
        c.drawImage(logo, x=w - x_margin - width, y=h - y_margin - height, width=width, height=height, preserveAspectRatio=True, mask="auto")

    c.setFillColor(teal);  c.setFont("Raleway-Bold", 19); c.drawString(margin, h - margin - 0.00 * inch, client_name)
    c.setFillColor(navy);  c.setFont("Raleway", 22);      c.drawString(margin, h - margin - 0.40 * inch, "Weekly Reporting")
    c.setFont("Raleway", 15); c.setFillColor(navy);       c.drawString(margin, h - margin - 0.71 * inch, report_date)

    # Grid
    card_w = 3.7 * inch; card_h = 3.6 * inch; gap = 0.15 * inch
    row1_top = h - margin - 1.0 * inch
    row2_top = row1_top - card_h - gap
    total_w = (2 * card_w) + gap
    side_margin = (w - total_w) / 2.0
    x_left  = side_margin
    x_right = x_left + card_w + gap

    def _draw_card_shell(x: float, y_top: float, w_: float, h_: float, title: str) -> None:
        c.setFillColor(panel_bg)
        c.roundRect(x, y_top - h_, w_, h_, radius=10, stroke=0, fill=1)
        c.setStrokeColor(teal)
        c.roundRect(x, y_top - h_, w_, h_, radius=10, stroke=1, fill=0)
        c.setFont("Raleway-Bold", 22); c.setFillColor(navy)
        c.drawCentredString(x + w_ / 2.0, y_top - 38, title)

    def _draw_bullets(x: float, y_top: float, w_: float, start_offset: float, bullets: list[str]) -> float:
        y = y_top - start_offset
        c.setFont("Raleway", 13)
        for line in bullets:
            c.setFillColor(navy); c.setStrokeColor(navy); c.circle(x + 16, y + 3.5, 3.5, fill=1)
            c.setFillColor(navy); c.drawString(x + 38, y - 2, line)
            y -= 24
        return y

    # Box 1 — Item Sales
    x, y_top = x_left, row1_top
    _draw_card_shell(x, y_top, card_w, card_h, "Item Sales")
    bullets_sales = [
        f"Units Sold: {units_total:,}",
        f"Auth Sales: ${sales_total:,.2f}",        # cents preserved
        f"Avg Conversion: {avg_conv_pct:.2f}%",    # two decimals
    ]
    y_cursor = _draw_bullets(x + 16, y_top, card_w - 32, start_offset=38 + 26, bullets=bullets_sales)
    _ = _writein_text(x + 16, y_cursor - 6, card_w - 32, "Top SKUs:", top_skus_text, min_height=80)

    # Box 2 — Inventory
    x, y_top = x_right, row1_top
    _draw_card_shell(x, y_top, card_w, card_h, "Inventory")
    bullets_inv = [
        f"In-Stock Rate: {in_stock_rate_pct}%",
        f"OOS SKUs: {oos_count}",
        f"At-Risk SKUs: {at_risk_count}",
    ]
    y_cursor = _draw_bullets(x + 16, y_top, card_w - 32, start_offset=38 + 26, bullets=bullets_inv)
    _ = _writein_text(x + 16, y_cursor - 6, card_w - 32, "Key Callouts:", inventory_callouts_text, min_height=100)

    # Box 3 — Search Insights
    x, y_top = x_left, row2_top
    _draw_card_shell(x, y_top, card_w, card_h, "Search Insights")
    bullets_search = [
        f"Avg Impressions Rank: {avg_impr_rank}",
        f"{top10_impr} SKUs in Top 10 Impressions",
        f"{top10_sales} SKUs in Top 10 Sales Rank",
    ]
    y_cursor = _draw_bullets(x + 16, y_top, card_w - 32, start_offset=38 + 26, bullets=bullets_search)
    _ = _writein_text(x + 16, y_cursor - 6, card_w - 32, "Highlights:", search_highlights_text, min_height=100)

    # Box 4 — Advertising (now calculated + notes)
    x, y_top = x_right, row2_top
    _draw_card_shell(x, y_top, card_w, card_h, "Advertising")
    bullets_adv = [
        f"Total Ad Spend: ${adv_spend_total:,.2f}",
        f"Total ROAS: ${adv_roas_total:,.2f}",
        f"Avg Conversion Rate: {adv_conv_avg:.2f}%",
    ]
    y_cursor = _draw_bullets(x + 16, y_top, card_w - 32, start_offset=38 + 26, bullets=bullets_adv)
    _ = _writein_text(x + 16, y_cursor - 6, card_w - 32, "Notes:", advertising_notes_text, min_height=100)

    # Footer
    c.setFont("Raleway", 8); c.setFillColor(colors.HexColor("#200453"))
    c.drawCentredString(w / 2, 0.45 * inch, f"Generated by Soapbox Retail • {datetime.now().strftime('%B %d, %Y')}")
    c.setFillColor(colors.black)

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
