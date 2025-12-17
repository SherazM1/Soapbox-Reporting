# =========================================
# File: main.py
# =========================================
# main.py

import os
import sys
import io
import json
import re
import unicodedata
import pathlib
from typing import Optional, Set, Tuple
from io import BytesIO
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resource_path(rel_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────────────────────────────────────

def _safe_register_font(name: str, rel_path: str) -> None:
    """Register a TTF font if present; never raise at import-time."""
    try:
        full_path = resource_path(os.path.join("fonts", rel_path))
        if os.path.isfile(full_path):
            pdfmetrics.registerFont(TTFont(name, full_path))
        else:
            print(f"[fonts] Skipping {name}: not found at {full_path}")
    except Exception as e:
        print(f"[fonts] Skipping {name}: {e}")

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
# Data loading (generic)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataframe(src) -> pd.DataFrame:
    """
    Read a CSV/XLS/XLSX from either a Streamlit UploadedFile-like object or a file path.
    """
    # Streamlit UploadedFile
    if hasattr(src, "read") and hasattr(src, "name"):
        data = src.getvalue()
        ext  = os.path.splitext(src.name)[1].lower()
        if ext == ".csv":
            return pd.read_csv(io.BytesIO(data), encoding="utf-8", engine="python", on_bad_lines="skip")
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(io.BytesIO(data))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # Regular file path
    ext = os.path.splitext(str(src))[1].lower()
    if ext == ".csv":
        return pd.read_csv(src, encoding="utf-8", engine="python", on_bad_lines="skip")
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(src)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
# Search Insights
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_INSIGHTS_REQUIRED = [
    "Item ID", "Item Name", "Impressions Rank", "Clicks Rank", "Added to Cart Rank", "Sales Rank",
]

def load_search_insights(src) -> pd.DataFrame:
    df = load_dataframe(src)

    missing = [c for c in SEARCH_INSIGHTS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Search Insights file is missing required columns: " + ", ".join(missing))

    df["Item ID"] = df["Item ID"].astype(str).str.strip()
    df["Item Name"] = df["Item Name"].astype(str).str.strip()

    for col in ["Impressions Rank", "Clicks Rank", "Added to Cart Rank", "Sales Rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df[SEARCH_INSIGHTS_REQUIRED].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Inventory
# ─────────────────────────────────────────────────────────────────────────────

INVENTORY_REQUIRED = [
    "Item ID", "Item Name", "Daily sales", "Daily units sold", "Stock status",
]

_INVENTORY_ALIASES = {
    "Item ID": {"item id", "item_id", "id"},
    "Item Name": {"item name", "item_name", "product name", "name"},
    "Daily sales": {"daily sales", "net sales", "sales ($)", "sales"},
    "Daily units sold": {"daily units sold", "units sold", "daily units"},
    "Stock status": {"stock status", "status"},
}

def _resolve_inventory_columns(cols):
    lc_to_actual = {str(c).strip().lower(): c for c in cols}
    mapping, missing = {}, []
    for canon, aliases in _INVENTORY_ALIASES.items():
        found = next((lc_to_actual[a] for a in aliases if a in lc_to_actual), None)
        if found is None:
            missing.append(canon)
        else:
            mapping[canon] = found
    return mapping, missing

def load_inventory(src) -> pd.DataFrame:
    df = load_dataframe(src)
    mapping, missing = _resolve_inventory_columns(df.columns)
    if missing:
        raise ValueError("Inventory file is missing required columns: " + ", ".join(missing))

    df = df.rename(columns={v: k for k, v in mapping.items()})[INVENTORY_REQUIRED].copy()

    df["Item ID"] = df["Item ID"].astype(str).str.strip()
    df["Item Name"] = df["Item Name"].astype(str).str.strip()

    for col in ("Daily sales", "Daily units sold"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
            errors="coerce"
        ).astype(float)

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


# ─────────────────────────────────────────────────────────────────────────────
# Item Sales
# ─────────────────────────────────────────────────────────────────────────────

ITEM_SALES_REQUIRED = [
    "Item ID", "Item Name", "Orders", "Units Sold", "Auth Sales", "Item pageviews", "Item conversion",
]

_ITEM_SALES_ALIASES = {
    "Item_id": {"item_id", "item id", "itemid"},
    "Base_Item_Id": {"base_item_id", "base item id", "baseitemid"},  # optional for gated fallback
    "Item Name": {"item_name", "item name"},
    "Orders": {"orders", "order"},
    "Units Sold": {"units_sold", "units sold"},
    "Auth Sales": {"auth_sales", "auth sales", "gmv", "net sales"},
    "Item pageviews": {"item_pageviews", "item pageviews"},
    "Item conversion": {"item_conversion", "item conversion", "conversion", "item_conver"},
    "SKU": {"sku", "sku #", "sku#", "sku number", "sku_no", "sku id", "skuid"},
}

def _normalize_header_map(cols):
    def norm(s): return str(s).strip().lower().replace("_", "").replace(" ", "")
    return {norm(c): c for c in cols}

def load_item_sales(src) -> pd.DataFrame:
    # For IDs, read as strings
    if hasattr(src, "read") and hasattr(src, "name"):
        data = src.getvalue()
        ext = os.path.splitext(src.name)[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        elif ext == ".csv":
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        ext = os.path.splitext(str(src))[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(src, dtype=str)
        elif ext == ".csv":
            df = pd.read_csv(src, dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    norm_map = _normalize_header_map(df.columns)

    # Resolve Item_id (canonical for Item ID)
    item_id_actual = None
    for candidate in _ITEM_SALES_ALIASES["Item_id"]:
        key = candidate.replace("_", "").replace(" ", "")
        if key in norm_map:
            item_id_actual = norm_map[key]
            break

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
    sku_actual          = resolve("SKU")

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
        raise ValueError("Item Sales file is missing required columns: " + ", ".join(missing))

    rename_map = {item_id_actual: "Item_id_raw"}
    if base_item_id_actual:
        rename_map[base_item_id_actual] = "Base_Item_Id"
    rename_map.update({
        item_name_actual: "Item Name",
        orders_actual: "Orders",
        units_actual: "Units Sold",
        auth_sales_actual: "Auth Sales",
        pageviews_actual: "Item pageviews",
        conversion_actual: "Item conversion",
    })
    if sku_actual:
        rename_map[sku_actual] = "SKU"

    df_work = df.rename(columns=rename_map)

    s_item = df_work.get("Item_id_raw")
    if s_item is not None:
        s_item = s_item.astype(str).str.strip()
    df_work["Item ID"] = s_item

    cols = ["Item ID", "Item Name", "Orders", "Units Sold", "Auth Sales", "Item pageviews", "Item conversion"]
    if "SKU" in df_work.columns:
        cols.append("SKU")
    if "Base_Item_Id" in df_work.columns:
        cols.append("Base_Item_Id")
    df_work = df_work[cols].copy()

    # Types/coercions
    df_work["Item ID"] = df_work["Item ID"].astype(str).str.strip()
    df_work["Item Name"] = df_work["Item Name"].astype(str).str.strip()
    if "SKU" in df_work.columns:
        df_work["SKU"] = df_work["SKU"].astype(str).str.strip()

    for col in ("Orders", "Units Sold", "Item pageviews"):
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce").astype("Int64")

    df_work["Auth Sales"] = (
        df_work["Auth Sales"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    )
    df_work["Auth Sales"] = pd.to_numeric(df_work["Auth Sales"], errors="coerce").astype(float)

    df_work["Item conversion"] = pd.to_numeric(df_work["Item conversion"], errors="coerce").astype(float)

    return df_work


# ─────────────────────────────────────────────────────────────────────────────
# Managed SKUs (IDs / Names) + filtering
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name_strict(s: str) -> str:
    """
    Canonical, sturdy normalization for names across reports:
      - Unicode NFKC, remove ® ™ artifacts, unify dashes
      - keep alnum + hyphen, collapse whitespace, casefold
    """
    if s is None: return ""
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
    def norm(s): return str(s).strip().lower().replace("_", "").replace(" ", "")
    return {norm(c): c for c in cols}

def _resolve_managed_columns(cols) -> Tuple[Optional[str], Optional[str]]:
    norm_map = _normalize_headers_map(cols)
    item_id_actual = None
    item_name_actual = None
    for a in _MANAGED_ALIASES["Item ID"] if "Item ID" in _MANAGED_ALIASES else []:
        k = a.replace("_", "").replace(" ", "")
        if k in norm_map:
            item_id_actual = norm_map[k]
            break
    for a in _MANAGED_ALIASES["Item Name"] if "Item Name" in _MANAGED_ALIASES else []:
        k = a.replace("_", "").replace(" ", "")
        if k in norm_map:
            item_name_actual = norm_map[k]
            break
    return item_id_actual, item_name_actual

def load_managed_keys(src) -> Tuple[Set[str], Set[str]]:
    df = load_dataframe(src)
    item_id_actual, item_name_actual = _resolve_managed_columns(df.columns)
    if not item_id_actual and not item_name_actual:
        raise ValueError("Managed SKUs file must include 'Item ID' or 'Item Name'.")

    ids_set: Set[str] = set()
    names_set: Set[str] = set()

    if item_id_actual:
        ids_set = set(df[item_id_actual].astype(str).str.strip().replace("nan", pd.NA).dropna().tolist())
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
    total = len(df)
    if total == 0:
        return df.copy(), {"total": 0, "matched": 0, "unmatched_count": len(ids_set), "unmatched_sample": []}

    id_series = df["Item ID"].astype(str).str.strip() if "Item ID" in df.columns else None
    name_series = df["Item Name"].astype(str) if "Item Name" in df.columns else None
    name_series_norm = name_series.map(_norm_name_strict) if name_series is not None else None
    base_series = df["Base_Item_Id"].astype(str).str.strip() if "Base_Item_Id" in df.columns else None

    ids_set_norm = {s.strip() for s in ids_set} if ids_set else set()
    names_set_norm = {s for s in names_set} if names_set else set()

    id_mask   = id_series.isin(ids_set_norm) if (id_series is not None and ids_set_norm) else False
    name_mask = name_series_norm.isin(names_set_norm) if (name_series_norm is not None and names_set_norm) else False
    primary_mask = id_mask | name_mask

    base_mask = False
    if base_series is not None and ids_set_norm and name_series_norm is not None and names_set_norm:
        base_mask = base_series.isin(ids_set_norm) & name_series_norm.isin(names_set_norm)

    mask = primary_mask | base_mask
    filtered = df[mask].copy()
    matched = int(mask.sum())

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
# Advertising column resolver (NEW + USED)
# ─────────────────────────────────────────────────────────────────────────────

# ------- Advertising header resolver (robust; used by backend & mirrored in UI) -------
_AD_ALIASES = {
    "Ad Spend": {
        "ad spend", "ad_spend", "spend", "ad spend ($)", "adspend",
        "total spend", "total ad spend"
    },
    "Conversion Rate": {
        "conversion rate", "conversion_rate", "conv rate", "conv_rate", "conversionrate",
        "conversion rate - 14 day", "conversion rate – 14 day", "conversion rate 14 day", "conversionrate14day"
    },
    "ROAS": {
        "roas", "return on ad spend", "returnonadspend",
        "roas - 14 day", "roas – 14 day", "roas 14 day", "roas14day"
    },
}

def _norm_hdr(s: str) -> str:
    """Lowercase and remove *all* non-alphanumerics so any dash/space/NBSP/commas vanish."""
    import re, unicodedata
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return re.sub(r"[^0-9a-z]+", "", s)

def _resolve_advertising_columns(cols):
    """
    Return dict mapping canonical -> actual header using flexible aliases.
    Missing canonicals map to None (we handle that gracefully).
    """
    lc_to_actual = { _norm_hdr(c): c for c in cols }
    mapping = { "Ad Spend": None, "Conversion Rate": None, "ROAS": None }
    for canon, aliases in _AD_ALIASES.items():
        # exact canonical first
        if _norm_hdr(canon) in lc_to_actual:
            mapping[canon] = lc_to_actual[_norm_hdr(canon)]
            continue
        # then aliases
        for a in aliases:
            na = _norm_hdr(a)
            if na in lc_to_actual:
                mapping[canon] = lc_to_actual[na]
                break
    return mapping
# ------- /resolver -------


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & tables for 1P
# ─────────────────────────────────────────────────────────────────────────────

def split_by_threshold(df: pd.DataFrame):
    rounded = df["Content Quality Score"].round(2)
    above = df[rounded >= THRESHOLD].copy()
    below = df[rounded <  THRESHOLD].copy()
    return below, above

def compute_buybox_ownership(df: pd.DataFrame) -> float:
    if "Buy Box Winner" not in df.columns:
        return 0.0
    total = len(df)
    if total == 0:
        return 0.0
    buybox_yes = df["Buy Box Winner"].str.strip().str.lower().eq("yes").sum()
    return buybox_yes / total

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
        "pct_above": pct_above,
        "avg_cqs":   avg_cqs_pct,
        "buybox":    buybox_pct,
        "threshold": int(round(THRESHOLD*100)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Tables for 1P (used by streamlit_app.py)
# ─────────────────────────────────────────────────────────────────────────────
TOP_N = 5  # keep consistent with your UI

def get_top_skus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the top N rows by Content Quality Score with the columns:
      Product Name | Item ID | Content Quality Score (as '%')
    """
    table = (
        df
        .sort_values("Content Quality Score", ascending=False)
        .head(TOP_N)[["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )
    # Convert Content Quality Score to percent and round to whole %
    table["Content Quality Score"] = (table["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"
    return table

def get_skus_below(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows below the THRESHOLD with the columns:
      Product Name | Item ID | Content Quality Score (as '%')
    Uses the same rounding rule as the metrics (round to 2 decimals before compare).
    """
    rounded = df["Content Quality Score"].round(2)
    table = (
        df[rounded < THRESHOLD]
        [["Product Name", "Item ID", "Content Quality Score"]]
        .copy()
    )
    table["Content Quality Score"] = (table["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"
    return table



# ─────────────────────────────────────────────────────────────────────────────
# Pie chart helper
# ─────────────────────────────────────────────────────────────────────────────

def make_pie_bytes(metrics: dict) -> BytesIO:
    threshold = int(metrics.get("threshold", .95))
    below = int(metrics.get("below", 0))
    above = int(metrics.get("above", 0))

    if below == 0 and above == 0:
        below = 1
        above = 0

    colors_pie = ["#002c47", "#4bc3cf"]

    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.pie(
        [below, above],
        colors=colors_pie,
        startangle=90,
        labels=None,
        autopct=None,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax.set(aspect="equal")
    ax.axis("off")

    buf = BytesIO()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.0)
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# PDF: 1P
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(
    data_src,
    client_name: str,
    report_date: str,
    client_notes: str,
    logo_path: str = None
) -> bytes:
    df      = load_dataframe(data_src)
    metrics = compute_metrics(df)
    top5    = (
        df.sort_values("Content Quality Score", ascending=False)
          .head(TOP_N)[["Product Name", "Item ID", "Content Quality Score"]]
          .copy()
    )
    top5["Content Quality Score"] = (top5["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"

    below_df = df.copy()
    rounded = below_df["Content Quality Score"].round(2)
    below_df = below_df[rounded < THRESHOLD][["Product Name", "Item ID", "Content Quality Score"]]
    below_df["Content Quality Score"] = (below_df["Content Quality Score"] * 100).round().astype(int).astype(str) + "%"

    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    margin      = inch * 0.75
    navy        = colors.HexColor("#002c47")
    teal        = colors.HexColor("#4bc3cf")
    panel_bg    = colors.HexColor("#ffffff")
    header_bg   = navy
    row_bg      = colors.HexColor("#eaf3fa")

    panel_w = 3.7 * inch
    panel_h = 3.6 * inch
    panel_y = h - margin - 1.0 * inch

    panels_total_width = 2 * panel_w + 0.15 * inch
    side_margin = (w - panels_total_width) / 2
    pie_panel_x = side_margin
    summary_panel_x = pie_panel_x + panel_w + 0.15 * inch

    base_path = pathlib.Path(__file__).parent.resolve()
    auto_logo = base_path / "logo.png"
    chosen = (pathlib.Path(logo_path) if logo_path else auto_logo) if (logo_path or auto_logo.is_file()) else None
    if chosen and chosen.is_file():
        logo = ImageReader(str(chosen))
        width = 1.3 * inch
        height = 1.3 * inch
        x_margin = 0.4 * inch
        y_margin = 0.45 * inch
        c.drawImage(logo, x=w - x_margin - width, y=h - y_margin - height,
                    width=width, height=height, preserveAspectRatio=True, mask="auto")

    c.setFillColor(teal);  c.setFont("Raleway-Bold", 19); c.drawString(pie_panel_x, h - margin - 0.00 * inch, client_name)
    c.setFillColor(navy);  c.setFont("Raleway", 22);      c.drawString(pie_panel_x, h - margin - 0.40 * inch, "Weekly Content Reporting")
    c.setFont("Raleway", 15); c.setFillColor(navy);       c.drawString(pie_panel_x, h - margin - 0.71 * inch, report_date)

    # Pie panel
    c.setFillColor(panel_bg)
    c.roundRect(pie_panel_x, panel_y - panel_h, panel_w, panel_h, radius=10, stroke=0, fill=1)
    c.setStrokeColor(navy)
    c.roundRect(pie_panel_x, panel_y - panel_h, panel_w, panel_h, radius=10, stroke=1, fill=0)

    title_fontsize = 22
    title_y_offset = 38
    c.setFont("Raleway-Bold", title_fontsize)
    c.setFillColor(navy)
    c.drawCentredString(pie_panel_x + panel_w / 2, panel_y - title_y_offset, "Score Distribution")

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
    c.setFillColor(navy); c.rect(below_box_x, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont("Raleway", 10)
    c.drawString(below_box_x + square_size + gap, legend_y + 1, f"Below {int(metrics['threshold'])}%")
    above_box_x = below_box_x + 120
    c.setFillColor(teal); c.rect(above_box_x, legend_y, square_size, square_size, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(above_box_x + square_size + gap, legend_y + 1, f"Above {int(metrics['threshold'])}%")

    # Summary panel
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
    for label, key in [("Average CQS", "avg_cqs"), ("SKUs Above 95%", "above"), ("SKUs Below 95%", "below"), ("Buybox Ownership", "buybox")]:
        val = metrics.get(key, "")
        if key in ("above", "below") and val != "":
            val = int(val)
        if key in ("avg_cqs", "buybox") and val != "":
            val = f"{val}%"
        c.setFillColor(navy); c.setStrokeColor(navy); c.circle(box_x + bullet_offset_x, y + 4, 4, fill=1)
        c.setFillColor(navy); c.setFont("Raleway", 16); c.drawString(box_x + text_offset_x, y, f"{label}: {val}")
        y -= line_height

    c.setFillColor(colors.black)

    # Top 5 table
    table_title_y = panel_y - panel_h - 32
    c.setFont("Raleway-Bold", 18)
    c.setFillColor(navy)
    c.drawString(pie_panel_x, table_title_y, "Top 5 SKUs by Content Quality Score")

    styles = getSampleStyleSheet()
    styleN = styles["Normal"]; styleN.fontName = "Raleway"; styleN.fontSize = 10
    data = [top5.columns.tolist()]
    for row in top5.astype(str).values.tolist():
        row[0] = Paragraph(row[0], styleN)
        data.append(row)

    table_w = w - 2 * pie_panel_x
    col_widths = [table_w * 0.5, table_w * 0.20, table_w * 0.28]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Raleway"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,-1), row_bg),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#002c47")),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("ALIGN", (2,1), (2,-1), "CENTER"),
    ]))
    tw, th = table.wrap(table_w, h)
    table.drawOn(c, pie_panel_x, table_title_y - 14 - th)

    # Content Notes
    table_bottom_y = table_title_y - 14 - th
    spacing = 48
    box_y = table_bottom_y - spacing

    convention_text = (client_notes or "").replace("\n", "<br/>")
    c.setFont("Raleway-Bold", 18); c.setFillColor(navy)
    title_y = box_y + 20
    c.drawString(pie_panel_x, title_y, "Content Updates")

    para_style = ParagraphStyle(name='ConventionBox', fontName="Raleway", fontSize=15, leading=20, textColor=navy, spaceAfter=0)
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

    c.setFont("Raleway", 8); c.setFillColor(colors.HexColor("#200453"))
    c.drawCentredString(w / 2, 0.45 * inch, f"Generated by Soapbox Retail • {datetime.now().strftime('%B %d, %Y')}")
    c.setFillColor(colors.black)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PDF: 3P (with Advertising integrated)
# ─────────────────────────────────────────────────────────────────────────────

def generate_3p_report(
    item_sales_src,
    inventory_src,
    search_insights_src,
    managed_src,
    mode: str,                                # "catalog" | "managed"
    client_name: str,
    report_date: str,
    top_skus_text: str = "",
    inventory_callouts_text: str = "",
    search_highlights_text: str = "",
    advertising_src=None,
    advertising_notes_text: str = "",
    logo_path: str | None = None
) -> bytes:
    # Load 3P data
    df_sales  = load_item_sales(item_sales_src)
    df_inv    = load_inventory(inventory_src)
    df_search = load_search_insights(search_insights_src)

    # Managed filter (ID or Name; gated Base-ID fallback handled inside filter)
    if str(mode).strip().lower() == "managed":
        ids_set, names_set = load_managed_keys(managed_src)
        df_sales,  _ = filter_by_managed(df_sales,  ids_set, names_set)
        df_inv,    _ = filter_by_managed(df_inv,    ids_set, names_set)
        df_search, _ = filter_by_managed(df_search, ids_set, names_set)

    # Helpers (metrics)
    def _mean_conversion_pp_excel(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        s = (
            series.astype(str)
                  .str.replace("%", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.strip()
        )
        s = pd.to_numeric(s, errors="coerce")
        nonnull = s.dropna()
        if nonnull.empty:
            return 0.0
        if (nonnull <= 1.0).mean() >= 0.8:
            nonnull = nonnull * 100.0
        return float(nonnull.mean())

    def _avg_impressions_rank_whole(series: pd.Series) -> int:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return int(round(float(s.mean()))) if len(s) else 0

    def _count_top_n(series: pd.Series, n: int) -> int:
        s = pd.to_numeric(series, errors="coerce")
        return int((s <= n).sum())

    def _writein_text(x: float, y: float, w_: float, title: str, body_text: str, min_height: float = 80.0) -> float:
        c.setFont("Raleway-Bold", 14); c.setFillColor(navy); c.drawString(x, y, title)
        content = (body_text or "").strip()
        used_h = 0.0
        if content:
            para_style = ParagraphStyle(name="ManualText", fontName="Raleway", fontSize=12, leading=16, textColor=navy)
            para = Paragraph(content.replace("\n", "<br/>"), para_style)
            usable_w = w_
            _, ph = para.wrap(usable_w, min_height)
            para.drawOn(c, x, y - 16 - ph)
            used_h = 16 + ph
        block_h = max(min_height, used_h or 16)
        return y - block_h

    # ---------- Advertising loader with header auto-detect + shared parsing ----------
    def _load_advertising_df(src) -> pd.DataFrame:
        if src is None:
            return pd.DataFrame(columns=["Ad Spend", "Conversion Rate – 14 Day", "RoAS – 14 Day"])

        import io
        import unicodedata, re

        def _norm_hdr_detect(s: str) -> str:
            s = unicodedata.normalize("NFKC", str(s or "")).lower()
            return re.sub(r"[^0-9a-z]+", "", s)

        expected_hdrs = {"adspend", "conversionrate14day", "roas14day"}

        # Try to find the real header row in the first ~20 rows
        df_raw = None
        try:
            if hasattr(src, "read") and hasattr(src, "name"):
                data = src.getvalue()
                ext  = os.path.splitext(src.name)[1].lower()
                if ext in (".xls", ".xlsx"):
                    tmp = pd.read_excel(io.BytesIO(data), header=None, nrows=25)
                else:
                    tmp = pd.read_csv(io.BytesIO(data), header=None, nrows=25, engine="python", encoding="utf-8", on_bad_lines="skip")
                hdr_idx = None
                for i in range(min(20, len(tmp))):
                    row_norm = {_norm_hdr_detect(x) for x in tmp.iloc[i].tolist()}
                    if expected_hdrs.issubset(row_norm):
                        hdr_idx = i
                        break
                if hdr_idx is not None:
                    if ext in (".xls", ".xlsx"):
                        df_raw = pd.read_excel(io.BytesIO(data), header=hdr_idx)
                    else:
                        df_raw = pd.read_csv(io.BytesIO(data), header=hdr_idx, engine="python", encoding="utf-8", on_bad_lines="skip")
            else:
                ext = os.path.splitext(str(src))[1].lower()
                if ext in (".xls", ".xlsx"):
                    tmp = pd.read_excel(src, header=None, nrows=25)
                else:
                    tmp = pd.read_csv(src, header=None, nrows=25, engine="python", encoding="utf-8", on_bad_lines="skip")
                hdr_idx = None
                for i in range(min(20, len(tmp))):
                    row_norm = {_norm_hdr_detect(x) for x in tmp.iloc[i].tolist()}
                    if expected_hdrs.issubset(row_norm):
                        hdr_idx = i
                        break
                if hdr_idx is not None:
                    if ext in (".xls", ".xlsx"):
                        df_raw = pd.read_excel(src, header=hdr_idx)
                    else:
                        df_raw = pd.read_csv(src, header=hdr_idx, engine="python", encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df_raw = None

        if df_raw is None:
            df_raw = load_dataframe(src)

        # Resolve columns using the shared resolver
        mapping = _resolve_advertising_columns(df_raw.columns)

        # Coercers (shared rules)
        def _to_currency(s: pd.Series) -> pd.Series:
            return pd.to_numeric(
                s.astype(str)
                 .str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.replace(" ", "", regex=False)
                 .str.strip(),
                errors="coerce"
            ).fillna(0.0)

        def _to_roas(s: pd.Series) -> pd.Series:
            cleaned = (
                s.astype(str)
                 .str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.replace("x", "", case=False, regex=False)  # strip 'x' suffix/prefix
                 .str.strip()
            )
            return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

        def _to_percent_vals(s: pd.Series) -> pd.Series:
            vals = (
                s.astype(str)
                 .str.replace("%", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.strip()
            )
            vals = pd.to_numeric(vals, errors="coerce")
            nonnull = vals.dropna()
            if not nonnull.empty and (nonnull <= 1.0).mean() >= 0.8:
                vals = vals * 100.0
            return vals

        out = pd.DataFrame()
        out["Ad Spend"] = _to_currency(df_raw[mapping["Ad Spend"]]) if mapping["Ad Spend"] else pd.Series([], dtype=float)
        out["RoAS – 14 Day"] = _to_roas(df_raw[mapping["ROAS"]]) if mapping["ROAS"] else pd.Series([], dtype=float)
        out["Conversion Rate – 14 Day"] = _to_percent_vals(df_raw[mapping["Conversion Rate"]]) if mapping["Conversion Rate"] else pd.Series([], dtype=float)
        return out
    # -------------------------------------------------------------------------

    # Item Sales metrics
    units_total = int(pd.to_numeric(df_sales.get("Units Sold"), errors="coerce").fillna(0).sum()) if not df_sales.empty else 0
    sales_total = float(pd.to_numeric(df_sales.get("Auth Sales"), errors="coerce").sum()) if not df_sales.empty else 0.0
    avg_conv_pct = _mean_conversion_pp_excel(df_sales.get("Item conversion", pd.Series(dtype=float))) if not df_sales.empty else 0.0

    # Inventory metrics
    status_col = "Status" if "Status" in df_inv.columns else ("Stock status" if "Stock status" in df_inv.columns else None)
    in_stock_rate_pct = 0; oos_count = 0; at_risk_count = 0
    if status_col and not df_inv.empty:
        status_norm = df_inv[status_col].astype(str).str.strip().str.lower()
        total_inv = len(status_norm)
        if total_inv > 0:
            in_stock_rate_pct = int(round((status_norm == "in stock").sum() / total_inv * 100))
        oos_count = int((status_norm == "out of stock").sum())
        at_risk_count = int((status_norm == "at risk").sum())

    # Search Insights metrics
    avg_impr_rank = _avg_impressions_rank_whole(df_search.get("Impressions Rank", pd.Series(dtype=float))) if not df_search.empty else 0
    top10_impr = _count_top_n(df_search.get("Impressions Rank", pd.Series(dtype=float)), 10) if not df_search.empty else 0
    top10_sales = _count_top_n(df_search.get("Sales Rank", pd.Series(dtype=float)), 10) if not df_search.empty else 0

    # Advertising (independent of managed/catalog)
    df_adv = _load_advertising_df(advertising_src)
    if df_adv.empty:
        adv_spend_total = 0.0
        adv_roas_total  = 0.0
        adv_conv_avg    = 0.0
    else:
        adv_spend_total = float(df_adv["Ad Spend"].sum())
        adv_roas_total  = float(df_adv["RoAS – 14 Day"].sum())
        conv_nonnull    = df_adv["Conversion Rate – 14 Day"].dropna()
        adv_conv_avg    = float(conv_nonnull.mean()) if len(conv_nonnull) else 0.0

    # ---------- PDF drawing ----------
    buf = BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    margin   = inch * 0.75
    navy     = colors.HexColor("#002c47")
    teal     = colors.HexColor("#4bc3cf")
    panel_bg = colors.white

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
        f"Auth Sales: ${sales_total:,.2f}",
        f"Avg Conversion: {avg_conv_pct:.2f}%",
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

    # Box 4 — Advertising (now populated)
    x, y_top = x_right, row2_top
    _draw_card_shell(x, y_top, card_w, card_h, "Advertising")
    bullets_adv = [
        f"Total Ad Spend: ${adv_spend_total:,.2f}",
        f"Total RoAS: ${adv_roas_total:,.2f}",
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dashboard PDF")
    parser.add_argument("input_file", help="CSV or Excel input")
    parser.add_argument("--client", required=True, help="Client name")
    parser.add_argument("--date",   required=True, help="Report date")
    parser.add_argument("--out",    default="dashboard.pdf", help="Output PDF")
    args = parser.parse_args()

    pdf = generate_full_report(args.input_file, args.client, args.date, client_notes="")
    with open(args.out, "wb") as f:
        f.write(pdf)
    print(f"Wrote {args.out}")
