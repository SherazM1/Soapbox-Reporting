from dataclasses import dataclass
from typing import Any


PDF_ROW_ORDER = (
    "on_model_image",
    "on_model_detail",
    "laydown_silo",
    "color_corrections",
    "post_production",
    "model_hours",
    "model_fitting",
    "ai_generation",
    "account_management",
)

PDF_ROW_LABELS = {
    "on_model_image": "On Model Image",
    "on_model_detail": "On-model detail",
    "laydown_silo": "Laydown Silo",
    "color_corrections": "Color Corrections From Existing Image",
    "post_production": "Post Production hourly time",
    "model_hours": "Model Hours",
    "model_fitting": "Model Fitting",
    "ai_generation": "AI Generation Markup",
    "account_management": "Account Management",
}


@dataclass(frozen=True)
class PdfPricingRow:
    label: str
    quantity: str
    unit_price: str
    total: str


@dataclass(frozen=True)
class Page2PricingPayload:
    rows: tuple[PdfPricingRow, ...]
    subtotal: str
    total: str


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _quantity(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _is_active_line(line: Any, derived_total_image_count: int) -> bool:
    code = getattr(line, "code", "")
    quantity = float(getattr(line, "quantity", 0) or 0)
    total = float(getattr(line, "total", 0) or 0)
    if code == "account_management":
        return total > 0 or derived_total_image_count > 0
    return quantity > 0 and total > 0


def build_page2_pricing_payload(quote: Any) -> Page2PricingPayload:
    lines_by_code = {getattr(line, "code", ""): line for line in getattr(quote, "line_items", ())}
    rows: list[PdfPricingRow] = []

    for code in PDF_ROW_ORDER:
        line = lines_by_code.get(code)
        if line is None or not _is_active_line(line, int(getattr(quote, "derived_total_image_count", 0) or 0)):
            continue
        rows.append(
            PdfPricingRow(
                label=PDF_ROW_LABELS.get(code, getattr(line, "label", "")),
                quantity=_quantity(float(getattr(line, "quantity", 0) or 0)),
                unit_price=_money(float(getattr(line, "unit_price", 0) or 0)),
                total=_money(float(getattr(line, "total", 0) or 0)),
            )
        )

    return Page2PricingPayload(
        rows=tuple(rows),
        subtotal=_money(float(getattr(quote, "subtotal", 0) or 0)),
        total=_money(float(getattr(quote, "total", 0) or 0)),
    )
