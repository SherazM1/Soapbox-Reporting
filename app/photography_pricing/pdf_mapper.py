from dataclasses import dataclass
from typing import Any


PDF_ROW_ORDER = (
    "on_model_image",
    "on_model_detail",
    "laydown_silo",
    "color_corrections",
    "post_production",
    "kid_model_hours",
    "adult_model_hours",
    "model_fitting",
    "ai_generation",
    "account_management",
)

PDF_ROW_LABELS = {
    "on_model_image": "On Model Image",
    "on_model_detail": "On-Model Detail",
    "laydown_silo": "Laydown Silo",
    "color_corrections": "Color Corrections From Existing Images",
    "post_production": "Post Production Hourly Time",
    "kid_model_hours": "Kid Model Hours",
    "adult_model_hours": "Adult Model Hours",
    "model_fitting": "Model Fitting",
    "ai_generation": "AI Generation Markup",
    "account_management": "Account Management",
}


@dataclass(frozen=True)
class PdfPricingRow:
    code: str
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


def _is_active_line(line: Any) -> bool:
    total = float(getattr(line, "total", 0) or 0)
    return total > 0


def _model_hours_code(line: Any, quote: Any) -> str:
    inputs = getattr(quote, "apparel_inputs", None)
    model_type = str(getattr(inputs, "model_type", "") or "").lower()
    unit_price = float(getattr(line, "unit_price", 0) or 0)

    if model_type == "kid":
        return "kid_model_hours"
    if model_type == "adult":
        return "adult_model_hours"
    if unit_price < 200:
        return "kid_model_hours"
    return "adult_model_hours"


def _pricing_lines_by_pdf_code(quote: Any) -> dict[str, Any]:
    lines: dict[str, Any] = {}

    for line in getattr(quote, "line_items", ()):
        code = getattr(line, "code", "")
        if code == "model_hours":
            code = _model_hours_code(line, quote)
        lines[code] = line

    return lines


def build_page2_pricing_payload(quote: Any) -> Page2PricingPayload:
    lines_by_code = _pricing_lines_by_pdf_code(quote)
    rows: list[PdfPricingRow] = []

    for code in PDF_ROW_ORDER:
        line = lines_by_code.get(code)
        if line is None or not _is_active_line(line):
            continue
        rows.append(
            PdfPricingRow(
                code=code,
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
