from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.photography_pricing.pdf_mapper import Page2PricingPayload, build_page2_pricing_payload


TEMPLATE_PATH = Path("templates/photographytemplate.pdf")
GOTHAM_MEDIUM_PATH = Path("fonts/Gotham-Medium.ttf")
GOTHAM_BOLD_PATH = Path("fonts/Gotham-Bold.ttf")
GOTHAM_MEDIUM = "Gotham-Medium"
GOTHAM_BOLD = "Gotham-Bold"

TEXT_TOP_Y = (382, 508, 663, 776, 935, 1090, 1180, 1295, 1406)
TEMPLATE_ROW_TOP_Y_BY_CODE = {
    "on_model_image": TEXT_TOP_Y[0],
    "laydown_silo": TEXT_TOP_Y[1],
    "color_corrections": TEXT_TOP_Y[2],
    "post_production": TEXT_TOP_Y[3],
    "model_hours": TEXT_TOP_Y[4],
    "account_management": TEXT_TOP_Y[5],
    "on_model_detail": TEXT_TOP_Y[6],
    "model_fitting": TEXT_TOP_Y[7],
    "ai_generation": TEXT_TOP_Y[8],
}

QUANTITY_RIGHT_X = 1160
UNIT_PRICE_RIGHT_X = 1452
TOTAL_RIGHT_X = 1682

SUBTOTAL_AMOUNT_Y = 1553
TOTAL_AMOUNT_Y = 1666
PAGE1_COMMENTS_LEFT_X = 300
PAGE1_COMMENTS_TOP_Y = 1920
PAGE1_COMMENTS_MAX_WIDTH = 1185
PAGE1_COMMENTS_LINE_STEP = 31
PAGE1_COMMENTS_MAX_LINES = 17

TEXT = HexColor("#002C47")
PAGE1_TEXT = HexColor("#F2F5F7")
TEMPLATE_COORDINATE_SCALE = 3
ROW_FONT_SIZE = 9.5 * TEMPLATE_COORDINATE_SCALE
SUBTOTAL_FONT_SIZE = 10 * TEMPLATE_COORDINATE_SCALE
TOTAL_FONT_SIZE = 11 * TEMPLATE_COORDINATE_SCALE
PAGE1_COMMENTS_FONT_SIZE = 8.5 * TEMPLATE_COORDINATE_SCALE


def _register_gotham_fonts() -> None:
    registered_fonts = set(pdfmetrics.getRegisteredFontNames())
    if GOTHAM_MEDIUM not in registered_fonts:
        pdfmetrics.registerFont(TTFont(GOTHAM_MEDIUM, str(GOTHAM_MEDIUM_PATH)))
    if GOTHAM_BOLD not in registered_fonts:
        pdfmetrics.registerFont(TTFont(GOTHAM_BOLD, str(GOTHAM_BOLD_PATH)))


def _pdf_y(page_height: float, top_y: float) -> float:
    return page_height - top_y


def _draw_row_numbers(c: canvas.Canvas, page_height: float, top_y: float, row) -> None:
    y = _pdf_y(page_height, top_y)
    c.setFillColor(TEXT)
    c.setFont(GOTHAM_MEDIUM, ROW_FONT_SIZE)
    c.drawRightString(QUANTITY_RIGHT_X, y, row.quantity)
    c.drawRightString(UNIT_PRICE_RIGHT_X, y, row.unit_price)
    c.drawRightString(TOTAL_RIGHT_X, y, row.total)


def _draw_totals(c: canvas.Canvas, page_height: float, payload: Page2PricingPayload) -> None:
    c.setFillColor(TEXT)
    c.setFont(GOTHAM_MEDIUM, SUBTOTAL_FONT_SIZE)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, SUBTOTAL_AMOUNT_Y), payload.subtotal)
    c.setFont(GOTHAM_BOLD, TOTAL_FONT_SIZE)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, TOTAL_AMOUNT_Y), payload.total)


def _wrap_text_line(text: str, max_width: float, font_name: str, font_size: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _comments_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    return str(payload.get("rendered_comments_block") or "").strip()


def _page1_overlay(page_width: float, page_height: float, comments_payload: dict[str, Any] | None) -> BytesIO:
    _register_gotham_fonts()
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(page_width, page_height))
    c.setFillColor(PAGE1_TEXT)
    c.setFont(GOTHAM_MEDIUM, PAGE1_COMMENTS_FONT_SIZE)

    lines: list[str] = []
    for raw_line in _comments_text(comments_payload).splitlines():
        if raw_line.strip():
            lines.extend(_wrap_text_line(raw_line.strip(), PAGE1_COMMENTS_MAX_WIDTH, GOTHAM_MEDIUM, PAGE1_COMMENTS_FONT_SIZE))
        else:
            lines.append("")

    for index, line in enumerate(lines[:PAGE1_COMMENTS_MAX_LINES]):
        c.drawString(
            PAGE1_COMMENTS_LEFT_X,
            _pdf_y(page_height, PAGE1_COMMENTS_TOP_Y + index * PAGE1_COMMENTS_LINE_STEP),
            line,
        )

    c.save()
    overlay.seek(0)
    return overlay


def _page2_overlay(page_width: float, page_height: float, payload: Page2PricingPayload) -> BytesIO:
    _register_gotham_fonts()
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(page_width, page_height))

    for row in payload.rows:
        top_y = TEMPLATE_ROW_TOP_Y_BY_CODE.get(row.code)
        if top_y is not None:
            _draw_row_numbers(c, page_height, top_y, row)

    _draw_totals(c, page_height, payload)

    c.save()
    overlay.seek(0)
    return overlay


def generate_page2_pricing_pdf(quote, template_path: Path = TEMPLATE_PATH, page1_comments_payload: dict[str, Any] | None = None) -> bytes:
    payload = build_page2_pricing_payload(quote)
    reader = PdfReader(str(template_path))
    writer = PdfWriter()

    for index, page in enumerate(reader.pages):
        if index == 0 and page1_comments_payload:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay_pdf = PdfReader(_page1_overlay(width, height, page1_comments_payload))
            page.merge_page(overlay_pdf.pages[0])
        elif index == 1:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay_pdf = PdfReader(_page2_overlay(width, height, payload))
            page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
