from io import BytesIO
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import date, datetime

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
PAGE1_COMMENTS_LEFT_X = 165
PAGE1_COMMENTS_TOP_Y = 1030
PAGE1_COMMENTS_MAX_WIDTH = 1450
PAGE1_COMMENTS_LINE_STEP = 38
PAGE1_COMMENTS_MAX_LINES = 29
PAGE1_HEADER_TITLE_X = 150
PAGE1_HEADER_TITLE_TOP_Y = 220
PAGE1_HEADER_TITLE_MAX_WIDTH = 870
PAGE1_HEADER_CLIENT_X = 165
PAGE1_HEADER_COMPANY_TOP_Y = 455
PAGE1_HEADER_CLIENT_NAME_TOP_Y = 525
PAGE1_HEADER_CLIENT_EMAIL_TOP_Y = 585
PAGE1_HEADER_CLIENT_MAX_WIDTH = 790
PAGE1_HEADER_RIGHT_X = 1205
PAGE1_HEADER_RIGHT_MAX_WIDTH = 445
PAGE1_HEADER_REFERENCE_TOP_Y = 190
PAGE1_HEADER_CREATED_TOP_Y = 294
PAGE1_HEADER_EXPIRES_TOP_Y = 398
PAGE1_HEADER_CREATED_BY_TOP_Y = 505
PAGE1_HEADER_CREATED_BY_TITLE_TOP_Y = 565
PAGE1_HEADER_CREATED_BY_EMAIL_TOP_Y = 622

TEXT = HexColor("#002C47")
PAGE1_TEXT = HexColor("#002C47")
PAGE1_HEADER_TEXT = HexColor("#FFFFFF")
TEMPLATE_COORDINATE_SCALE = 3
ROW_FONT_SIZE = 9.5 * TEMPLATE_COORDINATE_SCALE
SUBTOTAL_FONT_SIZE = 10 * TEMPLATE_COORDINATE_SCALE
TOTAL_FONT_SIZE = 11 * TEMPLATE_COORDINATE_SCALE
PAGE1_COMMENTS_FONT_SIZE = 8.5 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_TITLE_FONT_SIZE = 24 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_TITLE_MIN_FONT_SIZE = 15 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_COMPANY_FONT_SIZE = 13 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_NAME_FONT_SIZE = 11 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_SMALL_FONT_SIZE = 9.5 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_MIN_FONT_SIZE = 7.5 * TEMPLATE_COORDINATE_SCALE


@dataclass(frozen=True)
class Page1HeaderItem:
    text: str
    x: float
    top_y: float
    max_width: float
    font_name: str
    font_size: float
    min_font_size: float


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


def _parse_header_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _format_header_date(value: Any) -> str:
    parsed = _parse_header_date(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def build_page1_header_items(payload: dict[str, Any] | None) -> tuple[Page1HeaderItem, ...]:
    if not payload:
        return ()
    metadata = payload.get("quote_metadata", {}) or {}
    client = payload.get("selected_client", {}) or {}
    internal = payload.get("selected_internal", {}) or {}

    return (
        Page1HeaderItem(
            str(metadata.get("quote_title") or "").strip(),
            PAGE1_HEADER_TITLE_X,
            PAGE1_HEADER_TITLE_TOP_Y,
            PAGE1_HEADER_TITLE_MAX_WIDTH,
            GOTHAM_BOLD,
            PAGE1_HEADER_TITLE_FONT_SIZE,
            PAGE1_HEADER_TITLE_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(client.get("company_name") or "").strip(),
            PAGE1_HEADER_CLIENT_X,
            PAGE1_HEADER_COMPANY_TOP_Y,
            PAGE1_HEADER_CLIENT_MAX_WIDTH,
            GOTHAM_BOLD,
            PAGE1_HEADER_COMPANY_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(client.get("full_name") or "").strip(),
            PAGE1_HEADER_CLIENT_X,
            PAGE1_HEADER_CLIENT_NAME_TOP_Y,
            PAGE1_HEADER_CLIENT_MAX_WIDTH,
            GOTHAM_BOLD,
            PAGE1_HEADER_NAME_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(client.get("email") or "").strip(),
            PAGE1_HEADER_CLIENT_X,
            PAGE1_HEADER_CLIENT_EMAIL_TOP_Y,
            PAGE1_HEADER_CLIENT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(metadata.get("reference_number") or "").strip(),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_REFERENCE_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            _format_header_date(metadata.get("quote_created_date")),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_CREATED_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            _format_header_date(metadata.get("quote_expiration_date")),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_EXPIRES_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(internal.get("name") or "").strip(),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_CREATED_BY_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(internal.get("title") or "").strip(),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_CREATED_BY_TITLE_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            str(internal.get("email") or "").strip(),
            PAGE1_HEADER_RIGHT_X,
            PAGE1_HEADER_CREATED_BY_EMAIL_TOP_Y,
            PAGE1_HEADER_RIGHT_MAX_WIDTH,
            GOTHAM_MEDIUM,
            PAGE1_HEADER_SMALL_FONT_SIZE,
            PAGE1_HEADER_MIN_FONT_SIZE,
        ),
    )


def _fit_text(text: str, max_width: float, font_name: str, font_size: float, min_font_size: float) -> tuple[str, float]:
    clean = " ".join(str(text or "").split())
    size = font_size
    while size > min_font_size and pdfmetrics.stringWidth(clean, font_name, size) > max_width:
        size -= 1
    if pdfmetrics.stringWidth(clean, font_name, size) <= max_width:
        return clean, size

    ellipsis = "..."
    while clean and pdfmetrics.stringWidth(f"{clean}{ellipsis}", font_name, size) > max_width:
        clean = clean[:-1].rstrip()
    return f"{clean}{ellipsis}" if clean else "", size


def _draw_page1_header(c: canvas.Canvas, page_height: float, payload: dict[str, Any] | None) -> None:
    c.setFillColor(PAGE1_HEADER_TEXT)
    for item in build_page1_header_items(payload):
        if not item.text:
            continue
        text, font_size = _fit_text(item.text, item.max_width, item.font_name, item.font_size, item.min_font_size)
        c.setFont(item.font_name, font_size)
        c.drawString(item.x, _pdf_y(page_height, item.top_y), text)


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


def _page1_overlay(
    page_width: float,
    page_height: float,
    comments_payload: dict[str, Any] | None,
    header_payload: dict[str, Any] | None = None,
) -> BytesIO:
    _register_gotham_fonts()
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(page_width, page_height))
    _draw_page1_header(c, page_height, header_payload)
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


def generate_page2_pricing_pdf(
    quote,
    template_path: Path = TEMPLATE_PATH,
    page1_comments_payload: dict[str, Any] | None = None,
    page1_header_payload: dict[str, Any] | None = None,
) -> bytes:
    payload = build_page2_pricing_payload(quote)
    reader = PdfReader(str(template_path))
    writer = PdfWriter()

    for index, page in enumerate(reader.pages):
        if index == 0 and (page1_comments_payload or page1_header_payload):
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay_pdf = PdfReader(_page1_overlay(width, height, page1_comments_payload, page1_header_payload))
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
