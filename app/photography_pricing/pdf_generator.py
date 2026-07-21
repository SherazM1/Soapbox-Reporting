from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from app.photography_pricing.pdf_mapper import Page2PricingPayload, build_page2_pricing_payload


TEMPLATE_PATH = Path("templates/photographytemplate.pdf")

TEXT_TOP_Y = (382, 508, 663, 776, 935, 1090, 1180, 1295, 1406)

QUANTITY_RIGHT_X = 1160
UNIT_PRICE_RIGHT_X = 1452
TOTAL_RIGHT_X = 1682

SUBTOTAL_AMOUNT_Y = 1553
TOTAL_AMOUNT_Y = 1666

BACKGROUND = Color(0, 0, 0)
TEXT = Color(0.22, 0.22, 0.22)


def _pdf_y(page_height: float, top_y: float) -> float:
    return page_height - top_y


def _clear_numeric_cell(c: canvas.Canvas, page_height: float, left_x: float, top_y: float, width: float) -> None:
    c.setFillColor(BACKGROUND)
    c.rect(
        left_x,
        _pdf_y(page_height, top_y + 26),
        width,
        38,
        fill=1,
        stroke=0,
    )


def _clear_row_numbers(c: canvas.Canvas, page_height: float, top_y: float) -> None:
    _clear_numeric_cell(c, page_height, 1065, top_y, 110)
    _clear_numeric_cell(c, page_height, 1268, top_y, 205)
    _clear_numeric_cell(c, page_height, 1484, top_y, 210)


def _draw_row_numbers(c: canvas.Canvas, page_height: float, top_y: float, row) -> None:
    y = _pdf_y(page_height, top_y)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 26)
    c.drawRightString(QUANTITY_RIGHT_X, y, row.quantity)
    c.drawRightString(UNIT_PRICE_RIGHT_X, y, row.unit_price)
    c.drawRightString(TOTAL_RIGHT_X, y, row.total)


def _clear_totals(c: canvas.Canvas, page_height: float) -> None:
    c.setFillColor(BACKGROUND)
    c.rect(1450, _pdf_y(page_height, SUBTOTAL_AMOUNT_Y + 24), 244, 38, fill=1, stroke=0)
    c.rect(1450, _pdf_y(page_height, TOTAL_AMOUNT_Y + 28), 244, 42, fill=1, stroke=0)


def _draw_totals(c: canvas.Canvas, page_height: float, payload: Page2PricingPayload) -> None:
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 26)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, SUBTOTAL_AMOUNT_Y), payload.subtotal)
    c.setFont("Helvetica-Bold", 30)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, TOTAL_AMOUNT_Y), payload.total)


def _page2_overlay(page_width: float, page_height: float, payload: Page2PricingPayload) -> BytesIO:
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(page_width, page_height))

    for top_y in TEXT_TOP_Y:
        _clear_row_numbers(c, page_height, top_y)

    for row, top_y in zip(payload.rows, TEXT_TOP_Y):
        _draw_row_numbers(c, page_height, top_y, row)

    _clear_totals(c, page_height)
    _draw_totals(c, page_height, payload)

    c.save()
    overlay.seek(0)
    return overlay


def generate_page2_pricing_pdf(quote, template_path: Path = TEMPLATE_PATH) -> bytes:
    payload = build_page2_pricing_payload(quote)
    reader = PdfReader(str(template_path))
    writer = PdfWriter()

    for index, page in enumerate(reader.pages):
        if index == 1:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay_pdf = PdfReader(_page2_overlay(width, height, payload))
            page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
