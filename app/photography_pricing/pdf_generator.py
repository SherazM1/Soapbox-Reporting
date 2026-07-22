from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from app.photography_pricing.pdf_mapper import Page2PricingPayload, build_page2_pricing_payload


TEMPLATE_PATH = Path("templates/photographytemplate.pdf")

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

TEXT = HexColor("#262B3A")


def _pdf_y(page_height: float, top_y: float) -> float:
    return page_height - top_y


def _draw_row_numbers(c: canvas.Canvas, page_height: float, top_y: float, row) -> None:
    y = _pdf_y(page_height, top_y)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 26)
    c.drawRightString(QUANTITY_RIGHT_X, y, row.quantity)
    c.drawRightString(UNIT_PRICE_RIGHT_X, y, row.unit_price)
    c.drawRightString(TOTAL_RIGHT_X, y, row.total)


def _draw_totals(c: canvas.Canvas, page_height: float, payload: Page2PricingPayload) -> None:
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 26)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, SUBTOTAL_AMOUNT_Y), payload.subtotal)
    c.setFont("Helvetica-Bold", 30)
    c.drawRightString(TOTAL_RIGHT_X, _pdf_y(page_height, TOTAL_AMOUNT_Y), payload.total)


def _page2_overlay(page_width: float, page_height: float, payload: Page2PricingPayload) -> BytesIO:
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
