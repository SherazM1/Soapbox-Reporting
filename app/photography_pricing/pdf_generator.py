# app/photography_pricing/pdf_generator.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.photography_pricing.pdf_mapper import (
    Page2PricingPayload,
    build_page2_pricing_payload,
)


MAIN_TEMPLATE_PATH = Path("templates/photographytemplate.pdf")
PRICING_TEMPLATE_PATH = Path("templates/Page 2.pdf")
TEMPLATE_PATH = MAIN_TEMPLATE_PATH
GOTHAM_MEDIUM_PATH = Path("fonts/Gotham-Medium.ttf")
GOTHAM_BOLD_PATH = Path("fonts/Gotham-Bold.ttf")
GOTHAM_MEDIUM = "Gotham-Medium"
GOTHAM_BOLD = "Gotham-Bold"

PRICING_ROW_SLOT_TOP_Y = (
    394.23,
    506.87,
    664.47,
    781.03,
    934.75,
    1089.86,
    1199.94,
    1306.06,
    1417.88,
    1533.52,
)

PRICING_ROW_SEPARATOR_TOP_Y = (
    441.13,
    598.45,
    711.34,
    868.66,
    1025.99,
    1137.11,
    1241.09,
    1352.64,
    1456.66,
    1584.48,
)

PRICING_TABLE_MASK_LEFT_X = 100
PRICING_TABLE_MASK_RIGHT_X = 1700
PRICING_TABLE_MASK_TOP_Y = 345
PRICING_TABLE_MASK_BOTTOM_Y = 1605
PRICING_LABEL_X = 113.26

QUANTITY_RIGHT_X = 1160
UNIT_PRICE_RIGHT_X = 1452
TOTAL_RIGHT_X = 1682

SUBTOTAL_AMOUNT_Y = 1672.24
TOTAL_AMOUNT_Y = 1783.08

PAGE1_COMMENTS_LEFT_X = 165
PAGE1_COMMENTS_TOP_Y = 1030
PAGE1_COMMENTS_MAX_WIDTH = 1450
PAGE1_COMMENTS_LINE_STEP = 38
PAGE1_COMMENTS_MAX_LINES = 29

PAGE1_LOGO_REGION = (0, 0, 520, 360)

# Page 1 title — full-width single line
PAGE1_HEADER_TITLE_X = 165
PAGE1_HEADER_TITLE_TOP_Y = 440
PAGE1_HEADER_TITLE_MAX_WIDTH = 1515

# Page 1 header — left side
PAGE1_HEADER_CLIENT_X = 165
PAGE1_HEADER_COMPANY_TOP_Y = 645
PAGE1_HEADER_CLIENT_NAME_TOP_Y = 755
PAGE1_HEADER_CLIENT_EMAIL_TOP_Y = 810
PAGE1_HEADER_CLIENT_MAX_WIDTH = 790

# Page 1 header — right side
PAGE1_HEADER_RIGHT_X = 1115
PAGE1_HEADER_RIGHT_MAX_WIDTH = 585

PAGE1_HEADER_REFERENCE_TOP_Y = 535
PAGE1_HEADER_CREATED_TOP_Y = 590
PAGE1_HEADER_EXPIRES_TOP_Y = 645
PAGE1_HEADER_CREATED_BY_TOP_Y = 700
PAGE1_HEADER_CREATED_BY_TITLE_TOP_Y = 755
PAGE1_HEADER_CREATED_BY_EMAIL_TOP_Y = 810

PAGE1_HEADER_TITLE_LINE_STEP = 34
PAGE1_HEADER_EMAIL_ROW_GAP = 55

TEXT = HexColor("#002C47")
PAGE1_TEXT = HexColor("#002C47")
PAGE1_HEADER_TEXT = HexColor("#FFFFFF")
PAGE2_TABLE_BACKGROUND = HexColor("#FFFFFF")

TEMPLATE_COORDINATE_SCALE = 3

ROW_FONT_SIZE = 9.5 * TEMPLATE_COORDINATE_SCALE
SUBTOTAL_FONT_SIZE = 10 * TEMPLATE_COORDINATE_SCALE
TOTAL_FONT_SIZE = 11 * TEMPLATE_COORDINATE_SCALE

PAGE1_COMMENTS_FONT_SIZE = 8.5 * TEMPLATE_COORDINATE_SCALE

PAGE1_HEADER_TITLE_FONT_SIZE = 18 * TEMPLATE_COORDINATE_SCALE
PAGE1_HEADER_TITLE_MIN_FONT_SIZE = 9 * TEMPLATE_COORDINATE_SCALE
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
    max_lines: int = 1
    line_step: float = 0


def _register_gotham_fonts() -> None:
    registered_fonts = set(pdfmetrics.getRegisteredFontNames())

    if GOTHAM_MEDIUM not in registered_fonts:
        pdfmetrics.registerFont(
            TTFont(
                GOTHAM_MEDIUM,
                str(GOTHAM_MEDIUM_PATH),
            )
        )

    if GOTHAM_BOLD not in registered_fonts:
        pdfmetrics.registerFont(
            TTFont(
                GOTHAM_BOLD,
                str(GOTHAM_BOLD_PATH),
            )
        )


def _pdf_y(page_height: float, top_y: float) -> float:
    return page_height - top_y


def _draw_pricing_row(
    c: canvas.Canvas,
    page_height: float,
    top_y: float,
    row: Any,
) -> None:
    y = _pdf_y(page_height, top_y)

    c.setFillColor(TEXT)
    c.setFont(GOTHAM_MEDIUM, ROW_FONT_SIZE)
    c.drawString(PRICING_LABEL_X, y, row.label)
    c.drawRightString(QUANTITY_RIGHT_X, y, row.quantity)
    c.drawRightString(UNIT_PRICE_RIGHT_X, y, row.unit_price)
    c.drawRightString(TOTAL_RIGHT_X, y, row.total)


def _draw_totals(
    c: canvas.Canvas,
    page_height: float,
    payload: Page2PricingPayload,
) -> None:
    c.setFillColor(TEXT)

    c.setFont(GOTHAM_MEDIUM, SUBTOTAL_FONT_SIZE)
    c.drawRightString(
        TOTAL_RIGHT_X,
        _pdf_y(page_height, SUBTOTAL_AMOUNT_Y),
        payload.subtotal,
    )

    c.setFont(GOTHAM_BOLD, TOTAL_FONT_SIZE)
    c.drawRightString(
        TOTAL_RIGHT_X,
        _pdf_y(page_height, TOTAL_AMOUNT_Y),
        payload.total,
    )


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


def _readable_company_name(value: Any) -> str:
    return " ".join(str(value or "").split())


def _label_value(label: str, value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    return f"{label}: {text}"


def build_page1_header_items(
    payload: dict[str, Any] | None,
) -> tuple[Page1HeaderItem, ...]:
    if not payload:
        return ()

    metadata = payload.get("quote_metadata", {}) or {}
    client = payload.get("selected_client", {}) or {}
    internal = payload.get("selected_internal", {}) or {}

    internal_title = str(internal.get("title") or "").strip()

    internal_title_lines, _ = _wrap_fitted_text(
        text=internal_title,
        max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
        font_name=GOTHAM_MEDIUM,
        font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
        min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        max_lines=2,
    )

    internal_title_line_count = max(1, len(internal_title_lines))

    aligned_email_top_y = (
        PAGE1_HEADER_CREATED_BY_TITLE_TOP_Y
        + (
            internal_title_line_count - 1
        )
        * PAGE1_HEADER_TITLE_LINE_STEP
        + PAGE1_HEADER_EMAIL_ROW_GAP
    )

    return (
        Page1HeaderItem(
            text=str(metadata.get("quote_title") or "").strip(),
            x=PAGE1_HEADER_TITLE_X,
            top_y=PAGE1_HEADER_TITLE_TOP_Y,
            max_width=PAGE1_HEADER_TITLE_MAX_WIDTH,
            font_name=GOTHAM_BOLD,
            font_size=PAGE1_HEADER_TITLE_FONT_SIZE,
            min_font_size=PAGE1_HEADER_TITLE_MIN_FONT_SIZE,
            max_lines=1,
        ),
        Page1HeaderItem(
            text=_readable_company_name(client.get("company_name")),
            x=PAGE1_HEADER_CLIENT_X,
            top_y=PAGE1_HEADER_COMPANY_TOP_Y,
            max_width=PAGE1_HEADER_CLIENT_MAX_WIDTH,
            font_name=GOTHAM_BOLD,
            font_size=PAGE1_HEADER_COMPANY_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=str(client.get("full_name") or "").strip(),
            x=PAGE1_HEADER_CLIENT_X,
            top_y=PAGE1_HEADER_CLIENT_NAME_TOP_Y,
            max_width=PAGE1_HEADER_CLIENT_MAX_WIDTH,
            font_name=GOTHAM_BOLD,
            font_size=PAGE1_HEADER_NAME_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=str(client.get("email") or "").strip(),
            x=PAGE1_HEADER_CLIENT_X,
            top_y=aligned_email_top_y,
            max_width=PAGE1_HEADER_CLIENT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=_label_value(
                "Reference",
                metadata.get("reference_number"),
            ),
            x=PAGE1_HEADER_RIGHT_X,
            top_y=PAGE1_HEADER_REFERENCE_TOP_Y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=_label_value(
                "Quote created",
                _format_header_date(
                    metadata.get("quote_created_date")
                ),
            ),
            x=PAGE1_HEADER_RIGHT_X,
            top_y=PAGE1_HEADER_CREATED_TOP_Y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=_label_value(
                "Quote expires",
                _format_header_date(
                    metadata.get("quote_expiration_date")
                ),
            ),
            x=PAGE1_HEADER_RIGHT_X,
            top_y=PAGE1_HEADER_EXPIRES_TOP_Y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=_label_value(
                "Quote created by",
                internal.get("name"),
            ),
            x=PAGE1_HEADER_RIGHT_X,
            top_y=PAGE1_HEADER_CREATED_BY_TOP_Y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
        Page1HeaderItem(
            text=internal_title,
            x=PAGE1_HEADER_RIGHT_X,
            top_y=PAGE1_HEADER_CREATED_BY_TITLE_TOP_Y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
            max_lines=2,
            line_step=PAGE1_HEADER_TITLE_LINE_STEP,
        ),
        Page1HeaderItem(
            text=str(internal.get("email") or "").strip(),
            x=PAGE1_HEADER_RIGHT_X,
            top_y=aligned_email_top_y,
            max_width=PAGE1_HEADER_RIGHT_MAX_WIDTH,
            font_name=GOTHAM_MEDIUM,
            font_size=PAGE1_HEADER_SMALL_FONT_SIZE,
            min_font_size=PAGE1_HEADER_MIN_FONT_SIZE,
        ),
    )


def _wrap_fitted_text(
    text: str,
    max_width: float,
    font_name: str,
    font_size: float,
    min_font_size: float,
    max_lines: int = 1,
) -> tuple[list[str], float]:
    clean = " ".join(str(text or "").split())
    size = font_size

    def wrapped(current_size: float) -> list[str]:
        return _wrap_text_line(
            clean,
            max_width,
            font_name,
            current_size,
        )

    while size > min_font_size:
        lines = wrapped(size)

        if (
            len(lines) <= max_lines
            and all(
                pdfmetrics.stringWidth(
                    line,
                    font_name,
                    size,
                )
                <= max_width
                for line in lines
            )
        ):
            return lines, size

        size -= 1

    lines = wrapped(size)

    if (
        len(lines) <= max_lines
        and all(
            pdfmetrics.stringWidth(
                line,
                font_name,
                size,
            )
            <= max_width
            for line in lines
        )
    ):
        return lines, size

    if max_lines > 1 and len(lines) > max_lines:
        kept = lines[:max_lines]
        kept[-1] = " ".join(
            [
                kept[-1],
                *lines[max_lines:],
            ]
        ).strip()
        lines = kept

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    if (
        lines
        and pdfmetrics.stringWidth(
            lines[-1],
            font_name,
            size,
        )
        <= max_width
    ):
        return lines, size

    ellipsis = "..."
    clean = lines[-1] if lines else clean

    while (
        clean
        and pdfmetrics.stringWidth(
            f"{clean}{ellipsis}",
            font_name,
            size,
        )
        > max_width
    ):
        clean = clean[:-1].rstrip()

    if lines:
        lines[-1] = (
            f"{clean}{ellipsis}"
            if clean
            else ""
        )

    return lines, size


def _fit_text(
    text: str,
    max_width: float,
    font_name: str,
    font_size: float,
    min_font_size: float,
) -> tuple[str, float]:
    lines, size = _wrap_fitted_text(
        text=text,
        max_width=max_width,
        font_name=font_name,
        font_size=font_size,
        min_font_size=min_font_size,
        max_lines=1,
    )

    return (
        lines[0] if lines else "",
        size,
    )


def _draw_page1_header(
    c: canvas.Canvas,
    page_height: float,
    payload: dict[str, Any] | None,
) -> None:
    c.setFillColor(PAGE1_HEADER_TEXT)

    for item in build_page1_header_items(payload):
        if not item.text:
            continue

        lines, font_size = _wrap_fitted_text(
            text=item.text,
            max_width=item.max_width,
            font_name=item.font_name,
            font_size=item.font_size,
            min_font_size=item.min_font_size,
            max_lines=item.max_lines,
        )

        c.setFont(
            item.font_name,
            font_size,
        )

        line_step = (
            item.line_step
            or font_size * 1.15
        )

        for index, line in enumerate(
            lines[: item.max_lines]
        ):
            c.drawString(
                item.x,
                _pdf_y(
                    page_height,
                    item.top_y + index * line_step,
                ),
                line,
            )


def _wrap_text_line(
    text: str,
    max_width: float,
    font_name: str,
    font_size: float,
) -> list[str]:
    words = text.split()

    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"

        if (
            pdfmetrics.stringWidth(
                candidate,
                font_name,
                font_size,
            )
            <= max_width
        ):
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _comments_text(
    payload: dict[str, Any] | None,
) -> str:
    if not payload:
        return ""

    return str(
        payload.get("rendered_comments_block")
        or ""
    ).strip()


def _page1_overlay(
    page_width: float,
    page_height: float,
    comments_payload: dict[str, Any] | None,
    header_payload: dict[str, Any] | None = None,
) -> BytesIO:
    _register_gotham_fonts()

    overlay = BytesIO()
    c = canvas.Canvas(
        overlay,
        pagesize=(page_width, page_height),
    )

    _draw_page1_header(
        c,
        page_height,
        header_payload,
    )

    c.setFillColor(PAGE1_TEXT)
    c.setFont(
        GOTHAM_MEDIUM,
        PAGE1_COMMENTS_FONT_SIZE,
    )

    lines: list[str] = []

    for raw_line in _comments_text(
        comments_payload
    ).splitlines():
        if raw_line.strip():
            lines.extend(
                _wrap_text_line(
                    text=raw_line.strip(),
                    max_width=PAGE1_COMMENTS_MAX_WIDTH,
                    font_name=GOTHAM_MEDIUM,
                    font_size=PAGE1_COMMENTS_FONT_SIZE,
                )
            )
        else:
            lines.append("")

    for index, line in enumerate(
        lines[:PAGE1_COMMENTS_MAX_LINES]
    ):
        c.drawString(
            PAGE1_COMMENTS_LEFT_X,
            _pdf_y(
                page_height,
                PAGE1_COMMENTS_TOP_Y
                + index * PAGE1_COMMENTS_LINE_STEP,
            ),
            line,
        )

    c.save()
    overlay.seek(0)

    return overlay


def _page2_overlay(
    page_width: float,
    page_height: float,
    payload: Page2PricingPayload,
) -> BytesIO:
    _register_gotham_fonts()

    overlay = BytesIO()
    c = canvas.Canvas(
        overlay,
        pagesize=(page_width, page_height),
    )

    c.setFillColor(PAGE2_TABLE_BACKGROUND)
    c.rect(
        PRICING_TABLE_MASK_LEFT_X,
        _pdf_y(page_height, PRICING_TABLE_MASK_BOTTOM_Y),
        PRICING_TABLE_MASK_RIGHT_X - PRICING_TABLE_MASK_LEFT_X,
        PRICING_TABLE_MASK_BOTTOM_Y - PRICING_TABLE_MASK_TOP_Y,
        stroke=0,
        fill=1,
    )

    c.setStrokeColor(TEXT)
    c.setLineWidth(2)

    for index, row in enumerate(payload.rows):
        if index >= len(PRICING_ROW_SLOT_TOP_Y):
            break

        _draw_pricing_row(
            c,
            page_height,
            PRICING_ROW_SLOT_TOP_Y[index],
            row,
        )

        if index < len(PRICING_ROW_SEPARATOR_TOP_Y):
            separator_y = _pdf_y(
                page_height,
                PRICING_ROW_SEPARATOR_TOP_Y[index],
            )
            c.line(
                PRICING_TABLE_MASK_LEFT_X + 15,
                separator_y,
                PRICING_TABLE_MASK_RIGHT_X - 30,
                separator_y,
            )

    _draw_totals(
        c,
        page_height,
        payload,
    )

    c.save()
    overlay.seek(0)

    return overlay


def _page_contains_old_pricing_template(page: Any) -> bool:
    text = page.extract_text() or ""
    pricing_markers = (
        "On-model detail",
        "Model Fitting",
        "AI Gene",
    )
    return all(marker in text for marker in pricing_markers)


def _merge_page1_overlay(
    page: Any,
    page1_comments_payload: dict[str, Any] | None,
    page1_header_payload: dict[str, Any] | None,
) -> None:
    if not (page1_comments_payload or page1_header_payload):
        return

    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    overlay_pdf = PdfReader(
        _page1_overlay(
            page_width=width,
            page_height=height,
            comments_payload=page1_comments_payload,
            header_payload=page1_header_payload,
        )
    )

    page.merge_page(
        overlay_pdf.pages[0]
    )


def _merge_page2_overlay(
    page: Any,
    payload: Page2PricingPayload,
) -> None:
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    overlay_pdf = PdfReader(
        _page2_overlay(
            page_width=width,
            page_height=height,
            payload=payload,
        )
    )

    page.merge_page(
        overlay_pdf.pages[0]
    )


def generate_page2_pricing_pdf(
    quote: Any,
    template_path: Path = MAIN_TEMPLATE_PATH,
    page1_comments_payload: dict[str, Any] | None = None,
    page1_header_payload: dict[str, Any] | None = None,
    pricing_template_path: Path = PRICING_TEMPLATE_PATH,
) -> bytes:
    payload = build_page2_pricing_payload(quote)

    main_reader = PdfReader(str(template_path))
    pricing_reader = PdfReader(str(pricing_template_path))
    writer = PdfWriter()

    first_page = main_reader.pages[0]
    _merge_page1_overlay(
        first_page,
        page1_comments_payload,
        page1_header_payload,
    )
    writer.add_page(first_page)

    pricing_page = pricing_reader.pages[0]
    _merge_page2_overlay(
        pricing_page,
        payload,
    )
    writer.add_page(pricing_page)

    for index, page in enumerate(
        main_reader.pages[1:],
        start=1,
    ):
        if (
            index == 1
            and _page_contains_old_pricing_template(page)
        ):
            continue

        writer.add_page(page)

    output = BytesIO()
    writer.write(output)

    return output.getvalue()
