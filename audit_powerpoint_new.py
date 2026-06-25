from __future__ import annotations

import base64
import io
import math
import re
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt

from app.audit_helpers.image_guides import (
    get_image_guide_page,
    load_image_guide,
    resolve_image_guide_category,
)
from app.audit_helpers.slide4_findings import build_slide4_group_findings
from audit_powerpoint import _format_cover_date


PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")
ONLY_PLACEHOLDERS_RE = re.compile(r"^(?:\s*\{\{[^{}]+\}\}\s*)+$")
SLIDE4_SLOT_LABELS = {
    "silo_front": "Front Silo",
    "graphic_ingredients": "Ingredients",
    "silo_alt_in_pack": "Alt In-Pack",
    "lifestyle_in_use": "Lifestyle",
    "graphic_guarantee": "Guarantee",
    "feature_graphic": "Features",
    "dimensions": "Dimensions",
    "graphic_nutrition": "Nutrition",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _shape_text(shape: Any) -> str:
    """Get text from a shape, handling text frames."""
    if not getattr(shape, "has_text_frame", False):
        return ""
    return _safe_text(getattr(shape, "text", ""))


def _find_shape_contains(slide: Any, token: str) -> Any:
    """Find a shape by searching for token text (case-insensitive)."""
    token_l = token.lower()
    for shape in slide.shapes:
        text = _shape_text(shape).lower()
        if token_l in text:
            return shape
    return None


def _replace_shape_text_preserve_style(shape: Any, text: str) -> None:
    """Replace text in a shape while preserving existing run styling."""
    if not shape or not getattr(shape, "has_text_frame", False):
        return
    tf = shape.text_frame
    if not tf.paragraphs:
        shape.text = _safe_text(text)
        return
    first_para = tf.paragraphs[0]
    if first_para.runs:
        first_para.runs[0].text = _safe_text(text)
        for run in first_para.runs[1:]:
            run.text = ""
        for para in tf.paragraphs[1:]:
            for run in para.runs:
                run.text = ""
        return
    shape.text = _safe_text(text)


def _replace_paragraph_text_preserve_style(paragraph: Any, text: str) -> None:
    """Replace text in a paragraph while preserving styling."""
    if paragraph.runs:
        paragraph.runs[0].text = _safe_text(text)
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.text = _safe_text(text)


def _replace_text_in_paragraph(paragraph: Any, replacements: dict[str, str]) -> dict[str, int]:
    """Replace template text within a paragraph without creating new shapes."""
    counts = {target: 0 for target in replacements}
    if not paragraph:
        return counts

    for run in getattr(paragraph, "runs", []) or []:
        run_text = run.text or ""
        if not run_text:
            continue
        new_text = run_text
        for target, replacement in replacements.items():
            found = new_text.count(target)
            if found:
                counts[target] += found
                new_text = new_text.replace(target, replacement)
        if new_text != run_text:
            run.text = new_text

    paragraph_text = paragraph.text or ""
    if not any(target in paragraph_text for target in replacements):
        return counts

    new_text = paragraph_text
    for target, replacement in replacements.items():
        found = new_text.count(target)
        if found:
            counts[target] += found
            new_text = new_text.replace(target, replacement)

    _replace_paragraph_text_preserve_style(paragraph, new_text)
    return counts


def _merge_counts(total: dict[str, int], increment: dict[str, int]) -> None:
    for target, count in increment.items():
        total[target] = total.get(target, 0) + int(count or 0)


def _walk_shapes(shapes: Any) -> list[Any]:
    found = []
    for shape in shapes:
        found.append(shape)
        if hasattr(shape, "shapes"):
            found.extend(_walk_shapes(shape.shapes))
    return found


def _find_slide_by_title(prs: Any, title_text: str) -> Any:
    """Find a slide by exact title text first, then fall back to containment."""
    title_l = title_text.strip().lower()
    for slide in prs.slides:
        for shape in _walk_shapes(slide.shapes):
            if _shape_text(shape).strip().lower() == title_l:
                return slide
    for slide in prs.slides:
        for shape in _walk_shapes(slide.shapes):
            text = _shape_text(shape).lower()
            if title_l in text:
                return slide
    return None


def _remove_slide_by_title(prs: Any, title_text: str) -> bool:
    """
    Remove a slide by its title text.
    Returns True if a slide was removed, False otherwise.
    """
    title_l = title_text.lower()
    slide_idx = None
    for idx, slide in enumerate(prs.slides):
        for shape in _walk_shapes(slide.shapes):
            text = _shape_text(shape).lower()
            if title_l in text:
                slide_idx = idx
                break
        if slide_idx is not None:
            break
    
    if slide_idx is not None:
        # Remove the slide via the slide layout's id
        rId = prs.slides._sldIdLst[slide_idx].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[slide_idx]
        return True
    return False


def replace_existing_template_text(prs: Any, replacements: dict[str, str]) -> dict[str, int]:
    """
    Replace existing template strings in-place across slides, groups, and tables.
    Returns a replacement count by target string for debug reporting.
    """
    counts = {target: 0 for target in replacements}
    for slide in prs.slides:
        for shape in _walk_shapes(slide.shapes):
            if getattr(shape, "has_text_frame", False):
                for paragraph in shape.text_frame.paragraphs:
                    _merge_counts(counts, _replace_text_in_paragraph(paragraph, replacements))
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            _merge_counts(counts, _replace_text_in_paragraph(paragraph, replacements))
    for target in replacements:
        if counts.get(target, 0):
            print(f"[audit_powerpoint_new] Replaced '{target}' {counts[target]} time(s).")
        else:
            print(f"[audit_powerpoint_new] Template text '{target}' was not found.")
    return counts


def _contains_unresolved_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text or ""))


def _only_unresolved_placeholders(text: str) -> bool:
    return bool(ONLY_PLACEHOLDERS_RE.fullmatch(text or ""))


def _remove_shape(shape: Any) -> bool:
    element = getattr(shape, "element", None)
    parent = element.getparent() if element is not None else None
    if parent is None:
        return False
    parent.remove(element)
    return True


def _clear_unresolved_placeholders_in_paragraph(paragraph: Any) -> int:
    text = paragraph.text or ""
    if not _contains_unresolved_placeholder(text):
        return 0

    if _only_unresolved_placeholders(text):
        _replace_paragraph_text_preserve_style(paragraph, "")
        return len(PLACEHOLDER_RE.findall(text))

    count = 0
    for run in getattr(paragraph, "runs", []) or []:
        run_text = run.text or ""
        if not _contains_unresolved_placeholder(run_text):
            continue
        if _only_unresolved_placeholders(run_text):
            count += len(PLACEHOLDER_RE.findall(run_text))
            run.text = ""
        else:
            cleaned = PLACEHOLDER_RE.sub("", run_text)
            count += len(PLACEHOLDER_RE.findall(run_text))
            run.text = cleaned

    # Fallback for placeholders split across runs.
    remaining = paragraph.text or ""
    if _contains_unresolved_placeholder(remaining):
        count += len(PLACEHOLDER_RE.findall(remaining))
        _replace_paragraph_text_preserve_style(paragraph, PLACEHOLDER_RE.sub("", remaining))
    return count


def clear_unresolved_placeholder_text(prs: Any) -> dict[str, int]:
    """Remove placeholder-only shapes and clear any remaining {{...}} text."""
    counts = {"removed_shapes": 0, "cleared_placeholders": 0}
    for slide in prs.slides:
        for shape in list(_walk_shapes(slide.shapes)):
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            counts["cleared_placeholders"] += _clear_unresolved_placeholders_in_paragraph(paragraph)

            if not getattr(shape, "has_text_frame", False):
                continue

            shape_text = getattr(shape, "text", "") or ""
            if _only_unresolved_placeholders(shape_text):
                if _remove_shape(shape):
                    counts["removed_shapes"] += 1
                continue

            for paragraph in shape.text_frame.paragraphs:
                counts["cleared_placeholders"] += _clear_unresolved_placeholders_in_paragraph(paragraph)

    print(
        "[audit_powerpoint_new] Removed "
        f"{counts['removed_shapes']} unresolved placeholder shape(s); cleared "
        f"{counts['cleared_placeholders']} unresolved placeholder token(s)."
    )
    return counts


def _find_pictures_in_region(slide: Any, region_left: float, region_top: float, region_right: float, region_bottom: float) -> list[Any]:
    """Find all picture shapes within a region."""
    pictures = []
    for shape in slide.shapes:
        if hasattr(shape, "image") or "picture" in str(shape.shape_type).lower():
            if hasattr(shape, "left") and hasattr(shape, "top"):
                shape_left = shape.left.inches if hasattr(shape.left, "inches") else shape.left
                shape_top = shape.top.inches if hasattr(shape.top, "inches") else shape.top
                if (shape_left >= region_left and shape_left <= region_right and
                    shape_top >= region_top and shape_top <= region_bottom):
                    pictures.append(shape)
    return pictures


def _load_image_from_url(url: str) -> bytes | None:
    """Download image from URL and return bytes. Returns None on failure."""
    try:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=10) as response:
            return response.read()
    except Exception as exc:
        print(f"[audit_powerpoint_new] Slide 4 image download failed: {url} ({exc})")
        return None


def _add_contained_picture(
    slide: Any,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    image_bytes: bytes,
) -> None:
    """Add an uncropped picture contained and centered within an EMU cell."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            return
        scale = min(width / image_width, height / image_height)
        rendered_width = max(1, int(image_width * scale))
        rendered_height = max(1, int(image_height * scale))
        rendered_left = left + (width - rendered_width) // 2
        rendered_top = top + (height - rendered_height) // 2
        slide.shapes.add_picture(
            io.BytesIO(image_bytes),
            left=rendered_left,
            top=rendered_top,
            width=rendered_width,
            height=rendered_height,
        )
    except Exception as exc:
        print(f"[audit_powerpoint_new] Slide 4 image placement failed: {exc}")


def _normalize_ordered_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    images = record.get("ordered_images")
    if not isinstance(images, list):
        images = record.get("images", []) or []
    normalized: list[dict[str, Any]] = []
    for position, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        normalized.append(
            {
                "index": image.get("index", position),
                "url": _safe_text(image.get("url", "")),
                "is_hero": bool(image.get("is_hero", position == 0)),
                "width": image.get("width"),
                "height": image.get("height"),
                "dimensions": image.get("dimensions", ""),
                "dimensions_text": image.get("dimensions_text", ""),
            }
        )
    return normalized[:12]


def _build_image_guide_match(category: str, product_type: str) -> dict[str, Any]:
    category_key = resolve_image_guide_category(category)
    page = get_image_guide_page(category_key, product_type)
    if not page:
        return {
            "matched": False,
            "category_key": category_key,
            "product_type": product_type,
        }
    guide = load_image_guide(category_key)
    slot_definitions = guide.get("slot_definitions", {}) or {}
    required_slots = list(page.get("required_slots", []) or [])
    return {
        "matched": True,
        "category_key": category_key,
        "product_type": product_type,
        "page_key": page.get("page_key"),
        "page_display_name": page.get("display_name", ""),
        "required_slots": required_slots,
        "required_slot_labels": [
            SLIDE4_SLOT_LABELS.get(slot)
            or _safe_text((slot_definitions.get(slot, {}) or {}).get("label"))
            or str(slot).replace("_", " ").title()
            for slot in required_slots
        ],
        "additional_recommendations": list(page.get("additional_recommendations", []) or []),
    }


def _parse_image_dimensions(image: dict[str, Any]) -> tuple[int, int] | None:
    try:
        width = int(float(str(image.get("width", "")).replace(",", "").strip()))
        height = int(float(str(image.get("height", "")).replace(",", "").strip()))
        if width > 0 and height > 0:
            return width, height
    except (TypeError, ValueError):
        pass
    raw = _safe_text(image.get("dimensions_text") or image.get("dimensions"))
    match = re.search(r"(\d{2,5})\s*(?:x|×)\s*(\d{2,5})", raw, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _build_slide4_bullets(images: list[dict[str, Any]], guide_match: dict[str, Any]) -> list[str]:
    bullets = [f"Carousel: {len(images)} ordered images"]
    dimensions = [dims for image in images if (dims := _parse_image_dimensions(image))]
    unique_dimensions = list(dict.fromkeys(dimensions))
    if len(unique_dimensions) == 1:
        width, height = unique_dimensions[0]
        suffix = " throughout" if len(dimensions) == len(images) else " detected"
        bullets.append(f"Dimensions: {width} x {height}{suffix}")
    elif len(unique_dimensions) > 1:
        bullets.append(f"Dimensions: {len(unique_dimensions)} detected asset sizes")

    if guide_match.get("matched"):
        labels = list(guide_match.get("required_slot_labels", []) or [])
        if labels:
            bullets.append("Recommended opening: " + " / ".join(labels[:3]))
        if len(labels) > 3:
            bullets.append("Recommended support: " + " / ".join(labels[3:6]))
        additional = list(guide_match.get("additional_recommendations", []) or [])
        if additional:
            opportunity = _safe_text(additional[0]).split(":", 1)[0]
            if opportunity:
                bullets.append(f"Guide opportunity: {opportunity}")
    return bullets[:5]


def _build_slide4_column(
    record: dict[str, Any],
    fallback_label: str,
    findings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not record:
        return {
            "label": fallback_label,
            "brand": "",
            "category": "",
            "product_type": "",
            "ordered_images": [],
            "image_count": 0,
            "image_guide_match": {"matched": False},
            "bullets": [],
            "findings": findings or {},
        }
    images = _normalize_ordered_images(record)
    product_type = _safe_text(record.get("product_type") or record.get("subcategory"))
    category = _safe_text(record.get("category"))
    guide_match = _build_image_guide_match(category, product_type)
    findings = findings or build_slide4_group_findings([record], fallback_label)
    finding_bullets = [
        _safe_text(bullet)
        for bullet in (findings.get("slide4_bullets", []) if isinstance(findings, dict) else [])
        if _safe_text(bullet)
    ]
    return {
        "label": _safe_text(record.get("brand")) or fallback_label,
        "brand": _safe_text(record.get("brand")),
        "category": category,
        "product_type": product_type,
        "ordered_images": images,
        "image_count": int(record.get("image_count", len(images)) or len(images)),
        "image_guide_match": guide_match,
        "bullets": finding_bullets[:5] or _build_slide4_bullets(images, guide_match),
        "findings": findings,
    }


def build_slide4_pdp_benchmark_payload(
    export_plan: dict, competitor_records: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build deterministic Slide 4 content from one client and two competitor PDPs."""
    metadata = export_plan.get("audit_metadata", {}) or {}
    product_pairs = export_plan.get("product_slide_pairs", []) or []
    client_company_name = _safe_text(metadata.get("client_company_name", ""))
    client_name = _safe_text(metadata.get("client_name", ""))
    client_record = (
        dict((product_pairs[0].get("pdp_slide", {}) or {}))
        if product_pairs
        else {}
    )
    client_fallback = (
        client_company_name
        or client_name
        or _safe_text(client_record.get("brand"))
        or "Client"
    )
    slide4_findings = export_plan.get("slide4_findings", {}) or {}
    client_column = _build_slide4_column(
        client_record,
        client_fallback,
        slide4_findings.get("client") if isinstance(slide4_findings, dict) else None,
    )
    client_column["label"] = client_company_name or client_name or client_column["label"]

    competitor_records = list(competitor_records or [])
    competitor_columns = [
        _build_slide4_column(
            competitor_records[index] if index < len(competitor_records) else {},
            f"Competitor {index + 1}",
            (
                slide4_findings.get(f"competitor_{index + 1}")
                if isinstance(slide4_findings, dict)
                else None
            ),
        )
        for index in range(2)
    ]
    return {"columns": [client_column, *competitor_columns], "slide4_findings": slide4_findings}


def _slide4_label_shapes(slide: Any) -> list[Any]:
    candidates = []
    for shape in _walk_shapes(slide.shapes):
        if not getattr(shape, "has_text_frame", False):
            continue
        text = _shape_text(shape)
        if not text or len(text) > 80:
            continue
        top = int(getattr(shape, "top", 0) or 0)
        if Inches(2.1) <= top <= Inches(2.8):
            candidates.append(shape)
    return sorted(candidates, key=lambda shape: int(getattr(shape, "left", 0) or 0))[:3]


def _slide4_bullet_shapes(slide: Any) -> list[Any]:
    candidates = [
        shape
        for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
        and int(getattr(shape, "top", 0) or 0) >= Inches(5.4)
        and int(getattr(shape, "height", 0) or 0) >= Inches(0.8)
    ]
    return sorted(candidates, key=lambda shape: int(getattr(shape, "left", 0) or 0))[:3]


def _replace_bullet_shape_text(shape: Any, bullets: list[str]) -> None:
    paragraphs = list(shape.text_frame.paragraphs)
    for index, paragraph in enumerate(paragraphs):
        _replace_paragraph_text_preserve_style(
            paragraph, bullets[index] if index < len(bullets) else ""
        )


def _fit_slide4_label(shape: Any) -> None:
    text_frame = shape.text_frame
    text_frame.word_wrap = False
    text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    label_text = _shape_text(shape)
    if len(label_text) > 12:
        font_size = Pt(10 if len(label_text) > 20 else 12)
        for paragraph in text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = font_size


def _slide4_picture_containers(prs: Any, slide: Any) -> list[Any]:
    slide_width = int(prs.slide_width)
    slide_height = int(prs.slide_height)
    pictures = [
        shape
        for shape in slide.shapes
        if hasattr(shape, "image") or "picture" in str(shape.shape_type).lower()
    ]
    on_slide = [
        shape
        for shape in pictures
        if int(shape.left) < slide_width
        and int(shape.top) < slide_height
        and int(shape.left + shape.width) > 0
        and int(shape.top + shape.height) > 0
    ]
    return sorted(on_slide, key=lambda shape: int(shape.left))[:3]


def _populate_slide4_carousel(
    slide: Any,
    *,
    container: tuple[int, int, int, int],
    images: list[dict[str, Any]],
) -> None:
    usable_images = [image for image in images[:12] if _safe_text(image.get("url"))]
    if not usable_images:
        return
    left, top, width, height = container
    columns = 3
    rows = max(1, math.ceil(len(usable_images) / columns))
    gutter = Inches(0.06)
    cell_width = int((width - gutter * (columns - 1)) / columns)
    cell_height = int((height - gutter * (rows - 1)) / rows)
    for position, image in enumerate(usable_images):
        row, column = divmod(position, columns)
        cell_left = int(left + column * (cell_width + gutter))
        cell_top = int(top + row * (cell_height + gutter))
        image_bytes = _load_image_from_url(_safe_text(image.get("url")))
        if image_bytes is None:
            continue
        _add_contained_picture(
            slide,
            left=cell_left,
            top=cell_top,
            width=cell_width,
            height=cell_height,
            image_bytes=image_bytes,
        )


def _apply_slide4_content(prs: Any, slide: Any, payload: dict[str, Any]) -> None:
    columns = list(payload.get("columns", []) or [])[:3]
    while len(columns) < 3:
        columns.append({"label": f"Competitor {len(columns)}", "ordered_images": [], "bullets": []})

    labels = _slide4_label_shapes(slide)
    bullets = _slide4_bullet_shapes(slide)
    containers = _slide4_picture_containers(prs, slide)
    if len(labels) < 3 or len(bullets) < 3 or len(containers) < 3:
        print(
            "[audit_powerpoint_new] Slide 4 template shapes incomplete; "
            "leaving the slide without floating replacement content."
        )
        return

    container_bounds = [
        (int(shape.left), int(shape.top), int(shape.width), int(shape.height))
        for shape in containers
    ]
    for shape in containers:
        _remove_shape(shape)
    for shape in list(slide.shapes):
        if not (hasattr(shape, "image") or "picture" in str(shape.shape_type).lower()):
            continue
        if (
            int(shape.left) < 0
            or int(shape.top) < 0
            or int(shape.left + shape.width) > int(prs.slide_width)
            or int(shape.top + shape.height) > int(prs.slide_height)
        ):
            _remove_shape(shape)

    for index, column in enumerate(columns):
        _replace_shape_text_preserve_style(labels[index], column.get("label", ""))
        _fit_slide4_label(labels[index])
        _replace_bullet_shape_text(bullets[index], list(column.get("bullets", []) or []))
        left, top, width, height = container_bounds[index]
        bullet_top = int(bullets[index].top)
        height = max(1, min(height, bullet_top - Inches(0.05) - top))
        _populate_slide4_carousel(
            slide,
            container=(left, top, width, height),
            images=list(column.get("ordered_images", []) or []),
        )


def _decode_data_image(value: Any) -> bytes | None:
    data_url = _safe_text(value)
    if not data_url.lower().startswith("data:image/") or "," not in data_url:
        return None
    header, encoded = data_url.split(",", 1)
    if ";base64" not in header.lower():
        return None
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        return image_bytes
    except Exception as exc:
        print(f"[audit_powerpoint_new] Slide 5 screenshot decode failed: {exc}")
        return None


def _slide5_side_shapes(prs: Any, slide: Any) -> dict[str, dict[str, Any]]:
    midpoint = int(prs.slide_width) // 2
    pictures = sorted(
        [
            shape
            for shape in slide.shapes
            if (hasattr(shape, "image") or "picture" in str(shape.shape_type).lower())
            and int(getattr(shape, "top", 0) or 0) >= Inches(2.5)
        ],
        key=lambda shape: int(shape.left),
    )
    bullet_shapes = sorted(
        [
            shape
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False)
            and int(getattr(shape, "top", 0) or 0) >= Inches(2.5)
            and int(getattr(shape, "height", 0) or 0) >= Inches(2.0)
            and len(shape.text_frame.paragraphs) >= 6
        ],
        key=lambda shape: int(shape.left),
    )
    left_header = next(
        (
            shape
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False)
            and _shape_text(shape) == "Current Structure"
        ),
        None,
    )
    right_header = next(
        (
            shape
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False)
            and _shape_text(shape) == "Competitive Benchmark"
        ),
        None,
    )
    lines = sorted(
        [
            shape
            for shape in slide.shapes
            if "LINE" in str(getattr(shape, "shape_type", "")).upper()
            and int(getattr(shape, "top", 0) or 0) >= Inches(2.0)
            and int(getattr(shape, "top", 0) or 0) <= Inches(3.0)
        ],
        key=lambda shape: int(shape.left),
    )
    return {
        "client": {
            "picture": next(
                (shape for shape in pictures if int(shape.left) < midpoint),
                None,
            ),
            "bullets": next(
                (shape for shape in bullet_shapes if int(shape.left) < midpoint),
                None,
            ),
            "header": left_header,
            "divider": lines[0] if lines else None,
        },
        "competitor": {
            "picture": next(
                (shape for shape in pictures if int(shape.left) >= midpoint),
                None,
            ),
            "bullets": next(
                (shape for shape in bullet_shapes if int(shape.left) >= midpoint),
                None,
            ),
            "header": right_header,
            "divider": lines[-1] if lines else None,
        },
    }


def _apply_slide5_no_brand_shop(
    prs: Any,
    slide: Any,
    payload: dict[str, Any],
    shapes: dict[str, dict[str, Any]],
) -> None:
    competitor = payload.get("competitor")
    if not isinstance(competitor, dict):
        print(
            "[audit_powerpoint_new] Slide 5 No Brand Shop mode lacked valid "
            "Competitor evidence; the slide was left unchanged."
        )
        return
    competitor_picture = shapes["competitor"].get("picture")
    competitor_bullets = shapes["competitor"].get("bullets")
    competitor_header = shapes["competitor"].get("header")
    competitor_divider = shapes["competitor"].get("divider")
    if any(
        shape is None
        for shape in (
            competitor_picture,
            competitor_bullets,
            competitor_header,
            competitor_divider,
        )
    ):
        print(
            "[audit_powerpoint_new] Slide 5 No Brand Shop template shapes were "
            "not resolved; the slide was left unchanged."
        )
        return
    image_bytes = _decode_data_image(competitor.get("screenshot"))
    bullets = [
        _safe_text(value)
        for value in (competitor.get("bullets", []) or [])
        if _safe_text(value)
    ]
    if image_bytes is None or len(bullets) != 6:
        print(
            "[audit_powerpoint_new] Slide 5 No Brand Shop payload was incomplete; "
            "the slide was left unchanged."
        )
        return

    for key in ("header", "divider", "picture", "bullets"):
        shape = shapes["client"].get(key)
        if shape is not None:
            _remove_shape(shape)

    competitor_header.left = Inches(3.45)
    competitor_header.width = Inches(3.2)
    competitor_divider.left = Inches(1.4)
    competitor_divider.width = Inches(11.2)

    _remove_shape(competitor_picture)
    _add_contained_picture(
        slide,
        left=Inches(1.4),
        top=Inches(2.88),
        width=Inches(7.4),
        height=Inches(3.77),
        image_bytes=image_bytes,
    )
    _replace_bullet_shape_text(competitor_bullets, bullets)


def _apply_slide5_brand_shop(prs: Any, slide: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    shapes = _slide5_side_shapes(prs, slide)
    if payload.get("mode") == "no_brand_shop":
        _apply_slide5_no_brand_shop(prs, slide, payload, shapes)
        return
    for side in ("client", "competitor"):
        side_payload = payload.get(side)
        if not isinstance(side_payload, dict):
            print(
                f"[audit_powerpoint_new] Slide 5 {side} evidence was unavailable; "
                "the template side was left unchanged."
            )
            continue
        picture = shapes[side].get("picture")
        bullet_shape = shapes[side].get("bullets")
        if picture is None or bullet_shape is None:
            print(
                f"[audit_powerpoint_new] Slide 5 {side} template shapes were not resolved; "
                "the template side was left unchanged."
            )
            continue
        image_bytes = _decode_data_image(side_payload.get("screenshot"))
        bullets = [
            _safe_text(value)
            for value in (side_payload.get("bullets", []) or [])
            if _safe_text(value)
        ]
        if image_bytes is None:
            print(
                f"[audit_powerpoint_new] Slide 5 {side} screenshot was invalid; "
                "the template picture was preserved."
            )
        else:
            bounds = (
                int(picture.left),
                int(picture.top),
                int(picture.width),
                int(picture.height),
            )
            _remove_shape(picture)
            _add_contained_picture(
                slide,
                left=bounds[0],
                top=bounds[1],
                width=bounds[2],
                height=bounds[3],
                image_bytes=image_bytes,
            )
        if len(bullets) == 6:
            _replace_bullet_shape_text(bullet_shape, bullets)
        else:
            print(
                f"[audit_powerpoint_new] Slide 5 {side} did not contain exactly six bullets; "
                "the template bullet box was preserved."
            )


def _slide2_text_shapes(slide: Any) -> list[Any]:
    return [
        shape
        for shape in _walk_shapes(slide.shapes)
        if getattr(shape, "has_text_frame", False) and _shape_text(shape)
    ]


def _shape_contains_any(shape: Any, tokens: tuple[str, ...]) -> bool:
    text = _shape_text(shape).lower()
    return any(token.lower() in text for token in tokens)


def _apply_slide2_summary(slide: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    shapes = _slide2_text_shapes(slide)
    intro_copy = _safe_text(payload.get("intro_copy"))
    if intro_copy:
        intro_shape = next(
            (
                shape
                for shape in shapes
                if _shape_contains_any(
                    shape,
                    (
                        "has built strong consumer trust",
                        "the next opportunity is translating",
                        "clean lifestyle categories",
                    ),
                )
            ),
            None,
        )
        if intro_shape is None:
            intro_candidates = [
                shape
                for shape in shapes
                if int(getattr(shape, "left", 0) or 0) < Inches(5.0)
                and int(getattr(shape, "top", 0) or 0) >= Inches(2.0)
                and int(getattr(shape, "height", 0) or 0) >= Inches(1.0)
            ]
            intro_shape = sorted(
                intro_candidates,
                key=lambda shape: int(getattr(shape, "top", 0) or 0),
            )[0] if intro_candidates else None
        if intro_shape is not None:
            _replace_shape_text_preserve_style(intro_shape, intro_copy)

    sections = payload.get("sections", {}) or {}
    rating_shapes = sorted(
        [
            shape
            for shape in shapes
            if _shape_text(shape) in {"Strong", "Significant", "Evolving", "Emerging", "Limited", "Meaningful", "Selective", "Competitive"}
            and int(getattr(shape, "left", 0) or 0) >= Inches(9.0)
        ],
        key=lambda shape: int(getattr(shape, "top", 0) or 0),
    )
    section_order = ("consumer_demand", "walmart_opportunity", "competitive_benchmark")
    for index, section_key in enumerate(section_order):
        if index < len(rating_shapes):
            rating = _safe_text((sections.get(section_key, {}) or {}).get("rating"))
            if rating:
                _replace_shape_text_preserve_style(rating_shapes[index], rating)

    bullet_shapes = sorted(
        [
            shape
            for shape in shapes
            if int(getattr(shape, "left", 0) or 0) >= Inches(5.0)
            and int(getattr(shape, "top", 0) or 0) >= Inches(1.45)
            and int(getattr(shape, "height", 0) or 0) >= Inches(0.7)
            and not _shape_contains_any(
                shape,
                (
                    "consumer demand",
                    "walmart opportunity",
                    "competitive benchmark",
                    "walmart ecommerce opportunity",
                ),
            )
            and _shape_text(shape) not in {"Strong", "Significant", "Evolving", "Emerging", "Limited", "Meaningful", "Selective", "Competitive"}
        ],
        key=lambda shape: int(getattr(shape, "top", 0) or 0),
    )[:3]
    for index, section_key in enumerate(section_order):
        if index >= len(bullet_shapes):
            continue
        bullets = [
            _safe_text(bullet)
            for bullet in ((sections.get(section_key, {}) or {}).get("bullets", []) or [])
            if _safe_text(bullet)
        ]
        _replace_bullet_shape_text(bullet_shapes[index], bullets[:4])


def _replace_cell_text_preserve_style(cell: Any, text: str) -> None:
    text_frame = getattr(cell, "text_frame", None)
    if text_frame is None or not text_frame.paragraphs:
        cell.text = _safe_text(text)
        return
    _replace_paragraph_text_preserve_style(text_frame.paragraphs[0], text)
    for paragraph in text_frame.paragraphs[1:]:
        _replace_paragraph_text_preserve_style(paragraph, "")


def _slide6_display_label(payload: dict[str, Any], value: Any, context: str) -> str:
    label = _safe_text(value)
    if label in {"Strong", "Moderate", "Partial", "Limited"}:
        return label
    warning = (
        f"{context} produced invalid visibility '{label or '<empty>'}'; "
        "replaced with Limited."
    )
    payload.setdefault("warnings", []).append(warning)
    print(f"[audit_powerpoint_new] Slide 6 warning: {warning}")
    return "Limited"


def _fit_slide6_client_header(cell: Any, client_label: str) -> None:
    if len(client_label) <= 18:
        return
    text_frame = cell.text_frame
    text_frame.word_wrap = True
    text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    font_size = 12 if len(client_label) <= 28 else 10 if len(client_label) <= 40 else 8
    for paragraph in text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(font_size)


def _apply_slide6_visibility(slide: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    segments = list(payload.get("segments", []) or [])
    if len(segments) != 6:
        print(
            "[audit_powerpoint_new] Slide 6 visibility payload did not contain "
            "exactly six segments; table replacement was skipped."
        )
        return

    table_shape = next(
        (
            shape
            for shape in _walk_shapes(slide.shapes)
            if getattr(shape, "has_table", False)
            and len(shape.table.rows) >= 7
            and len(shape.table.columns) >= 3
            and "search segment" in _safe_text(shape.table.cell(0, 0).text).lower()
        ),
        None,
    )
    if table_shape is None:
        print(
            "[audit_powerpoint_new] Slide 6 search-segment table was not found; "
            "visibility rows were skipped."
        )
    else:
        table = table_shape.table
        client_label = _safe_text(payload.get("client_label")) or "Client"
        _replace_cell_text_preserve_style(
            table.cell(0, 2),
            client_label,
        )
        _fit_slide6_client_header(table.cell(0, 2), client_label)
        for row_index, segment in enumerate(segments, start=1):
            _replace_cell_text_preserve_style(table.cell(row_index, 0), segment.get("segment", ""))
            _replace_cell_text_preserve_style(
                table.cell(row_index, 1),
                _slide6_display_label(
                    payload,
                    segment.get("competitor_visibility"),
                    f"row {row_index} competitor",
                ),
            )
            _replace_cell_text_preserve_style(
                table.cell(row_index, 2),
                _slide6_display_label(
                    payload,
                    segment.get("client_visibility"),
                    f"row {row_index} client",
                ),
            )

    text_shapes = [
        shape
        for shape in _walk_shapes(slide.shapes)
        if getattr(shape, "has_text_frame", False)
        and _shape_text(shape).strip().lower() != "digital shelf ownership"
    ]
    intro_shape = next(
        (
            shape
            for shape in text_shapes
            if "competitors currently own more walmart search paths" in _shape_text(shape).lower()
        ),
        None,
    )
    if intro_shape is None:
        intro_shape = min(
            (shape for shape in text_shapes if int(getattr(shape, "top", 0) or 0) < Inches(2.5)),
            key=lambda shape: int(getattr(shape, "top", 0) or 0),
            default=None,
        )
    if intro_shape is not None:
        _replace_shape_text_preserve_style(intro_shape, payload.get("intro", ""))
    else:
        print("[audit_powerpoint_new] Slide 6 intro shape was not found.")

    takeaway_shape = next(
        (
            shape
            for shape in text_shapes
            if "walmart shoppers are highly search-driven" in _shape_text(shape).lower()
        ),
        None,
    )
    if takeaway_shape is None:
        takeaway_shape = max(
            text_shapes,
            key=lambda shape: int(getattr(shape, "top", 0) or 0),
            default=None,
        )
    if takeaway_shape is not None:
        _replace_shape_text_preserve_style(takeaway_shape, payload.get("takeaway", ""))
    else:
        print("[audit_powerpoint_new] Slide 6 takeaway shape was not found.")


def generate_new_audit_powerpoint_from_template(
    *, export_plan: dict, template_path: str, include_slide_9: bool = False, competitor_records: list[dict[str, Any]] | None = None
) -> bytes:
    prs = Presentation(template_path)
    metadata = export_plan.get("audit_metadata", {}) or {}
    client_name = _safe_text(metadata.get("client_name", ""))
    client_company_name = _safe_text(metadata.get("client_company_name", "")) or client_name
    audit_date = _format_cover_date(_safe_text(metadata.get("audit_date", "")))
    
    # Handle Slide 9 removal if not included
    if not include_slide_9:
        _remove_slide_by_title(prs, "Walmart Cash Program Visibility")
    _remove_slide_by_title(prs, "If they only have bandwidth for 5 things:")
    
    # Existing template strings are the replacement targets. Do not require manual placeholders.
    template_replacements = {
        "The Honest Company": client_company_name,
        "May 27, 2026": audit_date,
        "Honest": client_name,
    }
    replace_existing_template_text(prs, template_replacements)

    slide2 = _find_slide_by_title(prs, "Walmart eCommerce Opportunity")
    if slide2 is None:
        print(
            "[audit_powerpoint_new] Slide 2 title was not found; "
            "Walmart eCommerce Opportunity summary was skipped."
        )
    else:
        _apply_slide2_summary(slide2, export_plan.get("slide2_summary", {}) or {})

    slide4 = _find_slide_by_title(prs, "PDP Content Benchmarking")
    if slide4 is None:
        print(
            "[audit_powerpoint_new] Slide 4 title was not found; "
            "PDP benchmarking was skipped."
        )
    else:
        slide4_payload = build_slide4_pdp_benchmark_payload(
            export_plan,
            competitor_records,
        )
        _apply_slide4_content(prs, slide4, slide4_payload)

    slide5 = _find_slide_by_title(prs, "Brand Shop Content Benchmarking")
    if slide5 is None:
        print(
            "[audit_powerpoint_new] Slide 5 title was not found; "
            "Brand Shop benchmarking was skipped."
        )
    else:
        _apply_slide5_brand_shop(
            prs,
            slide5,
            export_plan.get("slide5_brand_shop", {}) or {},
        )

    slide6 = _find_slide_by_title(prs, "Digital Shelf Ownership")
    if slide6 is None:
        print(
            "[audit_powerpoint_new] Slide 6 title was not found; "
            "digital shelf alignment was skipped."
        )
    else:
        _apply_slide6_visibility(
            slide6,
            export_plan.get("slide6_visibility", {}) or {},
        )

    clear_unresolved_placeholder_text(prs)
    
    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
