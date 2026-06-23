from __future__ import annotations

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
    """Find a slide by searching for title text in any shape."""
    title_l = title_text.lower()
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


def _build_slide4_column(record: dict[str, Any], fallback_label: str) -> dict[str, Any]:
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
        }
    images = _normalize_ordered_images(record)
    product_type = _safe_text(record.get("product_type") or record.get("subcategory"))
    category = _safe_text(record.get("category"))
    guide_match = _build_image_guide_match(category, product_type)
    return {
        "label": _safe_text(record.get("brand")) or fallback_label,
        "brand": _safe_text(record.get("brand")),
        "category": category,
        "product_type": product_type,
        "ordered_images": images,
        "image_count": int(record.get("image_count", len(images)) or len(images)),
        "image_guide_match": guide_match,
        "bullets": _build_slide4_bullets(images, guide_match),
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
    client_column = _build_slide4_column(client_record, client_fallback)
    client_column["label"] = client_company_name or client_name or client_column["label"]

    competitor_records = list(competitor_records or [])
    competitor_columns = [
        _build_slide4_column(
            competitor_records[index] if index < len(competitor_records) else {},
            f"Competitor {index + 1}",
        )
        for index in range(2)
    ]
    return {"columns": [client_column, *competitor_columns]}


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



def _count_recommendations(entry: dict[str, Any] | None) -> int:
    """Count total recommendations from an entry (images, description, key features)."""
    if not entry:
        return 0
    count = 0
    count += len(entry.get("image_recommendations", []) or [])
    count += len(entry.get("description_recommendations", []) or [])
    count += len(entry.get("key_features_recommendations", []) or [])
    return count


def _calculate_consumer_demand_label(product_pairs: list[dict[str, Any]]) -> str:
    """Determine consumer demand label based on ratings/reviews health."""
    if not product_pairs:
        return "Significant"
    
    total_ratings = 0
    healthy_count = 0
    
    for pair in product_pairs:
        pdp_slide = pair.get("pdp_slide", {}) or {}
        reviews = pdp_slide.get("reviews_summary", {}) or {}
        
        rating = reviews.get("average_rating")
        ratings_count = reviews.get("ratings_count")
        
        if isinstance(rating, (int, float)) and isinstance(ratings_count, int):
            total_ratings += 1
            # Consider "healthy" if rating >= 4.0 and has decent review volume
            if rating >= 4.0 and ratings_count >= 50:
                healthy_count += 1
    
    if total_ratings == 0:
        return "Significant"
    
    health_ratio = healthy_count / total_ratings
    if health_ratio >= 0.7:
        return "Strong"
    return "Significant"


def _calculate_walmart_opportunity_label(product_pairs: list[dict[str, Any]]) -> str:
    """Determine Walmart opportunity label based on content issue density."""
    if not product_pairs:
        return "Evolving"
    
    total_pairs = len(product_pairs)
    issues_per_pair = []
    
    for pair in product_pairs:
        content_slide = pair.get("content_optimization_slide", {}) or {}
        rec_count = _count_recommendations(content_slide)
        issues_per_pair.append(rec_count)
    
    if not issues_per_pair:
        return "Evolving"
    
    avg_issues = sum(issues_per_pair) / len(issues_per_pair)
    
    # If average issues per product is high (>2), significant opportunity
    if avg_issues > 2:
        return "Significant"
    # If any product has substantial issues, still meaningful opportunity
    if max(issues_per_pair) >= 2:
        return "Evolving"
    
    return "Evolving"


def _get_consumer_demand_bullets() -> str:
    """Generate consumer demand bullet points."""
    bullets = [
        "Strong shopper trust signals through ratings and reviews",
        "Positive consumer response across audited PDPs",
        "Established product presence within the Walmart category",
    ]
    return "\n".join(bullets)


def _get_walmart_opportunity_bullets(product_pairs: list[dict[str, Any]]) -> str:
    """Generate Walmart opportunity bullet points based on identified issues."""
    bullets = []
    
    # Check what types of issues exist
    has_title_issues = False
    has_image_issues = False
    has_description_issues = False
    has_key_features_issues = False
    
    for pair in product_pairs:
        content_slide = pair.get("content_optimization_slide", {}) or {}
        if content_slide.get("image_recommendations"):
            has_image_issues = True
        if content_slide.get("description_recommendations"):
            has_description_issues = True
        if content_slide.get("key_features_recommendations"):
            has_key_features_issues = True
        if content_slide.get("recommended_title"):
            has_title_issues = True
    
    # Build bullets based on detected issues
    if has_title_issues:
        bullets.append("Opportunity to strengthen product title structure")
    if has_description_issues:
        bullets.append("PDP content can be better aligned to Walmart style guide expectations")
    if has_image_issues:
        bullets.append("Image stack improvements can strengthen shopper education")
    
    # Fallback bullets if no issues detected
    if not bullets:
        bullets = [
            "Opportunity to improve Walmart-native PDP SEO",
            "Enhanced content and imagery optimization potential",
            "PDP content can be better aligned to Walmart style guide expectations",
        ]
    
    # Ensure we have exactly 3 bullets
    if len(bullets) > 3:
        bullets = bullets[:3]
    elif len(bullets) < 3:
        fallback = [
            "Opportunity to improve Walmart-native PDP SEO",
            "Enhanced content and imagery optimization potential",
            "Opportunity to strengthen product title structure",
            "PDP content can be better aligned to Walmart style guide expectations",
            "Image stack improvements can strengthen shopper education",
        ]
        for fb in fallback:
            if fb not in bullets and len(bullets) < 3:
                bullets.append(fb)
    
    return "\n".join(bullets[:3])


def _get_competitive_benchmark_bullets() -> str:
    """Generate competitive benchmark bullet points (generic for now)."""
    bullets = [
        "Competitors are building stronger full-funnel shopping journeys",
        "Competitive PDPs use stronger educational merchandising",
        "Category leaders are using more structured product storytelling",
    ]
    return "\n".join(bullets)


def build_slide2_summary_payload(export_plan: dict) -> dict[str, str]:
    """
    Build Slide 2 content placeholders from audit export plan.
    
    Returns a dictionary with slide2 placeholder keys and their values.
    """
    metadata = export_plan.get("audit_metadata", {}) or {}
    product_pairs = export_plan.get("product_slide_pairs", []) or []
    
    client_company_name = _safe_text(metadata.get("client_company_name", ""))
    client_name = _safe_text(metadata.get("client_name", ""))
    retailer = _safe_text(metadata.get("retailer", ""))
    
    # Fallback if client_company_name is empty
    company = client_company_name or client_name
    
    # Generate intro copy
    intro_copy = ""
    if company and retailer:
        intro_copy = (
            f"{company} has built a strong foundation on {retailer}, "
            "with clear opportunities to strengthen discoverability, PDP content quality, "
            "and competitive shelf ownership."
        )
    elif company:
        intro_copy = (
            f"{company} has strong product presence with clear opportunities to enhance "
            "PDP content quality and discoverability across key retail channels."
        )
    else:
        intro_copy = (
            "Your brand has built a solid foundation with clear opportunities to strengthen "
            "discoverability, PDP content quality, and competitive shelf ownership."
        )
    
    # Calculate labels
    consumer_demand_label = _calculate_consumer_demand_label(product_pairs)
    walmart_opportunity_label = _calculate_walmart_opportunity_label(product_pairs)
    competitive_benchmark_label = "Evolving"  # Safe default for now
    
    # Generate bullets
    consumer_demand_bullets = _get_consumer_demand_bullets()
    walmart_opportunity_bullets = _get_walmart_opportunity_bullets(product_pairs)
    competitive_benchmark_bullets = _get_competitive_benchmark_bullets()
    
    return {
        "{{slide2_intro_copy}}": intro_copy,
        "{{slide2_consumer_demand_label}}": consumer_demand_label,
        "{{slide2_consumer_demand_bullets}}": consumer_demand_bullets,
        "{{slide2_walmart_opportunity_label}}": walmart_opportunity_label,
        "{{slide2_walmart_opportunity_bullets}}": walmart_opportunity_bullets,
        "{{slide2_competitive_benchmark_label}}": competitive_benchmark_label,
        "{{slide2_competitive_benchmark_bullets}}": competitive_benchmark_bullets,
    }


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
    
    # Existing template strings are the replacement targets. Do not require manual placeholders.
    template_replacements = {
        "The Honest Company": client_company_name,
        "May 27, 2026": audit_date,
        "Honest": client_name,
    }
    replace_existing_template_text(prs, template_replacements)

    slide4 = _find_slide_by_title(prs, "PDP Content Benchmarking")
    if slide4 is None:
        print("[audit_powerpoint_new] Slide 4 'PDP Content Benchmarking' was not found.")
    else:
        slide4_payload = build_slide4_pdp_benchmark_payload(
            export_plan,
            competitor_records=competitor_records,
        )
        _apply_slide4_content(prs, slide4, slide4_payload)

    clear_unresolved_placeholder_text(prs)
    
    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
