from __future__ import annotations

import copy
import io
import os
from urllib.request import Request, urlopen
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches, Pt


def resolve_audit_template_path() -> str:
    candidates = [
        os.path.join("templates", "audit_template.pptx"),
        os.path.join("templates", "Audit Template.pptx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    templates_dir = "templates"
    if os.path.isdir(templates_dir):
        for name in os.listdir(templates_dir):
            if name.lower().endswith(".pptx"):
                return os.path.join(templates_dir, name)
    raise FileNotFoundError("No .pptx template found in templates/.")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _shape_text(shape: Any) -> str:
    if not getattr(shape, "has_text_frame", False):
        return ""
    return _safe_text(getattr(shape, "text", ""))


def _find_slide_by_text(prs: Presentation, token: str) -> Any:
    token_l = token.lower()
    for slide in prs.slides:
        for shape in slide.shapes:
            text = _shape_text(shape).lower()
            if token_l in text:
                return slide
    return None


def _find_shape_contains(slide: Any, token: str) -> Any:
    token_l = token.lower()
    for shape in slide.shapes:
        text = _shape_text(shape).lower()
        if token_l in text:
            return shape
    return None


def _set_shape_text(shape: Any, text: str, font_size: int | None = None) -> None:
    if not shape or not getattr(shape, "has_text_frame", False):
        return
    shape.text_frame.clear()
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.text = _safe_text(text)
    if font_size is not None:
        for run in paragraph.runs:
            run.font.size = Pt(font_size)


def _set_bullet_block(shape: Any, heading: str, bullets: list[str]) -> None:
    if not shape or not getattr(shape, "has_text_frame", False):
        return
    lines = [heading]
    lines.extend([f"- {_safe_text(b)}" for b in bullets if _safe_text(b)])
    _set_shape_text(shape, "\n".join(lines))


def _download_image_bytes(url: str) -> bytes | None:
    clean = _safe_text(url)
    if not clean:
        return None
    if clean.startswith("data:image"):
        try:
            header, payload = clean.split(",", 1)
            if ";base64" in header:
                import base64

                return base64.b64decode(payload)
        except Exception:
            return None

    try:
        req = Request(clean, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=12) as resp:
            return resp.read()
    except Exception:
        return None


def _insert_image_over_shape(slide: Any, target_shape: Any, image_url: str) -> bool:
    image_bytes = _download_image_bytes(image_url)
    if not image_bytes:
        return False
    try:
        left = target_shape.left
        top = target_shape.top
        width = target_shape.width
        height = target_shape.height
        slide.shapes.add_picture(io.BytesIO(image_bytes), left, top, width=width, height=height)
        return True
    except Exception:
        return False


def _largest_autoshape(slide: Any) -> Any:
    autoshapes = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    if not autoshapes:
        return None
    return max(autoshapes, key=lambda s: int(s.width) * int(s.height))


def _duplicate_slide(prs: Presentation, source_slide: Any) -> Any:
    blank_layout = prs.slide_layouts[6]
    new_slide = prs.slides.add_slide(blank_layout)

    for shape in source_slide.shapes:
        el = copy.deepcopy(shape.element)
        new_slide.shapes._spTree.insert_element_before(el, "p:extLst")

    for rel in source_slide.part.rels.values():
        if "notesSlide" in rel.reltype:
            continue
        try:
            new_slide.part.rels.add_relationship(rel.reltype, rel._target, rel.rId)
        except Exception:
            continue

    return new_slide


def _populate_pdp_slide(slide: Any, pair_payload: dict[str, Any]) -> None:
    pdp = pair_payload.get("pdp_slide", {}) or {}
    content = pair_payload.get("content_optimization_slide", {}) or {}

    title = _safe_text(pdp.get("product_title"))
    item_id = _safe_text(pdp.get("item_id"))
    image_recs = list(content.get("image_recommendations", []) or [])
    selected_img = (pdp.get("selected_primary_image", {}) or {}).get("url", "")

    title_shape = _find_shape_contains(slide, "(Product Title)")
    if title_shape:
        _set_shape_text(title_shape, f"{title}\n{item_id}" if item_id else title)

    rec_shape = _find_shape_contains(slide, "Image Recommendations")
    if rec_shape:
        _set_bullet_block(rec_shape, "Image Recommendations", image_recs)

    image_box = _largest_autoshape(slide)
    if image_box and _safe_text(selected_img):
        _insert_image_over_shape(slide, image_box, selected_img)


def _populate_content_slide(slide: Any, pair_payload: dict[str, Any]) -> None:
    content = pair_payload.get("content_optimization_slide", {}) or {}
    title = _safe_text(content.get("product_title"))
    item_id = _safe_text(content.get("item_id"))

    title_shape = _find_shape_contains(slide, "Title Recommendations")
    if title_shape:
        rec_title = _safe_text(content.get("recommended_title"))
        heading = "Title Recommendations:"
        value = rec_title or title
        if item_id:
            value = f"{value}\nItem ID: {item_id}"
        _set_shape_text(title_shape, f"{heading}\n{value}")

    desc_shape = _find_shape_contains(slide, "Description Recommendations")
    if desc_shape:
        _set_bullet_block(
            desc_shape,
            "Description Recommendations",
            list(content.get("description_recommendations", []) or []),
        )

    feat_shape = _find_shape_contains(slide, "Key Features Recommendations")
    if feat_shape:
        bullets = list(content.get("key_features_recommendations", []) or [])
        bullets.extend(list(content.get("top_priority_fixes", []) or []))
        _set_bullet_block(feat_shape, "Key Features Recommendations", bullets[:8])


def _sorted_competitor_slots(slide: Any) -> list[Any]:
    slots = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    slots.sort(key=lambda s: (int(s.top), int(s.left)))
    return slots[:10]


def _populate_competitor_graphics(slide: Any, ordered_assignments: list[dict[str, Any]], notes: str) -> None:
    slots = _sorted_competitor_slots(slide)
    by_order = {int(a.get("display_order", 0)): a for a in ordered_assignments if 1 <= int(a.get("display_order", 0)) <= 10}

    for order in range(1, 11):
        assignment = by_order.get(order)
        if not assignment:
            continue
        slot_idx = order - 1
        if slot_idx >= len(slots):
            continue
        _insert_image_over_shape(slide, slots[slot_idx], _safe_text(assignment.get("url")))

    notes_shape = _find_shape_contains(slide, "(image placeholder)")
    if notes_shape:
        _set_shape_text(notes_shape, _safe_text(notes) or "")


def _populate_shared_note_slide(slide: Any, heading: str, body_text: str) -> None:
    body = _safe_text(body_text)
    target = None
    for shape in slide.shapes:
        text = _shape_text(shape)
        if text and heading.lower() not in text.lower() and "(retailer)" not in text.lower():
            target = shape
            break

    if target and getattr(target, "has_text_frame", False):
        _set_shape_text(target, body)
        return

    textbox = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.5))
    _set_shape_text(textbox, body)


def _populate_retailer_tokens(prs: Presentation, retailer: str) -> None:
    value = _safe_text(retailer)
    if not value:
        return
    for slide in prs.slides:
        for shape in slide.shapes:
            text = _shape_text(shape)
            if "(Retailer)" in text:
                _set_shape_text(shape, text.replace("(Retailer)", value))


def _populate_cover_date(prs: Presentation, audit_date: str) -> None:
    if not _safe_text(audit_date) or len(prs.slides) < 1:
        return
    cover = prs.slides[0]
    for shape in cover.shapes:
        text = _shape_text(shape)
        if text and any(ch.isdigit() for ch in text) and len(text) <= 30:
            _set_shape_text(shape, audit_date)
            break


def generate_audit_powerpoint_from_template(*, export_plan: dict[str, Any], template_path: str) -> bytes:
    prs = Presentation(template_path)

    pair_payloads = list(export_plan.get("product_slide_pairs", []) or [])
    if not pair_payloads:
        raise ValueError("No included primary product entries were found for export.")

    pdp_template = _find_slide_by_text(prs, "Image Recommendations")
    content_template = _find_slide_by_text(prs, "Content Optimizations")
    competitor_slide = _find_slide_by_text(prs, "Competitor Graphics")
    retail_media_slide = _find_slide_by_text(prs, "Retail Media Optimizations")
    competitor_ad_slide = _find_slide_by_text(prs, "Competitor Ad Graphics")

    if pdp_template is None or content_template is None:
        raise ValueError("Could not find required primary product template slides.")

    _populate_pdp_slide(pdp_template, pair_payloads[0])
    _populate_content_slide(content_template, pair_payloads[0])

    for pair in pair_payloads[1:]:
        pdp_slide = _duplicate_slide(prs, pdp_template)
        content_slide = _duplicate_slide(prs, content_template)
        _populate_pdp_slide(pdp_slide, pair)
        _populate_content_slide(content_slide, pair)

    competitor_payload = export_plan.get("competitor_graphics_payload", {}) or {}
    shared_sections = export_plan.get("shared_sections_payload", {}) or {}
    if competitor_slide is not None:
        _populate_competitor_graphics(
            competitor_slide,
            list(competitor_payload.get("ordered_assignments", []) or []),
            _safe_text(shared_sections.get("competitor_graphics_notes", "")),
        )

    if retail_media_slide is not None:
        _populate_shared_note_slide(
            retail_media_slide,
            "Retail Media Optimizations",
            _safe_text(shared_sections.get("retail_media_optimizations", "")),
        )

    if competitor_ad_slide is not None:
        _populate_shared_note_slide(
            competitor_ad_slide,
            "Competitor Ad Graphics",
            _safe_text(shared_sections.get("competitor_ad_graphics_notes", "")),
        )

    metadata = export_plan.get("audit_metadata", {}) or {}
    _populate_retailer_tokens(prs, _safe_text(metadata.get("retailer", "")))
    _populate_cover_date(prs, _safe_text(metadata.get("audit_date", "")))

    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
