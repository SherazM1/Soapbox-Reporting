from __future__ import annotations

import io
from typing import Any
from urllib.request import urlopen

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

from audit_powerpoint import _format_cover_date


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _replace_runs_preserve_style(paragraph: Any, replacements: dict[str, str]) -> None:
    """Replace placeholders while keeping existing run formatting when possible."""
    if not getattr(paragraph, "runs", None):
        text = paragraph.text
        replaced = _replace_text(text, replacements)
        if replaced != text:
            paragraph.text = replaced
        return

    replaced_any_run = False
    for run in paragraph.runs:
        original = run.text
        replaced = _replace_text(original, replacements)
        if replaced != original:
            run.text = replaced
            replaced_any_run = True

    paragraph_text = "".join(run.text for run in paragraph.runs)
    if replaced_any_run and not _contains_placeholder(paragraph_text, replacements):
        return

    # PowerPoint can split a placeholder across several runs. In that case,
    # replace at paragraph level and keep the first run's style for the result.
    replaced_text = _replace_text(paragraph_text, replacements)
    if replaced_text == paragraph_text:
        return

    paragraph.runs[0].text = replaced_text
    for run in paragraph.runs[1:]:
        run.text = ""


def _replace_text(text: str, replacements: dict[str, str]) -> str:
    result = text
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


def _contains_placeholder(text: str, replacements: dict[str, str]) -> bool:
    return any(token in text for token in replacements)


def replace_text_frame_placeholders(text_frame: Any, replacements: dict[str, str]) -> None:
    """Safely replace placeholders inside a text frame without clearing it."""
    if not text_frame:
        return
    for paragraph in text_frame.paragraphs:
        _replace_runs_preserve_style(paragraph, replacements)


def _iter_text_frames_from_shapes(shapes: Any):
    for shape in shapes:
        if getattr(shape, "has_text_frame", False):
            yield shape.text_frame
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    yield cell.text_frame
        if hasattr(shape, "shapes"):
            yield from _iter_text_frames_from_shapes(shape.shapes)


def _find_slide_by_title(prs: Any, title_text: str) -> int | None:
    """
    Find a slide by searching for the title text in its shapes.
    Returns the slide index (0-based) if found, otherwise None.
    """
    for idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                if title_text in shape.text_frame.text:
                    return idx
    return None


def _remove_slide_by_title(prs: Any, title_text: str) -> bool:
    """
    Remove a slide by its title text.
    Returns True if a slide was removed, False otherwise.
    """
    slide_idx = _find_slide_by_title(prs, title_text)
    if slide_idx is not None:
        # Remove the slide via the slide layout's id
        rId = prs.slides._sldIdLst[slide_idx].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[slide_idx]
        return True
    return False


def _get_image_urls_from_record(record: dict[str, Any]) -> list[str]:
    """Extract up to 12 image URLs from a cached record."""
    images = record.get("images", []) or []
    urls = []
    for img in images:
        url = img.get("url", "").strip()
        if url and len(urls) < 12:
            urls.append(url)
    return urls


def _get_competitor_brand_images(
    competitor_payload: dict[str, Any], competitor_records: list[dict[str, Any]]
) -> tuple[str, list[str], str, list[str]]:
    """
    Extract brand names and image lists for up to 2 competitors.
    Returns: (brand1, images1, brand2, images2)
    """
    # Build a map of record_id to record for quick lookup
    record_map = {r.get("record_id", ""): r for r in competitor_records if r.get("record_id")}
    
    # Group images by record_id to identify unique competitors
    competitor_groups: dict[str, list[str]] = {}
    
    ordered_assignments = competitor_payload.get("ordered_assignments", []) or []
    for assignment in ordered_assignments:
        record_id = assignment.get("record_id", "")
        url = assignment.get("url", "").strip()
        if url and len(competitor_groups.get(record_id, [])) < 12:
            if record_id not in competitor_groups:
                competitor_groups[record_id] = []
            competitor_groups[record_id].append(url)
    
    # Extract brand names and images for first two competitors
    brand1, images1 = "", []
    brand2, images2 = "", []
    
    for idx, record_id in enumerate(competitor_groups.keys()):
        if idx >= 2:
            break
        record = record_map.get(record_id, {})
        brand = _safe_text(record.get("brand", ""))
        images = competitor_groups[record_id]
        
        if idx == 0:
            brand1 = brand or "Competitor 1"
            images1 = images
        else:
            brand2 = brand or "Competitor 2"
            images2 = images
    
    # If only one competitor, ensure brand2 is fallback label
    if brand1 and not brand2:
        brand2 = "Competitor 2"
    
    return brand1, images1, brand2, images2


def _calculate_grid_positions(container_left: float, container_top: float, container_width: float, container_height: float, cols: int = 3, rows: int = 4, gutter: float = 0.15) -> list[tuple[float, float, float, float]]:
    """
    Calculate positions for a grid of images inside a container.
    Returns list of (left, top, width, height) for each grid cell in column-major order.
    gutter is in inches.
    """
    # Account for gutters in available space
    available_width = container_width - (gutter * (cols - 1))
    available_height = container_height - (gutter * (rows - 1))
    
    cell_width = available_width / cols
    cell_height = available_height / rows
    
    positions = []
    for row in range(rows):
        for col in range(cols):
            left = container_left + (col * (cell_width + gutter))
            top = container_top + (row * (cell_height + gutter))
            positions.append((left, top, cell_width, cell_height))
    
    return positions


def _load_image_from_url(url: str) -> bytes | None:
    """Download image from URL and return bytes. Returns None on failure."""
    try:
        with urlopen(url, timeout=5) as response:
            return response.read()
    except Exception:
        return None


def _find_or_create_picture(slide: Any, left: float, top: float, width: float, height: float, image_bytes: bytes) -> None:
    """Add a picture to a slide at the specified position."""
    try:
        import io as io_module
        pic = slide.shapes.add_picture(
            io_module.BytesIO(image_bytes),
            left=Inches(left),
            top=Inches(top),
            width=Inches(width),
            height=Inches(height),
        )
    except Exception:
        pass


def _populate_slide4_image_grid(slide: Any, positions: list[tuple[float, float, float, float]], image_urls: list[str]) -> None:
    """
    Populate image grid on slide with images from URLs.
    Respects the provided positions list (up to 12 grid cells).
    """
    for grid_idx, (left, top, width, height) in enumerate(positions[:12]):
        if grid_idx >= len(image_urls):
            break
        
        url = image_urls[grid_idx]
        if not url:
            continue
        
        image_bytes = _load_image_from_url(url)
        if image_bytes:
            _find_or_create_picture(slide, left, top, width, height, image_bytes)


def _find_slide_by_title_text(prs: Any, title_text: str) -> Any:
    """Find and return a slide object by its title text."""
    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                if title_text in shape.text_frame.text:
                    return slide
    return None


def _find_shape_by_name(slide: Any, shape_name: str) -> Any:
    """Find a shape by its name."""
    for shape in slide.shapes:
        if hasattr(shape, "name") and shape_name in shape.name:
            return shape
    return None


def build_slide4_pdp_benchmark_payload(export_plan: dict, competitor_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Build Slide 4 content: labels and image lists for client and competitors.
    
    Returns a dictionary with:
    - slide4_client_label: Client/company name
    - slide4_competitor_1_label: First competitor brand or fallback
    - slide4_competitor_2_label: Second competitor brand or fallback
    - slide4_client_images: List of image URLs (up to 12)
    - slide4_competitor_1_images: List of image URLs (up to 12)
    - slide4_competitor_2_images: List of image URLs (up to 12)
    """
    metadata = export_plan.get("audit_metadata", {}) or {}
    product_pairs = export_plan.get("product_slide_pairs", []) or []
    competitor_payload = export_plan.get("competitor_graphics_payload", {}) or {}
    
    client_company_name = _safe_text(metadata.get("client_company_name", ""))
    client_name = _safe_text(metadata.get("client_name", ""))
    company = client_company_name or client_name
    
    # Extract client images from first product
    client_images = []
    if product_pairs:
        first_pair = product_pairs[0]
        pdp_slide = first_pair.get("pdp_slide", {}) or {}
        record_id = pdp_slide.get("record_id", "")
        
        # Try to get images from selected_primary_images first
        primary_images = pdp_slide.get("selected_primary_images", []) or []
        if primary_images:
            client_images = [img.get("url", "") for img in primary_images if img.get("url")]
    
    # Extract competitor images and brands
    competitor_records = competitor_records or []
    brand1, images1, brand2, images2 = _get_competitor_brand_images(competitor_payload, competitor_records)
    
    # Ensure fallback labels
    label1 = brand1 or "Competitor 1"
    label2 = brand2 or "Competitor 2"
    
    return {
        "slide4_client_label": company,
        "slide4_competitor_1_label": label1,
        "slide4_competitor_2_label": label2,
        "slide4_client_images": client_images,
        "slide4_competitor_1_images": images1,
        "slide4_competitor_2_images": images2,
    }


def _apply_slide4_placeholders(slide: Any, payload: dict[str, Any]) -> None:
    """Replace text placeholders on Slide 4 with label values."""
    replacements = {
        "{{slide4_client_label}}": payload.get("slide4_client_label", ""),
        "{{slide4_competitor_1_label}}": payload.get("slide4_competitor_1_label", ""),
        "{{slide4_competitor_2_label}}": payload.get("slide4_competitor_2_label", ""),
    }
    
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            for paragraph in shape.text_frame.paragraphs:
                _replace_runs_preserve_style(paragraph, replacements)


def _apply_slide4_images(slide: Any, payload: dict[str, Any]) -> None:
    """
    Populate Slide 4 image grids.
    
    Attempts to find named shapes for image grids:
    - slide4_client_image_grid
    - slide4_competitor_1_image_grid
    - slide4_competitor_2_image_grid
    
    Falls back to hardcoded positions if shapes not found.
    """
    # Try to find named shapes first
    client_shape = _find_shape_by_name(slide, "slide4_client_image_grid")
    comp1_shape = _find_shape_by_name(slide, "slide4_competitor_1_image_grid")
    comp2_shape = _find_shape_by_name(slide, "slide4_competitor_2_image_grid")
    
    # Fallback positions if named shapes not found (3 columns, 4 rows each grid)
    # These are estimated based on typical Slide 4 layout: 3 grids side-by-side
    fallback_positions = {
        "client": _calculate_grid_positions(0.5, 1.5, 2.0, 3.0, cols=3, rows=4),
        "competitor1": _calculate_grid_positions(2.7, 1.5, 2.0, 3.0, cols=3, rows=4),
        "competitor2": _calculate_grid_positions(4.9, 1.5, 2.0, 3.0, cols=3, rows=4),
    }
    
    # Populate client images
    if client_shape and getattr(client_shape, "left", None) is not None:
        positions = _calculate_grid_positions(
            client_shape.left.inches,
            client_shape.top.inches,
            client_shape.width.inches,
            client_shape.height.inches,
        )
    else:
        positions = fallback_positions["client"]
    
    client_images = payload.get("slide4_client_images", []) or []
    _populate_slide4_image_grid(slide, positions, client_images)
    
    # Populate competitor 1 images
    if comp1_shape and getattr(comp1_shape, "left", None) is not None:
        positions = _calculate_grid_positions(
            comp1_shape.left.inches,
            comp1_shape.top.inches,
            comp1_shape.width.inches,
            comp1_shape.height.inches,
        )
    else:
        positions = fallback_positions["competitor1"]
    
    comp1_images = payload.get("slide4_competitor_1_images", []) or []
    _populate_slide4_image_grid(slide, positions, comp1_images)
    
    # Populate competitor 2 images
    if comp2_shape and getattr(comp2_shape, "left", None) is not None:
        positions = _calculate_grid_positions(
            comp2_shape.left.inches,
            comp2_shape.top.inches,
            comp2_shape.width.inches,
            comp2_shape.height.inches,
        )
    else:
        positions = fallback_positions["competitor2"]
    
    comp2_images = payload.get("slide4_competitor_2_images", []) or []
    _populate_slide4_image_grid(slide, positions, comp2_images)



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
    retailer = _safe_text(metadata.get("retailer", ""))
    
    # Handle Slide 9 removal if not included
    if not include_slide_9:
        _remove_slide_by_title(prs, "Walmart Cash Program Visibility")
    
    # Global replacements
    replacements = {
        "{{client_name}}": client_name,
        "{{client_company_name}}": client_company_name,
        "{{audit_date}}": audit_date,
        "{{retailer}}": retailer,
    }
    
    # Add Slide 2 placeholders
    slide2_payload = build_slide2_summary_payload(export_plan)
    replacements.update(slide2_payload)

    # Process all slides for placeholder replacement
    for slide in prs.slides:
        for text_frame in _iter_text_frames_from_shapes(slide.shapes):
            replace_text_frame_placeholders(text_frame, replacements)
    
    # Populate Slide 4 with images and labels
    slide4 = _find_slide_by_title_text(prs, "PDP Content Benchmarking")
    if slide4:
        slide4_payload = build_slide4_pdp_benchmark_payload(
            export_plan, competitor_records=competitor_records or []
        )
        _apply_slide4_placeholders(slide4, slide4_payload)
        _apply_slide4_images(slide4, slide4_payload)

    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
