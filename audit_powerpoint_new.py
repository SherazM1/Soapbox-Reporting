from __future__ import annotations

import io
from typing import Any
from urllib.request import urlopen

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE

from audit_powerpoint import _format_cover_date


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


def _replace_placeholder_in_shape(shape: Any, placeholder: str, replacement: str) -> bool:
    """
    Replace a placeholder string in a shape with a replacement value.
    Only replaces if the placeholder actually exists in the shape.
    Returns True if replacement was made.
    """
    if not shape or not getattr(shape, "has_text_frame", False):
        return False
    
    tf = shape.text_frame
    for paragraph in tf.paragraphs:
        if placeholder in paragraph.text:
            new_text = paragraph.text.replace(placeholder, replacement)
            _replace_paragraph_text_preserve_style(paragraph, new_text)
            return True
    
    return False


def _find_slide_by_title(prs: Any, title_text: str) -> Any:
    """Find a slide by searching for title text in any shape."""
    title_l = title_text.lower()
    for slide in prs.slides:
        for shape in slide.shapes:
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
        for shape in slide.shapes:
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


def _apply_all_replacements_to_slide(slide: Any, replacements: dict[str, str]) -> None:
    """Apply all placeholder replacements to all shapes in a slide."""
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            for placeholder, replacement in replacements.items():
                _replace_placeholder_in_shape(shape, placeholder, replacement)
        # Also handle text in tables
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    for placeholder, replacement in replacements.items():
                        for paragraph in cell.text_frame.paragraphs:
                            if placeholder in paragraph.text:
                                new_text = paragraph.text.replace(placeholder, replacement)
                                _replace_paragraph_text_preserve_style(paragraph, new_text)


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





def _clear_pictures_from_slide(slide: Any) -> None:
    """Remove all picture shapes from a slide."""
    shapes_to_remove = []
    for shape in slide.shapes:
        if hasattr(shape, "image") or "picture" in str(shape.shape_type).lower():
            shapes_to_remove.append(shape)
    
    for shape in shapes_to_remove:
        sp = shape.element
        sp.getparent().remove(sp)


def _populate_slide4_images(slide: Any, payload: dict[str, Any]) -> None:
    """
    Populate Slide 4 images by placing them in a simple 3x4 grid layout.
    Uses hard-coded positions for the three grid areas.
    """
    # Define approximate grid areas for the three competitors
    # Format: (left_inches, top_inches, width_inches, height_inches)
    grid_areas = {
        "client": (0.5, 2.0, 2.2, 3.0),
        "competitor1": (2.9, 2.0, 2.2, 3.0),
        "competitor2": (5.3, 2.0, 2.2, 3.0),
    }
    
    # Get image lists
    client_images = payload.get("slide4_client_images", []) or []
    comp1_images = payload.get("slide4_competitor_1_images", []) or []
    comp2_images = payload.get("slide4_competitor_2_images", []) or []
    
    all_image_lists = [
        ("client", client_images),
        ("competitor1", comp1_images),
        ("competitor2", comp2_images),
    ]
    
    # For each grid area, place images
    for grid_name, image_urls in all_image_lists:
        grid_left, grid_top, grid_width, grid_height = grid_areas[grid_name]
        
        # Calculate 3x4 grid positions
        cell_width = grid_width / 3
        cell_height = grid_height / 4
        
        image_idx = 0
        for row in range(4):
            for col in range(3):
                if image_idx >= len(image_urls):
                    break
                
                url = image_urls[image_idx]
                if not url:
                    image_idx += 1
                    continue
                
                left = grid_left + (col * cell_width) + 0.05
                top = grid_top + (row * cell_height) + 0.05
                width = cell_width - 0.1
                height = cell_height - 0.1
                
                image_bytes = _load_image_from_url(url)
                if image_bytes:
                    _find_or_create_picture(slide, left, top, width, height, image_bytes)
                
                image_idx += 1
            
            if image_idx >= len(image_urls):
                break



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
    """Replace Slide 4 label placeholders in existing shapes."""
    replacements = {
        "{{slide4_client_label}}": payload.get("slide4_client_label", ""),
        "{{slide4_competitor_1_label}}": payload.get("slide4_competitor_1_label", ""),
        "{{slide4_competitor_2_label}}": payload.get("slide4_competitor_2_label", ""),
    }
    
    _apply_all_replacements_to_slide(slide, replacements)


def _apply_slide4_images(slide: Any, payload: dict[str, Any]) -> None:
    """Populate Slide 4 image grids."""
    # Clear existing pictures and add new ones
    _clear_pictures_from_slide(slide)
    _populate_slide4_images(slide, payload)



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
    global_replacements = {
        "{{client_name}}": client_name,
        "{{client_company_name}}": client_company_name,
        "{{audit_date}}": audit_date,
        "{{retailer}}": retailer,
    }
    
    # Add Slide 2 placeholders
    slide2_payload = build_slide2_summary_payload(export_plan)
    global_replacements.update(slide2_payload)
    
    # Process all slides for placeholder replacement (text only)
    for slide in prs.slides:
        _apply_all_replacements_to_slide(slide, global_replacements)
    
    # Populate Slide 4 with labels (text replacement)
    slide4 = _find_slide_by_title(prs, "PDP Content Benchmarking")
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
