from __future__ import annotations

import io
from typing import Any

from pptx import Presentation

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


def generate_new_audit_powerpoint_from_template(*, export_plan: dict, template_path: str) -> bytes:
    prs = Presentation(template_path)
    metadata = export_plan.get("audit_metadata", {}) or {}
    client_name = _safe_text(metadata.get("client_name", ""))
    client_company_name = _safe_text(metadata.get("client_company_name", "")) or client_name
    audit_date = _format_cover_date(_safe_text(metadata.get("audit_date", "")))
    retailer = _safe_text(metadata.get("retailer", ""))
    
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

    for slide in prs.slides:
        for text_frame in _iter_text_frames_from_shapes(slide.shapes):
            replace_text_frame_placeholders(text_frame, replacements)

    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
