from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned


def title_length(text: str) -> int:
    return len((text or "").strip())


def combine_description(description_body: str, description_bullets: list[str]) -> str:
    body = (description_body or "").strip()
    bullets = [b.strip() for b in (description_bullets or []) if b and b.strip()]
    if not body and not bullets:
        return ""
    if not bullets:
        return body
    return "\n".join([body] + [f"- {b}" for b in bullets]).strip()


def make_image(index: int, url: str, is_hero: bool = False) -> dict[str, Any]:
    return {"index": int(index), "url": url, "is_hero": bool(is_hero)}


def make_key_feature(index: int, text: str) -> dict[str, Any]:
    return {"index": int(index), "text": (text or "").strip()}


def make_reviews_summary(
    average_rating: float | None = None,
    ratings_count: int | None = None,
    review_count: int | None = None,
    top_review_snippets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "average_rating": average_rating,
        "ratings_count": ratings_count,
        "review_count": review_count,
        "top_review_snippets": list(top_review_snippets or []),
    }


def create_cached_pdp_record(
    *,
    client_name: str = "",
    retailer: str = "",
    source_url: str = "",
    source_type: str = "primary",
    item_id: str = "",
    brand: str = "",
    product_title: str = "",
    category: str = "",
    subcategory: str = "",
    current_title: str = "",
    current_description_body: str = "",
    current_description_bullets: list[str] | None = None,
    description_section_labels: list[str] | None = None,
    current_key_features: list[dict[str, Any]] | None = None,
    key_features_section_label: str = "",
    images: list[dict[str, Any]] | None = None,
    reviews_summary: dict[str, Any] | None = None,
    extraction_status: str = "mock_extracted",
    extraction_errors: list[str] | None = None,
    fetched_at: str | None = None,
    last_used_at: str | None = None,
    user_overrides: dict[str, Any] | None = None,
    ingest_metadata: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    current_description_bullets = list(current_description_bullets or [])
    combined = combine_description(current_description_body, current_description_bullets)
    images = list(images or [])
    current_title = current_title or product_title
    normalized = normalize_title(current_title)
    return {
        "record_id": record_id or f"pdp-{uuid4().hex[:12]}",
        "client_name": client_name,
        "retailer": retailer,
        "source_url": source_url,
        "source_type": source_type,
        "item_id": item_id,
        "brand": brand,
        "product_title": product_title,
        "category": category,
        "subcategory": subcategory,
        "current_title": current_title,
        "normalized_title": normalized,
        "title_length": title_length(current_title),
        "current_description_body": current_description_body,
        "current_description_bullets": current_description_bullets,
        "current_description_combined": combined,
        "description_section_labels": list(description_section_labels or []),
        "current_key_features": list(current_key_features or []),
        "key_features_section_label": key_features_section_label,
        "images": images,
        "image_count": len(images),
        "reviews_summary": reviews_summary or make_reviews_summary(),
        "extraction_status": extraction_status,
        "extraction_errors": list(extraction_errors or []),
        "fetched_at": fetched_at or _utc_now_iso(),
        "last_used_at": last_used_at or _utc_now_iso(),
        "user_overrides": dict(user_overrides or {}),
        "ingest_metadata": dict(ingest_metadata or {}),
    }


def create_selected_primary_image(record_id: str, image_index: int, url: str) -> dict[str, Any]:
    return {"record_id": record_id, "image_index": int(image_index), "url": url}


def create_unselected_primary_image(record_id: str) -> dict[str, Any]:
    return {"record_id": record_id, "image_index": None, "url": ""}


def create_generated_output_shell() -> dict[str, Any]:
    return {
        "image_recommendations": [],
        "recommended_title": "",
        "description_recommendations": [],
        "key_features_recommendations": [],
        "top_priority_fixes": [],
    }


def create_product_audit_entry(
    *,
    record_id: str,
    product_title: str,
    item_id: str = "",
    selected_primary_image: dict[str, Any] | None = None,
    generated_outputs: dict[str, Any] | None = None,
    edited_outputs: dict[str, Any] | None = None,
    rule_findings: list[dict[str, Any]] | None = None,
    include_in_export: bool = True,
    status: str = "ready_for_generation",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "product_title": product_title,
        "item_id": item_id,
        "selected_primary_image": selected_primary_image
        or create_unselected_primary_image(record_id),
        "generated_outputs": generated_outputs or create_generated_output_shell(),
        "edited_outputs": edited_outputs or create_generated_output_shell(),
        "rule_findings": list(rule_findings or []),
        "include_in_export": bool(include_in_export),
        "status": status,
    }


def create_competitor_graphics_assignment(
    *,
    record_id: str,
    image_index: int,
    url: str,
    display_order: int = 0,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "image_index": int(image_index),
        "url": url,
        "display_order": int(display_order),
    }


def create_audit_result_record(
    *,
    client_name: str = "",
    retailer: str = "",
    audit_date: str = "",
    status: str = "draft",
    tone: str = "Standard",
    audit_goal: str = "SEO",
    product_audit_entries: list[dict[str, Any]] | None = None,
    competitor_graphics_assignments: list[dict[str, Any]] | None = None,
    competitor_graphics_notes: str = "",
    retail_media_optimizations: str = "",
    competitor_ad_graphics_notes: str = "",
    audit_id: str | None = None,
) -> dict[str, Any]:
    return {
        "audit_id": audit_id or f"audit-{uuid4().hex[:12]}",
        "client_name": client_name,
        "retailer": retailer,
        "audit_date": audit_date,
        "status": status,
        "tone": tone,
        "audit_goal": audit_goal,
        "product_audit_entries": list(product_audit_entries or []),
        "competitor_graphics_assignments": list(competitor_graphics_assignments or []),
        "competitor_graphics_notes": competitor_graphics_notes,
        "retail_media_optimizations": retail_media_optimizations,
        "competitor_ad_graphics_notes": competitor_ad_graphics_notes,
    }
