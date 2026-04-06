from __future__ import annotations

from typing import Any


def _text_has_value(value: Any) -> bool:
    return bool(str(value or "").strip())


def _list_has_value(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(_text_has_value(v) for v in value)


def _outputs_meaningful(outputs: dict[str, Any] | None) -> bool:
    if not isinstance(outputs, dict):
        return False
    if _text_has_value(outputs.get("recommended_title")):
        return True
    for key in (
        "image_recommendations",
        "description_recommendations",
        "key_features_recommendations",
        "top_priority_fixes",
    ):
        if _list_has_value(outputs.get(key)):
            return True
    return False


def resolve_effective_outputs(entry: dict[str, Any]) -> dict[str, Any]:
    edited = entry.get("edited_outputs", {})
    generated = entry.get("generated_outputs", {})
    return edited if _outputs_meaningful(edited) else generated


def resolve_primary_image_payload(entry: dict[str, Any]) -> dict[str, Any]:
    record = entry.get("cached_record", {}) or {}
    images = record.get("images", []) or []
    selected = entry.get("selected_primary_image", {}) or {}

    selected_index = selected.get("image_index")
    selected_url = str(selected.get("url", "") or "").strip()
    selected_record_id = selected.get("record_id") or record.get("record_id", "")

    if selected_url and isinstance(selected_index, int) and selected_index >= 0:
        return {
            "record_id": selected_record_id,
            "image_index": selected_index,
            "url": selected_url,
            "selection_source": "user_selected",
        }

    hero = next((img for img in images if img.get("is_hero") and img.get("url")), None)
    if hero is not None:
        return {
            "record_id": record.get("record_id", ""),
            "image_index": int(hero.get("index", 0)),
            "url": hero.get("url", ""),
            "selection_source": "hero_fallback",
        }

    if images:
        first = images[0]
        return {
            "record_id": record.get("record_id", ""),
            "image_index": int(first.get("index", 0)),
            "url": first.get("url", ""),
            "selection_source": "first_image_fallback",
        }

    return {
        "record_id": record.get("record_id", ""),
        "image_index": None,
        "url": "",
        "selection_source": "missing",
    }


def build_product_slide_pair(entry: dict[str, Any], pair_order: int) -> dict[str, Any]:
    record = entry.get("cached_record", {}) or {}
    outputs = resolve_effective_outputs(entry)
    primary_image = resolve_primary_image_payload(entry)

    pdp_slide = {
        "slide_type": "pdp_image_info",
        "pair_order": pair_order,
        "record_id": record.get("record_id", entry.get("record_id", "")),
        "product_title": entry.get("product_title", ""),
        "item_id": entry.get("item_id", ""),
        "source_url": record.get("source_url", ""),
        "selected_primary_image": primary_image,
        "image_count": int(record.get("image_count", 0) or 0),
        "extraction_status": record.get("extraction_status", ""),
        "reviews_summary": record.get("reviews_summary", {}),
    }

    content_slide = {
        "slide_type": "content_optimization",
        "pair_order": pair_order,
        "record_id": record.get("record_id", entry.get("record_id", "")),
        "product_title": entry.get("product_title", ""),
        "item_id": entry.get("item_id", ""),
        "image_recommendations": list(outputs.get("image_recommendations", []) or []),
        "recommended_title": outputs.get("recommended_title", ""),
        "description_recommendations": list(outputs.get("description_recommendations", []) or []),
        "key_features_recommendations": list(outputs.get("key_features_recommendations", []) or []),
        "top_priority_fixes": list(outputs.get("top_priority_fixes", []) or []),
        "output_source": "edited" if outputs is entry.get("edited_outputs") else "generated",
    }

    return {
        "pair_order": pair_order,
        "product_entry_id": entry.get("entry_id", f"entry-{pair_order}"),
        "record_id": record.get("record_id", entry.get("record_id", "")),
        "pdp_slide": pdp_slide,
        "content_optimization_slide": content_slide,
    }


def build_competitor_graphics_payload(
    assignments: list[dict[str, Any]],
    competitor_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    competitor_records = competitor_records or []
    record_meta = {
        r.get("record_id", ""): {
            "title": r.get("product_title", ""),
            "brand": r.get("brand", ""),
            "source_url": r.get("source_url", ""),
        }
        for r in competitor_records
        if r.get("record_id")
    }

    cleaned = []
    for a in assignments or []:
        display_order = int(a.get("display_order", 0) or 0)
        if display_order <= 0:
            continue
        record_id = a.get("record_id", "")
        row = {
            "record_id": record_id,
            "image_index": int(a.get("image_index", 0) or 0),
            "url": a.get("url", ""),
            "display_order": display_order,
        }
        row.update(record_meta.get(record_id, {}))
        cleaned.append(row)

    cleaned.sort(key=lambda x: (x["display_order"], x["record_id"], x["image_index"]))
    return {
        "assignment_count": len(cleaned),
        "ordered_assignments": cleaned,
    }


def build_audit_export_plan(
    *,
    audit_record: dict[str, Any] | None,
    primary_entries: list[dict[str, Any]],
    competitor_assignments: list[dict[str, Any]],
    competitor_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    included_entries = [e for e in (primary_entries or []) if e.get("include_in_export", True)]
    product_pairs = [
        build_product_slide_pair(entry, i + 1)
        for i, entry in enumerate(included_entries)
    ]
    competitor_payload = build_competitor_graphics_payload(
        assignments=competitor_assignments,
        competitor_records=competitor_records or [],
    )

    metadata_src = audit_record or {}
    return {
        "audit_metadata": {
            "audit_id": metadata_src.get("audit_id", ""),
            "client_name": metadata_src.get("client_name", ""),
            "retailer": metadata_src.get("retailer", ""),
            "audit_date": metadata_src.get("audit_date", ""),
            "status": metadata_src.get("status", ""),
            "tone": metadata_src.get("tone", ""),
            "audit_goal": metadata_src.get("audit_goal", ""),
        },
        "summary": {
            "included_primary_entry_count": len(included_entries),
            "product_slide_pair_count": len(product_pairs),
            "total_mapped_slides": len(product_pairs) * 2,
            "competitor_graphics_assignment_count": competitor_payload["assignment_count"],
        },
        "product_slide_pairs": product_pairs,
        "competitor_graphics_payload": competitor_payload,
        "shared_sections_payload": {
            "competitor_graphics_notes": metadata_src.get("competitor_graphics_notes", ""),
            "retail_media_optimizations": metadata_src.get("retail_media_optimizations", ""),
            "competitor_ad_graphics_notes": metadata_src.get("competitor_ad_graphics_notes", ""),
        },
    }
