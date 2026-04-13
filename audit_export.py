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
    *,
    slide_mode: str = "single_pdp",
    mode_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    competitor_records = competitor_records or []
    mode_payload = mode_payload or {}
    record_meta = {
        r.get("record_id", ""): {
            "product_title": r.get("product_title", ""),
            "title": r.get("product_title", ""),
            "brand": r.get("brand", ""),
            "item_id": r.get("item_id", ""),
            "source_url": r.get("source_url", ""),
            "images": list(r.get("images", []) or []),
        }
        for r in competitor_records
        if r.get("record_id")
    }

    cleaned = []
    for a in assignments or []:
        display_order = int(a.get("display_order", 0) or 0)
        if display_order <= 0 or display_order > 10:
            continue
        record_id = a.get("record_id", "")
        row = {
            "record_id": record_id,
            "image_index": int(a.get("image_index", 0) or 0),
            "url": a.get("url", ""),
            "display_order": display_order,
            "product_title": a.get("product_title", ""),
            "title": a.get("title", a.get("product_title", "")),
            "brand": a.get("brand", ""),
            "item_id": a.get("item_id", ""),
            "source_url": a.get("source_url", ""),
        }
        fallback = record_meta.get(record_id, {})
        for key, value in fallback.items():
            if not _text_has_value(row.get(key)):
                row[key] = value
        cleaned.append(row)

    cleaned.sort(key=lambda x: (x["display_order"], x["record_id"], x["image_index"]))
    cleaned_by_record: dict[str, list[dict[str, Any]]] = {}
    for row in cleaned:
        rid = str(row.get("record_id", ""))
        if not rid:
            continue
        cleaned_by_record.setdefault(rid, []).append(row)

    normalized_mode = str(slide_mode or "").strip().lower()
    if normalized_mode not in {"single_pdp", "combined", "per_pdp"}:
        normalized_mode = "single_pdp"

    group_summary = list(mode_payload.get("group_summary", []) or [])
    ordered_record_ids: list[str] = []
    for group in group_summary:
        rid = str(group.get("record_id", "") or "").strip()
        if rid and rid not in ordered_record_ids:
            ordered_record_ids.append(rid)
    if not ordered_record_ids:
        ordered_record_ids = [str(r.get("record_id", "")).strip() for r in competitor_records if r.get("record_id")]

    slide_specs: list[dict[str, Any]] = []
    if normalized_mode == "per_pdp":
        selected_ids_by_record = mode_payload.get("selected_image_ids_by_record", {}) or {}
        multi_orders_by_record = mode_payload.get("multi_image_orders_by_record", {}) or {}
        for rid in ordered_record_ids:
            record = record_meta.get(rid, {})
            image_by_index: dict[int, dict[str, Any]] = {}
            for i, image in enumerate(record.get("images", []) or []):
                img_index = int(image.get("index", i) or i)
                image_by_index[img_index] = image

            explicit_orders = multi_orders_by_record.get(rid, {}) if isinstance(multi_orders_by_record, dict) else {}
            selected_ids = selected_ids_by_record.get(rid, []) if isinstance(selected_ids_by_record, dict) else []

            selected_rows: list[dict[str, Any]] = []
            if isinstance(explicit_orders, dict) and explicit_orders:
                ordered_ids = sorted(explicit_orders.items(), key=lambda x: int(x[1] or 0))
                for image_id, order in ordered_ids:
                    image_id = str(image_id or "")
                    if "|" not in image_id:
                        continue
                    try:
                        _, idx_raw = image_id.rsplit("|", 1)
                        image_index = int(idx_raw)
                    except Exception:
                        continue
                    image = image_by_index.get(image_index, {})
                    image_url = str(image.get("url", "") or "").strip()
                    if not image_url:
                        continue
                    selected_rows.append(
                        {
                            "record_id": rid,
                            "image_index": image_index,
                            "url": image_url,
                            "display_order": int(order or 0),
                            "product_title": record.get("product_title", ""),
                            "title": record.get("title", record.get("product_title", "")),
                            "brand": record.get("brand", ""),
                            "item_id": record.get("item_id", ""),
                            "source_url": record.get("source_url", ""),
                        }
                    )
            elif isinstance(selected_ids, list) and selected_ids:
                for order, image_id in enumerate(selected_ids, start=1):
                    image_id = str(image_id or "")
                    if "|" not in image_id:
                        continue
                    try:
                        _, idx_raw = image_id.rsplit("|", 1)
                        image_index = int(idx_raw)
                    except Exception:
                        continue
                    image = image_by_index.get(image_index, {})
                    image_url = str(image.get("url", "") or "").strip()
                    if not image_url:
                        continue
                    selected_rows.append(
                        {
                            "record_id": rid,
                            "image_index": image_index,
                            "url": image_url,
                            "display_order": int(order),
                            "product_title": record.get("product_title", ""),
                            "title": record.get("title", record.get("product_title", "")),
                            "brand": record.get("brand", ""),
                            "item_id": record.get("item_id", ""),
                            "source_url": record.get("source_url", ""),
                        }
                    )
            else:
                selected_rows = list(cleaned_by_record.get(rid, []) or [])

            selected_rows = [
                row
                for row in selected_rows
                if 1 <= int(row.get("display_order", 0) or 0) <= 10 and _text_has_value(row.get("url"))
            ]
            selected_rows.sort(key=lambda x: (int(x.get("display_order", 0) or 0), int(x.get("image_index", 0) or 0)))
            selected_rows = selected_rows[:10]
            for order, row in enumerate(selected_rows, start=1):
                row["display_order"] = order

            slide_specs.append(
                {
                    "record_id": rid,
                    "product_title": record.get("product_title", ""),
                    "assignment_count": len(selected_rows),
                    "ordered_assignments": selected_rows,
                }
            )
    else:
        active_rows = cleaned[:10]
        for order, row in enumerate(active_rows, start=1):
            row["display_order"] = order
        slide_specs.append(
            {
                "record_id": "",
                "product_title": "",
                "assignment_count": len(active_rows),
                "ordered_assignments": active_rows,
            }
        )

    total_slide_assignments = sum(len(spec.get("ordered_assignments", []) or []) for spec in slide_specs)
    return {
        "assignment_count": total_slide_assignments,
        "raw_assignment_count": len(cleaned),
        "ordered_assignments": cleaned,
        "slide_mode": normalized_mode,
        "slides": slide_specs,
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
    metadata_src = audit_record or {}
    competitor_payload = build_competitor_graphics_payload(
        assignments=competitor_assignments,
        competitor_records=competitor_records or [],
        slide_mode=metadata_src.get("competitor_graphics_mode", "single_pdp"),
        mode_payload=metadata_src.get("competitor_graphics_mode_payload", {}) or {},
    )
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
