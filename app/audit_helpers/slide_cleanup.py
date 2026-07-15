from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


SLIDE_CLEANUP_SEQUENCE: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("slide6_visibility", "Slide 6", lambda payload: cleanup_slide6(payload)),
    ("slide4_findings", "Slide 4", lambda payload: cleanup_slide4(payload)),
    ("slide3_search_benchmark", "Slide 3", lambda payload: cleanup_slide3(payload)),
    ("slide5_brand_shop", "Slide 5", lambda payload: cleanup_slide5(payload)),
)


def cleanup_slide6(payload: Any) -> Any:
    return payload


def cleanup_slide4(payload: Any) -> Any:
    return payload


def cleanup_slide3(payload: Any) -> Any:
    return payload


def cleanup_slide5(payload: Any) -> Any:
    return payload


def cleanup_generated_audit_plan(export_plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "has_run": True,
        "succeeded": False,
        "active": False,
        "slides_cleaned": [],
        "slides_skipped": [],
        "warnings": [],
    }

    try:
        cleaned_plan = deepcopy(export_plan or {})
    except Exception as exc:
        metadata["warnings"].append(f"Unable to clone generated audit plan: {exc}")
        return export_plan or {}, metadata

    for payload_key, slide_label, cleanup_hook in SLIDE_CLEANUP_SEQUENCE:
        original_payload = cleaned_plan.get(payload_key)
        try:
            cleaned_plan[payload_key] = cleanup_hook(deepcopy(original_payload))
            metadata["slides_cleaned"].append(slide_label)
        except Exception as exc:
            cleaned_plan[payload_key] = original_payload
            metadata["slides_skipped"].append(slide_label)
            metadata["warnings"].append(f"{slide_label} cleanup skipped: {exc}")

    metadata["succeeded"] = True
    metadata["active"] = True
    return cleaned_plan, metadata
