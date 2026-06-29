from __future__ import annotations

import re
from typing import Any


VARIATION_PREFIXES = (
    "Category evidence shows",
    "PDP evidence suggests",
    "Captured evidence shows",
    "Available evidence suggests",
)


def normalize_bullet_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u201c\u201d\"']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_unique_bullet_text(text: str, used: set[str], *, fallback_subject: str = "evidence") -> tuple[str, bool]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = normalize_bullet_text(clean)
    if clean and normalized not in used:
        used.add(normalized)
        return clean, False

    for prefix in VARIATION_PREFIXES:
        candidate = f"{prefix} {clean[:1].lower()}{clean[1:]}" if clean else f"{prefix} {fallback_subject}"
        normalized = normalize_bullet_text(candidate)
        if normalized not in used:
            used.add(normalized)
            return candidate, True

    candidate = f"{fallback_subject.title()} evidence creates a focused content opportunity"
    normalized = normalize_bullet_text(candidate)
    suffix = 2
    while normalized in used:
        candidate = f"{fallback_subject.title()} evidence creates focused content opportunity {suffix}"
        normalized = normalize_bullet_text(candidate)
        suffix += 1
    used.add(normalized)
    return candidate, True


def dedupe_bullet_debug(
    bullets: list[dict[str, Any]],
    *,
    fallback_subject: str = "evidence",
) -> tuple[list[dict[str, Any]], list[str]]:
    used: set[str] = set()
    deduped: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in bullets:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        original = str(copied.get("text") or "").strip()
        unique, changed = make_unique_bullet_text(original, used, fallback_subject=fallback_subject)
        copied["text"] = unique
        if changed:
            signals = list(copied.get("signals", []) or [])
            signals.append("duplicate_bullet_reworded")
            copied["signals"] = signals
            copied["reason"] = (
                str(copied.get("reason") or "").strip()
                + " Duplicate wording was rephrased with controlled evidence-safe language."
            ).strip()
            warnings.append(f"Duplicate bullet reworded: {original}")
        deduped.append(copied)
    return deduped, warnings
