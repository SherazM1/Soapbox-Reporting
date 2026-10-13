"""Short, traceable shopper bullets from captured listing content across categories."""
from __future__ import annotations
import re
from typing import Any


def product_evidence_bullets(record: dict[str, Any]) -> list[dict[str, Any]]:
    # Fix note: Use listing content, never extractor scoring notes or category-guide suggestions.
    texts = []
    for source in (record.get("data", {}), record):
        if not isinstance(source, dict):
            continue
        for key in ("descriptionBody", "description", "descriptionText", "descriptionBullets",
                    "description_bullets", "keyFeaturesText", "key_features", "keyFeatures", "features"):
            value = source.get(key)
            for item in value if isinstance(value, list) else [value]:
                text = item.get("text", "") if isinstance(item, dict) else item
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    content = " ".join(dict.fromkeys(texts)).lower()
    rules = (
        ("dimensions", r"\b\d+(?:\.\d+)?\s*(?:inches|inch|cm|mm|ft|feet)\b", "Size details help shoppers assess product fit", "detail_compliance"),
        ("compatibility", r"\b(?:compatible with|fits models?|works with)\b", "Compatibility details help shoppers choose suitable options", "detail_compliance"),
        ("assembly", r"\b(?:assembly required|assembly instructions|assemble|installation instructions)\b", "Setup instructions help shoppers prepare for use", "usage_education"),
        ("care", r"\b(?:hand wash|machine washable|wipe clean|wipe with|dishwasher safe|clean with)\b", "Care details help shoppers choose for everyday use", "usage_education"),
        ("contents", r"\b(?:includes?|contains?)\s+\d+\b", "Included contents help shoppers compare what they receive", "detail_compliance"),
        ("power", r"\b(?:battery[- ]operated|battery[- ]powered|usb[- ]powered|rechargeable)\b", "Power details help shoppers assess everyday use", "detail_compliance"),
    )
    bullets = []
    for dimension, pattern, text, family in rules:
        matches = list(re.finditer(pattern, content))
        usable = [m for m in matches if not re.search(r"\b(?:not|no|without|lacks?)\b", content[max(0,m.start()-25):m.start()])]
        if not usable:
            continue
        match = usable[0]
        excerpt = content[max(0, match.start()-25):min(len(content),match.end()+65)]
        bullets.append({"text": text, "dimension": dimension, "family": family,
                        "evidence": excerpt, "source": "captured_listing_content"})
    return bullets
