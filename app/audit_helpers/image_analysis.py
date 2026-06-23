from __future__ import annotations

import copy
import hashlib
import io
import math
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

from app.audit_helpers.image_guides import (
    get_image_guide_page,
    resolve_image_guide_category,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on runtime packaging
    cv2 = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - depends on runtime packaging
    pytesseract = None


ANALYSIS_VERSION = "local_ocr_v1"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_WORKING_DIMENSION = 1600
_CACHE_LIMIT = 256
_ANALYSIS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

ProgressCallback = Callable[[int, int, int, int], None]

_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "ingredients": ("ingredient", "ingredients", "contains", "allergen"),
    "nutrition": ("nutrition facts", "calories", "serving size", "daily value"),
    "protein_or_nutrition_benefit": (
        "protein",
        "fiber",
        "vitamin",
        "mineral",
        "nutrient",
        "grams of",
    ),
    "organic_or_certification": (
        "organic",
        "certified",
        "non gmo",
        "kosher",
        "fair trade",
        "usda",
    ),
    "clinical_or_dermatologist": (
        "clinical",
        "clinically",
        "dermatologist",
        "hypoallergenic",
    ),
    "usage_or_instructions": (
        "how to use",
        "directions",
        "step 1",
        "step 2",
        "apply",
        "prepare",
        "instructions",
    ),
    "recipe_or_serving": (
        "recipe",
        "baking",
        "spread",
        "serve",
        "serving suggestion",
    ),
    "routine_or_regimen": ("routine", "regimen", "morning", "night", "daily use"),
    "feature_or_benefit_claim": (
        "helps",
        "benefit",
        "benefits",
        "improves",
        "supports",
        "long lasting",
    ),
    "size_or_count": ("net wt", "net weight", "count", "pack", "ounces", "ounce"),
    "assortment_or_variants": (
        "available in",
        "flavors",
        "flavours",
        "sizes",
        "colors",
        "colours",
        "collection",
        "variety",
    ),
    "sustainability_or_recycling": (
        "recycle",
        "recyclable",
        "recycled",
        "sustainable",
        "compostable",
    ),
    "guarantee": ("guarantee", "guaranteed", "money back", "satisfaction"),
}

_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:inches?|in|feet|foot|ft|centimeters?|cm|millimeters?|mm|"
    r"width|wide|height|high|length|long)\b",
    flags=re.IGNORECASE,
)
_USEFUL_TOKEN_RE = re.compile(
    r"^(?:[a-z]{2,}|[0-9]+(?:\.[0-9]+)?|[0-9]+%|[0-9]+(?:oz|lb|g|kg|ml|l|cm|mm|in|ft))$",
    flags=re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)


def clear_image_analysis_cache() -> None:
    _ANALYSIS_CACHE.clear()


def image_analysis_cache_size() -> int:
    return len(_ANALYSIS_CACHE)


def _cache_key(image_hash: str) -> str:
    return f"{ANALYSIS_VERSION}:{image_hash}"


def _cache_get(image_hash: str) -> dict[str, Any] | None:
    key = _cache_key(image_hash)
    cached = _ANALYSIS_CACHE.get(key)
    if cached is None:
        return None
    _ANALYSIS_CACHE.move_to_end(key)
    return _json_copy(cached)


def _cache_set(image_hash: str, result: dict[str, Any]) -> None:
    if result.get("status") != "analyzed":
        return
    key = _cache_key(image_hash)
    _ANALYSIS_CACHE[key] = _json_copy(result)
    _ANALYSIS_CACHE.move_to_end(key)
    while len(_ANALYSIS_CACHE) > _CACHE_LIMIT:
        _ANALYSIS_CACHE.popitem(last=False)


def normalize_ocr_text(text: str) -> tuple[str, list[str]]:
    text = str(text or "").replace("\x0c", " ")
    text = re.sub(r"[|_~`^]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[.'%-][A-Za-z0-9]+)*", text.lower())
    tokens = [
        token.strip(".'-")
        for token in raw_tokens
        if _USEFUL_TOKEN_RE.match(token.strip(".'-"))
    ]
    normalized = " ".join(tokens)
    return normalized, tokens


def detect_ocr_signals(text: str, tokens: list[str] | None = None) -> list[str]:
    normalized = str(text or "").lower()
    signals = [
        signal
        for signal, patterns in _SIGNAL_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    ]
    if _MEASUREMENT_RE.search(normalized):
        signals.append("dimensions_or_scale")
    return list(dict.fromkeys(signals))


def _download_image(url: str) -> tuple[bytes, Image.Image]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        },
    )
    chunks: list[bytes] = []
    total = 0
    with urlopen(request, timeout=15) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("image exceeds download size limit")
        while True:
            chunk = response.read(min(1024 * 1024, MAX_DOWNLOAD_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("image exceeds download size limit")
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        raise ValueError("downloaded image is empty")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"download is not a readable raster image: {exc}") from exc
    return raw, image


def _working_image(image: Image.Image) -> Image.Image:
    working = image.copy()
    working.thumbnail((MAX_WORKING_DIMENSION, MAX_WORKING_DIMENSION), Image.Resampling.LANCZOS)
    return working


def _best_ocr(working: Image.Image) -> tuple[str, list[str], list[str]]:
    if pytesseract is None:
        return "", [], ["pytesseract is unavailable"]
    variants = [
        working,
        ImageOps.grayscale(working),
        ImageEnhance.Contrast(ImageOps.grayscale(working)).enhance(2.0).point(
            lambda value: 255 if value > 165 else 0
        ),
    ]
    candidates: list[tuple[str, list[str]]] = []
    errors: list[str] = []
    for variant in variants:
        try:
            raw_text = pytesseract.image_to_string(variant, config="--psm 6")
            normalized, tokens = normalize_ocr_text(raw_text)
            candidates.append((normalized, tokens))
        except Exception as exc:  # Tesseract executable may be unavailable
            errors.append(f"OCR unavailable: {exc}")
            break
    if not candidates:
        return "", [], list(dict.fromkeys(errors))
    best_text, best_tokens = max(candidates, key=lambda item: (len(item[1]), len(item[0])))
    return best_text, best_tokens, list(dict.fromkeys(errors))


def _white_background_ratio(rgb: np.ndarray) -> float:
    near_white = np.all(rgb >= 245, axis=2)
    height, width = near_white.shape
    border = max(1, int(min(height, width) * 0.08))
    border_mask = np.zeros_like(near_white, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_ratio = float(near_white[border_mask].mean()) if border_mask.any() else 0.0
    overall_ratio = float(near_white.mean())
    return round((border_ratio * 0.7) + (overall_ratio * 0.3), 4)


def _text_density(gray: np.ndarray) -> float:
    dark = gray < 105
    return round(float(dark.mean()), 4)


def _edge_density(gray: np.ndarray) -> float:
    if cv2 is not None:
        edges = cv2.Canny(gray, 80, 180)
        return round(float((edges > 0).mean()), 4)
    pil_gray = Image.fromarray(gray)
    edges = np.asarray(pil_gray.filter(ImageFilter.FIND_EDGES))
    return round(float((edges > 35).mean()), 4)


def _colorfulness(rgb: np.ndarray) -> float:
    values = rgb.astype(np.float32)
    red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    score = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    score += 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return round(min(score / 100.0, 1.0), 4)


def _difference_hash(image: Image.Image) -> str:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def classify_probable_format(
    *,
    white_background_ratio: float,
    text_density: float,
    edge_density: float,
    colorfulness: float,
    detected_signals: list[str],
    ocr_word_count: int,
) -> tuple[str, list[str], float]:
    reasons: list[str] = []
    signal_set = set(detected_signals)
    if signal_set & {"nutrition", "ingredients", "protein_or_nutrition_benefit"}:
        reasons.append("nutrition or ingredient OCR signal")
        return "nutrition_or_ingredients", reasons, 0.82
    if signal_set & {"dimensions_or_scale", "usage_or_instructions"}:
        reasons.append("measurement or instruction OCR signal")
        return "dimensions_or_instructions", reasons, 0.78
    if white_background_ratio >= 0.72 and text_density <= 0.08 and ocr_word_count <= 8:
        reasons.extend(["high white-background ratio", "limited detected text"])
        return "product_silo", reasons, 0.76
    if text_density >= 0.12 or ocr_word_count >= 18:
        reasons.append("substantial text coverage")
        return "text_heavy_graphic", reasons, 0.72
    if white_background_ratio >= 0.45 and (text_density >= 0.05 or ocr_word_count >= 6):
        reasons.append("product-style background with meaningful text")
        return "mixed_product_graphic", reasons, 0.64
    if white_background_ratio < 0.35 and text_density < 0.09 and edge_density >= 0.04:
        reasons.extend(["limited white background", "photographic structural complexity"])
        return "lifestyle_or_scene", reasons, 0.58
    return "unknown", ["signals were not strong enough for a broad format estimate"], 0.25


def _base_result(image: dict[str, Any], expected_slot: str | None) -> dict[str, Any]:
    position = int(image.get("position") or int(image.get("index", 0) or 0) + 1)
    return {
        "status": "failed",
        "position": position,
        "url": str(image.get("url", "") or ""),
        "image_hash": "",
        "width": image.get("width"),
        "height": image.get("height"),
        "aspect_ratio": 0.0,
        "ocr_text": "",
        "ocr_tokens": [],
        "ocr_word_count": 0,
        "text_density": 0.0,
        "white_background_ratio": 0.0,
        "edge_density": 0.0,
        "colorfulness": 0.0,
        "perceptual_hash": "",
        "probable_format": "unknown",
        "detected_signals": [],
        "expected_slot": expected_slot or "",
        "slot_evidence": [],
        "format_reasons": [],
        "confidence": 0.0,
        "errors": [],
    }


def analyze_pdp_image(
    image: dict[str, Any],
    *,
    category: str,
    product_type: str,
    expected_slot: str | None,
) -> dict[str, Any]:
    result = _base_result(image, expected_slot)
    url = result["url"]
    if not url:
        result["errors"] = ["image URL is empty"]
        return result
    try:
        raw, original = _download_image(url)
        image_hash = hashlib.sha256(raw).hexdigest()
        cached = _cache_get(image_hash)
        if cached is not None:
            cached["position"] = result["position"]
            cached["url"] = url
            cached["expected_slot"] = expected_slot or ""
            cached["slot_evidence"] = (
                [f"guide expects {expected_slot} at carousel position"]
                if expected_slot
                else []
            )
            cached["cache_hit"] = True
            return cached

        width, height = original.size
        working = _working_image(original)
        rgb = np.asarray(working)
        gray = np.asarray(ImageOps.grayscale(working))
        ocr_text, ocr_tokens, ocr_errors = _best_ocr(working)
        signals = detect_ocr_signals(ocr_text, ocr_tokens)
        white_ratio = _white_background_ratio(rgb)
        density = _text_density(gray)
        edge_density = _edge_density(gray)
        colorfulness = _colorfulness(rgb)
        probable_format, reasons, confidence = classify_probable_format(
            white_background_ratio=white_ratio,
            text_density=density,
            edge_density=edge_density,
            colorfulness=colorfulness,
            detected_signals=signals,
            ocr_word_count=len(ocr_tokens),
        )
        slot_evidence = []
        if expected_slot:
            slot_evidence.append(f"guide expects {expected_slot} at carousel position")
        result.update(
            {
                "status": "analyzed",
                "image_hash": image_hash,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if height else 0.0,
                "ocr_text": ocr_text,
                "ocr_tokens": ocr_tokens,
                "ocr_word_count": len(ocr_tokens),
                "text_density": density,
                "white_background_ratio": white_ratio,
                "edge_density": edge_density,
                "colorfulness": colorfulness,
                "perceptual_hash": _difference_hash(working),
                "probable_format": probable_format,
                "detected_signals": signals,
                "slot_evidence": slot_evidence,
                "format_reasons": reasons,
                "confidence": confidence,
                "errors": ocr_errors,
                "cache_hit": False,
            }
        )
        _cache_set(image_hash, result)
        return result
    except Exception as exc:
        result["errors"] = [str(exc)]
        return result


def _stack_signal_counts(images: list[dict[str, Any]]) -> dict[str, Any]:
    analyzed = [image for image in images if image.get("status") == "analyzed"]
    formats = [
        "product_silo",
        "text_heavy_graphic",
        "lifestyle_or_scene",
        "mixed_product_graphic",
        "nutrition_or_ingredients",
        "dimensions_or_instructions",
        "unknown",
    ]
    signals = {
        f"{format_name}_count": sum(
            1 for image in analyzed if image.get("probable_format") == format_name
        )
        for format_name in formats
    }
    for signal in _SIGNAL_PATTERNS:
        signals[f"{signal}_count"] = sum(
            1 for image in analyzed if signal in (image.get("detected_signals") or [])
        )
    signals["dimensions_or_scale_count"] = sum(
        1
        for image in analyzed
        if "dimensions_or_scale" in (image.get("detected_signals") or [])
    )
    dimensions = {
        (image.get("width"), image.get("height"))
        for image in analyzed
        if image.get("width") and image.get("height")
    }
    signals["consistent_dimensions"] = len(dimensions) <= 1 if analyzed else False
    return signals


def analyze_pdp_image_stack(
    record: dict[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    category = str(record.get("category", "") or "")
    product_type = str(record.get("product_type") or record.get("subcategory") or "")
    guide_key = resolve_image_guide_category(category)
    guide_page = get_image_guide_page(guide_key, product_type)
    expected_slots = list((guide_page or {}).get("required_slots", []) or [])
    source_images = list(record.get("images", []) or [])
    results: list[dict[str, Any]] = []
    for index, source_image in enumerate(source_images):
        image = dict(source_image or {})
        image["position"] = index + 1
        expected_slot = expected_slots[index] if index < len(expected_slots) else None
        if progress_callback:
            progress_callback(index + 1, len(source_images))
        results.append(
            analyze_pdp_image(
                image,
                category=category,
                product_type=product_type,
                expected_slot=expected_slot,
            )
        )

    duplicate_count = 0
    prior_hashes: list[tuple[int, str]] = []
    for result in results:
        fingerprint = str(result.get("perceptual_hash", "") or "")
        if result.get("status") != "analyzed" or not fingerprint:
            continue
        duplicate_of = next(
            (
                position
                for position, prior in prior_hashes
                if _hamming_distance(fingerprint, prior) <= 5
            ),
            None,
        )
        if duplicate_of is not None:
            result["duplicate_of_position"] = duplicate_of
            duplicate_count += 1
        prior_hashes.append((int(result.get("position", 0) or 0), fingerprint))

    analyzed_count = sum(1 for result in results if result.get("status") == "analyzed")
    failed_count = len(results) - analyzed_count
    errors: list[str] = []
    if pytesseract is None:
        errors.append("pytesseract is unavailable; OCR was skipped")
    elif any(
        any("OCR unavailable" in str(error) for error in result.get("errors", []))
        for result in results
    ):
        errors.append("Tesseract is unavailable; OCR was skipped")
    if cv2 is None:
        errors.append("OpenCV is unavailable; Pillow edge analysis was used")
    stack_signals = _stack_signal_counts(results)
    stack_signals["duplicate_image_count"] = duplicate_count
    analysis = {
        "status": "complete" if analyzed_count or not source_images else "failed",
        "analyzed_at": _utc_now_iso(),
        "analysis_version": ANALYSIS_VERSION,
        "category": category,
        "product_type": product_type,
        "guide_key": guide_key,
        "guide_page_key": (guide_page or {}).get("page_key", ""),
        "expected_slots": expected_slots,
        "image_count": len(source_images),
        "analyzed_image_count": analyzed_count,
        "failed_image_count": failed_count,
        "images": results,
        "stack_signals": stack_signals,
        "errors": errors,
    }
    record["image_analysis"] = analysis
    return analysis


def analyze_pdp_records(
    records: list[dict[str, Any]],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    total_analyzed = 0
    total_failed = 0
    warning_messages: list[str] = []
    total_pdps = len(records)
    for pdp_index, record in enumerate(records, start=1):
        def on_image(image_index: int, total_images: int) -> None:
            if progress_callback:
                progress_callback(pdp_index, total_pdps, image_index, total_images)

        analysis = analyze_pdp_image_stack(record, progress_callback=on_image)
        total_analyzed += int(analysis.get("analyzed_image_count", 0) or 0)
        total_failed += int(analysis.get("failed_image_count", 0) or 0)
        warning_messages.extend(analysis.get("errors", []) or [])
    return {
        "pdp_count": total_pdps,
        "analyzed_image_count": total_analyzed,
        "failed_image_count": total_failed,
        "warnings": list(dict.fromkeys(warning_messages)),
    }
