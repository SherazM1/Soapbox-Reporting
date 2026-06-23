from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image, ImageDraw

from app.audit_helpers import image_analysis as ia
from audit_helpers import (
    process_competitor_audit_extract_sheet,
    process_primary_audit_extract_sheet,
)


def _save_image(path: Path, color: str = "white", text: str = "") -> str:
    image = Image.new("RGB", (240, 240), color)
    draw = ImageDraw.Draw(image)
    if text:
        draw.rectangle((15, 70, 225, 170), fill="white")
        draw.text((25, 95), text, fill="black")
    else:
        draw.rectangle((95, 95, 145, 145), fill="black")
    image.save(path)
    return path.resolve().as_uri()


def _sheet(url: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Product URL": "https://example.test/pdp",
                "Product ID": "1",
                "Product Title": "Test Product",
                "Brand": "Test Brand",
                "Category": "Food/Pantry/Peanut butter & spreads",
                "Product Type": "Nut Butters & Spreads",
                "Image 1": url,
            }
        ]
    )


class ImageAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        ia.clear_image_analysis_cache()

    def test_ocr_normalization_removes_garbage_and_keeps_useful_tokens(self) -> None:
        text, tokens = ia.normalize_ocr_text("  INGREDIENTS: peanuts || 12 oz\n@@@ % Daily Value  ")
        self.assertIn("ingredients", tokens)
        self.assertIn("peanuts", tokens)
        self.assertIn("12", tokens)
        self.assertIn("oz", tokens)
        self.assertNotIn("@@@", text)

    def test_white_background_product_image_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "silo.png")
            with patch.object(ia, "_best_ocr", return_value=("", [], [])):
                result = ia.analyze_pdp_image(
                    {"url": url, "index": 0},
                    category="Food",
                    product_type="Nut Butters & Spreads",
                    expected_slot="silo_front",
                )
        self.assertEqual(result["status"], "analyzed")
        self.assertEqual(result["probable_format"], "product_silo")
        self.assertGreater(result["white_background_ratio"], 0.7)

    def test_text_heavy_graphic_classification(self) -> None:
        fmt, reasons, confidence = ia.classify_probable_format(
            white_background_ratio=0.3,
            text_density=0.2,
            edge_density=0.08,
            colorfulness=0.2,
            detected_signals=[],
            ocr_word_count=25,
        )
        self.assertEqual(fmt, "text_heavy_graphic")
        self.assertGreater(confidence, 0.7)
        self.assertTrue(reasons)

    def test_nutrition_ocr_signal_detection(self) -> None:
        signals = ia.detect_ocr_signals("nutrition facts calories serving size daily value")
        self.assertIn("nutrition", signals)

    def test_dimensions_ocr_signal_detection(self) -> None:
        signals = ia.detect_ocr_signals("12 in width 8 inches height")
        self.assertIn("dimensions_or_scale", signals)

    def test_duplicate_image_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "dup.png")
            record = {
                "category": "Food",
                "subcategory": "Nut Butters & Spreads",
                "images": [{"url": url, "index": 0}, {"url": url, "index": 1}],
            }
            with patch.object(ia, "_best_ocr", return_value=("", [], [])):
                analysis = ia.analyze_pdp_image_stack(record)
        self.assertEqual(analysis["stack_signals"]["duplicate_image_count"], 1)

    def test_ordered_expected_slot_mapping_preserves_guide_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            urls = [_save_image(Path(tmp) / f"img-{i}.png") for i in range(3)]
            record = {
                "category": "Food/Pantry/Peanut butter & spreads",
                "subcategory": "Nut Butters & Spreads",
                "images": [{"url": url, "index": i} for i, url in enumerate(urls)],
            }
            with patch.object(ia, "_best_ocr", return_value=("", [], [])):
                analysis = ia.analyze_pdp_image_stack(record)
        self.assertEqual(
            [image["expected_slot"] for image in analysis["images"]],
            ["silo_front", "graphic_ingredients", "silo_alt_in_pack"],
        )

    def test_cache_reuse_skips_reanalysis_for_same_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "cache.png")
            with patch.object(ia, "_best_ocr", return_value=("", [], [])) as ocr:
                first = ia.analyze_pdp_image(
                    {"url": url, "index": 0},
                    category="Food",
                    product_type="Nut Butters & Spreads",
                    expected_slot="silo_front",
                )
                second = ia.analyze_pdp_image(
                    {"url": url, "index": 1},
                    category="Food",
                    product_type="Nut Butters & Spreads",
                    expected_slot="graphic_ingredients",
                )
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(ocr.call_count, 1)

    def test_one_failed_image_does_not_fail_pdp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good_url = _save_image(Path(tmp) / "good.png")
            bad_url = (Path(tmp) / "missing.png").resolve().as_uri()
            record = {
                "category": "Food",
                "subcategory": "Nut Butters & Spreads",
                "images": [{"url": good_url, "index": 0}, {"url": bad_url, "index": 1}],
            }
            with patch.object(ia, "_best_ocr", return_value=("", [], [])):
                analysis = ia.analyze_pdp_image_stack(record)
        self.assertEqual(analysis["analyzed_image_count"], 1)
        self.assertEqual(analysis["failed_image_count"], 1)
        self.assertEqual(analysis["status"], "complete")

    def test_missing_tesseract_does_not_fail_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "no-tess.png")
            with patch.object(ia, "pytesseract", None):
                result = ia.analyze_pdp_image(
                    {"url": url, "index": 0},
                    category="Food",
                    product_type="Nut Butters & Spreads",
                    expected_slot=None,
                )
        self.assertEqual(result["status"], "analyzed")
        self.assertIn("pytesseract is unavailable", result["errors"])

    def test_primary_and_competitor_processing_can_store_image_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "sheet.png")
            with patch.object(ia, "_best_ocr", return_value=("", [], [])):
                primary_entries, _, primary_messages = process_primary_audit_extract_sheet(
                    df_uploaded=_sheet(url)
                )
                competitor_records, _, competitor_messages = process_competitor_audit_extract_sheet(
                    df_uploaded=_sheet(url)
                )
                ia.analyze_pdp_records([entry["cached_record"] for entry in primary_entries])
                ia.analyze_pdp_records(competitor_records)
        self.assertTrue(all(message.startswith("Warning:") for message in primary_messages))
        self.assertTrue(all(message.startswith("Warning:") for message in competitor_messages))
        self.assertIn("image_analysis", primary_entries[0]["cached_record"])
        self.assertIn("image_analysis", competitor_records[0])

    def test_sheet_ingestion_still_works_when_analysis_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = _save_image(Path(tmp) / "unavailable.png")
            primary_entries, records, messages = process_primary_audit_extract_sheet(
                df_uploaded=_sheet(url)
            )
        self.assertTrue(all(message.startswith("Warning:") for message in messages))
        self.assertEqual(len(primary_entries), 1)
        self.assertEqual(len(records), 1)
        self.assertNotIn("image_analysis", primary_entries[0]["cached_record"])


if __name__ == "__main__":
    unittest.main()
