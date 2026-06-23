import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.audit_helpers.slide4_findings import build_slide4_group_findings


ROOT = Path(__file__).resolve().parents[1]


def _analysis(
    *,
    image_count: int = 6,
    formats: list[str] | None = None,
    signals: list[list[str]] | None = None,
    tokens: list[list[str]] | None = None,
    duplicate_count: int = 0,
) -> dict:
    formats = formats or ["product_silo", "text_heavy_graphic", "nutrition_or_ingredients"]
    signals = signals or [["ingredients"], ["feature_or_benefit_claim"], ["nutrition_or_ingredients"]]
    tokens = tokens or [["ingredient", "flavor"], ["benefit"], ["nutrition"]]
    images = []
    for index, fmt in enumerate(formats):
        images.append(
            {
                "status": "analyzed",
                "position": index + 1,
                "probable_format": fmt,
                "white_background_ratio": 0.86 if index == 0 else 0.2,
                "detected_signals": signals[index] if index < len(signals) else [],
                "ocr_tokens": tokens[index] if index < len(tokens) else [],
            }
        )
    return {
        "status": "complete",
        "image_count": image_count,
        "analyzed_image_count": len(images),
        "images": images,
        "stack_signals": {"duplicate_image_count": duplicate_count},
    }


def _record(
    *,
    category: str = "Food/Pantry/Jams, Jellies & Preserves",
    product_type: str = "Jams, Jellies & Preserves",
    analysis: dict | None = None,
) -> dict:
    return {
        "brand": "Test Brand",
        "product_title": "Strawberry Preserves",
        "category": category,
        "product_type": product_type,
        "image_count": (analysis or {}).get("image_count", 6),
        "image_analysis": analysis or _analysis(),
    }


class Slide4FindingsTest(unittest.TestCase):
    def test_missing_lifestyle_requires_strict_majority(self) -> None:
        majority_records = [_record() for _ in range(6)] + [
            _record(analysis=_analysis(formats=["product_silo", "lifestyle_or_scene"]))
            for _ in range(4)
        ]
        majority = build_slide4_group_findings(majority_records, "test")
        self.assertEqual(majority["analyzed_pdp_count"], 10)
        self.assertEqual(majority["majority_threshold"], 6)
        self.assertIn(
            "Opportunity to strengthen breakfast, snack, and pairing use cases",
            majority["slide4_bullets"],
        )

        non_majority_records = [_record() for _ in range(5)] + [
            _record(analysis=_analysis(formats=["product_silo", "lifestyle_or_scene"]))
            for _ in range(5)
        ]
        non_majority = build_slide4_group_findings(non_majority_records, "test")
        self.assertEqual(non_majority["majority_threshold"], 6)
        self.assertNotIn(
            "Opportunity to strengthen breakfast, snack, and pairing use cases",
            non_majority["slide4_bullets"],
        )

    def test_category_wording_for_jam_jelly_preserves(self) -> None:
        findings = build_slide4_group_findings([_record() for _ in range(3)], "test")
        all_text = " | ".join(
            item["text"]
            for item in findings["strengths"] + findings["opportunities"]
        )
        self.assertIn("Strong flavor-forward visual identity", all_text)
        self.assertIn("Opportunity to expand recipe-led serving inspiration", all_text)

    def test_category_wording_for_nut_butter_spreads(self) -> None:
        findings = build_slide4_group_findings(
            [
                _record(
                    category="Food/Pantry/Peanut butter & spreads",
                    product_type="Nut Butters & Spreads",
                )
                for _ in range(3)
            ],
            "test",
        )
        all_text = " | ".join(
            item["text"]
            for item in findings["strengths"] + findings["opportunities"]
        )
        self.assertIn("Strong protein and ingredient-led benefit communication", all_text)
        self.assertIn("Opportunity to expand snack, breakfast, and recipe-based usage storytelling", all_text)

    def test_category_wording_for_baby_care(self) -> None:
        baby_analysis = _analysis(
            formats=["product_silo", "text_heavy_graphic"],
            signals=[["routine_or_regimen"], ["feature_or_benefit_claim"]],
            tokens=[["routine"], ["gentle"]],
        )
        findings = build_slide4_group_findings(
            [
                _record(
                    category="Baby/Bath & Skin Care",
                    product_type="Baby Wash",
                    analysis=baby_analysis,
                )
                for _ in range(3)
            ],
            "test",
        )
        all_text = " | ".join(
            item["text"]
            for item in findings["strengths"] + findings["opportunities"]
        )
        self.assertIn("Routine-based merchandising approach", all_text)
        self.assertIn("Opportunity to expand parent-focused reassurance and bath-time storytelling", all_text)

    def test_records_without_image_analysis_return_empty_bullets_for_fallback(self) -> None:
        findings = build_slide4_group_findings([{"brand": "No Evidence"}], "test")
        self.assertEqual(findings["analyzed_pdp_count"], 0)
        self.assertEqual(findings["majority_threshold"], 0)
        self.assertEqual(findings["slide4_bullets"], [])

    def test_no_paid_or_model_dependencies_are_required(self) -> None:
        requirement_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        package_text = (
            (ROOT / "packages.txt").read_text(encoding="utf-8").lower()
            if (ROOT / "packages.txt").exists()
            else ""
        )
        combined = requirement_text + "\n" + package_text
        for forbidden in ("openai", "torch", "clip", "transformers"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
