import json
import unittest
from pathlib import Path

from audit_powerpoint_new import (
    SLIDE4_SCOPE_FALLBACKS,
    _slide4_category_context,
    _slide4_product_fallback,
    _slide4_scope_family,
    _slide4_theme_bullet,
    _slide4_language_allowed,
    _slide4_evidence_theme,
    _slide4_build_candidate_pool,
)


class Slide4ScopeFallbackTest(unittest.TestCase):
    def test_cooking_appliances_do_not_generate_food_bullets(self):
        for product in ("Air Fryer", "Toaster Oven", "Convection Oven"):
            with self.subTest(product=product):
                facts = {
                    "title": f"Gourmia {product}",
                    "category": "Food Preparation",
                    "description": "Cook protein for breakfast and toast nuts.",
                }
                self.assertEqual(_slide4_scope_family(facts), "home")
                self.assertEqual(_slide4_category_context(facts, {"category_key": "food_beverage"}), "home")
                self.assertEqual(_slide4_evidence_theme(facts), "product")
                self.assertNotIn("nutrition", _slide4_theme_bullet("pack_or_spec_detail", "strength", facts, {}))
                record = {
                    "product_title": facts["title"],
                    "category": facts["category"],
                    "description_body": facts["description"],
                    "key_features": ["Adjustable temperature", "Cooking timer"],
                }
                candidates, _, _ = _slide4_build_candidate_pool(record, side="competitor", existing_texts=set())
                text = " ".join(candidate["text"] for candidate in candidates).lower()
                self.assertNotIn("nutrition", text)
                self.assertNotIn("breakfast occasions", text)
                self.assertNotIn("peanut butter", text)

    def test_nutrition_detail_remains_available_for_food(self):
        facts = {"category": "Food", "title": "Peanut Butter"}
        self.assertIn("nutrition", _slide4_theme_bullet("pack_or_spec_detail", "strength", facts, {}))
        self.assertFalse(_slide4_language_allowed("Clear pack and nutrition detail", "home")[0])

    def test_every_configured_category_has_a_scope(self):
        guides = Path(__file__).resolve().parents[1] / "config" / "style_guides"
        for path in guides.glob("*.json"):
            display = json.loads(path.read_text(encoding="utf-8-sig"))["category"]
            for category in (path.stem, display):
                with self.subTest(category=category):
                    family = _slide4_scope_family({"category": category})
                    self.assertNotEqual(family, "generic")
                    self.assertTrue(SLIDE4_SCOPE_FALLBACKS[family]["family"])

    def test_air_fryer_uses_home_scope_and_product_fallback(self):
        facts = {"category": "Home", "title": "Digital Air Fryer"}
        self.assertEqual(_slide4_scope_family(facts), "home")
        self.assertEqual(_slide4_product_fallback(facts, "home"), "air fryer")
        self.assertEqual(_slide4_category_context(facts, {"category_key": "home"}), "home")

    def test_explicit_category_takes_priority_over_title(self):
        self.assertEqual(
            _slide4_scope_family({"category": "Toys", "title": "Toy food kitchen"}),
            "toys",
        )

    def test_unknown_category_keeps_generic_fallback(self):
        self.assertEqual(_slide4_scope_family({"category": "Unclassified"}), "generic")
