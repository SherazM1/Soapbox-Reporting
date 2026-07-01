from __future__ import annotations

import unittest

from app.audit_helpers.audit_language_resolver import (
    cue_language_label,
    language_profile,
    recommendation_phrase,
    strategic_bullet_text,
)


class AuditLanguageResolverTests(unittest.TestCase):
    def test_priority_category_profiles_are_commercially_specific(self) -> None:
        cases = {
            "food_beverage": ("pantry shelf", "nutrition", "breakfast"),
            "beauty": ("beauty shelf", "formula", "regimen"),
            "health_personal_care": ("wellness shelf", "dosage", "symptom"),
            "animals": ("pet care shelf", "feeding", "life-stage"),
            "electronics": ("electronics shelf", "compatibility", "setup"),
        }
        for category_key, expected_terms in cases.items():
            with self.subTest(category_key=category_key):
                profile = language_profile(
                    {
                        "category_key": category_key,
                        "category_display": "Category",
                        "product_type_display": "Test Product",
                    }
                )
                joined = " ".join(profile.values()).lower()
                for term in expected_terms:
                    self.assertIn(term, joined)

    def test_slide_aware_translation_changes_expression_by_slide(self) -> None:
        identity = {
            "category_key": "beauty",
            "category_display": "Beauty",
            "family_display": "Skin Care",
            "product_type_display": "Face Moisturizers",
            "shopping_context_phrase": "Face Moisturizers shopping journey",
        }
        candidate = {"cue_key": "shopper_education", "classification": "opportunity"}
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide2").lower(),
        )
        self.assertIn(
            "discovery",
            strategic_bullet_text({"cue_key": "discoverability"}, identity, slide_key="slide3").lower(),
        )
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide4").lower(),
        )
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide5").lower(),
        )

    def test_recommendation_phrases_use_product_and_category_context(self) -> None:
        identity = {
            "category_key": "electronics",
            "product_type_display": "Bluetooth Speakers",
        }
        self.assertIn("Bluetooth Speakers SEO", recommendation_phrase(identity, "seo"))
        self.assertIn("compatibility", recommendation_phrase(identity, "attributes"))
        self.assertIn("device discovery", recommendation_phrase(identity, "discovery"))
        self.assertEqual(cue_language_label(identity, "pack_or_spec_detail"), "compatibility and spec detail")


if __name__ == "__main__":
    unittest.main()
