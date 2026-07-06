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
        self.assertIn("compatibility", cue_language_label(identity, "pack_or_spec_detail"))
        self.assertIn("spec detail", cue_language_label(identity, "pack_or_spec_detail"))

    def test_product_type_overlays_make_language_more_specific(self) -> None:
        cleanser = language_profile(
            {
                "category_key": "beauty",
                "category_display": "Beauty",
                "family_display": "Skin Care",
                "product_type_display": "Facial Cleansers",
            }
        )
        peanut_butter = language_profile(
            {
                "category_key": "food_beverage",
                "category_display": "Food",
                "product_type_display": "Peanut Butter",
            }
        )
        speakers = language_profile(
            {
                "category_key": "electronics",
                "category_display": "Electronics",
                "product_type_display": "Bluetooth Speakers",
            }
        )
        self.assertIn("facial cleanser shelf", cleanser["product_context"])
        self.assertIn("sensitive-skin", cleanser["formula"])
        self.assertIn("nut butter shelf", peanut_butter["product_context"])
        self.assertIn("allergen", peanut_butter["formula"])
        self.assertIn("audio device shelf", speakers["product_context"])
        self.assertIn("battery", speakers["formula"])

    def test_category_invalid_language_guards_block_cross_category_leakage(self) -> None:
        beauty = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        food = {
            "category_key": "food_beverage",
            "product_type_display": "Peanut Butter",
        }
        electronics = {
            "category_key": "electronics",
            "product_type_display": "Bluetooth Speakers",
        }
        self.assertNotIn(
            "nutrition detail",
            strategic_bullet_text(
                {"cue_key": "ingredient_or_formula_communication"},
                beauty,
                slide_key="slide4",
                evidence_terms={"detail": "Clear nutrition detail"},
            ).lower(),
        )
        self.assertNotIn("regimen", recommendation_phrase(food, "education").lower())
        self.assertNotIn(
            "ingredient detail",
            strategic_bullet_text(
                {"cue_key": "ingredient_or_formula_communication"},
                electronics,
                slide_key="slide4",
                evidence_terms={"detail": "Clear ingredient detail"},
            ).lower(),
        )

    def test_cue_language_families_are_separated(self) -> None:
        identity = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        labels = {
            cue: cue_language_label(identity, cue)
            for cue in (
                "keyword_alignment",
                "discoverability",
                "assortment_segmentation",
                "category_grouping",
                "discovery_pathways",
                "cross_category_navigation",
                "review_or_trust_signals",
                "conversion_guidance",
            )
        }
        self.assertEqual(len(set(labels.values())), len(labels))
        self.assertIn("search intent", labels["keyword_alignment"])
        self.assertIn("shelf visibility", labels["discoverability"])
        self.assertIn("segmentation", labels["assortment_segmentation"])
        self.assertIn("cross-shopping", labels["cross_category_navigation"])

    def test_recommendation_phrases_are_commercial_actions(self) -> None:
        identity = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        self.assertIn("titles and PDP language", recommendation_phrase(identity, "search_intent"))
        self.assertIn("Brand Shop modules", recommendation_phrase(identity, "brand_shop"))
        self.assertIn("priority attributes", recommendation_phrase(identity, "attributes"))
        self.assertIn("decision points", recommendation_phrase(identity, "conversion"))


if __name__ == "__main__":
    unittest.main()
