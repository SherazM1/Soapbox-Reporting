import unittest

from app.audit_helpers.audit_language_resolver import (
    INVALID_LANGUAGE_GUARDS,
    LANGUAGE_PROFILES,
    _guard_language,
    strategic_bullet_text,
)


class LanguageGuardCoverageTest(unittest.TestCase):
    def test_every_profile_has_guards(self):
        self.assertEqual(set(LANGUAGE_PROFILES), set(INVALID_LANGUAGE_GUARDS))
        for category, rules in INVALID_LANGUAGE_GUARDS.items():
            self.assertTrue(rules, category)

    def test_new_categories_correct_wrong_shelf_without_cascading(self):
        existing = {"beauty", "food_beverage", "electronics", "business_industrial", "health_personal_care"}
        for category, profile in LANGUAGE_PROFILES.items():
            if category in existing:
                continue
            with self.subTest(category=category):
                guarded = _guard_language("BEAUTY SHELF", {"category_key": category})
                self.assertEqual(guarded, profile["context"])
                self.assertEqual(_guard_language(guarded, {"category_key": category}), guarded)

    def test_home_nutrition_evidence_is_replaced_in_slide_copy(self):
        text = strategic_bullet_text(
            {"cue_key": "pack_or_spec_detail", "classification": "strength"},
            {"category_key": "home", "product_type_display": "Air Fryers"},
            slide_key="slide4",
            evidence_terms={"detail": "Clear pack and nutritional details"},
        )
        self.assertNotIn("nutriti", text.lower())

    def test_valid_cross_category_language_is_preserved(self):
        cases = {
            "animals": "Pet nutrition detail and pet feeding guidance",
            "baby": "Infant formula, nutrition detail, and feeding guidance",
            "health_general": "Nutrition detail and dosage guidance",
            "sports_outdoors": "Nutrition detail and protein supplements",
            "home": "Cooking recipes, breakfast preparation, and food capacity",
            "household_clean": "Cleaning formula, ingredients, and dilution guidance",
            "garden_patio": "Plant feeding and soil nutrients",
            "toys": "Pretend kitchen and toy food",
        }
        for category, text in cases.items():
            with self.subTest(category=category):
                self.assertEqual(_guard_language(text, {"category_key": category}), text)

    def test_slide4_uses_guards_for_every_scope(self):
        from audit_powerpoint_new import (
            SLIDE4_LANGUAGE_GUARD_CATEGORIES,
            SLIDE4_SCOPE_FALLBACKS,
            _slide4_language_allowed,
        )
        for family in set(SLIDE4_SCOPE_FALLBACKS) - {"generic"}:
            category = SLIDE4_LANGUAGE_GUARD_CATEGORIES.get(family, family)
            with self.subTest(family=family):
                self.assertIn(category, INVALID_LANGUAGE_GUARDS)
                phrase = INVALID_LANGUAGE_GUARDS[category][0][0]
                self.assertFalse(_slide4_language_allowed(phrase, family)[0])
