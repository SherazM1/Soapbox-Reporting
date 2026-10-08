import unittest

from app.audit_helpers.slide4_findings import CATEGORY_WORDS, _category_family, _category_words
from app.audit_helpers.strategic_identity import STYLE_GUIDE_FILES


class CategoryWordsTest(unittest.TestCase):
    def test_every_supported_category_has_complete_wording(self):
        for key in STYLE_GUIDE_FILES:
            with self.subTest(category=key):
                self.assertIn(key, CATEGORY_WORDS)
                self.assertEqual(set(CATEGORY_WORDS[key]), {"product", "benefit", "usage", "trust"})
                self.assertTrue(all(CATEGORY_WORDS[key].values()))

    def test_supported_departments_are_routed_to_their_wording(self):
        for key in STYLE_GUIDE_FILES:
            expected = {"baby": "baby_care", "food_beverage": "food_generic"}.get(key, key)
            with self.subTest(category=key):
                self.assertEqual(_category_family(key, "", ""), expected)

    def test_explicit_department_wins_over_incidental_title_word(self):
        self.assertEqual(_category_family("Fashion", "T-Shirts", "Baby Blue T-Shirt"), "fashion")
        self.assertEqual(_category_family("Furniture", "Outdoor Chairs", "Garden Chair"), "furniture")

    def test_food_specializations_and_unknown_fallback_remain(self):
        self.assertEqual(_category_family("Food", "Peanut Butter", "Peanut Butter"), "nut_butter_spreads")
        self.assertEqual(_category_family("Food", "Strawberry Preserves", "Strawberry Preserves"), "jam_preserves")
        self.assertEqual(_category_words("unrecognized"), CATEGORY_WORDS["generic"])


if __name__ == "__main__":
    unittest.main()
