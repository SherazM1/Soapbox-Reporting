import unittest

from app.audit_helpers.strategic_identity import STYLE_GUIDE_FILES, resolve_category_key


class StrategicCategoryMatchingTests(unittest.TestCase):
    def test_canonical_categories_remain_distinct(self):
        for category in STYLE_GUIDE_FILES:
            with self.subTest(category=category):
                self.assertEqual(resolve_category_key(category), category)

    def test_specific_product_phrases(self):
        cases = {
            "digital camera": "photography",
            "camera bag": "photography",
            "dog food": "animals",
            "baby food": "baby",
            "office chair": "furniture",
            "toy car": "toys",
            "computer keyboard": "electronics",
            "musical keyboard": "musical_instruments",
            "medical equipment": "health_general",
            "Health & Wellness vitamins": "health_personal_care",
            "notebooks": "office_stationery",
            "bookcases": "furniture",
            "scrapbooking": "arts_crafts",
        }
        for text, category in cases.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_category_key(text), category)

    def test_partial_words_remain_unknown(self):
        for text in (None, "", "education", "catalog", "tired"):
            with self.subTest(text=text):
                self.assertEqual(resolve_category_key(text), "")

    def test_multiple_categories_use_first_matching_condition(self):
        self.assertEqual(resolve_category_key("camera and shampoo"), "photography")

    def test_category_aliases(self):
        cases = {
            "Arts & Crafts": "arts_crafts",
            "Office & Stationery": "office_stationery",
            "vehicle parts and accessories": "vehicle",
            "Electronics & Photography": "electronics",
            "photographyimg.json": "photography",
            "Home Improvement": "home_improvement",
            "Health General": "health_general",
        }
        for text, category in cases.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_category_key(text), category)


if __name__ == "__main__":
    unittest.main()
