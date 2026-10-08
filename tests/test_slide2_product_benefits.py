import unittest

from app.audit_helpers.slide2_summary import build_slide2_summary_payload


def record(kind, title, benefits=None):
    return {"category": "Food", "product_type": kind, "product_title": title,
            "benefitTerms": benefits or [],
            "reviews_summary": {"average_rating": 4.6, "ratings_count": 120}}


class ProductBenefitsTest(unittest.TestCase):
    def line(self, records, competitors=None):
        payload = build_slide2_summary_payload(records, competitors)
        bullets = payload["sections"]["consumer_demand"]["bullet_debug"]
        matches = [b for b in bullets if b["template_id"] == "product_specific_benefits"]
        self.assertEqual(len(matches), 1)
        return matches[0]["text"]

    def test_seafood_retains_complete_identity_and_client_benefit(self):
        text = self.line([record("Canned and Jarred Seafood", "Canned Salmon", ["protein"])])
        self.assertIn("seafood", text.lower())
        self.assertIn("protein", text.lower())
        self.assertNotIn("peanut", text.lower())

    def test_different_products_have_different_lines(self):
        fish = self.line([record("Canned and Jarred Seafood", "Canned Salmon")])
        soda = self.line([record("Soda Pop", "Orange Soda")])
        self.assertNotEqual(fish, soda)

    def test_same_type_uses_distinct_observed_benefits(self):
        first = self.line([record("Soda Pop", "Orange Soda", ["refreshing flavor"])])
        second = self.line([record("Soda Pop", "Diet Cola", ["zero sugar"])])
        self.assertNotEqual(first, second)

    def test_competitor_benefits_do_not_leak(self):
        text = self.line([record("Canned and Jarred Seafood", "Canned Salmon")],
                         [record("Peanut Butter", "Peanut Butter", ["creamy texture"])])
        self.assertNotIn("creamy", text)
        self.assertNotIn("peanut", text.lower())

    def test_range_does_not_generalize_one_products_benefit(self):
        text = self.line([record("Soda Pop", "Orange Soda", ["zero sugar"]),
                          record("Canned and Jarred Seafood", "Canned Salmon", ["protein"])])
        self.assertIn("product range", text)
        self.assertNotIn("zero sugar", text)
        self.assertNotIn("protein", text)

    def test_shoe_description_supplies_benefit_without_extracted_terms(self):
        shoe = record("Shoes", "Men's Running Shoes")
        shoe["category"] = "Fashion"
        shoe["description"] = "Breathable mesh with a cushioned sole"
        text = self.line([shoe])
        self.assertIn("breathability", text.lower())
        self.assertIn("shoes", text.lower())

    def test_shoe_fallback_is_about_sizing_and_use(self):
        shoe = record("Shoes", "Running Shoes")
        shoe["category"] = "Fashion"
        text = self.line([shoe])
        self.assertIn("sizing", text)
        self.assertNotIn("clarify benefits", text)

    def test_shoes_do_not_keep_generic_benefit_led_bullet(self):
        shoe = record("Shoes", "Running Shoes")
        shoe["category"] = "Fashion"
        shoe["reviews_summary"] = {}
        competitor = record("Shoes", "No Boundaries Shoes", ["comfort"])
        competitor["category"] = "Fashion"
        bullets = build_slide2_summary_payload([shoe], [competitor])["sections"]["consumer_demand"]["bullets"]
        self.assertTrue(any("sizing" in b for b in bullets))
        self.assertFalse(any("benefit-led" in b.lower() for b in bullets))

    def test_negated_feature_is_not_promoted_as_benefit(self):
        shoe = record("Shoes", "Running Shoes")
        shoe["category"] = "Fashion"
        shoe["description"] = "These shoes are not waterproof"
        text = self.line([shoe])
        self.assertNotIn("waterproofing", text)


if __name__ == "__main__":
    unittest.main()
