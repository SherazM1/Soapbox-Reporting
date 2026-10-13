import unittest
from app.audit_helpers.product_evidence import product_evidence_bullets
from app.audit_helpers.slide2_summary import build_slide2_summary_payload

class ProductEvidenceTest(unittest.TestCase):
    def test_actual_listing_details_are_traceable(self):
        bullets = product_evidence_bullets({"data": {"descriptionBody": "Measures 24 inches. Wipe clean. Includes 2 chairs."}})
        self.assertEqual({b["dimension"] for b in bullets}, {"dimensions", "care", "contents"})
        self.assertTrue(all(b["evidence"] for b in bullets))

    def test_scoring_notes_and_negation_do_not_create_strengths(self):
        self.assertEqual(product_evidence_bullets({"descriptionNotes": "Add compatible with details"}), [])
        self.assertEqual(product_evidence_bullets({"description": "Not compatible with older models"}), [])

    def test_unknown_content_does_not_invent_evidence(self):
        self.assertEqual(product_evidence_bullets({"description": "A great product"}), [])

    def test_multicategory_evidence_works_without_guide_identity(self):
        for description, dimension in [("Battery-operated alarm", "power"),
                                       ("Machine washable fabric", "care"),
                                       ("Compatible with model X", "compatibility")]:
            self.assertEqual(product_evidence_bullets({"description": description})[0]["dimension"], dimension)

    def test_range_does_not_generalize_one_products_attribute(self):
        records = [{"category": "Furniture", "product_type": "Chairs", "description": "Measures 24 inches"},
                   {"category": "Furniture", "product_type": "Chairs", "description": "Wood chair"}]
        bullets = build_slide2_summary_payload(records)["sections"]["consumer_demand"]["bullets"]
        self.assertNotIn("Size details help shoppers assess product fit", bullets)

if __name__ == "__main__":
    unittest.main()
