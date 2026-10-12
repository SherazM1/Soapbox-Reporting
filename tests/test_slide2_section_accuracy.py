import unittest

from app.audit_helpers.slide2_summary import build_slide2_summary_payload, resolve_slide2_phrases


def product(kind="Shoes", category="Fashion", **extra):
    return {"product_title": kind, "product_type": kind, "category": category, **extra}


class Slide2AccuracyTest(unittest.TestCase):
    def test_patio_seating_benefit_is_complete_without_clipped_product_name(self):
        record = product("Outdoor Conversation Sets", "Garden & Patio", description="Cushioned seating")
        line = build_slide2_summary_payload([record])["sections"]["consumer_demand"]["bullets"][0]
        self.assertEqual(line, "Cushioning helps shoppers choose comfortable patio seating")

    def test_health_mouthguard_gets_specific_identity(self):
        from app.audit_helpers.strategic_identity import resolve_strategic_identity
        identity = resolve_strategic_identity([product("Health Products", "Health & Personal Care", product_title="DenTek Professional Fit Dental Guard")])
        self.assertEqual(identity["product_type_display"], "Mouthguards")

    def test_broad_patio_identity_does_not_assume_gardening(self):
        from app.audit_helpers.strategic_identity import resolve_strategic_identity
        from app.audit_helpers.strategic_cues import _category_vocab
        identity = resolve_strategic_identity([product("Patio Products", "Garden & Patio")])
        education = _category_vocab(identity)["education"]
        self.assertNotIn("plant care", education)
        self.assertNotIn("maintenance", education)

    def test_outdoor_furniture_does_not_inherit_plant_care(self):
        from app.audit_helpers.strategic_identity import resolve_strategic_identity
        identity = resolve_strategic_identity([product("Outdoor Dining Chairs", "Garden & Patio")])
        from app.audit_helpers.strategic_cues import _category_vocab
        self.assertEqual(_category_vocab(identity)["education"], "furniture setup and care guidance")
        self.assertIn("furniture setup and care guidance", identity["education_cues"])
        self.assertFalse(any("plant care" in cue for cue in identity["education_cues"]))

    def test_tshirt_line_explains_concrete_shopping_factors(self):
        payload = build_slide2_summary_payload([product("T-Shirts", "Fashion")])
        line = payload["sections"]["consumer_demand"]["bullets"][0]
        self.assertIn("Sizing and fabric", line)
        self.assertNotIn("clarify benefits", line)
        self.assertNotIn(":", line)

    def test_benchmark_openings_are_varied_without_superiority_claims(self):
        payload = build_slide2_summary_payload([product("T-Shirts", "Fashion")],
                    [product("T-Shirts", "Fashion", description="Cotton T-shirts")])
        bullets = payload["sections"]["competitive_benchmark"]["bullets"]
        openings = [b.split()[0].lower() for b in bullets]
        self.assertEqual(len(openings), len(set(openings)))
        self.assertFalse(any(b.startswith("Compare") for b in bullets))

    def test_description_cannot_change_seafood_to_peanut_butter(self):
        fish = product("Canned and Jarred Seafood", "Food", description="Serve with peanut butter sauce; baby meal recipe")
        phrases = resolve_slide2_phrases([fish])
        self.assertNotIn("butter", phrases["category_phrase"])
        self.assertNotEqual(phrases["category_phrase"], "baby care")

    def test_empty_competition_is_limited_without_superiority_claims(self):
        payload = build_slide2_summary_payload([product()])
        benchmark = payload["sections"]["competitive_benchmark"]
        self.assertEqual(benchmark["rating"], "Limited")
        self.assertFalse(any(word in " ".join(benchmark["bullets"]).lower()
                             for word in ("higher", "stronger", "category leaders")))

    def test_competitor_names_alone_cannot_establish_benchmark_depth(self):
        benchmark = build_slide2_summary_payload([product()], [product(), product()])["sections"]["competitive_benchmark"]
        self.assertEqual(benchmark["rating"], "Limited")

    def test_intro_does_not_claim_established_demand_without_reviews(self):
        text = build_slide2_summary_payload([product()])["intro_copy"]
        self.assertIn("insufficient", text)
        self.assertNotIn("established shopper relevance", text)

    def test_unrelated_review_count_cannot_make_demand_strong(self):
        records = [product(averageRating=4.8), product(reviewCount=1000)]
        demand = build_slide2_summary_payload(records)["sections"]["consumer_demand"]
        self.assertEqual(demand["rating"], "Limited")

    def test_review_weighted_rating_reflects_majority_feedback(self):
        records = [product(averageRating=5, reviewCount=1),
                   product(averageRating=3.5, reviewCount=1000)]
        demand = build_slide2_summary_payload(records)["sections"]["consumer_demand"]
        self.assertEqual(demand["rating"], "Limited")

    def test_gap_rating_tracks_evidence(self):
        for gaps, expected in [(0, "Selective"), (1, "Meaningful"), (4, "Significant")]:
            with self.subTest(gaps=gaps):
                section = build_slide2_summary_payload([product(_slide2_gap_count=gaps)])["sections"]["walmart_opportunity"]
                self.assertEqual(section["rating"], expected)

    def test_product_matrix_has_valid_distinct_section_reads(self):
        for kind, category in [("Shoes", "Fashion"), ("Soda Pop", "Food"),
                               ("Canned and Jarred Seafood", "Food"), ("Squishy Toys", "Toys"),
                               ("Baby Wash", "Baby"), ("Laptop", "Electronics"),
                               ("Skin Care", "Beauty"), ("Dog Food", "Animals")]:
            with self.subTest(kind=kind):
                record = product(kind, category)
                payload = build_slide2_summary_payload([record], [product(kind, category)])
                seen = set()
                for section in payload["sections"].values():
                    self.assertGreaterEqual(len(section["bullets"]), 3)
                    for bullet in section["bullets"]:
                        self.assertLessEqual(len(bullet), 64)
                        self.assertNotIn(bullet.lower(), seen)
                        seen.add(bullet.lower())


if __name__ == "__main__":
    unittest.main()
