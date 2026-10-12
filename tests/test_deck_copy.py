import unittest

from app.audit_helpers.deck_copy import simplify_deck_copy


class DeckCopyTest(unittest.TestCase):
    def test_novelty_taxonomy_becomes_simple_toy_language(self):
        self.assertEqual(simplify_deck_copy("Title could identify the novelty and gag toys more clearly"),
                         "Title could identify the toys more clearly")
        self.assertEqual(simplify_deck_copy("Novelty & gag toys details"), "toys details")
        self.assertEqual(simplify_deck_copy("the sensory toys and novelty and gag toys category"),
                         "the sensory toys category")
        self.assertEqual(simplify_deck_copy("LEGO building blocks"), "LEGO building blocks")

    def test_common_jargon_becomes_plain_language(self):
        self.assertEqual(simplify_deck_copy("Clearer PDP guidance can reduce purchase friction"),
                         "Clearer product page guidance can reduce shopping uncertainty")
        self.assertEqual(simplify_deck_copy("Cohesive PDP visual education"),
                         "Images tell a consistent product story")

    def test_missing_evidence_and_qualified_claims_remain_qualified(self):
        for text in ("Limited competitor evidence available for benchmarking",
                     "A Brand Shop could help shoppers explore products",
                     "No Brand Shop could be confirmed"):
            self.assertEqual(simplify_deck_copy(text), text)

    def test_product_names_and_numbers_remain(self):
        self.assertEqual(simplify_deck_copy("DenTek mouthguard has 13 images and a 4.2 rating"),
                         "DenTek mouthguard has 13 images and a 4.2 rating")

    def test_general_summary_stays_high_level(self):
        text = "Strong Walmart PDP patterns educate shoppers through benefit-driven merchandising, routine-based storytelling, and conversion-focused educational content."
        result = simplify_deck_copy(text)
        self.assertEqual(result, "Clear product pages help shoppers understand the product and decide whether it meets their needs.")
        self.assertNotIn("Step", result)

    def test_slide_titles_are_easy_to_scan(self):
        self.assertEqual(simplify_deck_copy("PDP Content Benchmarking"), "Product Page Comparison")
        self.assertEqual(simplify_deck_copy("Digital Shelf Ownership"), "Online Shopping Presence")

    def test_substitution_is_stable_on_repeat_runs(self):
        once = simplify_deck_copy("PDP merchandising uses proof points to support conversion")
        self.assertEqual(simplify_deck_copy(once), once)


if __name__ == "__main__":
    unittest.main()
