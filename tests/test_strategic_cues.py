from __future__ import annotations

import unittest

from app.audit_helpers.strategic_cues import (
    aggregate_pdp_cues,
    brand_shop_cue_context,
    search_cue_context,
    translate_cues,
)


class StrategicCueTests(unittest.TestCase):
    def test_food_identity_uses_existing_style_and_image_guides(self) -> None:
        context = aggregate_pdp_cues(
            [
                {
                    "category": "Food/Pantry/Nut Butters & Spreads",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Peanut Butter Spread with Protein",
                    "description_body": "Great for breakfast, snacks, and recipes.",
                    "images": [{"position": 1, "alt_text": "front white background"}],
                    "review_count": 120,
                }
            ]
        )
        identity = context["identity"]
        self.assertEqual(identity["category_key"], "food_beverage")
        self.assertEqual(identity["family_key"], "dips_spreads")
        self.assertEqual(identity["product_type_display"], "Nut Butters & Spreads")
        self.assertIn("config\\style_guides\\food_beverage.json", context["debug"]["guide_files_used"])
        self.assertTrue(context["debug"]["candidate_cues"])

    def test_aggregation_tracks_gap_and_coverage_ratios(self) -> None:
        context = aggregate_pdp_cues(
            [
                {
                    "category": "Food",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Peanut Butter",
                    "description_body": "Protein and nutrition detail for breakfast.",
                    "review_count": 50,
                },
                {
                    "category": "Food",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Plain Spread",
                },
            ]
        )
        cues = {item["cue"]: item for item in context["candidate_cues"]}
        self.assertGreater(cues["review_or_trust_signals"]["gap_ratio"], 0)
        self.assertGreater(cues["ingredient_or_formula_communication"]["coverage_ratio"], 0)
        bullets, debug = translate_cues(
            context,
            slide_key="slide4",
            count=4,
            preferred_order=("strength", "opportunity", "context", "pressure"),
            side="client",
        )
        self.assertEqual(len(bullets), 4)
        self.assertEqual(len(debug), 4)

    def test_search_and_brand_shop_contexts_are_category_aware(self) -> None:
        search_context = search_cue_context(
            "nut butter spread",
            [
                {"title": "Almond Nut Butter", "brand": "Brand A", "reviewCount": 90},
                {"title": "Cashew Nut Butter", "brand": "Brand B", "reviewCount": 25},
            ],
            client_brand="Client Brand",
            side="benchmark",
        )
        self.assertTrue(
            any(item["classification"] == "pressure" for item in search_context["candidate_cues"])
        )

        brand_context = brand_shop_cue_context(
            {
                "role": "Client",
                "modules": [{"type": "HeroPOV", "heading": "Skin Care Routine"}],
                "categoryNavigation": ["Skin Care", "Face Care"],
                "destinationLinks": [{"label": "Skin Care"}, {"label": "Face Care"}],
            }
        )
        self.assertEqual(brand_context["identity"]["category_key"], "beauty")
        self.assertEqual(brand_context["identity"]["family_display"], "Skin Care")

    def test_final_translation_polishes_banned_patterns_and_repeated_starters(self) -> None:
        context = {
            "identity": {
                "category_display": "Food & Beverage",
                "family_display": "Spreads",
                "product_type_display": "Nut Butters & Spreads",
            },
            "candidate_cues": [
                {
                    "cue": "shopper_education",
                    "classification": "opportunity",
                    "label": "shopper education",
                    "coverage_ratio": 0.1,
                    "gap_ratio": 0.9,
                },
                {
                    "cue": "usage_storytelling",
                    "classification": "opportunity",
                    "label": "usage storytelling",
                    "coverage_ratio": 0.1,
                    "gap_ratio": 0.8,
                },
                {
                    "cue": "discoverability",
                    "classification": "pressure",
                    "label": "discovery paths",
                    "coverage_ratio": 0.8,
                    "gap_ratio": 0.1,
                },
                {
                    "cue": "product_positioning",
                    "classification": "pressure",
                    "label": "benchmark cue",
                    "coverage_ratio": 0.7,
                    "gap_ratio": 0.1,
                },
            ],
        }

        bullets, _debug = translate_cues(
            context,
            slide_key="slide3",
            count=4,
            preferred_order=("opportunity", "pressure", "strength", "context"),
            side="benchmark",
        )

        self.assertEqual(len(bullets), 4)
        lower = " ".join(bullets).lower()
        for banned in ("cue", "evidence", "benchmark 2", "secondary benchmark", "competitive pressure"):
            self.assertNotIn(banned, lower)
        starters = [bullet.split(" ", 1)[0].lower() for bullet in bullets]
        self.assertLessEqual(max(starters.count(starter) for starter in starters), 1)


if __name__ == "__main__":
    unittest.main()
