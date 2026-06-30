from __future__ import annotations

import unittest

from app.audit_helpers.strategic_cue_engine import (
    APPROVED_CUE_DEFINITIONS,
    aggregate_strategic_cues,
)
from audit_export import build_audit_export_plan


def _cue(context: dict, cue_key: str) -> dict:
    return next(item for item in context["candidate_cues"] if item["cue_key"] == cue_key)


class StrategicCueEngineTests(unittest.TestCase):
    def test_approved_cue_set_is_complete(self) -> None:
        self.assertEqual(
            set(APPROVED_CUE_DEFINITIONS),
            {
                "product_positioning",
                "benefit_communication",
                "ingredient_or_formula_communication",
                "shopper_education",
                "usage_storytelling",
                "visual_identity",
                "pack_or_spec_detail",
                "keyword_alignment",
                "discoverability",
                "assortment_segmentation",
                "category_grouping",
                "discovery_pathways",
                "cross_category_navigation",
                "review_or_trust_signals",
                "conversion_guidance",
            },
        )

    def test_structured_evidence_generates_strength_candidate(self) -> None:
        context = aggregate_strategic_cues(
            [
                {
                    "record_id": "client-1",
                    "category": "Food/Pantry/Peanut butter & spreads",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Creamy Peanut Butter Spread",
                    "description_body": "Protein benefit and nutrition detail for breakfast snacks.",
                    "key_features": ["Protein", "Gluten free"],
                    "review_count": 120,
                },
                {
                    "record_id": "client-2",
                    "category": "Food/Pantry/Peanut butter & spreads",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Crunchy Peanut Butter Spread",
                    "description_body": "Protein benefit and nutrition detail for recipes.",
                    "key_features": ["Protein"],
                    "review_count": 80,
                },
            ]
        )
        benefit = _cue(context, "benefit_communication")
        self.assertEqual(benefit["classification"], "strength")
        self.assertEqual(benefit["confidence_tier"], 1)
        self.assertGreaterEqual(benefit["coverage_ratio"], 0.55)
        self.assertIn("structured", benefit["evidence_sources"])

    def test_guide_expectation_comparison_tracks_matches_and_gaps(self) -> None:
        context = aggregate_strategic_cues(
            [
                {
                    "record_id": "client-1",
                    "category": "Food/Pantry/Peanut butter & spreads",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Nut Butters & Spreads",
                    "description_body": "",
                    "images": [{"position": 1, "alt_text": "front white background"}],
                }
            ]
        )
        positioning = _cue(context, "product_positioning")
        visual = _cue(context, "visual_identity")
        self.assertEqual(positioning["confidence_tier"], 2)
        self.assertTrue(positioning["matched_guide_rules"])
        self.assertTrue(visual["missing_guide_rules"])
        self.assertIn("guide_expectation", positioning["evidence_sources"])

    def test_group_aggregation_suppresses_single_record_outlier(self) -> None:
        records = [
            {
                "record_id": f"client-{index}",
                "category": "Food",
                "product_type": "Nut Butters & Spreads",
                "product_title": "Plain Spread",
                "description_body": "Pantry spread.",
            }
            for index in range(10)
        ]
        records[0]["description_body"] = "Protein benefit and nutrition detail."
        context = aggregate_strategic_cues(records)
        benefit = _cue(context, "benefit_communication")
        self.assertEqual(benefit["classification"], "opportunity")
        self.assertLessEqual(benefit["coverage_ratio"], 0.2)
        self.assertNotEqual(benefit["classification"], "strength")

    def test_competitive_environment_can_create_pressure_candidate(self) -> None:
        context = aggregate_strategic_cues(
            [
                {
                    "record_id": "client-1",
                    "category": "Food",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Plain Spread",
                }
            ],
            competitor_records=[
                {
                    "record_id": "comp-1",
                    "category": "Food",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Peanut Butter Spread",
                    "review_count": 500,
                    "description_body": "Protein benefit with nutrition detail.",
                },
                {
                    "record_id": "comp-2",
                    "category": "Food",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Almond Butter Spread",
                    "review_count": 450,
                    "description_body": "Protein benefit with breakfast usage.",
                },
            ],
        )
        reviews = _cue(context, "review_or_trust_signals")
        self.assertEqual(reviews["classification"], "pressure")
        self.assertGreaterEqual(reviews["competitive_delta"], 0.35)

    def test_export_plan_includes_backend_cue_debug_only_hook(self) -> None:
        record = {
            "record_id": "client-1",
            "product_title": "Creamy Peanut Butter Spread",
            "item_id": "123",
            "brand": "Client Brand",
            "category": "Food/Pantry/Peanut butter & spreads",
            "product_type": "Nut Butters & Spreads",
            "images": [],
            "description_body": "Protein and nutrition detail for breakfast.",
            "key_features": ["Protein", "Pantry spread"],
        }
        plan = build_audit_export_plan(
            audit_record={"client_name": "Client Brand", "retailer": "Walmart"},
            primary_entries=[
                {
                    "entry_id": "entry-1",
                    "record_id": "client-1",
                    "product_title": record["product_title"],
                    "item_id": record["item_id"],
                    "cached_record": record,
                    "include_in_export": True,
                }
            ],
            competitor_assignments=[],
            competitor_records=[],
        )
        debug = plan["strategic_cue_debug"]
        self.assertEqual(debug["resolved_identity"]["category_key"], "food_beverage")
        self.assertTrue(debug["candidate_cues"])
        candidate = debug["candidate_cues"][0]
        for key in (
            "cue_key",
            "classification",
            "confidence_tier",
            "coverage_ratio",
            "gap_ratio",
            "strength_ratio",
            "consistency",
            "competitive_delta",
            "evidence_sources",
            "matched_guide_rules",
            "missing_guide_rules",
            "slide_objective_tags",
            "debug_reason",
        ):
            self.assertIn(key, candidate)
        self.assertEqual(plan["audit_metadata"]["strategic_cue_debug"], debug)


if __name__ == "__main__":
    unittest.main()
