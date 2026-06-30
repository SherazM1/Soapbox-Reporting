from __future__ import annotations

import unittest

from app.audit_helpers.strategic_identity import (
    build_identity_phrases,
    identity_debug_payload,
    load_normalized_guides,
    normalize_image_story_guide,
    normalize_style_title_guide,
    resolve_category_key,
    resolve_strategic_identity,
)
from audit_export import build_audit_export_plan


class StrategicIdentityTests(unittest.TestCase):
    def test_priority_guide_files_normalize_to_shared_schema(self) -> None:
        for category_key in (
            "food_beverage",
            "beauty",
            "health_personal_care",
            "animals",
            "electronics",
        ):
            with self.subTest(category_key=category_key):
                guides = load_normalized_guides(category_key)
                self.assertEqual(guides["category_key"], category_key)
                self.assertTrue(guides["style"]["families"])
                self.assertTrue(guides["image"]["pages"])
                self.assertTrue(
                    any("config" in path and "style_guides" in path for path in guides["source_guides_used"])
                )
                self.assertTrue(
                    any("config" in path and "image_guides" in path for path in guides["source_guides_used"])
                )

    def test_style_and_image_adapters_expose_cue_source_fields(self) -> None:
        style = normalize_style_title_guide("food_beverage")
        dips = style["families"]["dips_spreads"]
        nut_butters = dips["product_types"]["nut_butters_spreads"]
        self.assertIn("attribute_cues", nut_butters)
        self.assertIn("benefit_cues", nut_butters)
        self.assertIn("recommended_title_priorities", nut_butters)

        image = normalize_image_story_guide("food_beverage")
        page = image["pages"]["savory_dips_spreads_nut_butters_jarred_single"]
        self.assertIn("image_story_cues", page)
        self.assertIn("recommended_visual_priorities", page)
        self.assertIn("Silo, Front", page["recommended_visual_priorities"])

    def test_identity_resolver_outputs_display_safe_phrases(self) -> None:
        identity = resolve_strategic_identity(
            [
                {
                    "category": "Food/Pantry/Peanut butter & spreads",
                    "product_type": "Nut Butters & Spreads",
                    "product_title": "Creamy Peanut Butter Spread with Protein",
                    "description_body": "Protein and nutrition detail for breakfast and snacks.",
                }
            ]
        )
        self.assertEqual(identity["category_key"], "food_beverage")
        self.assertEqual(identity["family_key"], "dips_spreads")
        self.assertEqual(identity["product_type_display"], "Nut Butters & Spreads")
        self.assertIn("Nut Butters & Spreads", identity["combined_category_phrase"])
        self.assertEqual(identity["shopping_context_phrase"], "Nut Butters & Spreads shopping journey")
        self.assertEqual(identity["product_type_focus_phrase"], "Nut Butters & Spreads positioning")
        self.assertTrue(identity["source_guides_used"])
        debug = identity_debug_payload(identity)
        self.assertIn("recommended_visual_priorities", debug)
        self.assertNotIn("normalized_guides", debug)

    def test_category_aliases_and_phrase_builder_are_stable(self) -> None:
        self.assertEqual(resolve_category_key("Health & Wellness vitamins"), "health_personal_care")
        phrases = build_identity_phrases(
            category_display="Beauty",
            family_display="Skin Care",
            product_type_display="Face Moisturizers",
        )
        self.assertEqual(phrases["combined_category_phrase"], "the Skin Care and Face Moisturizers category")
        self.assertEqual(phrases["combined_category_phrase_alt"], "the Face Moisturizers segment within Beauty")
        self.assertEqual(phrases["shopping_context_phrase"], "Face Moisturizers shopping journey")

    def test_export_plan_includes_debug_only_strategic_identity_hook(self) -> None:
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

        debug = plan["strategic_identity_debug"]
        self.assertEqual(debug["category_key"], "food_beverage")
        self.assertEqual(debug["family_key"], "dips_spreads")
        self.assertIn("source_guides_used", debug)
        self.assertEqual(plan["audit_metadata"]["strategic_identity_debug"], debug)


if __name__ == "__main__":
    unittest.main()
