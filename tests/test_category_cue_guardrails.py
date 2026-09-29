import unittest

from app.audit_helpers.audit_language_resolver import LANGUAGE_PROFILES
from app.audit_helpers.strategic_cue_engine import (
    APPROVED_CUE_DEFINITIONS,
    CATEGORY_CUE_GUARDRAILS,
    _category_guardrail_multiplier,
)


class CategoryCueGuardrailTests(unittest.TestCase):
    def test_every_category_has_valid_guardrails(self):
        self.assertEqual(set(CATEGORY_CUE_GUARDRAILS), set(LANGUAGE_PROFILES))
        for category, rules in CATEGORY_CUE_GUARDRAILS.items():
            with self.subTest(category=category):
                self.assertTrue(rules["penalized_terms"])
                self.assertTrue(rules["cue_boosts"])
                self.assertLessEqual(set(rules["cue_boosts"]), set(APPROVED_CUE_DEFINITIONS))

    def test_new_categories_boost_relevant_evidence_and_demote_mismatches(self):
        for category, rules in CATEGORY_CUE_GUARDRAILS.items():
            if category in {"beauty", "electronics", "food_beverage"}:
                continue
            identity = {"category_key": category}
            with self.subTest(category=category):
                for cue, terms in rules["cue_boosts"].items():
                    self.assertGreater(_category_guardrail_multiplier(identity, cue, {"features": terms[0]}), 1)
                self.assertLess(_category_guardrail_multiplier(
                    identity, "pack_or_spec_detail", {"features": rules["penalized_terms"][0]}
                ), 1)
                self.assertEqual(_category_guardrail_multiplier(identity, "pack_or_spec_detail", {}), 1)

    def test_valid_food_related_evidence_is_preserved(self):
        for category, evidence in (
            ("home", "cooking breakfast recipes"),
            ("animals", "nutrition detail and feeding"),
            ("baby", "infant formula and nutrition detail"),
            ("health_general", "nutrition detail"),
            ("sports_outdoors", "nutrition detail and protein content"),
            ("garden_patio", "plant feeding and soil nutrients"),
            ("household_clean", "cleaning formula ingredients"),
        ):
            with self.subTest(category=category):
                self.assertGreaterEqual(_category_guardrail_multiplier(
                    {"category_key": category}, "pack_or_spec_detail", {"features": evidence}
                ), 1)
