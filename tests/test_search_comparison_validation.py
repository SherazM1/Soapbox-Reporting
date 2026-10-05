import unittest

from app.audit_helpers.slide3_search_benchmark import (
    build_slide3_search_benchmark,
    validate_search_comparison,
    validate_search_relevance,
)


class SearchComparisonValidationTests(unittest.TestCase):
    def test_unrelated_captured_searches_block_payload_generation(self):
        evidence = {
            "current": [{"role": "Current", "data": {"searchTerm": "baby formula"}}],
            "benchmark": [{"role": "Benchmark", "data": {"searchTerm": "face wash"}}],
        }
        with self.assertRaisesRegex(ValueError, "baby formula.*face wash"):
            build_slide3_search_benchmark(evidence)

    def test_normalized_query_sets_match(self):
        validate_search_comparison([" Baby Formula ", "toddler formula"],
                                   ["toddler formula", "baby+formula"])

    def test_different_queries_and_query_counts_are_allowed(self):
        for current, benchmark in ((["baby formula"], ["infant nutrition", "toddler formula"]),
                                   (["nut butter spread"], ["low sugar spreads"]),
                                   (["dog food"], ["cat litter"]),
                                   (["unknown brand"], ["another brand"])):
            validate_search_comparison(current, benchmark)
            validate_search_relevance(current, benchmark)

    def test_presence_validator_does_not_judge_relevance(self):
        validate_search_comparison(["baby formula"], ["face wash"])

    def test_missing_terms_are_rejected_on_either_side(self):
        for terms in ([], [""], ["   "], ["category search"], ["baby formula", ""]):
            for current, benchmark in ((terms, ["baby formula"]), (["baby formula"], terms)):
                with self.subTest(current=current, benchmark=benchmark), self.assertRaises(ValueError):
                    validate_search_comparison(current, benchmark)

    def test_known_conflict_is_symmetric_and_checks_all_captures(self):
        for current, benchmark in ((["infant nutrition"], ["gentle facial cleanser"]),
                                   (["face wash"], ["baby food"]),
                                   (["baby formula"], ["infant formula", "face wash"])):
            with self.subTest(current=current), self.assertRaises(ValueError):
                validate_search_relevance(current, benchmark)


if __name__ == "__main__":
    unittest.main()
