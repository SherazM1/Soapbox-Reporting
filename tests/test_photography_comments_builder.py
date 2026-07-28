import unittest

from app.photography_pricing.comments_builder import build_page1_comments_payload


class PhotographyCommentsBuilderTests(unittest.TestCase):
    def test_custom_notes_only_omits_empty_estimate_boilerplate(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={
                "id": "ashley-watson",
                "name": "Ashley Watson",
                "title": "Photography Producer",
                "email": "ashley.watson@soapbox.com",
            },
            estimate_subject="",
            subtitle_line="",
            project_entries=[],
            custom_notes="Please prioritize hero selects.",
        )

        self.assertEqual(payload.project_count, 0)
        self.assertEqual(payload.project_count_label, "")
        self.assertEqual(payload.intro_text, "")
        self.assertEqual(payload.rendered_comments_block, "Comments from Ashley Watson\n\nPlease prioritize hero selects.")
        self.assertNotIn("Photography Estimate:", payload.rendered_comments_block)
        self.assertNotIn("Estimate includes the following projects:", payload.rendered_comments_block)
        self.assertNotIn("0 projects", payload.rendered_comments_block)

    def test_custom_notes_with_empty_default_project_row_omits_estimate_boilerplate(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={"name": "Ashley Watson"},
            estimate_subject="",
            subtitle_line="",
            project_entries=[
                {
                    "project_name": "",
                    "on_model": 0,
                    "laydown_detail": 0,
                    "color_correct": 0,
                    "post": 0,
                    "model_hours": 0,
                }
            ],
            custom_notes="Please prioritize hero selects.",
        )

        self.assertEqual(payload.project_entries, ())
        self.assertEqual(payload.rendered_comments_block, "Comments from Ashley Watson\n\nPlease prioritize hero selects.")
        self.assertNotIn("Photography Estimate:", payload.rendered_comments_block)
        self.assertNotIn("0 projects", payload.rendered_comments_block)

    def test_subject_plus_custom_notes_preserves_full_comments_format(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={"name": "Ashley Watson"},
            estimate_subject="Apparel Refresh",
            subtitle_line="",
            project_entries=[],
            custom_notes="Rush timing requested.",
        )

        self.assertEqual(
            payload.rendered_comments_block,
            "Comments from Ashley Watson\n\n"
            "Photography Estimate for Apparel Refresh:\n"
            "Estimate includes the following projects:\n\n"
            "Rush timing requested.\n\n"
            "0 projects=",
        )

    def test_subtitle_plus_custom_notes_preserves_full_comments_format(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={"name": "Ashley Watson"},
            estimate_subject="",
            subtitle_line="Spring27 - Bangladesh",
            project_entries=[],
            custom_notes="Rush timing requested.",
        )

        self.assertEqual(
            payload.rendered_comments_block,
            "Comments from Ashley Watson\n\n"
            "Photography Estimate:\n"
            "Spring27 - Bangladesh\n"
            "Estimate includes the following projects:\n\n"
            "Rush timing requested.\n\n"
            "0 projects=",
        )

    def test_meaningful_project_plus_custom_notes_preserves_full_comments_format(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={"name": "Ashley Watson"},
            estimate_subject="Sam's Club Kids Apparel Project",
            subtitle_line="Spring27 - Bangladesh",
            project_entries=[
                {
                    "project_name": "Kids Denim",
                    "on_model": 12,
                    "laydown_detail": 0,
                    "color_correct": 3,
                    "post": 0,
                    "model_hours": 4,
                }
            ],
            custom_notes="Rush timing requested.",
        )

        self.assertEqual(
            payload.rendered_comments_block,
            "Comments from Ashley Watson\n\n"
            "Photography Estimate for Sam's Club Kids Apparel Project:\n"
            "Spring27 - Bangladesh\n"
            "Estimate includes the following projects:\n\n"
            "Kids Denim\n"
            "On Model= 12, Color correct: 3, Model hrs= 4\n\n"
            "Rush timing requested.\n\n"
            "1 project=",
        )

    def test_completely_empty_comments_omit_estimate_boilerplate(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={"name": "Ashley Watson"},
            estimate_subject="",
            subtitle_line="",
            project_entries=[],
            custom_notes="",
        )

        self.assertEqual(payload.rendered_comments_block, "Comments from Ashley Watson")
        self.assertEqual(payload.project_count_label, "")
        self.assertNotIn("Photography Estimate:", payload.rendered_comments_block)
        self.assertNotIn("0 projects", payload.rendered_comments_block)

    def test_single_project_rendering_uses_singular_count_and_sparse_details(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={
                "id": "ashley-watson",
                "name": "Ashley Watson",
                "title": "Photography Producer",
                "email": "ashley.watson@soapbox.com",
            },
            estimate_subject="Sam's Club Kids Apparel Project",
            subtitle_line="Spring27 - Bangladesh",
            project_entries=[
                {
                    "project_name": "Kids Denim",
                    "on_model": 12,
                    "laydown_detail": 0,
                    "color_correct": 3,
                    "post": 0,
                    "model_hours": 4,
                }
            ],
            custom_notes="Rush timing requested.",
        )

        self.assertEqual(payload.project_count, 1)
        self.assertEqual(payload.project_count_label, "1 project=")
        self.assertIn("Comments from Ashley Watson", payload.rendered_comments_block)
        self.assertIn("Photography Estimate for Sam's Club Kids Apparel Project:", payload.rendered_comments_block)
        self.assertIn("Spring27 - Bangladesh", payload.rendered_comments_block)
        self.assertIn("Kids Denim", payload.rendered_comments_block)
        self.assertIn("On Model= 12, Color correct: 3, Model hrs= 4", payload.rendered_comments_block)
        self.assertNotIn("Laydown/Detail=0", payload.rendered_comments_block)
        self.assertIn("Rush timing requested.", payload.rendered_comments_block)

    def test_many_project_rendering_uses_plural_count(self) -> None:
        payload = build_page1_comments_payload(
            selected_internal_contact={
                "id": "morgan-lee",
                "name": "Morgan Lee",
                "title": "Creative Operations Manager",
                "email": "morgan.lee@soapbox.com",
            },
            estimate_subject="Apparel Refresh",
            subtitle_line="",
            project_entries=[
                {"project_name": "Project A", "on_model": 2},
                {"project_name": "Project B", "laydown_detail": 5},
            ],
            custom_notes="",
        )

        self.assertEqual(payload.project_count, 2)
        self.assertEqual(payload.project_count_label, "2 projects=")
        self.assertIn("Project A", payload.rendered_comments_block)
        self.assertIn("Project B", payload.rendered_comments_block)
        self.assertTrue(payload.rendered_comments_block.endswith("2 projects="))


if __name__ == "__main__":
    unittest.main()
