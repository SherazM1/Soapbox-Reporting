import unittest

from app.photography_pricing.comments_builder import build_page1_comments_payload


class PhotographyCommentsBuilderTests(unittest.TestCase):
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
