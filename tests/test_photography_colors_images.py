import unittest
from io import BytesIO

from pypdf import PdfReader

from app.photography_pricing.comments_builder import build_page1_comments_payload, normalize_project_entry, render_project_detail_line
from app.photography_pricing.draft_service import serialize_draft_payload, restore_draft_payload_to_state, update_project_rows, apparel_inputs_from_draft_payload
from app.photography_pricing.quote_builder import build_apparel_quote
from app.photography_pricing.pdf_generator import generate_page2_pricing_pdf


def comments(entries, total):
    return build_page1_comments_payload(selected_internal_contact={"name": "Producer"},
        estimate_subject="", subtitle_line="", project_entries=entries, custom_notes="", total_images=total)


class ColorsImagesTests(unittest.TestCase):
    def test_colors_only_and_zero_output(self):
        self.assertEqual(0, normalize_project_entry({}).colors)
        self.assertNotIn("Colors=", render_project_detail_line(normalize_project_entry({"colors": 0})))
        payload = comments([{"colors": 3, "color_correct": 2}], 194)
        self.assertIn("Color correct: 2, Colors=3", payload.rendered_comments_block)
        self.assertEqual(1, comments([{"colors": 3}], 0).project_count)
        self.assertTrue(payload.rendered_comments_block.endswith("1 project= 194 images total"))
        self.assertTrue(comments([], 0).rendered_comments_block.endswith("0 projects= 0 images total"))
        self.assertTrue(comments([], 40).rendered_comments_block.endswith("0 projects= 40 images total"))
        self.assertNotIn("194 images total", payload.rendered_comments_block.splitlines())

    def test_colors_draft_roundtrip_reindex_and_no_pricing_effect(self):
        old = {}
        restore_draft_payload_to_state({}, old)
        self.assertEqual(0, old["photo_pricing_comments_colors_0"])
        for removed in range(3):
            state = {"photo_pricing_project_rows": [{}, {}, {}], "photo_pricing_on_model_image_quantity": 5}
            for index in range(3):
                state[f"photo_pricing_comments_colors_{index}"] = float(index + 1)
            before = build_apparel_quote(apparel_inputs_from_draft_payload(serialize_draft_payload(state)))
            update_project_rows(state, removed)
            saved = serialize_draft_payload(state)
            restored = {}
            restore_draft_payload_to_state(saved, restored)
            self.assertEqual([float(i + 1) for i in range(3) if i != removed],
                [restored[f"photo_pricing_comments_colors_{i}"] for i in range(2)])
            self.assertEqual(before, build_apparel_quote(apparel_inputs_from_draft_payload(saved)))
            self.assertNotIn("total_images", saved)

    def test_pdf_continuation_keeps_all_colors_and_quote_total(self):
        entries = [{"project_name": f"Project {index} with a long descriptive name for wrapping",
                    "colors": index + 1, "laydown_detail": 6, "on_model_detail": 2,
                    "color_correct": 3, "post": 4, "model_hours": 5} for index in range(18)]
        quote = build_apparel_quote(apparel_inputs_from_draft_payload({"pricing": {"on_model_image_quantity": 194}}))
        payload = comments(entries, quote.derived_total_image_count)
        reader = PdfReader(BytesIO(generate_page2_pricing_pdf(quote, page1_comments_payload=payload.to_payload())))
        self.assertGreater(len(reader.pages), 4)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for index in range(18):
            self.assertIn(f"Colors={index + 1}", text)
        self.assertIn("18 projects= 194 images total", text)
        self.assertIn("Laydown=6", text)
        self.assertNotIn("Laydown/Detail", text)


if __name__ == "__main__":
    unittest.main()
