import unittest
from io import BytesIO

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from app.photography_pricing.draft_service import (
    PROJECT_FIELDS, update_project_rows, serialize_draft_payload, restore_draft_payload_to_state,
)
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.quote_builder import build_apparel_quote
from app.photography_pricing.pdf_generator import generate_page2_pricing_pdf


def project_state():
    state = {"photo_pricing_project_rows": [{}, {}, {}],
             "photo_pricing_comments_custom_notes": "Keep notes",
             "photo_pricing_comments_estimate_subject": "Keep subject",
             "photo_pricing_comments_subtitle_line": "Keep subtitle",
             "photo_pricing_on_model_image_quantity": 7}
    for index, name in enumerate(("Project Alpha", "Project Bravo", "Project Charlie")):
        for offset, field in enumerate(PROJECT_FIELDS):
            state[f"photo_pricing_comments_{field}_{index}"] = name if offset == 0 else float(index * 10 + offset)
    return state


class ProjectRowTests(unittest.TestCase):
    def test_first_middle_last_removal_preserves_all_fields_and_drafts(self):
        for removed in range(3):
            with self.subTest(removed=removed):
                state = project_state()
                original = dict(state)
                state["photo_pricing_comments_on_model_9"] = 999.0
                state["photo_pricing_generated_pdf"] = b"stale"
                state["photo_pricing_page1_comments_payload"] = {"stale": True}
                update_project_rows(state, removed)
                self.assertEqual([{}, {}], state["photo_pricing_project_rows"])
                for new_index, old_index in enumerate(i for i in range(3) if i != removed):
                    for field in PROJECT_FIELDS:
                        self.assertEqual(original[f"photo_pricing_comments_{field}_{old_index}"],
                                         state[f"photo_pricing_comments_{field}_{new_index}"])
                for field in PROJECT_FIELDS:
                    self.assertNotIn(f"photo_pricing_comments_{field}_2", state)
                self.assertNotIn("photo_pricing_comments_on_model_9", state)
                self.assertNotIn("photo_pricing_generated_pdf", state)
                self.assertNotIn("photo_pricing_page1_comments_payload", state)
                for key in ("custom_notes", "estimate_subject", "subtitle_line"):
                    self.assertEqual(original[f"photo_pricing_comments_{key}"], state[f"photo_pricing_comments_{key}"])
                self.assertEqual(7, state["photo_pricing_on_model_image_quantity"])
                saved = serialize_draft_payload(state)
                self.assertEqual(2, len(saved["comments"]["project_entries"]))
                restored = {}
                restore_draft_payload_to_state(saved, restored)
                self.assertEqual(saved["comments"], serialize_draft_payload(restored)["comments"])

    def test_repeated_cycles_and_clean_add(self):
        state = project_state()
        update_project_rows(state, 1)
        update_project_rows(state, 0)
        for _ in range(3):
            state["photo_pricing_comments_project_name_1"] = "obsolete"
            update_project_rows(state)
            for field in PROJECT_FIELDS:
                self.assertEqual("" if field == "project_name" else 0.0,
                                 state[f"photo_pricing_comments_{field}_1"])
            update_project_rows(state, 1)
        self.assertEqual("Project Charlie", state["photo_pricing_comments_project_name_0"])
        before = dict(state)
        update_project_rows(state, 0)
        self.assertEqual(before, state)

    def test_real_widgets_remove_middle_then_add_and_generate_pdf(self):
        app = AppTest.from_string(
            "from app.photography_pricing.apparel_estimator import _render_comments_composer\n"
            "_render_comments_composer({'name': 'Producer'})\n"
        )
        original = project_state()
        for key, value in original.items():
            app.session_state[key] = value
        app.run()
        app.button(key="photo_pricing_comments_remove_1").click().run()
        self.assertEqual(0, len(app.exception))
        for new_index, old_index in enumerate((0, 2)):
            for field in PROJECT_FIELDS:
                self.assertEqual(original[f"photo_pricing_comments_{field}_{old_index}"],
                                 app.session_state[f"photo_pricing_comments_{field}_{new_index}"])
        self.assertEqual("Laydown", app.number_input(key="photo_pricing_comments_laydown_detail_1").label)
        comments = app.session_state["photo_pricing_page1_comments_payload"]
        self.assertEqual(2, comments["project_count"])
        self.assertNotIn("Project Bravo", app.text[0].value)
        self.assertLess(app.text[0].value.index("Project Alpha"), app.text[0].value.index("Project Charlie"))
        pdf = generate_page2_pricing_pdf(build_apparel_quote(ApparelInputs()), page1_comments_payload=comments)
        text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
        self.assertNotIn("Project Bravo", text)
        self.assertIn("Project Charlie", text)
        self.assertIn("2 projects=", text)
        app.button(key="photo_pricing_comments_add_project").click().run()
        self.assertEqual(0, len(app.exception))
        for field in PROJECT_FIELDS:
            self.assertEqual("" if field == "project_name" else 0.0,
                             app.session_state[f"photo_pricing_comments_{field}_2"])
        self.assertEqual(2, app.session_state["photo_pricing_page1_comments_payload"]["project_count"])


if __name__ == "__main__":
    unittest.main()
