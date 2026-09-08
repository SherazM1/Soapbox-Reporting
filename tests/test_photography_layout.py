import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing import apparel_estimator, draft_ui
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.quote_builder import build_apparel_quote


class PhotographyLayoutTests(unittest.TestCase):
    def test_visual_order_and_existing_data_flow(self):
        client = ClientContact("client-1", None, "Company", "Ada", "Lovelace", "ada@example.com")
        internal = InternalContact("internal-1", "Producer", "Lead", "producer@example.com")

        def contact_management():
            with st.expander("Contact Management"):
                st.caption("Contact management fixture")

        app = AppTest.from_string(
            "from app.photography_pricing.apparel_estimator import render_photography_pricing\n"
            "render_photography_pricing()\n"
        )
        app.session_state["photo_pricing_quote_title"] = "Layout quote"
        app.session_state["photo_pricing_on_model_image_quantity"] = 5
        app.session_state["photo_pricing_comments_project_name_0"] = "Project A"
        app.session_state["photo_pricing_comments_laydown_detail_0"] = 12.0
        expected_quote = build_apparel_quote(ApparelInputs(on_model_image_quantity=5))
        with (
            patch.object(apparel_estimator, "render_contact_management", side_effect=contact_management),
            patch("app.contact_management.contact_ui.safe_list_active_client_contacts", return_value=[client]),
            patch("app.contact_management.contact_ui.safe_list_active_internal_contacts", return_value=[internal]),
            patch.object(draft_ui, "_safe_list_drafts", return_value=[]),
            patch.object(apparel_estimator, "render_drafts_section", wraps=draft_ui.render_drafts_section) as drafts,
            patch("app.photography_pricing.pdf_generator.generate_page2_pricing_pdf", return_value=b"pdf fixture") as generate,
        ):
            app.run()
            self.assertEqual(0, len(app.exception))
            drafts.assert_called_with(selected_client=client, selected_internal=internal)
            elements = list(app.main)

            def position(kind, text):
                return next(i for i, element in enumerate(elements)
                            if element.type == kind and
                            (getattr(element, "label", None) == text or getattr(element, "value", None) == text))

            positions = [
                position("selectbox", "Job Type"),
                position("subheader", "Quote Setup"),
                position("expander", "Contact Management"),
                position("expander", "Drafts"),
                position("selectbox", "Client Contact"),
                position("text_input", "Quote Title"),
                position("subheader", "Page 1 Comments"),
                position("subheader", "Summary"),
                position("button", "Generate PDF"),
                position("subheader", "Pricing Rows"),
            ]
            self.assertEqual(sorted(positions), positions)
            self.assertEqual(1, sum(e.type == "subheader" and e.value == "Summary" for e in elements))
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual("$1,375.00", metrics["Final Total"])
            self.assertEqual("$1,375.00", metrics["Running Subtotal"])
            self.assertEqual("5", metrics["Image Count For Account Management"])
            self.assertEqual(apparel_estimator._line_table_rows(expected_quote.to_payload()),
                             app.dataframe[0].value.to_dict("records"))
            self.assertEqual("Laydown", app.number_input(key="photo_pricing_comments_laydown_detail_0").label)
            app.text_area(key="photo_pricing_comments_custom_notes").set_value("Preserve these notes").run()
            app.button(key="photo_pricing_generate_pdf").click().run()
            self.assertEqual(0, len(app.exception))
            self.assertEqual("Project A", app.text_input(key="photo_pricing_comments_project_name_0").value)
            self.assertEqual("Preserve these notes", app.text_area(key="photo_pricing_comments_custom_notes").value)
            generate.assert_called_once()
            self.assertEqual(expected_quote, generate.call_args.args[0])
            comments = generate.call_args.kwargs["page1_comments_payload"]
            self.assertEqual(app.session_state["photo_pricing_page1_comments_payload"], comments)
            self.assertIn("Laydown=12", comments["rendered_comments_block"])
            self.assertEqual("client-1", generate.call_args.kwargs["page1_header_payload"]["selected_client"]["id"])
            self.assertEqual(b"pdf fixture", app.session_state["photo_pricing_generated_pdf"])


if __name__ == "__main__":
    unittest.main()
