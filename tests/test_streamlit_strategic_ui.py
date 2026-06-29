import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "streamlitapp.py"


def _source() -> str:
    return STREAMLIT_APP.read_text(encoding="utf-8")


def _function_body(source: str, function_name: str) -> str:
    pattern = rf"^def {re.escape(function_name)}\([^\n]*\)[^\n]*:\n(?P<body>.*?)(?=^def |\Z)"
    match = re.search(pattern, source, flags=re.M | re.S)
    if not match:
        raise AssertionError(f"{function_name} was not found")
    return match.group("body")


class StreamlitStrategicUiTests(unittest.TestCase):
    def test_content_auditing_routes_only_to_combined_strategic_workflow(self) -> None:
        body = _function_body(_source(), "render_content_auditing")

        self.assertIn("render_combined_strategic_audit_upload_v2()", body)
        self.assertIn("render_strategic_evidence_summary_v2()", body)
        self.assertIn('st.session_state["audit_template_version"] = "New Strategic Template"', body)
        self.assertNotIn("st.selectbox(", body)
        self.assertNotIn("render_primary_pdp_upload_v2()", body)
        self.assertNotIn("render_competitor_pdp_upload_v2()", body)
        self.assertNotIn("render_extracted_primary_product_entries_v2()", body)
        self.assertNotIn("render_extracted_competitor_entries_v2()", body)
        self.assertNotIn("render_mocked_audit_results_v2()", body)

    def test_powerpoint_export_always_uses_new_strategic_generator(self) -> None:
        source = _source()
        body = _function_body(source, "render_audit_powerpoint_export_v2")

        self.assertNotIn("from audit_powerpoint import", source)
        self.assertIn('os.path.join("templates", "Audit_Template_New.pptx")', body)
        self.assertIn("generate_new_audit_powerpoint_from_template(", body)
        self.assertNotIn("generate_audit_powerpoint_from_template(", body)
        self.assertNotIn("resolve_audit_template_path", body)
        self.assertNotIn("Current Audit Template", body)
        self.assertNotIn("Template Version", body)

    def test_legacy_manual_control_labels_are_not_in_strategic_renderers(self) -> None:
        source = _source()
        strategic_bodies = "\n".join(
            [
                _function_body(source, "render_content_auditing"),
                _function_body(source, "render_strategic_evidence_summary_v2"),
                _function_body(source, "render_generate_audit_v2"),
                _function_body(source, "render_audit_powerpoint_export_v2"),
            ]
        )

        legacy_labels = [
            "Primary Audit Extract Upload",
            "Upload Primary Audit Extract Sheet",
            "Fallback URL Mode",
            "Select All for Export",
            "Include in Export",
            "Selected primary image preview",
            "Competitor Audit Extract Upload",
            "Competitor Graphics Generation Mode",
            "Select Default 10",
            "Clear All",
            "Ordered competitor image preview",
            "Competitor Graphics Notes",
            "Retail Media Optimizations",
            "Competitor Ad Graphics Notes",
        ]
        for label in legacy_labels:
            self.assertNotIn(label, strategic_bodies)


if __name__ == "__main__":
    unittest.main()
