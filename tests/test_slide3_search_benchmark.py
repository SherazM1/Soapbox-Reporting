from __future__ import annotations

import io
import unittest
from pathlib import Path

from pptx import Presentation

from app.audit_helpers.slide3_search_benchmark import build_slide3_search_benchmark
from audit_powerpoint_new import generate_new_audit_powerpoint_from_template

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "Audit_Template_New.pptx"


class Slide3SearchBenchmarkTests(unittest.TestCase):
    def test_build_slide3_search_benchmark_selects_earliest_valid_evidence(self) -> None:
        payload = build_slide3_search_benchmark(
            {
                "current": [
                    {
                        "role": "Current",
                        "sourceRow": 7,
                        "searchTerm": "nut butter spread",
                        "screenshotDataUrl": "data:image/jpeg;base64,first",
                        "orderedMainResultProducts": [
                            {"position": 1, "title": "Client Brand Nut Butter", "brand": "Client Brand", "sponsored": False},
                            {"position": 2, "title": "Natural Peanut Butter", "brand": "Natural Co", "sponsored": True},
                        ],
                    },
                    {
                        "role": "Current",
                        "sourceRow": 5,
                        "searchTerm": "nut butter spread",
                        "screenshotDataUrl": "data:image/jpeg;base64,selected",
                        "orderedMainResultProducts": [
                            {"position": 1, "title": "Client Brand Nut Butter", "brand": "Client Brand", "sponsored": False},
                            {"position": 2, "title": "Natural Peanut Butter", "brand": "Natural Co", "sponsored": True},
                        ],
                    },
                ],
                "benchmark": [
                    {
                        "role": "Benchmark",
                        "sourceRow": 6,
                        "searchTerm": "low sugar spreads",
                        "screenshotDataUrl": "data:image/jpeg;base64,benchmark",
                        "orderedMainResultProducts": [
                            {"position": 1, "title": "Low Sugar Fruit Spread", "brand": "Brand A", "sponsored": False},
                            {"position": 2, "title": "Organic Jam", "brand": "Brand B", "sponsored": True},
                        ],
                    }
                ],
                "all": [],
            },
            client_name="Client Brand",
        )
        self.assertEqual(payload["current"]["source_row"], 5)
        self.assertEqual(payload["current"]["search_term"], "nut butter spread")
        self.assertEqual(payload["current"]["category_phrase"], "nut butter spreads")
        self.assertEqual(payload["benchmark"]["source_row"], 6)
        self.assertEqual(payload["benchmark"]["search_term"], "low sugar spreads")
        self.assertEqual(payload["benchmark"]["category_phrase"], "low sugar spreads")
        self.assertEqual(len(payload["current"]["bullets"]), 5)
        self.assertEqual(len(payload["benchmark"]["bullets"]), 5)
        self.assertTrue(any("selected row" in warning.lower() for warning in payload["warnings"]))

    def test_search_term_fallbacks_to_url_query_and_label(self) -> None:
        payload = build_slide3_search_benchmark(
            {
                "current": [
                    {
                        "role": "Current",
                        "sourceRow": 1,
                        "URL": "https://example.test/search?q=strawberry%20jam",
                        "label": "strawberry jam",
                        "orderedMainResultProducts": [],
                    }
                ],
                "benchmark": [
                    {
                        "role": "Benchmark",
                        "sourceRow": 2,
                        "URL": "https://example.test/search?query=low%20sugar%20fruit%20spread",
                        "orderedMainResultProducts": [],
                    }
                ],
                "all": [],
            },
            client_name="Client Brand",
        )
        self.assertEqual(payload["current"]["search_term"], "strawberry jam")
        self.assertEqual(payload["current"]["category_phrase"], "strawberry jam")
        self.assertEqual(payload["benchmark"]["search_term"], "low sugar fruit spread")
        self.assertEqual(payload["benchmark"]["category_phrase"], "low sugar fruit spreads")

    @unittest.skipUnless(TEMPLATE.exists(), "New strategic template is unavailable")
    def test_slide3_generation_populates_screenshots_and_text_without_changing_constants(self) -> None:
        payload = build_slide3_search_benchmark(
            {
                "current": [
                    {
                        "role": "Current",
                        "sourceRow": 3,
                        "searchTerm": "nut butter spread",
                        "screenshotDataUrl": "data:image/jpeg;base64,left",
                        "orderedMainResultProducts": [
                            {"position": 1, "title": "Client Brand Nut Butter", "brand": "Client Brand", "reviewCount": 120, "sponsored": False},
                        ],
                    }
                ],
                "benchmark": [
                    {
                        "role": "Benchmark",
                        "sourceRow": 4,
                        "searchTerm": "low sugar spreads",
                        "screenshotDataUrl": "data:image/jpeg;base64,right",
                        "orderedMainResultProducts": [
                            {"position": 1, "title": "Low Sugar Fruit Spread", "brand": "Brand A", "reviewCount": 90, "sponsored": True},
                        ],
                    }
                ],
                "all": [],
            },
            client_name="Client Brand",
        )
        deck_bytes = generate_new_audit_powerpoint_from_template(
            export_plan={
                "audit_metadata": {"client_name": "Client Brand", "client_company_name": "Client Brand"},
                "slide3_search_benchmark": payload,
            },
            template_path=str(TEMPLATE),
            include_slide_9=True,
        )
        presentation = Presentation(io.BytesIO(deck_bytes))
        slide3 = next(
            slide
            for slide in presentation.slides
            if any(
                "Search & Discoverability Benchmarking" in (shape.text or "")
                for shape in slide.shapes
                if getattr(shape, "has_text_frame", False)
            )
        )
        texts = [shape.text for shape in slide3.shapes if getattr(shape, "has_text_frame", False)]
        self.assertTrue(any('“nut butter spread”' in text for text in texts))
        self.assertTrue(any('“low sugar spreads”' in text for text in texts))
        self.assertTrue(any("nut butter spreads" in text for text in texts))
        self.assertTrue(any("low sugar spreads" in text for text in texts))
        self.assertTrue(any("Search & Discoverability Benchmarking" in text for text in texts))
        self.assertTrue(any("Current Visibility Structure" in text for text in texts))
        self.assertTrue(any("Competitive Search Benchmark" in text for text in texts))
        self.assertTrue(any("Competitive Walmart search environments" in text for text in texts))
        pictures = [shape for shape in slide3.shapes if hasattr(shape, "image")]
        self.assertEqual(len(pictures), 2)


if __name__ == "__main__":
    unittest.main()
