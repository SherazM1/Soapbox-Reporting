from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pptx import Presentation

from app.audit_helpers.image_guides import get_image_guide_page
from audit_export import build_audit_export_plan
from audit_powerpoint_new import (
    build_slide4_pdp_benchmark_payload,
    generate_new_audit_powerpoint_from_template,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "Audit_Template_New.pptx"


def _walk_shapes(shapes):
    for shape in shapes:
        yield shape
        if hasattr(shape, "shapes"):
            yield from _walk_shapes(shape.shapes)


def _make_images(directory: Path, prefix: str, count: int, size: tuple[int, int]) -> list[dict]:
    images = []
    for index in range(count):
        path = directory / f"{prefix}-{index + 1}.png"
        Image.new(
            "RGB",
            size,
            color=((index * 31) % 255, (index * 53) % 255, (index * 79) % 255),
        ).save(path)
        images.append(
            {
                "index": index,
                "url": path.resolve().as_uri(),
                "is_hero": index == 0,
                "width": 2200,
                "height": 2200,
                "dimensions": "2200 x 2200",
                "dimensions_text": "2200 x 2200",
            }
        )
    return images


def _record(brand: str, images: list[dict], record_id: str) -> dict:
    return {
        "record_id": record_id,
        "brand": brand,
        "product_title": f"{brand} Nut Butter",
        "item_id": record_id,
        "source_url": f"https://example.test/{record_id}",
        "category": "Food/Pantry/Peanut butter & spreads",
        "subcategory": "Nut Butters & Spreads",
        "images": images,
        "image_count": len(images),
        "ingest_metadata": {"content_score": "95"},
        "reviews_summary": {},
    }


class Slide4IntegrationTest(unittest.TestCase):
    def test_slide4_uses_full_carousels_and_image_guide_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            client_record = _record(
                "Client Brand", _make_images(temp_dir, "client", 10, (800, 800)), "client-1"
            )
            competitor_1 = _record(
                "Competitor Alpha", _make_images(temp_dir, "comp1", 7, (800, 600)), "comp-1"
            )
            competitor_2 = _record(
                "Competitor Beta", _make_images(temp_dir, "comp2", 6, (600, 800)), "comp-2"
            )
            primary_entry = {
                "entry_id": "entry-client-1",
                "record_id": "client-1",
                "product_title": client_record["product_title"],
                "item_id": client_record["item_id"],
                "cached_record": client_record,
                "selected_primary_image": {
                    "record_id": "client-1",
                    "image_index": 0,
                    "url": client_record["images"][0]["url"],
                },
                "selected_primary_images": [
                    {
                        "record_id": "client-1",
                        "image_index": 0,
                        "url": client_record["images"][0]["url"],
                    }
                ],
                "include_in_export": True,
            }
            plan = build_audit_export_plan(
                audit_record={
                    "client_name": "Client Company",
                    "retailer": "Walmart",
                    "audit_date": "2026-06-23",
                    "status": "generated_mvp",
                },
                primary_entries=[primary_entry],
                competitor_assignments=[],
                competitor_records=[competitor_1, competitor_2],
            )

            pdp_slide = plan["product_slide_pairs"][0]["pdp_slide"]
            self.assertEqual(len(pdp_slide["ordered_images"]), 10)
            self.assertEqual(pdp_slide["ordered_images"][0]["index"], 0)
            self.assertEqual(pdp_slide["ordered_images"][-1]["index"], 9)
            self.assertEqual(pdp_slide["brand"], "Client Brand")
            self.assertEqual(
                pdp_slide["category"],
                "Food/Pantry/Peanut butter & spreads",
            )
            self.assertEqual(pdp_slide["product_type"], "Nut Butters & Spreads")
            self.assertIn("style_guide_match", pdp_slide)
            self.assertEqual(pdp_slide["ingest_metadata"], {"content_score": "95"})
            self.assertEqual(
                pdp_slide["ordered_images"][0],
                {
                    "index": 0,
                    "url": client_record["images"][0]["url"],
                    "is_hero": True,
                    "width": 2200,
                    "height": 2200,
                    "dimensions": "2200 x 2200",
                    "dimensions_text": "2200 x 2200",
                },
            )

            guide_page = get_image_guide_page(
                "Food/Pantry/Peanut butter & spreads", "Nut Butters & Spreads"
            )
            self.assertIsNotNone(guide_page)
            self.assertEqual(
                guide_page["page_key"],
                "savory_dips_spreads_nut_butters_jarred_single",
            )
            self.assertEqual(
                guide_page["required_slots"],
                [
                    "silo_front",
                    "graphic_ingredients",
                    "silo_alt_in_pack",
                    "lifestyle_in_use",
                    "graphic_guarantee",
                    "feature_graphic",
                    "dimensions",
                    "graphic_nutrition",
                ],
            )

            payload = build_slide4_pdp_benchmark_payload(
                plan, competitor_records=[competitor_1, competitor_2]
            )
            self.assertEqual(
                [column["label"] for column in payload["columns"]],
                ["Client Company", "Competitor Alpha", "Competitor Beta"],
            )
            self.assertEqual(
                [len(column["ordered_images"]) for column in payload["columns"]],
                [10, 7, 6],
            )
            self.assertEqual(
                payload["columns"][0]["image_guide_match"]["required_slots"],
                guide_page["required_slots"],
            )

            deck_bytes = generate_new_audit_powerpoint_from_template(
                export_plan=plan,
                template_path=str(TEMPLATE),
                include_slide_9=True,
                competitor_records=[competitor_1, competitor_2],
            )
            presentation = Presentation(io.BytesIO(deck_bytes))
            deck_text = "\n".join(
                shape.text
                for slide in presentation.slides
                for shape in _walk_shapes(slide.shapes)
                if getattr(shape, "has_text_frame", False)
            )
            self.assertNotIn("{{", deck_text)
            self.assertTrue(
                any(
                    any(
                        "Walmart Cash Program Visibility" in (shape.text or "")
                        for shape in _walk_shapes(slide.shapes)
                        if getattr(shape, "has_text_frame", False)
                    )
                    for slide in presentation.slides
                )
            )
            slide4 = next(
                slide
                for slide in presentation.slides
                if any(
                    "PDP Content Benchmarking" in (shape.text or "")
                    for shape in slide.shapes
                    if getattr(shape, "has_text_frame", False)
                )
            )
            all_text = "\n".join(
                shape.text
                for shape in _walk_shapes(slide4.shapes)
                if getattr(shape, "has_text_frame", False)
            )
            for expected in ("Client Company", "Competitor Alpha", "Competitor Beta"):
                self.assertIn(expected, all_text)
            for sample in ("Honest", "CeraVe", "Jergens"):
                self.assertNotIn(sample, all_text)
            self.assertNotIn("{{", all_text)
            self.assertIn("Carousel: 10 ordered images", all_text)
            self.assertIn("Dimensions: 2200 x 2200 throughout", all_text)
            self.assertIn("Recommended opening:", all_text)
            self.assertIn("Guide opportunity: Solutions Graphic", all_text)

            pictures = [shape for shape in slide4.shapes if hasattr(shape, "image")]
            self.assertEqual(len(pictures), 23)
            intro_bottom = max(
                shape.top + shape.height
                for shape in slide4.shapes
                if getattr(shape, "has_text_frame", False)
                and "Best-in-class Walmart PDPs" in (shape.text or "")
            )
            bullet_shapes = sorted(
                (
                    shape
                    for shape in slide4.shapes
                    if getattr(shape, "has_text_frame", False)
                    and "Carousel:" in (shape.text or "")
                ),
                key=lambda shape: shape.left,
            )
            self.assertEqual(len(bullet_shapes), 3)
            bullet_centers = [
                (
                    shape.left + shape.width / 2,
                    shape.top,
                )
                for shape in bullet_shapes
            ]
            for picture in pictures:
                self.assertGreaterEqual(picture.left, 0)
                self.assertGreaterEqual(picture.top, intro_bottom)
                self.assertLessEqual(picture.left + picture.width, presentation.slide_width)
                self.assertLessEqual(picture.top + picture.height, presentation.slide_height)
                picture_center = picture.left + picture.width / 2
                column_index = min(
                    range(3),
                    key=lambda index: abs(picture_center - bullet_centers[index][0]),
                )
                self.assertLessEqual(
                    picture.top + picture.height,
                    bullet_centers[column_index][1],
                )
                expected_ratio = (1.0, 4 / 3, 3 / 4)[column_index]
                self.assertAlmostEqual(
                    picture.width / picture.height,
                    expected_ratio,
                    places=2,
                )

            without_slide_9 = generate_new_audit_powerpoint_from_template(
                export_plan=plan,
                template_path=str(TEMPLATE),
                include_slide_9=False,
                competitor_records=[competitor_1, competitor_2],
            )
            presentation_without_slide_9 = Presentation(io.BytesIO(without_slide_9))
            self.assertFalse(
                any(
                    any(
                        "Walmart Cash Program Visibility" in (shape.text or "")
                        for shape in _walk_shapes(slide.shapes)
                        if getattr(shape, "has_text_frame", False)
                    )
                    for slide in presentation_without_slide_9.slides
                )
            )

    def test_missing_second_competitor_is_not_duplicated(self) -> None:
        plan = {
            "audit_metadata": {"client_name": "Client Company"},
            "product_slide_pairs": [{"pdp_slide": {"brand": "Client Brand", "ordered_images": []}}],
        }
        payload = build_slide4_pdp_benchmark_payload(
            plan,
            competitor_records=[{"brand": "Only Competitor", "images": []}],
        )
        self.assertEqual(payload["columns"][1]["label"], "Only Competitor")
        self.assertEqual(payload["columns"][2]["label"], "Competitor 2")
        self.assertEqual(payload["columns"][2]["ordered_images"], [])
        self.assertEqual(payload["columns"][2]["bullets"], [])


if __name__ == "__main__":
    unittest.main()
