from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from app.audit_helpers.combined_audit_extract import (
    attach_combined_evidence_to_record,
    combined_pdp_to_dataframe,
    parse_combined_audit_html,
    reset_combined_audit_state,
)
from audit_export import build_audit_export_plan
from audit_helpers import (
    initialize_auditing_session_state,
    parse_audit_extract_upload_to_dataframe,
    process_competitor_audit_extract_sheet,
    process_primary_audit_extract_sheet,
)
from streamlitapp import _process_combined_pdp_records_v2


class _Upload(io.BytesIO):
    name = "combined-audit.html"


def _html(payload: dict) -> _Upload:
    encoded = json.dumps(payload).replace("</script>", "<\\/script>")
    return _Upload(
        (
            "<html><body>"
            "<table><tr><td>Visible legacy text must not be parsed</td></tr></table>"
            f'<script id="soapbox-audit-data" type="application/json">{encoded}</script>'
            "</body></html>"
        ).encode("utf-8")
    )


def _pdp(role: str, row: int, title: str, brand: str) -> dict:
    return {
        "sourceRow": row,
        "role": role,
        "inputBrandName": brand,
        "productUrl": f"https://example.com/{row}",
        "productId": f"item-{row}",
        "productTitle": title,
        "brand": brand,
        "resolvedCategory": "Jams and Fruit Spreads",
        "resolvedFamily": "Food",
        "resolvedProductType": "Fruit Spread",
        "description": f"{title} description",
        "descriptionBullets": ["Breakfast use", "Fruit ingredients"],
        "keyFeatures": ["Organic fruit", "Reduced sugar"],
        "images": [
            {"url": f"data:image/png;base64,ROW{row}A"},
            {"url": f"https://images.example.com/{row}-2.png"},
        ],
        "seller": "Example Seller",
        "soldByWalmart": row % 2 == 0,
        "shippedByWalmart": True,
        "enhancedBrandContentStatus": "present",
    }


def _valid_payload() -> dict:
    return {
        "schemaVersion": "2.0",
        "pdpEvidence": [
            _pdp("Client", 4, "Client Jam One", "Client Brand"),
            _pdp("Competitor", 7, "Competitor Jam One", "Competitor Brand"),
            _pdp("Client", 9, "Client Jam Two", "Client Brand"),
            _pdp("Competitor", 12, "Competitor Jam Two", "Competitor Brand"),
        ],
        "searchEvidence": [
            {
                "sourceRow": 15,
                "role": "Current",
                "searchTerm": "strawberry jam",
                "resultCount": 42,
                "products": [{"title": "A"}],
                "sponsoredProductsDetected": True,
                "screenshotDataUrl": "data:image/png;base64,CURRENT",
                "extractionStatus": "success",
            },
            {
                "sourceRow": 16,
                "role": "Benchmark",
                "searchTerm": "fruit spread",
                "resultCount": 38,
                "products": [{"title": "B"}, {"title": "C"}],
                "screenshotDataUrl": "data:image/png;base64,BENCHMARK",
                "extractionStatus": "partial",
            },
        ],
        "brandShopEvidence": [
            {
                "sourceRow": 18,
                "role": "Client",
                "inputBrandName": "Client Brand",
                "modules": [{"type": "hero"}, {"type": "product-carousel"}],
                "categoryNavigation": ["Jam"],
                "videoPresent": True,
                "products": [{"title": "Client Jam"}],
                "screenshotDataUrl": "data:image/png;base64,CLIENTSHOP",
                "extractionStatus": "success",
            },
            {
                "sourceRow": 19,
                "role": "Competitor",
                "inputBrandName": "Competitor Brand",
                "modules": [{"type": "hero"}],
                "categoryNavigation": ["Spreads"],
                "videoPresent": False,
                "products": [{"title": "Competitor Jam"}],
                "screenshotDataUrl": "data:image/png;base64,COMPSHOP",
                "extractionStatus": "success",
            },
        ],
        "warnings": [{"source": "report", "message": "Top-level warning"}],
        "errors": [],
    }


def _nested_schema_payload() -> dict:
    def pdp(role: str, row: int, product_id: str, title: str) -> dict:
        return {
            "sourceRow": row,
            "evidenceType": "PDP",
            "role": role,
            "originalRole": role,
            "inputBrandName": "Nutella" if role == "Client" else "Jif",
            "brandName": "Nutella" if role == "Client" else "Jif",
            "status": "success",
            "warnings": [],
            "errors": [],
            "data": {
                "url": f"https://example.com/{product_id}",
                "productId": product_id,
                "productTitle": title,
                "brand": "Nutella" if role == "Client" else "Jif",
                "categoryPathName": "Food / Nut Butters and Spreads",
                "productType": "Nut Butters & Spreads",
                "imageCount": 2,
                "images": [
                    {
                        "index": 5,
                        "url": f"https://images.example.com/{product_id}-1.jpg",
                        "width": 2200,
                        "height": 2200,
                    },
                    {
                        "index": 7,
                        "url": f"https://images.example.com/{product_id}-2.jpg",
                        "width": 1800,
                        "height": 1200,
                    },
                ],
                "descriptionBody": f"{title} description",
                "descriptionBullets": [],
                "keyFeatures": ["Smooth spread", "Pantry staple"],
                "averageRating": 4.7,
                "ratingsCount": 41000,
                "reviewCount": 39525,
                "sellerName": "Walmart.com",
                "soldByWalmart": True,
                "shippedByWalmart": True,
                "enhancedBrandContentPresent": True,
            },
        }

    def search(role: str, row: int, term: str) -> dict:
        return {
            "sourceRow": row,
            "role": role,
            "originalRole": role,
            "status": "success",
            "warnings": [],
            "errors": [],
            "data": {
                "url": f"https://walmart.com/search?q={term}",
                "searchTerm": term,
                "resultCount": 120,
                "productsCaptured": 50,
                "sponsoredProductsDetected": True,
                "topBrands": ["Nutella", "Jif"],
                "products": [{"title": "Product A"}],
                "screenshotDataUrl": f"data:image/png;base64,{role.upper()}SEARCH",
            },
        }

    def brand_shop(role: str, row: int, brand: str) -> dict:
        return {
            "sourceRow": row,
            "role": role,
            "originalRole": role,
            "inputBrandName": brand,
            "status": "success",
            "warnings": [],
            "errors": [],
            "data": {
                "url": f"https://walmart.com/brand/{brand.lower()}",
                "brandName": brand,
                "moduleCount": 2,
                "moduleTypes": ["HeroPov", "ItemCarousel"],
                "productCount": 34,
                "videoPresent": False,
                "screenshotDataUrl": f"data:image/png;base64,{role.upper()}SHOP",
                "modules": [
                    {"type": "HeroPov", "heading": f"{brand} hero"},
                    {"type": "ItemCarousel"},
                ],
            },
        }

    return {
        "schemaVersion": "2.0",
        "pdpEvidence": [
            pdp("Client", 2, "10451273", "Nutella Hazelnut Spread"),
            pdp("Competitor", 3, "20000001", "Jif Peanut Butter"),
        ],
        "searchEvidence": [
            search("Current", 4, "hazelnut spread"),
            search("Benchmark", 5, "peanut butter"),
        ],
        "brandShopEvidence": [
            brand_shop("Client", 6, "Nutella"),
            brand_shop("Competitor", 7, "Jif"),
        ],
        "warnings": [],
        "errors": [],
    }


class CombinedAuditExtractTests(unittest.TestCase):
    def test_brand_shop_control_defaults_checked_and_reaches_export_metadata(self) -> None:
        state: dict = {}
        initialize_auditing_session_state(state)
        self.assertTrue(state["audit_client_has_brand_shop"])
        plan = build_audit_export_plan(
            audit_record={
                "client_name": "Client",
                "client_has_brand_shop": False,
            },
            primary_entries=[],
            competitor_assignments=[],
            competitor_records=[],
            brand_shop_evidence={"client": [], "competitor": [], "all": []},
        )
        self.assertFalse(plan["audit_metadata"]["client_has_brand_shop"])
        self.assertEqual(plan["slide5_brand_shop"]["mode"], "no_brand_shop")

    def test_valid_schema_separates_all_evidence_and_preserves_order(self) -> None:
        result = parse_combined_audit_html(_html(_valid_payload()))
        self.assertEqual(result["schema_version"], "2.0")
        self.assertEqual(
            [record["sourceRow"] for record in result["client_pdps"]],
            [4, 9],
        )
        self.assertEqual(
            [record["sourceRow"] for record in result["competitor_pdps"]],
            [7, 12],
        )
        self.assertEqual(len(result["current_searches"]), 1)
        self.assertEqual(len(result["benchmark_searches"]), 1)
        self.assertEqual(len(result["client_brand_shops"]), 1)
        self.assertEqual(len(result["competitor_brand_shops"]), 1)
        self.assertEqual(
            result["current_searches"][0]["screenshotDataUrl"],
            "data:image/png;base64,CURRENT",
        )
        self.assertEqual(
            result["client_brand_shops"][0]["modules"][1]["type"],
            "product-carousel",
        )

    def test_nested_schema_2_envelopes_populate_existing_records_and_evidence(self) -> None:
        result = parse_combined_audit_html(_html(_nested_schema_payload()))
        self.assertEqual(len(result["client_pdps"]), 1)
        self.assertEqual(len(result["competitor_pdps"]), 1)
        self.assertEqual(len(result["current_searches"]), 1)
        self.assertEqual(len(result["benchmark_searches"]), 1)
        self.assertEqual(len(result["client_brand_shops"]), 1)
        self.assertEqual(len(result["competitor_brand_shops"]), 1)
        self.assertEqual(result["client_pdps"][0]["productId"], "10451273")
        self.assertEqual(
            result["client_pdps"][0]["productTitle"],
            "Nutella Hazelnut Spread",
        )
        self.assertEqual(result["current_searches"][0]["searchTerm"], "hazelnut spread")
        self.assertEqual(result["current_searches"][0]["productsCaptured"], 50)
        self.assertTrue(result["current_searches"][0]["screenshotDataUrl"])
        self.assertEqual(result["client_brand_shops"][0]["moduleCount"], 2)
        self.assertEqual(
            result["client_brand_shops"][0]["moduleTypes"],
            ["HeroPov", "ItemCarousel"],
        )
        self.assertEqual(result["client_brand_shops"][0]["productCount"], 34)
        self.assertTrue(result["client_brand_shops"][0]["screenshotDataUrl"])
        self.assertEqual(result["errors"], [])

        primary_entries, primary_map, primary_messages = process_primary_audit_extract_sheet(
            df_uploaded=combined_pdp_to_dataframe(result["client_pdps"][0]),
            client_name="Nutella",
            retailer="Walmart",
            schema_version="2.0",
        )
        competitor_records, competitor_map, competitor_messages = process_competitor_audit_extract_sheet(
            df_uploaded=combined_pdp_to_dataframe(result["competitor_pdps"][0]),
            client_name="Nutella",
            retailer="Walmart",
            schema_version="2.0",
        )
        self.assertEqual(primary_messages, [])
        self.assertEqual(competitor_messages, [])
        self.assertFalse(
            any("No Image N columns" in message for message in [*primary_messages, *competitor_messages])
        )
        self.assertFalse(
            any("missing required value" in message.lower() for message in [*primary_messages, *competitor_messages])
        )
        attach_combined_evidence_to_record(
            primary_entries[0]["cached_record"],
            result["client_pdps"][0],
        )
        attach_combined_evidence_to_record(
            competitor_records[0],
            result["competitor_pdps"][0],
        )
        client_record = primary_entries[0]["cached_record"]
        self.assertEqual(client_record["item_id"], "10451273")
        self.assertEqual(client_record["product_title"], "Nutella Hazelnut Spread")
        self.assertEqual(client_record["subcategory"], "Nut Butters & Spreads")
        self.assertEqual(client_record["extraction_status"], "success")
        self.assertEqual(client_record["images"][0]["index"], 5)
        self.assertEqual(client_record["images"][0]["width"], 2200)
        self.assertEqual(client_record["images"][1]["height"], 1200)
        self.assertEqual(client_record["ingest_metadata"]["seller"], "Walmart.com")
        self.assertTrue(client_record["ingest_metadata"]["sold_by_walmart"])
        self.assertTrue(client_record["ingest_metadata"]["shipped_by_walmart"])
        self.assertTrue(
            client_record["ingest_metadata"]["enhanced_brand_content_status"]
        )
        self.assertIn(client_record["record_id"], primary_map)
        self.assertIn(competitor_records[0]["record_id"], competitor_map)

        search_evidence = {
            "current": result["current_searches"],
            "benchmark": result["benchmark_searches"],
            "all": [*result["current_searches"], *result["benchmark_searches"]],
        }
        brand_shop_evidence = {
            "client": result["client_brand_shops"],
            "competitor": result["competitor_brand_shops"],
            "all": [*result["client_brand_shops"], *result["competitor_brand_shops"]],
        }
        plan = build_audit_export_plan(
            audit_record={"client_name": "Nutella", "retailer": "Walmart"},
            primary_entries=primary_entries,
            competitor_assignments=[],
            competitor_records=competitor_records,
            search_evidence=search_evidence,
            brand_shop_evidence=brand_shop_evidence,
        )
        self.assertEqual(plan["search_evidence"]["current"][0]["productsCaptured"], 50)
        self.assertEqual(plan["brand_shop_evidence"]["client"][0]["moduleCount"], 2)
        self.assertEqual(
            plan["product_slide_pairs"][0]["pdp_slide"]["item_id"],
            "10451273",
        )

    def test_streamlit_schema2_processing_bypasses_legacy_sheet_processors(self) -> None:
        result = parse_combined_audit_html(_html(_nested_schema_payload()))
        with patch(
            "streamlitapp.process_primary_audit_extract_sheet",
            side_effect=AssertionError("legacy primary processor must not be called"),
        ), patch(
            "streamlitapp.process_competitor_audit_extract_sheet",
            side_effect=AssertionError("legacy competitor processor must not be called"),
        ):
            primary_entries, primary_map, primary_messages = _process_combined_pdp_records_v2(
                result["client_pdps"],
                role="Client",
                schema_version="2.0",
                client_name="Nutella",
                retailer="Walmart",
            )
            competitor_entries, competitor_map, competitor_messages = _process_combined_pdp_records_v2(
                result["competitor_pdps"],
                role="Competitor",
                schema_version="2.0",
                client_name="Nutella",
                retailer="Walmart",
            )
        self.assertEqual(len(primary_entries), 1)
        self.assertEqual(len(primary_map), 1)
        self.assertEqual(primary_messages, [])
        self.assertEqual(len(competitor_entries), 1)
        self.assertEqual(len(competitor_map), 1)
        self.assertEqual(competitor_messages, [])
        primary_record = primary_entries[0]["cached_record"]
        competitor_record = competitor_entries[0]
        self.assertEqual(primary_record["item_id"], "10451273")
        self.assertEqual(primary_record["product_title"], "Nutella Hazelnut Spread")
        self.assertEqual(primary_record["images"][0]["index"], 5)
        self.assertEqual(primary_record["images"][0]["width"], 2200)
        self.assertEqual(competitor_record["item_id"], "20000001")
        self.assertEqual(competitor_record["product_title"], "Jif Peanut Butter")
        for message in [*primary_messages, *competitor_messages]:
            self.assertNotIn("missing Product ID", message)
            self.assertNotIn("missing Product Title", message)
            self.assertNotIn("No Image N columns", message)

    def test_streamlit_legacy_flat_processing_still_uses_legacy_processor(self) -> None:
        flat_record = {
            "role": "Client",
            "Product URL": "https://example.com/legacy",
            "Product ID": "legacy-1",
            "Product Title": "Legacy Product",
        }
        sentinel_entry = {"record_id": "legacy-record", "cached_record": {}}
        with patch(
            "streamlitapp.process_primary_audit_extract_sheet",
            return_value=([sentinel_entry], {"legacy-record": {}}, []),
        ) as legacy_processor:
            entries, mapped, messages = _process_combined_pdp_records_v2(
                [flat_record],
                role="Client",
                schema_version="",
                client_name="Client",
                retailer="Walmart",
            )
        legacy_processor.assert_called_once()
        self.assertEqual(entries, [sentinel_entry])
        self.assertEqual(mapped, {"legacy-record": {}})
        self.assertEqual(messages, [])

    def test_streamlit_schema2_malformed_record_does_not_abort_valid_records(self) -> None:
        valid = parse_combined_audit_html(_html(_nested_schema_payload()))["client_pdps"][0]
        malformed = {
            "sourceRow": 99,
            "role": "Client",
            "status": "success",
            "data": {"productId": "", "productTitle": ""},
        }
        entries, mapped, messages = _process_combined_pdp_records_v2(
            [malformed, valid],
            role="Client",
            schema_version="2.0",
            client_name="Nutella",
            retailer="Walmart",
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(mapped), 1)
        self.assertTrue(any("productId" in message for message in messages))
        self.assertTrue(any("productTitle" in message for message in messages))

    def test_legacy_html_table_parser_remains_available(self) -> None:
        table_html = b"""
        <html><body><table>
        <tr><th>Product URL</th><th>Product ID</th><th>Product Title</th></tr>
        <tr><td>https://example.com/legacy</td><td>legacy-1</td><td>Legacy Product</td></tr>
        </table></body></html>
        """
        upload = _Upload(table_html)
        dataframe, messages = parse_audit_extract_upload_to_dataframe(upload)
        self.assertEqual(dataframe.iloc[0]["Product ID"], "legacy-1")
        self.assertTrue(any("visible HTML table" in message for message in messages))

    def test_collection_wrappers_are_supported_without_role_guessing(self) -> None:
        payload = _valid_payload()
        payload["pdpEvidence"] = {"records": payload["pdpEvidence"]}
        payload["searchEvidence"] = {"captures": payload["searchEvidence"]}
        payload["brandShopEvidence"] = {"items": payload["brandShopEvidence"]}
        result = parse_combined_audit_html(_html(payload))
        self.assertEqual(len(result["client_pdps"]), 2)
        self.assertEqual(len(result["competitor_pdps"]), 2)
        self.assertEqual(len(result["current_searches"]), 1)
        self.assertEqual(len(result["competitor_brand_shops"]), 1)

    def test_embedded_json_is_used_instead_of_visible_table_text(self) -> None:
        result = parse_combined_audit_html(_html(_valid_payload()))
        self.assertNotIn(
            "Visible legacy text",
            json.dumps(result["raw_payload"]),
        )

    def test_malformed_json(self) -> None:
        upload = _Upload(
            b'<script id="soapbox-audit-data" type="application/json">{"bad":</script>'
        )
        result = parse_combined_audit_html(upload)
        self.assertTrue(any("malformed" in str(error).lower() for error in result["errors"]))

    def test_missing_script_is_legacy_and_does_not_guess_roles(self) -> None:
        result = parse_combined_audit_html(_Upload(b"<html><table></table></html>"))
        self.assertTrue(result["is_legacy"])
        self.assertEqual(result["client_pdps"], [])
        self.assertEqual(result["competitor_pdps"], [])
        self.assertTrue(any("legacy" in str(error).lower() for error in result["errors"]))

    def test_unsupported_schema_version(self) -> None:
        payload = _valid_payload()
        payload["schemaVersion"] = "3.0"
        result = parse_combined_audit_html(_html(payload))
        self.assertEqual(result["client_pdps"], [])
        self.assertTrue(any("unsupported" in str(error).lower() for error in result["errors"]))

    def test_row_level_warnings_errors_continue_and_blank_role_is_not_assigned(self) -> None:
        payload = _valid_payload()
        payload["pdpEvidence"].insert(
            1,
            {
                **_pdp("", 5, "Unassigned Product", "Unknown Brand"),
                "warnings": ["Missing optional content"],
                "errors": ["Image capture failed"],
            },
        )
        result = parse_combined_audit_html(_html(payload))
        self.assertEqual(len(result["client_pdps"]), 2)
        self.assertEqual(len(result["competitor_pdps"]), 2)
        self.assertTrue(any("not assigned" in str(item) for item in result["warnings"]))
        self.assertTrue(any("Missing optional content" in str(item) for item in result["warnings"]))
        self.assertTrue(any("Image capture failed" in str(item) for item in result["errors"]))

    def test_missing_optional_evidence_warns_without_blocking_pdps(self) -> None:
        payload = _valid_payload()
        payload["searchEvidence"] = []
        payload["brandShopEvidence"] = []
        result = parse_combined_audit_html(_html(payload))
        self.assertEqual(len(result["client_pdps"]), 2)
        self.assertEqual(len(result["competitor_pdps"]), 2)
        self.assertTrue(any("Slide 3" in str(item) for item in result["warnings"]))
        self.assertTrue(any("Slide 5" in str(item) for item in result["warnings"]))

    def test_existing_primary_competitor_processors_receive_compatible_records(self) -> None:
        result = parse_combined_audit_html(_html(_valid_payload()))
        primary_entries = []
        competitor_records = []
        for source in result["client_pdps"]:
            entries, _, messages = process_primary_audit_extract_sheet(
                df_uploaded=combined_pdp_to_dataframe(source),
                client_name="Client",
                retailer="Walmart",
            )
            self.assertFalse([message for message in messages if "missing required" in message.lower()])
            attach_combined_evidence_to_record(entries[0]["cached_record"], source)
            primary_entries.extend(entries)
        for source in result["competitor_pdps"]:
            records, _, messages = process_competitor_audit_extract_sheet(
                df_uploaded=combined_pdp_to_dataframe(source),
                client_name="Client",
                retailer="Walmart",
            )
            self.assertFalse([message for message in messages if "missing required" in message.lower()])
            attach_combined_evidence_to_record(records[0], source)
            competitor_records.extend(records)

        self.assertEqual(len(primary_entries), 2)
        self.assertEqual(len(competitor_records), 2)
        self.assertEqual(
            primary_entries[0]["cached_record"]["images"][0]["url"],
            "data:image/png;base64,ROW4A",
        )
        self.assertEqual(
            primary_entries[0]["cached_record"]["resolved_product_type"],
            "Fruit Spread",
        )
        self.assertEqual(
            primary_entries[0]["cached_record"]["ingest_metadata"]["input_brand_name"],
            "Client Brand",
        )

        plan = build_audit_export_plan(
            audit_record={"client_name": "Client", "retailer": "Walmart"},
            primary_entries=primary_entries,
            competitor_assignments=[],
            competitor_records=competitor_records,
            search_evidence={
                "current": result["current_searches"],
                "benchmark": result["benchmark_searches"],
                "all": [*result["current_searches"], *result["benchmark_searches"]],
            },
            brand_shop_evidence={
                "client": result["client_brand_shops"],
                "competitor": result["competitor_brand_shops"],
                "all": [*result["client_brand_shops"], *result["competitor_brand_shops"]],
            },
        )
        self.assertIn("slide2_summary", plan)
        self.assertIn("slide4_findings", plan)
        self.assertIn("slide5_brand_shop", plan)
        self.assertIn("slide6_visibility", plan)
        self.assertEqual(len(plan["slide5_brand_shop"]["client"]["bullets"]), 6)
        self.assertEqual(len(plan["slide5_brand_shop"]["competitor"]["bullets"]), 6)
        self.assertEqual(len(plan["search_evidence"]["all"]), 2)
        self.assertEqual(len(plan["brand_shop_evidence"]["all"]), 2)
        self.assertEqual(
            plan["search_evidence"]["current"][0]["screenshotDataUrl"],
            "data:image/png;base64,CURRENT",
        )

    def test_reset_clears_audit_artifacts_but_preserves_metadata(self) -> None:
        state = {
            "audit_client_name": "Client",
            "audit_retailer": "Walmart",
            "audit_date": "2026-06-24",
            "audit_template_version": "New Strategic Template",
            "audit_primary_entries": [{"old": True}],
            "audit_competitor_entries": [{"old": True}],
            "audit_search_evidence": {"all": [{"old": True}]},
            "audit_brand_shop_evidence": {"all": [{"old": True}]},
            "audit_cached_pdp_records": {"old": {}},
            "audit_generated": True,
            "audit_ppt_bytes": b"old",
            "audit_v2_primary_select_for_pdp_old_0": True,
            "audit_competitor_image_orders": {"old": 1},
            "unrelated_state": "keep",
        }
        reset_combined_audit_state(state)
        self.assertEqual(state["audit_client_name"], "Client")
        self.assertEqual(state["audit_retailer"], "Walmart")
        self.assertEqual(state["audit_date"], "2026-06-24")
        self.assertEqual(state["audit_template_version"], "New Strategic Template")
        self.assertEqual(state["audit_primary_entries"], [])
        self.assertEqual(state["audit_competitor_entries"], [])
        self.assertEqual(state["audit_search_evidence"]["all"], [])
        self.assertEqual(state["audit_brand_shop_evidence"]["all"], [])
        self.assertNotIn("audit_v2_primary_select_for_pdp_old_0", state)
        self.assertFalse(state["audit_generated"])
        self.assertIsNone(state["audit_ppt_bytes"])
        self.assertEqual(state["unrelated_state"], "keep")


if __name__ == "__main__":
    unittest.main()
