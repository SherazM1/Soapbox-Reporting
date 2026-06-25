from __future__ import annotations

import io
import json
import unittest

from app.audit_helpers.combined_audit_extract import (
    attach_combined_evidence_to_record,
    combined_pdp_to_dataframe,
    parse_combined_audit_html,
    reset_combined_audit_state,
)
from audit_export import build_audit_export_plan
from audit_helpers import (
    initialize_auditing_session_state,
    process_competitor_audit_extract_sheet,
    process_primary_audit_extract_sheet,
)


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
