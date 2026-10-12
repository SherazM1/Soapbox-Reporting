import io
import json
import unittest

from app.audit_helpers.combined_audit_extract import parse_combined_audit_html


def report(title):
    payload = {"schemaVersion": "2.2", "pdpEvidence": [{"sourceRow": 2, "role": "Client",
               "inputBrandName": "Kidde", "status": "success",
               "data": {"productId": "2720795611", "productTitle": title}}]}
    return io.BytesIO(('<script id="soapbox-audit-data" type="application/json">' + json.dumps(payload) + '</script>').encode())


class EmptyClientCaptureTest(unittest.TestCase):
    def test_empty_success_capture_explains_how_to_recover(self):
        result = parse_combined_audit_html(report(""))
        text = " ".join(str(e) for e in result["errors"])
        self.assertIn("Row 2: Kidde", text)
        self.assertIn("no product title", text)
        self.assertIn("capture it again", text)

    def test_valid_client_title_remains_accepted(self):
        result = parse_combined_audit_html(report("Kidde Carbon Monoxide Alarm"))
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["client_pdps"]), 1)


if __name__ == "__main__":
    unittest.main()
