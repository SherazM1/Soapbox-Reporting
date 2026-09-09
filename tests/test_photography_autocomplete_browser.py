"""Optional browser regression: run with Playwright available on PYTHONPATH."""
import importlib.util
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = '''
import sys
sys.path.insert(0, ROOT_PATH)
import streamlit as st
from app.contact_management import contact_ui
from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing import draft_ui, apparel_estimator
from app.photography_pricing.draft_models import QuoteDraft, QuoteDraftVersion
from app.photography_pricing.draft_service import serialize_draft_payload

clients = [
    ClientContact("a", None, "Alpha", "Ada", "Owen", "ada@test.com"),
    ClientContact("lisa", None, "Beta", "Lisa", "Gao", "lisa@test.com"),
    ClientContact("laura", None, "Beta", "Laura", "Xu", "laura@test.com"),
    ClientContact("lee", None, "Gamma", "Arthur", "Lee", "arthur@test.com"),
]
internal = InternalContact("producer", "Producer", "Lead", "producer@test.com")
contact_ui.safe_list_active_client_contacts = lambda: clients
contact_ui.safe_list_active_internal_contacts = lambda: [internal]
draft_ui._contact_id_sets = lambda: ({c.id for c in clients}, {internal.id})
apparel_estimator.render_contact_management = lambda: None
saved = serialize_draft_payload({"photo_pricing_client_contact_id": "lee", "photo_pricing_quote_title": "Saved quote"})
draft = QuoteDraft("draft", "Saved draft", "draft", "lee", "producer", 1, None, None)
version = QuoteDraftVersion("v1", "draft", 1, saved, "producer", None, None)
draft_ui.draft_repository.list_drafts = lambda: [draft]
draft_ui.draft_repository.list_versions = lambda _id: [version]
draft_ui.draft_repository.get_latest_version = lambda _id: version
apparel_estimator.render_photography_pricing()
st.caption("Selected ID: " + st.session_state["photo_pricing_client_contact_id"])
'''


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Optional Playwright browser tooling is not installed")
class AutocompleteBrowserTests(unittest.TestCase):
    def test_live_selection_dropdown_draft_reset_and_summary(self):
        from playwright.sync_api import sync_playwright, expect
        with tempfile.TemporaryDirectory(prefix="photo-autocomplete-") as directory:
            fixture = Path(directory) / "photo_fixture.py"
            fixture.write_text(FIXTURE.replace("ROOT_PATH", repr(str(ROOT))), encoding="utf-8")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            log_path = Path(directory) / "streamlit.log"
            with log_path.open("w", encoding="utf-8") as log:
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(ROOT / ".venv" / "Lib" / "site-packages")
                server = subprocess.Popen([sys._base_executable, "-m", "streamlit", "run", str(fixture),
                    "--server.headless=true", f"--server.port={port}", "--server.address=127.0.0.1",
                    "--browser.gatherUsageStats=false"], cwd=ROOT, stdout=log, stderr=log, env=environment)
                try:
                    url = f"http://127.0.0.1:{port}"
                    for _ in range(100):
                        try:
                            with urlopen(url + "/_stcore/health", timeout=1):
                                break
                        except OSError:
                            if server.poll() is not None:
                                self.fail(log_path.read_text())
                            time.sleep(.1)
                    with sync_playwright() as playwright:
                        browser_path = os.environ.get("PHOTO_TEST_CHROMIUM")
                        browser = playwright.chromium.launch(headless=True, **(
                            {"executable_path": browser_path} if browser_path else {}))
                        page = browser.new_page(viewport={"width": 1500, "height": 1000})
                        page_errors = []
                        page.on("pageerror", lambda error: page_errors.append(str(error)))
                        page.goto(url)
                        search = page.get_by_role("combobox", name="Client Contact Search", exact=True)
                        normal = page.get_by_role("combobox", name=re.compile(r"Client Contact$"))
                        expect(search).to_be_visible(timeout=20000)
                        expect(page.get_by_text("Selected ID: a", exact=True)).to_be_visible()
                        search.press_sequentially("L")
                        options = page.locator('[role="listbox"] [role="option"]')
                        expect(options).to_have_count(4)
                        self.assertIn("Laura Xu", options.nth(0).inner_text())
                        self.assertIn("Lisa Gao", options.nth(1).inner_text())
                        self.assertIn("Arthur Lee", options.nth(2).inner_text())
                        expect(page.get_by_text("Selected ID: a", exact=True)).to_be_visible()
                        search.press_sequentially("i")
                        expect(options).to_have_count(1)
                        options.first.click()
                        expect(page.get_by_text("Selected ID: lisa", exact=True)).to_be_visible()
                        expect(page.locator('[data-testid="stSelectbox"]').filter(has=normal)).to_contain_text("Lisa Gao")
                        normal.click()
                        page.get_by_role("option").filter(has_text="Ada Owen").click()
                        expect(page.get_by_text("Selected ID: a", exact=True)).to_be_visible()
                        search.fill("not a real contact")
                        expect(page.get_by_text("No matching client contacts.", exact=True)).to_be_visible()
                        expect(page.get_by_text("Selected ID: a", exact=True)).to_be_visible()
                        page.get_by_text("Drafts", exact=True).click()
                        page.get_by_role("button", name="Load Draft", exact=True).click()
                        expect(page.get_by_text("Selected ID: lee", exact=True)).to_be_visible()
                        expect(search).to_have_value("")
                        search.fill("Lisa")
                        expect(options).to_have_count(1)
                        options.first.click()
                        expect(page.get_by_text("Selected ID: lisa", exact=True)).to_be_visible()
                        page.get_by_role("button", name="Start New Quote", exact=True).click()
                        expect(page.get_by_text("Selected ID: a", exact=True)).to_be_visible()
                        expect(search).to_have_value("")
                        expect(page.locator(".katex")).to_have_count(0)
                        summary = page.locator('[data-testid="stMarkdownContainer"]').filter(
                            has_text="Automatic Account Management Fee:")
                        expect(summary).to_contain_text("Automatic Account Management Fee: $175.00")
                        expect(summary).to_contain_text("Account Management Fee Used: $175.00")
                        self.assertNotIn("**", summary.inner_text())
                        expect(page.get_by_text("Final Total", exact=True)).to_be_visible()
                        self.assertEqual([], page_errors)
                        browser.close()
                finally:
                    server.terminate()
                    server.wait(timeout=15)


if __name__ == "__main__":
    unittest.main()
