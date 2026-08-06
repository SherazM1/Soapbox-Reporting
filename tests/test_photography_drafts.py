import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from app.photography_pricing import draft_ui
from app.photography_pricing.draft_models import QuoteDraftVersion
from app.photography_pricing.draft_repository import (
    DuplicateDraftVersionError,
    create_draft,
    create_version,
    get_latest_version,
    get_version,
    list_versions,
    restore_version_as_latest,
)
from app.photography_pricing.draft_service import (
    apparel_inputs_from_draft_payload,
    build_draft_name,
    normalize_draft_payload,
    restore_draft_payload_to_state,
    serialize_draft_payload,
)
from app.photography_pricing.pdf_generator import generate_page2_pricing_pdf
from app.photography_pricing.quote_builder import build_apparel_quote


ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, params=()):
        self.conn.executed.append((sql, params))
        if self.conn.error:
            raise self.conn.error
        self.conn.current = self.conn.rows.pop(0) if self.conn.rows else None

    def fetchone(self):
        return self.conn.current

    def fetchall(self):
        return list(self.conn.current or [])


class FakeConnection:
    def __init__(self, rows=None, error=None):
        self.rows = list(rows or [])
        self.error = error
        self.current = None
        self.executed = []
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class UniqueViolationLike(Exception):
    sqlstate = "23505"


def draft_row(version=0):
    return (
        "draft-1",
        "Spring Apparel Estimate — SunVilla — 2026-07-28",
        "draft",
        "client-1",
        "internal-1",
        version,
        datetime(2026, 7, 28, 10),
        datetime(2026, 7, 28, 10),
    )


def version_row(version, payload=None):
    return (
        f"version-{version}",
        "draft-1",
        version,
        payload or sample_payload(),
        "internal-1",
        None,
        datetime(2026, 7, 28, 10, version),
    )


def sample_state():
    return {
        "photo_pricing_quote_title": "Spring Apparel Estimate",
        "photo_pricing_reference_number": "REF-123",
        "photo_pricing_quote_created_date": date(2026, 7, 28),
        "photo_pricing_quote_expiration_date": date(2026, 10, 28),
        "photo_pricing_expiration_overridden": True,
        "photo_pricing_client_contact_id": "client-1",
        "photo_pricing_internal_contact_id": "internal-1",
        "photo_pricing_on_model_image_quantity": 10,
        "photo_pricing_on_model_detail_quantity": 2,
        "photo_pricing_laydown_silo_type": "shoes",
        "photo_pricing_laydown_silo_quantity": 3,
        "photo_pricing_color_corrections_quantity": 4,
        "photo_pricing_post_production_hours": 1.25,
        "photo_pricing_model_type": "both",
        "photo_pricing_adult_model_hours": 2.5,
        "photo_pricing_kid_model_hours": 1.5,
        "photo_pricing_model_fitting_quantity": 2,
        "photo_pricing_ai_generation_quantity": 5,
        "photo_pricing_account_management_mode": "manual",
        "photo_pricing_manual_account_management_amount": 425.0,
        "photo_pricing_comments_estimate_subject": "",
        "photo_pricing_comments_subtitle_line": "",
        "photo_pricing_comments_custom_notes": "Custom notes only.",
        "photo_pricing_project_rows": [{}, {}],
        "photo_pricing_comments_project_name_0": "",
        "photo_pricing_comments_on_model_0": 0,
        "photo_pricing_comments_laydown_detail_0": 0,
        "photo_pricing_comments_color_correct_0": 0,
        "photo_pricing_comments_post_0": 0,
        "photo_pricing_comments_model_hours_0": 0,
        "photo_pricing_comments_project_name_1": "Hero",
        "photo_pricing_comments_on_model_1": 5,
        "photo_pricing_comments_laydown_detail_1": 1,
        "photo_pricing_comments_color_correct_1": 2,
        "photo_pricing_comments_post_1": 3,
        "photo_pricing_comments_model_hours_1": 4,
    }


def sample_payload():
    return serialize_draft_payload(sample_state())


class PhotographyDraftTests(unittest.TestCase):
    def test_schema_contains_draft_tables_constraints_and_indexes(self):
        schema = (ROOT / "database" / "contacts_schema.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS quote_drafts", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS quote_draft_versions", schema)
        self.assertIn("payload JSONB NOT NULL", schema)
        self.assertIn("UNIQUE (draft_id, version_number)", schema)
        self.assertIn("quote_drafts_updated_at_idx", schema)
        self.assertIn("quote_draft_versions_draft_version_idx", schema)
        self.assertIn("photography_quote_drafts_v1", schema)

    def test_payload_round_trips_all_editable_values(self):
        payload = sample_payload()
        normalized = normalize_draft_payload(payload)

        self.assertEqual(1, normalized["schema_version"])
        self.assertEqual("Spring Apparel Estimate", normalized["quote_metadata"]["quote_title"])
        self.assertEqual("REF-123", normalized["quote_metadata"]["reference_number"])
        self.assertEqual("2026-07-28", normalized["quote_metadata"]["quote_created_date"])
        self.assertEqual("2026-10-28", normalized["quote_metadata"]["quote_expiration_date"])
        self.assertTrue(normalized["quote_metadata"]["expiration_overridden"])
        self.assertEqual("client-1", normalized["contacts"]["client_contact_id"])
        self.assertEqual("internal-1", normalized["contacts"]["internal_contact_id"])
        self.assertEqual("both", normalized["pricing"]["model_hours_mode"])
        self.assertEqual("2.50", normalized["pricing"]["adult_model_hours"])
        self.assertEqual("1.50", normalized["pricing"]["kid_model_hours"])
        self.assertEqual(2, normalized["pricing"]["model_fitting_quantity"])
        self.assertEqual("manual", normalized["pricing"]["account_management_mode"])
        self.assertEqual("425.00", normalized["pricing"]["manual_account_management_amount"])
        self.assertEqual("Custom notes only.", normalized["comments"]["custom_notes"])
        self.assertEqual("Hero", normalized["comments"]["project_entries"][1]["project_name"])

    def test_draft_name_uses_title_company_and_created_date(self):
        self.assertEqual(
            "Spring Apparel Estimate — SunVilla — 2026-07-28",
            build_draft_name(sample_payload(), "SunVilla"),
        )
        self.assertEqual(
            "Untitled Quote — No Client — 2026-07-28",
            build_draft_name({"quote_metadata": {"quote_created_date": "2026-07-28"}}),
        )

    def test_draft_ui_uses_requested_draft_name_when_saving_new_draft(self):
        self.assertEqual(
            "test 4",
            draft_ui._draft_name_for_save(sample_payload(), None, " test 4 "),
        )

    def test_draft_ui_falls_back_to_generated_draft_name(self):
        self.assertIn(
            "Spring Apparel Estimate",
            draft_ui._draft_name_for_save(sample_payload(), None, ""),
        )

    def test_draft_ui_migrates_legacy_state_without_overwriting_explicit_state(self):
        state = {
            "photo_pricing_loaded_draft_id": "legacy-active",
            "photo_pricing_loaded_draft_version": 2,
            "photo_pricing_open_draft_id": "legacy-selected",
        }

        draft_ui._migrate_legacy_draft_state(state)

        self.assertEqual("legacy-active", state[draft_ui.ACTIVE_DRAFT_ID_KEY])
        self.assertEqual(2, state[draft_ui.ACTIVE_VERSION_KEY])
        self.assertEqual("legacy-selected", state[draft_ui.SELECTED_DRAFT_ID_KEY])

        state[draft_ui.ACTIVE_DRAFT_ID_KEY] = "explicit-active"
        state[draft_ui.SELECTED_DRAFT_ID_KEY] = "explicit-selected"
        draft_ui._migrate_legacy_draft_state(state)

        self.assertEqual("explicit-active", state[draft_ui.ACTIVE_DRAFT_ID_KEY])
        self.assertEqual("explicit-selected", state[draft_ui.SELECTED_DRAFT_ID_KEY])

    def test_draft_ui_active_draft_can_be_cleared_without_clearing_selected_draft(self):
        state = {
            draft_ui.ACTIVE_DRAFT_ID_KEY: "active-draft",
            draft_ui.ACTIVE_VERSION_KEY: 3,
            draft_ui.ACTIVE_DRAFT_NAME_KEY: "Active Draft",
            draft_ui.SELECTED_DRAFT_ID_KEY: "selected-draft",
            "photo_pricing_draft_name": "Active Draft",
        }

        with patch("app.photography_pricing.draft_ui.st.session_state", state):
            draft_ui._set_active_draft(None, None)

        self.assertNotIn(draft_ui.ACTIVE_DRAFT_ID_KEY, state)
        self.assertNotIn(draft_ui.ACTIVE_VERSION_KEY, state)
        self.assertNotIn(draft_ui.ACTIVE_DRAFT_NAME_KEY, state)
        self.assertEqual("selected-draft", state[draft_ui.SELECTED_DRAFT_ID_KEY])
        self.assertEqual("Active Draft", state["photo_pricing_draft_name"])

    def test_draft_ui_loading_active_draft_syncs_selected_draft(self):
        state = {}

        with patch("app.photography_pricing.draft_ui.st.session_state", state):
            draft_ui._set_active_draft("draft-2", 1, "test 4")

        self.assertEqual("draft-2", state[draft_ui.ACTIVE_DRAFT_ID_KEY])
        self.assertEqual("draft-2", state[draft_ui.SELECTED_DRAFT_ID_KEY])
        self.assertEqual(1, state[draft_ui.ACTIVE_VERSION_KEY])
        self.assertEqual("test 4", state[draft_ui.ACTIVE_DRAFT_NAME_KEY])
        self.assertNotIn("photo_pricing_draft_name", state)

    def test_draft_ui_pending_load_can_sync_draft_name_before_widgets_render(self):
        state = {}

        with patch("app.photography_pricing.draft_ui.st.session_state", state):
            draft_ui._set_active_draft(
                "draft-2",
                1,
                "test 4",
                sync_draft_name_input=True,
            )

        self.assertEqual("test 4", state["photo_pricing_draft_name"])

    def test_missing_optional_fields_receive_defaults(self):
        normalized = normalize_draft_payload({})

        self.assertEqual("adult", normalized["pricing"]["model_hours_mode"])
        self.assertEqual("automatic", normalized["pricing"]["account_management_mode"])
        self.assertEqual("0.00", normalized["pricing"]["manual_account_management_amount"])
        self.assertEqual([{"project_name": "", "on_model": "0.00", "laydown_detail": "0.00", "color_correct": "0.00", "post": "0.00", "model_hours": "0.00"}], normalized["comments"]["project_entries"])

    def test_restore_reconstructs_session_state_keys_and_preserves_dates(self):
        state = {"photo_pricing_comments_project_name_9": "stale"}
        warnings = restore_draft_payload_to_state(
            sample_payload(),
            state,
            available_client_ids={"client-1"},
            available_internal_ids={"internal-1"},
        )

        self.assertEqual([], warnings)
        self.assertEqual("REF-123", state["photo_pricing_reference_number"])
        self.assertEqual(date(2026, 7, 28), state["photo_pricing_quote_created_date"])
        self.assertEqual(date(2026, 10, 28), state["photo_pricing_quote_expiration_date"])
        self.assertTrue(state["photo_pricing_expiration_overridden"])
        self.assertEqual(date(2026, 7, 28), state["photo_pricing_previous_created_date"])
        self.assertEqual("both", state["photo_pricing_model_type"])
        self.assertEqual(2.5, state["photo_pricing_adult_model_hours_single"])
        self.assertEqual(1.5, state["photo_pricing_kid_model_hours_single"])
        self.assertEqual([{}, {}], state["photo_pricing_project_rows"])
        self.assertEqual("Hero", state["photo_pricing_comments_project_name_1"])
        self.assertNotIn("photo_pricing_comments_project_name_9", state)

    def test_restore_handles_unavailable_contacts(self):
        state = {}
        warnings = restore_draft_payload_to_state(
            sample_payload(),
            state,
            available_client_ids=set(),
            available_internal_ids=set(),
        )

        self.assertIn("Saved client contact is no longer available.", warnings)
        self.assertIn("Saved internal contact is no longer available.", warnings)
        self.assertNotIn("photo_pricing_client_contact_id", state)
        self.assertNotIn("photo_pricing_internal_contact_id", state)

    def test_repository_creates_draft_and_version_one(self):
        conn = FakeConnection(rows=[draft_row(0), version_row(1), draft_row(1)])

        draft, version = create_draft(
            draft_name="Draft",
            payload=sample_payload(),
            client_contact_id="client-1",
            internal_contact_id="internal-1",
            saved_by_contact_id="internal-1",
            conn=conn,
        )

        self.assertEqual(1, draft.latest_version_number)
        self.assertEqual(1, version.version_number)
        self.assertIn("INSERT INTO quote_drafts", conn.executed[0][0])
        self.assertIn("INSERT INTO quote_draft_versions", conn.executed[1][0])

    def test_repository_saves_versions_two_and_three(self):
        version2 = create_version("draft-1", payload=sample_payload(), conn=FakeConnection(rows=[draft_row(1), version_row(2), draft_row(2)]))
        version3 = create_version("draft-1", payload=sample_payload(), conn=FakeConnection(rows=[draft_row(2), version_row(3), draft_row(3)]))

        self.assertEqual(2, version2.version_number)
        self.assertEqual(3, version3.version_number)

    def test_repository_gets_latest_and_orders_history(self):
        payload = sample_payload()
        latest = get_latest_version("draft-1", conn=FakeConnection(rows=[version_row(3, payload)]))
        versions = list_versions("draft-1", conn=FakeConnection(rows=[ [version_row(3, payload), version_row(2, payload), version_row(1, payload)] ]))

        self.assertEqual(3, latest.version_number)
        self.assertEqual([3, 2, 1], [version.version_number for version in versions])

    def test_duplicate_version_numbers_are_rejected(self):
        with self.assertRaises(DuplicateDraftVersionError):
            create_version(
                "draft-1",
                payload=sample_payload(),
                version_number=2,
                conn=FakeConnection(rows=[draft_row(1)], error=UniqueViolationLike("duplicate key")),
            )

    def test_restore_older_version_creates_new_latest_and_keeps_payload(self):
        old_payload = sample_payload()
        restored = restore_version_as_latest(
            "draft-1",
            1,
            conn=FakeConnection(rows=[version_row(1, old_payload), draft_row(3), version_row(4, old_payload), draft_row(4)]),
        )

        self.assertEqual(4, restored.version_number)
        self.assertEqual(old_payload["quote_metadata"]["reference_number"], restored.payload["quote_metadata"]["reference_number"])

    def test_version_history_keeps_old_versions_after_restore(self):
        versions = list_versions(
            "draft-1",
            conn=FakeConnection(rows=[[version_row(4), version_row(3), version_row(2), version_row(1)]]),
        )

        self.assertEqual([4, 3, 2, 1], [version.version_number for version in versions])

    def test_get_specific_version(self):
        version = get_version("draft-1", 2, conn=FakeConnection(rows=[version_row(2)]))

        self.assertEqual(2, version.version_number)

    @unittest.skipUnless(Path("templates/photographytemplate.pdf").exists() and Path("templates/Page 2.pdf").exists(), "photography PDF templates are not present")
    def test_pdf_can_be_generated_from_reopened_draft_payload(self):
        restored_state = {}
        restore_draft_payload_to_state(sample_payload(), restored_state)
        quote = build_apparel_quote(apparel_inputs_from_draft_payload(sample_payload()))
        pdf_bytes = generate_page2_pricing_pdf(quote)

        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()
