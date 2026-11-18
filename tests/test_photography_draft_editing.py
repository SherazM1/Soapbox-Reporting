import re
import unittest
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch

from app.photography_pricing import draft_repository as repository, draft_ui
from app.photography_pricing.draft_models import QuoteDraft
from app.photography_pricing.draft_service import normalize_draft_payload
from test_photography_drafts import FakeConnection, draft_row, sample_payload, version_row


class EditableDraftRepositoryTests(unittest.TestCase):
    def assert_history_not_written(self, connection):
        for sql, _ in connection.executed:
            self.assertIsNone(re.search(
                r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE)\s+quote_draft_versions\b",
                sql, re.IGNORECASE,
            ))

    def test_new_draft_has_current_contents_and_no_version_records(self):
        connection = FakeConnection(rows=[None, draft_row(0), None])
        draft = repository.create_editable_draft(draft_name="New quote", payload=sample_payload(), conn=connection)
        self.assertEqual(0, draft.latest_version_number)
        self.assertIn("INSERT INTO quote_drafts", connection.executed[1][0])
        self.assertIn("INSERT INTO quote_draft_working_copies", connection.executed[2][0])
        self.assert_history_not_written(connection)

    def test_save_only_upserts_current_contents_without_changing_legacy_draft_or_versions(self):
        connection = FakeConnection(rows=[None, ("draft-1",), None])
        repository.save_editable_draft("draft-1", payload=sample_payload(), conn=connection)
        self.assertIn("FOR UPDATE", connection.executed[1][0])
        sql, params = connection.executed[2]
        self.assertIn("ON CONFLICT (draft_id) DO UPDATE", sql)
        self.assertEqual("draft-1", params[0])
        self.assertFalse(any(re.search(r"\bUPDATE\s+quote_drafts\b", sql, re.IGNORECASE)
                             for sql, _ in connection.executed))
        self.assert_history_not_written(connection)

    def test_edit_prefers_saved_current_contents(self):
        payload = sample_payload()
        connection = FakeConnection(rows=[None, (payload,)])
        with patch.object(repository, "get_latest_version") as read_legacy:
            loaded = repository.get_editable_draft_payload("draft-1", conn=connection)
        self.assertEqual(normalize_draft_payload(payload), loaded)
        read_legacy.assert_not_called()
        self.assert_history_not_written(connection)

    def test_old_draft_loads_latest_version_without_mutating_it(self):
        payload = sample_payload()
        original = deepcopy(payload)
        connection = FakeConnection(rows=[None, None, version_row(3, payload)])
        loaded = repository.get_editable_draft_payload("draft-1", conn=connection)
        self.assertEqual(normalize_draft_payload(original), loaded)
        self.assertEqual(original, payload)
        self.assertIn("ORDER BY version_number DESC", connection.executed[-1][0])
        self.assert_history_not_written(connection)

    def test_failed_save_rolls_back_and_never_writes_history(self):
        connection = FakeConnection(rows=[None, ("draft-1",)])
        with (
            patch.object(repository, "get_connection", return_value=nullcontext(connection)),
            patch.object(repository, "_write_working_copy", side_effect=RuntimeError("unavailable")),
        ):
            with self.assertRaises(repository.DraftRepositoryError):
                repository.save_editable_draft("draft-1", payload=sample_payload())
        self.assertTrue(connection.rolled_back)
        self.assertFalse(connection.committed)
        self.assert_history_not_written(connection)

    def test_missing_draft_cannot_create_an_orphan_current_copy(self):
        connection = FakeConnection(rows=[None, None])
        with self.assertRaises(repository.DraftRepositoryError):
            repository.save_editable_draft("missing", payload=sample_payload(), conn=connection)
        self.assertFalse(any("INSERT INTO quote_draft_working_copies" in sql for sql, _ in connection.executed))
        self.assert_history_not_written(connection)


class EditDraftUiTests(unittest.TestCase):
    def test_edit_loads_legacy_contents_then_save_updates_current_copy_and_reopens_it(self):
        from streamlit.testing.v1 import AppTest
        from app.contact_management.models import ClientContact, InternalContact

        legacy = QuoteDraft(*draft_row(3))
        historical = sample_payload()
        untouched = deepcopy(historical)
        current = {}
        client = ClientContact("client-1", None, "SunVilla", "First", "Last", "client@example.com")
        internal = InternalContact("internal-1", "Producer", "Lead", "producer@example.com")
        app = AppTest.from_string(
            "from app.photography_pricing.apparel_estimator import render_photography_pricing\n"
            "render_photography_pricing()\n"
        )

        def save(draft_id, *, payload):
            current[draft_id] = deepcopy(payload)

        with (
            patch("app.contact_management.client_autocomplete._component"),
            patch("app.photography_pricing.apparel_estimator.render_contact_management"),
            patch("app.contact_management.contact_ui.safe_list_active_client_contacts", return_value=[client]),
            patch("app.contact_management.contact_ui.safe_list_active_internal_contacts", return_value=[internal]),
            patch.object(draft_ui, "_contact_id_sets", return_value=({client.id}, {internal.id})),
            patch.object(draft_ui, "draft_repository") as repo,
        ):
            repo.list_drafts.return_value = [legacy]
            repo.get_editable_draft_payload.side_effect = lambda draft_id: deepcopy(current.get(draft_id, historical))
            repo.save_editable_draft.side_effect = save
            app.run()
            self.assertFalse(app.exception)
            app.button(key="photo_pricing_open_draft").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(10, app.number_input(key="photo_pricing_on_model_image_quantity").value)
            self.assertEqual("Custom notes only.", app.text_area(key="photo_pricing_comments_custom_notes").value)
            app.number_input(key="photo_pricing_on_model_image_quantity").set_value(25).run()
            app.button(key="photo_pricing_save_draft").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(25, current[legacy.id]["pricing"]["on_model_image_quantity"])
            self.assertEqual(untouched, historical)
            self.assertEqual(3, legacy.latest_version_number)

            app.number_input(key="photo_pricing_on_model_image_quantity").set_value(99).run()
            app.button(key="photo_pricing_open_draft").click().run()
            self.assertEqual(25, app.number_input(key="photo_pricing_on_model_image_quantity").value)
            self.assertFalse(app.exception)
            # Merely browsing a different draft must not redirect Save to it.
            other = QuoteDraft("other", "Another saved draft", "draft", client.id, internal.id, 2, None, None)
            repo.list_drafts.return_value = [legacy, other]
            app.run()
            app.selectbox(key=draft_ui.SELECTED_DRAFT_ID_KEY).set_value(other.id).run()
            app.button(key="photo_pricing_save_draft").click().run()
            self.assertFalse(app.exception)
            self.assertEqual((legacy.id,), repo.save_editable_draft.call_args.args)
            self.assertNotIn(other.id, current)
            self.assertNotIn("Save Version", [button.label for button in app.button])
            self.assertNotIn("Delete Draft", [button.label for button in app.button])
            self.assertNotIn("Version Note", [widget.label for widget in app.text_input])
            self.assertEqual([legacy.draft_name, other.draft_name], app.selectbox(key=draft_ui.SELECTED_DRAFT_ID_KEY).options)
            repo.create_editable_draft.assert_not_called()
            repo.create_version.assert_not_called()
            repo.list_versions.assert_not_called()
            repo.restore_version_as_latest.assert_not_called()
            repo.delete_draft.assert_not_called()


if __name__ == "__main__":
    unittest.main()
