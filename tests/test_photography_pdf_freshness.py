import unittest
from datetime import date
from unittest.mock import patch

import test_photography_editor_fragment as fixtures
from app.photography_pricing import apparel_estimator as editor


class PdfFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EditorFragmentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.app = self.fixture.app
        self.app.run()

    def generate(self):
        self.app.button(key='photo_pricing_generate_pdf').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(b'pdf', self.app.session_state['photo_pricing_generated_pdf'])

    def assert_invalid(self):
        self.assertFalse(self.app.exception)
        self.assertNotIn('photo_pricing_generated_pdf', self.app.session_state)
        self.assertEqual(0, len(self.app.get('download_button')))

    def test_pricing_edits_invalidate_and_regenerate_current_quote(self):
        for key, value in (
            ('on_model_image_quantity', 35), ('on_model_detail_quantity', 2),
            ('laydown_silo_quantity', 3), ('color_corrections_quantity', 4),
            ('ai_generation_quantity', 5), ('post_production_hours', 1.0),
            ('adult_model_hours_single', 2.0), ('model_fitting_quantity', 1),
        ):
            with self.subTest(key=key):
                self.generate()
                self.app.number_input(key='photo_pricing_' + key).set_value(value).run()
                self.assert_invalid()
        self.generate()
        self.app.selectbox(key='photo_pricing_laydown_silo_type').set_value('shoes').run()
        self.assert_invalid()
        self.generate()
        self.app.radio(key='photo_pricing_model_type').set_value('kid').run()
        self.assert_invalid()
        self.generate()
        self.app.radio(key='photo_pricing_account_management_mode').set_value('manual').run()
        self.assert_invalid()
        self.generate()
        self.app.number_input(key='photo_pricing_manual_account_management_amount').set_value(500.0).run()
        self.assert_invalid()
        self.generate()
        quote = self.fixture.generate.call_args.args[0]
        self.assertEqual(35, quote.apparel_inputs.on_model_image_quantity)
        self.assertEqual(500, quote.account_management_amount_used)

    def test_all_project_and_comment_fields_invalidate(self):
        for field in ('on_model', 'on_model_detail', 'laydown_detail', 'color_correct', 'colors', 'post', 'model_hours'):
            with self.subTest(field=field):
                self.generate()
                self.app.number_input(key=f'photo_pricing_comments_{field}_0').set_value(2.0).run()
                self.assert_invalid()
        for key in ('project_name_0', 'estimate_subject', 'subtitle_line'):
            self.generate()
            self.app.text_input(key='photo_pricing_comments_' + key).set_value('Changed').run()
            self.assert_invalid()
        self.generate()
        self.app.text_area(key='photo_pricing_comments_custom_notes').set_value('Fresh notes').run()
        self.assert_invalid()
        self.generate()
        self.app.button(key='photo_pricing_comments_add_project').click().run()
        self.assert_invalid()
        self.generate()
        self.app.button(key='photo_pricing_comments_remove_1').click().run()
        self.assert_invalid()

    def test_contacts_and_metadata_invalidate(self):
        from app.contact_management.models import ClientContact, InternalContact
        clients = [self.fixture.client, ClientContact('other', None, 'Other', 'Grace', 'Hopper', 'grace@example.com')]
        internals = [self.fixture.internal, InternalContact('other', 'Other Producer', 'Lead', 'other@example.com')]
        with patch('app.contact_management.contact_ui.safe_list_active_client_contacts', return_value=clients), patch('app.contact_management.contact_ui.safe_list_active_internal_contacts', return_value=internals):
            self.app.run()
            for key in ('client_contact_id', 'internal_contact_id'):
                self.generate()
                self.app.selectbox(key='photo_pricing_' + key).set_value('other').run()
                self.assert_invalid()
            for key in ('quote_title', 'reference_number'):
                self.generate()
                self.app.text_input(key='photo_pricing_' + key).set_value('Changed').run()
                self.assert_invalid()
            for key in ('quote_created_date', 'quote_expiration_date'):
                self.generate()
                self.app.date_input(key='photo_pricing_' + key).set_value(date(2030, 1, 1)).run()
                self.assert_invalid()

    def test_unchanged_reruns_and_draft_ui_state_preserve_pdf_and_download_contract(self):
        with patch.object(editor.st, 'download_button', wraps=editor.st.download_button) as download:
            self.generate()
            kwargs = download.call_args.kwargs
            self.assertEqual('ignore', kwargs['on_click'])
            self.assertEqual(b'pdf', kwargs['data'])
            self.assertEqual('photography_pricing_quote.pdf', kwargs['file_name'])
            self.assertEqual('application/pdf', kwargs['mime'])
        self.assertTrue(self.app.get('download_button')[0].proto.ignore_rerun)
        self.app.run()
        self.assertEqual(b'pdf', self.app.session_state['photo_pricing_generated_pdf'])
        self.app.session_state['photo_pricing_active_version_number'] = 99
        self.app.session_state['photo_pricing_draft_name'] = 'UI-only draft name'
        self.app.run()
        self.assertEqual(b'pdf', self.app.session_state['photo_pricing_generated_pdf'])

    def test_legacy_bytes_without_snapshot_are_invalidated(self):
        state = {'photo_pricing_generated_pdf': b'old'}
        editor._invalidate_stale_generated_pdf(state, {'quote': {}})
        self.assertNotIn('photo_pricing_generated_pdf', state)

    def test_save_preserves_pdf_and_start_new_clears_bytes_and_snapshot(self):
        from app.photography_pricing import draft_ui
        from app.photography_pricing.draft_models import QuoteDraft, QuoteDraftVersion
        draft = QuoteDraft('draft', 'Saved', 'draft', 'client', 'internal', 1, None, None)
        version = QuoteDraftVersion('version', 'draft', 1, {}, 'internal', None, None)
        self.generate()
        with patch.object(draft_ui.draft_repository, 'create_draft', return_value=(draft, version)) as save:
            self.app.button(key='photo_pricing_save_draft').click().run()
            save.assert_called_once()
        self.assertFalse(self.app.exception)
        self.assertEqual(b'pdf', self.app.session_state['photo_pricing_generated_pdf'])
        self.app.button(key='photo_pricing_start_new_draft').click().run()
        self.assert_invalid()
        self.assertNotIn('photo_pricing_generated_pdf_inputs', self.app.session_state)
