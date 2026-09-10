"""Editor behavior and actual ScriptRunner fragment reruns (test-only hooks)."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner
from streamlit.runtime.scriptrunner import RerunData
from streamlit.proto.WidgetStates_pb2 import WidgetStates

from app.contact_management.models import ClientContact, InternalContact
from app.photography_pricing import apparel_estimator as editor, draft_ui
from app.photography_pricing.models import ApparelInputs
from app.photography_pricing.quote_builder import build_apparel_quote


class EditorFragmentTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.client = ClientContact('client', None, 'Company', 'Ada', 'Lovelace', 'ada@example.com')
        self.internal = InternalContact('internal', 'Producer', 'Lead', 'producer@example.com')
        self.stack.enter_context(patch('app.contact_management.client_autocomplete._component'))
        self.management = self.stack.enter_context(patch.object(editor, 'render_contact_management'))
        self.stack.enter_context(patch('app.contact_management.contact_ui.safe_list_active_client_contacts', return_value=[self.client]))
        self.stack.enter_context(patch('app.contact_management.contact_ui.safe_list_active_internal_contacts', return_value=[self.internal]))
        self.drafts = self.stack.enter_context(patch.object(draft_ui, '_safe_list_drafts', return_value=[]))
        self.generate = self.stack.enter_context(patch('app.photography_pricing.pdf_generator.generate_page2_pricing_pdf', return_value=b'pdf'))
        self.app = AppTest.from_string('from app.photography_pricing.apparel_estimator import render_photography_pricing\nrender_photography_pricing()\n')
        self.app.session_state['photo_pricing_quote_title'] = 'Current quote'
        self.app.session_state['photo_pricing_comments_project_name_0'] = 'Project A'

    def test_actual_fragment_rerun_skips_setup_and_refreshes_all_derived_outputs(self):
        original = LocalScriptRunner._on_script_finished
        scheduled = []
        actions = [
            ('photo_pricing_on_model_image_quantity', 'double_value', 35),
            ('photo_pricing_comments_add_project', 'trigger_value', True),
            ('photo_pricing_comments_custom_notes', 'string_value', 'Fragment notes'),
            ('photo_pricing_generate_pdf', 'trigger_value', True),
        ]

        def finished(runner, ctx, event, premature_stop):
            original(runner, ctx, event, premature_stop)
            if len(scheduled) == len(actions) or not ctx.new_fragment_ids:
                return
            key, field, value = actions[len(scheduled)]
            scheduled.append(key)
            states = WidgetStates()
            states.widgets.extend(runner._session_state.get_widget_states())
            for widget in states.widgets:
                if widget.id.endswith('-' + key):
                    setattr(widget, field, value)
            runner.request_rerun(RerunData(widget_states=states, fragment_id_queue=list(ctx.new_fragment_ids)))

        with patch.object(LocalScriptRunner, '_on_script_finished', finished):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual([action[0] for action in actions], scheduled)
        self.assertEqual(1, self.management.call_count)
        self.assertEqual(1, self.drafts.call_count)
        expected = build_apparel_quote(ApparelInputs(on_model_image_quantity=35))
        self.assertEqual(editor._money(expected.total), self.app.metric[0].value)
        self.assertEqual(editor._line_table_rows(expected.to_payload()), self.app.dataframe[0].value.to_dict('records'))
        self.assertIn('1 project= 35 images total', self.app.session_state['photo_pricing_page1_comments_payload']['rendered_comments_block'])
        self.assertEqual(2, len(self.app.session_state['photo_pricing_project_rows']))
        self.generate.assert_called_once()
        self.assertEqual(expected, self.generate.call_args.args[0])
        self.assertIn('Fragment notes', self.generate.call_args.kwargs['page1_comments_payload']['rendered_comments_block'])
        # An outside widget still performs a full run and refreshes arguments.
        self.app.text_input(key='photo_pricing_quote_title').set_value('Full refresh').run()
        self.assertEqual(2, self.management.call_count)
        self.assertEqual(2, self.drafts.call_count)

    def test_tiers_model_hours_and_current_pdf_arguments(self):
        self.app.run()
        for quantity in (34, 35, 64, 65):
            self.app.number_input(key='photo_pricing_on_model_image_quantity').set_value(quantity).run()
            expected = build_apparel_quote(ApparelInputs(on_model_image_quantity=quantity))
            self.assertEqual(editor._money(expected.total), self.app.metric[0].value)
            self.assertEqual(editor._line_table_rows(expected.to_payload()), self.app.dataframe[0].value.to_dict('records'))
            self.assertIn(f'{quantity} images total', self.app.text[0].value)
        self.app.radio(key='photo_pricing_model_type').set_value('both').run()
        self.app.number_input(key='photo_pricing_adult_model_hours').set_value(2.0).run()
        self.app.number_input(key='photo_pricing_kid_model_hours').set_value(1.0).run()
        self.app.number_input(key='photo_pricing_comments_colors_0').set_value(8.0).run()
        self.app.text_area(key='photo_pricing_comments_custom_notes').set_value('Latest notes').run()
        self.app.text_input(key='photo_pricing_quote_title').set_value('Updated title').run()
        self.app.button(key='photo_pricing_generate_pdf').click().run()
        self.assertFalse(self.app.exception)
        expected = build_apparel_quote(ApparelInputs(on_model_image_quantity=65, model_type='both', adult_model_hours=2.0, kid_model_hours=1.0))
        self.assertEqual(expected, self.generate.call_args.args[0])
        kwargs = self.generate.call_args.kwargs
        self.assertEqual('Updated title', kwargs['page1_header_payload']['quote_metadata']['quote_title'])
        self.assertEqual('client', kwargs['page1_header_payload']['selected_client']['id'])
        self.assertEqual('internal', kwargs['page1_header_payload']['selected_internal']['id'])
        block = kwargs['page1_comments_payload']['rendered_comments_block']
        for text in ('Colors=8', 'Latest notes', '65 images total'):
            self.assertIn(text, block)

    def test_fragment_project_callbacks_preserve_reindexed_colors(self):
        self.app.run()
        for index in range(3):
            if index:
                self.app.button(key='photo_pricing_comments_add_project').click().run()
            self.app.text_input(key=f'photo_pricing_comments_project_name_{index}').set_value(f'Project {index}').run()
            self.app.number_input(key=f'photo_pricing_comments_colors_{index}').set_value(float(index + 1)).run()
        for removed, names, colors in ((1, ['Project 0', 'Project 2'], [1.0, 3.0]), (0, ['Project 2'], [3.0])):
            self.app.button(key=f'photo_pricing_comments_remove_{removed}').click().run()
            self.assertFalse(self.app.exception)
            for index, name in enumerate(names):
                self.assertEqual(name, self.app.text_input(key=f'photo_pricing_comments_project_name_{index}').value)
                self.assertEqual(colors[index], self.app.number_input(key=f'photo_pricing_comments_colors_{index}').value)
        self.app.button(key='photo_pricing_comments_add_project').click().run()
        self.app.button(key='photo_pricing_comments_remove_1').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(1, self.app.session_state['photo_pricing_page1_comments_payload']['project_count'])
