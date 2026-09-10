import unittest
from contextlib import ExitStack
from unittest.mock import patch

from app.contact_management import contact_cache as cache, repositories as repo
from app.contact_management import contact_ui, import_service


class ContactCacheTests(unittest.TestCase):
    def setUp(self):
        cache.invalidate_client_contacts()
        cache.invalidate_internal_contacts()
        self.addCleanup(cache.invalidate_client_contacts)
        self.addCleanup(cache.invalidate_internal_contacts)

    def test_all_four_lists_hit_repository_once_and_preserve_order(self):
        for kind in ('client', 'internal'):
            for scope in ('active', 'all'):
                name = f'list_{scope}_{kind}_contacts'
                with self.subTest(name=name), patch.object(repo, name, return_value=['z', 'a']) as read:
                    first = getattr(cache, name)()
                    first.append('local change')
                    self.assertEqual(['z', 'a'], getattr(cache, name)())
                    self.assertEqual(1, read.call_count)

    def test_all_four_lists_retry_errors_and_cache_recovery(self):
        for kind in ('client', 'internal'):
            for scope in ('active', 'all'):
                name = f'list_{scope}_{kind}_contacts'
                with self.subTest(name=name), patch.object(repo, name, side_effect=[RuntimeError('offline'), ['recovered']]) as read:
                    with self.assertRaises(RuntimeError):
                        getattr(cache, name)()
                    self.assertEqual(['recovered'], getattr(cache, name)())
                    self.assertEqual(['recovered'], getattr(cache, name)())
                    self.assertEqual(2, read.call_count)

    def test_safe_wrappers_do_not_cache_empty_error_fallback(self):
        for kind in ('client', 'internal'):
            name = f'list_active_{kind}_contacts'
            with self.subTest(kind=kind), patch.object(repo, name, side_effect=[RuntimeError('offline'), ['ok']]) as read:
                safe = getattr(contact_ui, 'safe_' + name)
                self.assertEqual([], safe())
                self.assertEqual(['ok'], safe())
                self.assertEqual(['ok'], safe())
                self.assertEqual(2, read.call_count)

    def test_successful_empty_list_is_cached(self):
        with patch.object(repo, 'list_active_client_contacts', return_value=[]) as read:
            self.assertEqual([], cache.list_active_client_contacts())
            self.assertEqual([], cache.list_active_client_contacts())
            self.assertEqual(1, read.call_count)

    def _assert_mutation_refreshes_only_kind(self, kind, mutate):
        with ExitStack() as stack:
            reads = {}
            for group in ('client', 'internal'):
                for scope in ('active', 'all'):
                    name = f'list_{scope}_{group}_contacts'
                    reads[name] = stack.enter_context(patch.object(repo, name, return_value=['before']))
                    getattr(cache, name)()
                    reads[name].return_value = ['after']
            mutate()
            for name, read in reads.items():
                affected = f'_{kind}_' in name
                self.assertEqual(['after'] if affected else ['before'], getattr(cache, name)())
                self.assertEqual(2 if affected else 1, read.call_count)

    def test_every_repository_mutation_invalidates_both_related_lists(self):
        for kind in ('client', 'internal'):
            operations = ['create', 'update', 'deactivate', 'reactivate']
            if kind == 'client':
                operations.append('upsert')
            for op in operations:
                with self.subTest(kind=kind, operation=op):
                    cache.invalidate_client_contacts()
                    cache.invalidate_internal_contacts()
                    name = f'{op}_{kind}_contact' if op != 'upsert' else 'upsert_client_contact_from_hubspot'
                    args = ('id',) if op in ('update', 'deactivate', 'reactivate') else ()
                    kwargs = {}
                    if op in ('create', 'update', 'upsert'):
                        kwargs = dict(company_name='Acme', first_name='A', last_name='B', email='a@b.com') if kind == 'client' else dict(name='A', title='B', email='a@b.com')
                    row = ('id', None, 'Acme', 'A', 'B', 'a@b.com', True) if kind == 'client' else ('id', 'A', 'B', 'a@b.com', True)
                    with patch.object(repo, '_execute_returning', return_value=row):
                        self._assert_mutation_refreshes_only_kind(kind, lambda: getattr(repo, name)(*args, **kwargs))

    def test_failed_mutation_does_not_clear_warm_cache(self):
        with patch.object(repo, 'list_active_client_contacts', return_value=['before']) as read:
            cache.list_active_client_contacts()
            with patch.object(repo, '_execute_returning', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    repo.deactivate_client_contact('id')
            self.assertEqual(['before'], cache.list_active_client_contacts())
            self.assertEqual(1, read.call_count)

    def test_internal_seed_create_and_update_invalidate(self):
        for existing in (None, repo.InternalContact('id', 'A', 'B', 'a@b.com')):
            with self.subTest(existing=existing):
                cache.invalidate_client_contacts()
                cache.invalidate_internal_contacts()
                with patch.object(import_service, 'load_internal_seed_rows', return_value=[dict(name='A', title='B', email='a@b.com')]), patch.object(repo, 'get_internal_contact_by_email', return_value=existing), patch.object(repo, '_execute_returning', return_value=('id', 'A', 'B', 'a@b.com', True)):
                    self._assert_mutation_refreshes_only_kind('internal', lambda: import_service.seed_internal_contacts('unused.json'))

    def test_hubspot_import_invalidates_client_lists(self):
        from unittest.mock import MagicMock
        workbook = MagicMock()
        with patch('openpyxl.load_workbook', return_value=workbook), patch.object(import_service, '_open_hubspot_rows', return_value=({}, [['row']])), patch.object(import_service, 'normalize_hubspot_row', return_value=import_service.HubspotContactRow(None, 'Acme', 'A', 'B', 'a@b.com')), patch.object(repo, 'get_client_contact_by_hubspot_or_email', return_value=None), patch.object(repo, '_execute_returning', return_value=('id', None, 'Acme', 'A', 'B', 'a@b.com', True)):
            self._assert_mutation_refreshes_only_kind('client', lambda: import_service.import_hubspot_contacts('unused.xlsx'))
