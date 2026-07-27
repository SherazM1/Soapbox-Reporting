import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook

from app.contact_management.import_service import header_mapping, import_hubspot_contacts, normalize_hubspot_row
from app.contact_management.models import ClientContact, InternalContact
from app.contact_management.repositories import (
    ContactValidationError,
    DuplicateContactError,
    create_client_contact,
    list_active_client_contacts,
    normalize_email,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, rows=None, error=None):
        self.rows = list(rows or [])
        self.error = error
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows=None, error=None):
        self.cursor_obj = FakeCursor(rows=rows, error=error)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class UniqueViolationLike(Exception):
    sqlstate = "23505"


class ContactManagementTests(unittest.TestCase):
    def test_schema_contains_contact_tables_triggers_and_migration(self):
        schema = (ROOT / "database" / "contacts_schema.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS client_contacts", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS internal_contacts", schema)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS client_contacts_email_lower_uidx", schema)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS internal_contacts_email_lower_uidx", schema)
        self.assertIn("contact_management_set_updated_at", schema)
        self.assertIn("contacts_v1", schema)

    def test_email_normalization_and_validation(self):
        self.assertEqual("person@example.com", normalize_email("  PERSON@Example.COM  "))
        with self.assertRaises(ContactValidationError):
            normalize_email("not-an-email")

    def test_dropdown_labels(self):
        client = ClientContact(
            id="client-id",
            hubspot_record_id="hs-1",
            company_name="Acme",
            first_name="Ada",
            last_name="Lovelace",
            email="ada@example.com",
        )
        internal = InternalContact(
            id="internal-id",
            name="Grace Hopper",
            title="Creative Lead",
            email="grace@example.com",
        )

        self.assertEqual("Ada Lovelace", client.full_name)
        self.assertEqual("Acme — Ada Lovelace — ada@example.com", client.dropdown_label)
        self.assertEqual("Grace Hopper — Creative Lead — grace@example.com", internal.dropdown_label)

    def test_hubspot_header_matching_and_row_normalization(self):
        mapping = header_mapping(
            ["HubSpot Record ID", "First Name", "Last Name", "Email Address", "Associated Company"]
        )
        row = normalize_hubspot_row([" 123 ", " Jane ", " Doe ", " JANE@EXAMPLE.COM ", " Brand Co "], mapping)

        self.assertEqual("123", row.hubspot_record_id)
        self.assertEqual("Brand Co", row.company_name)
        self.assertEqual("Jane", row.first_name)
        self.assertEqual("Doe", row.last_name)
        self.assertEqual("jane@example.com", row.email)

    def test_hubspot_excel_import_counts_empty_and_invalid_rows_without_printing_contacts(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hubspot.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Record ID", "First", "Last", "Email", "Company Name"])
            sheet.append(["1", "Jane", "Doe", "jane@example.com", "Brand"])
            sheet.append([None, None, None, None, None])
            sheet.append(["2", "", "Missing", "bad-email", "Brand"])
            workbook.save(path)

            conn = FakeConnection(
                rows=[
                    (
                        "id-1",
                        "1",
                        "Brand",
                        "Jane",
                        "Doe",
                        "jane@example.com",
                        True,
                    )
                ]
            )
            counts = import_hubspot_contacts(path, conn=conn)

        self.assertEqual(0, counts.inserted)
        self.assertEqual(1, counts.updated)
        self.assertEqual(2, counts.skipped)
        self.assertEqual(0, counts.errors)


    def test_active_client_filtering_uses_active_where_and_order(self):
        conn = FakeConnection(
            rows=[
                (
                    "id-1",
                    None,
                    "Company",
                    "First",
                    "Last",
                    "first@example.com",
                    True,
                )
            ]
        )

        contacts = list_active_client_contacts(conn=conn)
        sql, params = conn.cursor_obj.executed[0]

        self.assertEqual(1, len(contacts))
        self.assertIn("WHERE active = TRUE", sql)
        self.assertIn("ORDER BY company_name, last_name, first_name", sql)
        self.assertEqual((), params)

    def test_duplicate_error_handling(self):
        conn = FakeConnection(error=UniqueViolationLike("duplicate key"))

        with self.assertRaises(DuplicateContactError):
            create_client_contact(
                company_name="Brand",
                first_name="First",
                last_name="Last",
                email="first@example.com",
                conn=conn,
            )

    def test_repository_sql_uses_parameters(self):
        conn = FakeConnection(
            rows=[
                (
                    "id-1",
                    None,
                    "Brand",
                    "First",
                    "Last",
                    "first@example.com",
                    True,
                )
            ]
        )

        create_client_contact(
            company_name="Brand",
            first_name="First",
            last_name="Last",
            email="FIRST@EXAMPLE.COM",
            conn=conn,
        )
        sql, params = conn.cursor_obj.executed[0]

        self.assertIn("VALUES (%s, %s, %s, %s, %s, %s)", sql)
        self.assertNotIn("FIRST@EXAMPLE.COM", sql)
        self.assertEqual((None, "Brand", "First", "Last", "first@example.com", True), params)


if __name__ == "__main__":
    unittest.main()
