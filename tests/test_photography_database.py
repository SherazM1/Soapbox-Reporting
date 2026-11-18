import os
import unittest
from unittest.mock import MagicMock, patch

from app.contact_management import database


class CloudDatabaseTests(unittest.TestCase):
    def test_missing_configuration_fails_without_creating_a_local_database(self):
        with (
            patch.object(database, "_database_url_from_streamlit_secrets", return_value=""),
            patch.dict(os.environ, {}, clear=True),
            patch("sqlite3.connect") as local_connect,
        ):
            with self.assertRaisesRegex(database.DatabaseConfigurationError, "DATABASE_URL is not configured"):
                with database.get_connection():
                    self.fail("Missing configuration must not yield a connection.")
        local_connect.assert_not_called()

    def test_streamlit_secrets_take_precedence_over_environment(self):
        with (
            patch.object(database, "_database_url_from_streamlit_secrets", return_value="postgresql://cloud-secrets"),
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://cloud-environment"}),
        ):
            self.assertEqual("postgresql://cloud-secrets", database.resolve_database_url())

    def test_environment_configuration_is_used_when_secrets_are_absent(self):
        with (
            patch.object(database, "_database_url_from_streamlit_secrets", return_value=""),
            patch.dict(os.environ, {"DATABASE_URL": "  postgresql://cloud-environment  "}),
        ):
            self.assertEqual("postgresql://cloud-environment", database.resolve_database_url())

    def test_opening_cloud_connection_does_not_initialize_or_import_data(self):
        driver = MagicMock()
        connection = driver.connect.return_value.__enter__.return_value
        with (
            patch.object(database, "resolve_database_url", return_value="postgresql://existing-cloud"),
            patch.dict("sys.modules", {"psycopg": driver}),
            patch("sqlite3.connect") as local_connect,
        ):
            with database.get_connection() as actual:
                self.assertIs(connection, actual)
        driver.connect.assert_called_once_with("postgresql://existing-cloud")
        connection.cursor.assert_not_called()
        local_connect.assert_not_called()

    def test_cloud_connection_failure_never_falls_back_or_exposes_credentials(self):
        class DriverError(Exception):
            pass

        driver = MagicMock()
        driver.Error = DriverError
        driver.connect.side_effect = DriverError("private connection details")
        with (
            patch.object(database, "resolve_database_url", return_value="postgresql://existing-cloud"),
            patch.dict("sys.modules", {"psycopg": driver}),
            patch("sqlite3.connect") as local_connect,
        ):
            with self.assertRaises(database.DatabaseOperationError) as error:
                with database.get_connection():
                    self.fail("A failed cloud connection must not yield a connection.")
        self.assertNotIn("private connection details", str(error.exception))
        local_connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
