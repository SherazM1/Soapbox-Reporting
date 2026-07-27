from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ContactManagementError(Exception):
    """Base error for contact-management failures."""


class DatabaseConfigurationError(ContactManagementError):
    """Raised when database configuration is missing or unusable."""


class DatabaseOperationError(ContactManagementError):
    """Raised when a database operation fails without exposing credentials."""


def resolve_database_url() -> str:
    url = _database_url_from_streamlit_secrets() or os.environ.get("DATABASE_URL", "")
    url = url.strip()
    if not url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")
    return url


def _database_url_from_streamlit_secrets() -> str:
    try:
        import streamlit as st

        value = st.secrets.get("DATABASE_URL", "")
    except Exception:
        return ""
    return str(value or "")


@contextmanager
def get_connection() -> Iterator[object]:
    try:
        import psycopg
    except Exception as exc:
        raise DatabaseConfigurationError("psycopg is not installed.") from None

    try:
        with psycopg.connect(resolve_database_url()) as conn:
            yield conn
    except ContactManagementError:
        raise
    except Exception:
        raise DatabaseOperationError("Database operation failed.") from None


def schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "database" / "contacts_schema.sql"


def initialize_schema() -> None:
    sql = schema_path().read_text(encoding="utf-8")
    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise DatabaseOperationError("Contact schema initialization failed.") from None
