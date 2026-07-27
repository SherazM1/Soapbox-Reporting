from __future__ import annotations

import re
from contextlib import nullcontext
from typing import Any, Iterable, Optional

from app.contact_management.database import DatabaseOperationError, get_connection
from app.contact_management.models import ClientContact, InternalContact


class ContactValidationError(ValueError):
    """Raised when contact input fails validation."""


class DuplicateContactError(ContactValidationError):
    """Raised when a contact violates a unique email or HubSpot constraint."""


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

CLIENT_COLUMNS = "id, hubspot_record_id, company_name, first_name, last_name, email, active"
INTERNAL_COLUMNS = "id, name, title, email, active"


def _clean_required(value: Any, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ContactValidationError(f"{label} is required.")
    return cleaned


def normalize_email(value: Any) -> str:
    email = _clean_required(value, "Email").lower()
    if not EMAIL_RE.match(email):
        raise ContactValidationError("Email must be a valid email address.")
    return email


def normalize_hubspot_record_id(value: Any) -> Optional[str]:
    cleaned = str(value or "").strip()
    return cleaned or None


def _is_unique_violation(exc: Exception) -> bool:
    sqlstate = getattr(exc, "sqlstate", "")
    return sqlstate == "23505" or "duplicate" in str(exc).lower() or "unique" in str(exc).lower()


def _translate_db_error(exc: Exception) -> Exception:
    if _is_unique_violation(exc):
        return DuplicateContactError("A contact with that email or HubSpot record ID already exists.")
    return DatabaseOperationError("Contact database operation failed.")


def _connection_context(conn: object | None):
    return nullcontext(conn) if conn is not None else get_connection()


def _fetchall(sql: str, params: tuple[Any, ...] = (), *, conn: object | None = None) -> list[tuple[Any, ...]]:
    with _connection_context(conn) as active_conn:
        with active_conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


def _fetchone(sql: str, params: tuple[Any, ...] = (), *, conn: object | None = None) -> Optional[tuple[Any, ...]]:
    with _connection_context(conn) as active_conn:
        with active_conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def _execute_returning(sql: str, params: tuple[Any, ...], *, conn: object | None = None) -> tuple[Any, ...]:
    with _connection_context(conn) as active_conn:
        try:
            with active_conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            if conn is None:
                active_conn.commit()
            if row is None:
                raise DatabaseOperationError("Contact operation returned no row.")
            return row
        except Exception as exc:
            if conn is None:
                active_conn.rollback()
            translated = _translate_db_error(exc)
            raise translated from exc


def _client_from_row(row: tuple[Any, ...]) -> ClientContact:
    return ClientContact(
        id=str(row[0]),
        hubspot_record_id=row[1],
        company_name=row[2],
        first_name=row[3],
        last_name=row[4],
        email=row[5],
        active=bool(row[6]),
    )


def _internal_from_row(row: tuple[Any, ...]) -> InternalContact:
    return InternalContact(id=str(row[0]), name=row[1], title=row[2], email=row[3], active=bool(row[4]))


def _client_params(
    *,
    company_name: Any,
    first_name: Any,
    last_name: Any,
    email: Any,
    hubspot_record_id: Any = None,
    active: bool = True,
) -> tuple[Any, ...]:
    return (
        normalize_hubspot_record_id(hubspot_record_id),
        _clean_required(company_name, "Company name"),
        _clean_required(first_name, "First name"),
        _clean_required(last_name, "Last name"),
        normalize_email(email),
        bool(active),
    )


def _internal_params(*, name: Any, title: Any, email: Any, active: bool = True) -> tuple[Any, ...]:
    return (_clean_required(name, "Name"), _clean_required(title, "Title"), normalize_email(email), bool(active))


def list_active_client_contacts(*, conn: object | None = None) -> list[ClientContact]:
    rows = _fetchall(
        f"SELECT {CLIENT_COLUMNS} FROM client_contacts WHERE active = TRUE ORDER BY company_name, last_name, first_name",
        conn=conn,
    )
    return [_client_from_row(row) for row in rows]


def list_all_client_contacts(*, conn: object | None = None) -> list[ClientContact]:
    rows = _fetchall(
        f"SELECT {CLIENT_COLUMNS} FROM client_contacts ORDER BY active DESC, company_name, last_name, first_name",
        conn=conn,
    )
    return [_client_from_row(row) for row in rows]


def get_client_contact(contact_id: str, *, conn: object | None = None) -> Optional[ClientContact]:
    row = _fetchone(f"SELECT {CLIENT_COLUMNS} FROM client_contacts WHERE id = %s", (str(contact_id),), conn=conn)
    return _client_from_row(row) if row else None


def create_client_contact(
    *,
    company_name: Any,
    first_name: Any,
    last_name: Any,
    email: Any,
    hubspot_record_id: Any = None,
    active: bool = True,
    conn: object | None = None,
) -> ClientContact:
    params = _client_params(
        hubspot_record_id=hubspot_record_id,
        company_name=company_name,
        first_name=first_name,
        last_name=last_name,
        email=email,
        active=active,
    )
    row = _execute_returning(
        f"""
        INSERT INTO client_contacts (hubspot_record_id, company_name, first_name, last_name, email, active)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING {CLIENT_COLUMNS}
        """,
        params,
        conn=conn,
    )
    return _client_from_row(row)


def update_client_contact(
    contact_id: str,
    *,
    company_name: Any,
    first_name: Any,
    last_name: Any,
    email: Any,
    hubspot_record_id: Any = None,
    active: bool = True,
    conn: object | None = None,
) -> ClientContact:
    params = (
        *_client_params(
            hubspot_record_id=hubspot_record_id,
            company_name=company_name,
            first_name=first_name,
            last_name=last_name,
            email=email,
            active=active,
        ),
        str(contact_id),
    )
    row = _execute_returning(
        f"""
        UPDATE client_contacts
        SET hubspot_record_id = %s, company_name = %s, first_name = %s, last_name = %s, email = %s, active = %s
        WHERE id = %s
        RETURNING {CLIENT_COLUMNS}
        """,
        params,
        conn=conn,
    )
    return _client_from_row(row)


def deactivate_client_contact(contact_id: str, *, conn: object | None = None) -> Optional[ClientContact]:
    row = _execute_returning(
        f"UPDATE client_contacts SET active = FALSE WHERE id = %s RETURNING {CLIENT_COLUMNS}",
        (str(contact_id),),
        conn=conn,
    )
    return _client_from_row(row) if row else None


def reactivate_client_contact(contact_id: str, *, conn: object | None = None) -> Optional[ClientContact]:
    row = _execute_returning(
        f"UPDATE client_contacts SET active = TRUE WHERE id = %s RETURNING {CLIENT_COLUMNS}",
        (str(contact_id),),
        conn=conn,
    )
    return _client_from_row(row) if row else None


def get_client_contact_by_hubspot_or_email(
    *, hubspot_record_id: Any = None, email: Any = None, conn: object | None = None
) -> Optional[ClientContact]:
    normalized_hubspot_id = normalize_hubspot_record_id(hubspot_record_id)
    normalized_email = normalize_email(email) if email else ""
    if normalized_hubspot_id:
        row = _fetchone(
            f"SELECT {CLIENT_COLUMNS} FROM client_contacts WHERE hubspot_record_id = %s",
            (normalized_hubspot_id,),
            conn=conn,
        )
        if row:
            return _client_from_row(row)
    if normalized_email:
        row = _fetchone(
            f"SELECT {CLIENT_COLUMNS} FROM client_contacts WHERE LOWER(email) = %s",
            (normalized_email,),
            conn=conn,
        )
        return _client_from_row(row) if row else None
    return None


def upsert_client_contact_from_hubspot(
    *,
    company_name: Any,
    first_name: Any,
    last_name: Any,
    email: Any,
    hubspot_record_id: Any = None,
    active: bool = True,
    conn: object | None = None,
) -> ClientContact:
    params = _client_params(
        hubspot_record_id=hubspot_record_id,
        company_name=company_name,
        first_name=first_name,
        last_name=last_name,
        email=email,
        active=active,
    )
    if params[0]:
        conflict_clause = "ON CONFLICT (hubspot_record_id) WHERE hubspot_record_id IS NOT NULL DO UPDATE"
    else:
        conflict_clause = "ON CONFLICT ((LOWER(email))) DO UPDATE"
    row = _execute_returning(
        f"""
        INSERT INTO client_contacts (hubspot_record_id, company_name, first_name, last_name, email, active)
        VALUES (%s, %s, %s, %s, %s, %s)
        {conflict_clause}
        SET hubspot_record_id = COALESCE(EXCLUDED.hubspot_record_id, client_contacts.hubspot_record_id),
            company_name = EXCLUDED.company_name,
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            email = EXCLUDED.email,
            active = EXCLUDED.active
        RETURNING {CLIENT_COLUMNS}
        """,
        params,
        conn=conn,
    )
    return _client_from_row(row)


def list_active_internal_contacts(*, conn: object | None = None) -> list[InternalContact]:
    rows = _fetchall(
        f"SELECT {INTERNAL_COLUMNS} FROM internal_contacts WHERE active = TRUE ORDER BY name",
        conn=conn,
    )
    return [_internal_from_row(row) for row in rows]


def list_all_internal_contacts(*, conn: object | None = None) -> list[InternalContact]:
    rows = _fetchall(
        f"SELECT {INTERNAL_COLUMNS} FROM internal_contacts ORDER BY active DESC, name",
        conn=conn,
    )
    return [_internal_from_row(row) for row in rows]


def get_internal_contact(contact_id: str, *, conn: object | None = None) -> Optional[InternalContact]:
    row = _fetchone(f"SELECT {INTERNAL_COLUMNS} FROM internal_contacts WHERE id = %s", (str(contact_id),), conn=conn)
    return _internal_from_row(row) if row else None


def create_internal_contact(
    *, name: Any, title: Any, email: Any, active: bool = True, conn: object | None = None
) -> InternalContact:
    row = _execute_returning(
        f"""
        INSERT INTO internal_contacts (name, title, email, active)
        VALUES (%s, %s, %s, %s)
        RETURNING {INTERNAL_COLUMNS}
        """,
        _internal_params(name=name, title=title, email=email, active=active),
        conn=conn,
    )
    return _internal_from_row(row)


def update_internal_contact(
    contact_id: str,
    *,
    name: Any,
    title: Any,
    email: Any,
    active: bool = True,
    conn: object | None = None,
) -> InternalContact:
    row = _execute_returning(
        f"""
        UPDATE internal_contacts
        SET name = %s, title = %s, email = %s, active = %s
        WHERE id = %s
        RETURNING {INTERNAL_COLUMNS}
        """,
        (*_internal_params(name=name, title=title, email=email, active=active), str(contact_id)),
        conn=conn,
    )
    return _internal_from_row(row)


def deactivate_internal_contact(contact_id: str, *, conn: object | None = None) -> Optional[InternalContact]:
    row = _execute_returning(
        f"UPDATE internal_contacts SET active = FALSE WHERE id = %s RETURNING {INTERNAL_COLUMNS}",
        (str(contact_id),),
        conn=conn,
    )
    return _internal_from_row(row) if row else None


def reactivate_internal_contact(contact_id: str, *, conn: object | None = None) -> Optional[InternalContact]:
    row = _execute_returning(
        f"UPDATE internal_contacts SET active = TRUE WHERE id = %s RETURNING {INTERNAL_COLUMNS}",
        (str(contact_id),),
        conn=conn,
    )
    return _internal_from_row(row) if row else None


def get_internal_contact_by_email(email: Any, *, conn: object | None = None) -> Optional[InternalContact]:
    normalized_email = normalize_email(email)
    row = _fetchone(
        f"SELECT {INTERNAL_COLUMNS} FROM internal_contacts WHERE LOWER(email) = %s",
        (normalized_email,),
        conn=conn,
    )
    return _internal_from_row(row) if row else None
