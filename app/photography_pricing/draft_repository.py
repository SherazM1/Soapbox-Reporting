from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

from app.contact_management.database import DatabaseOperationError, get_connection
from app.photography_pricing.draft_models import QuoteDraft, QuoteDraftVersion
from app.photography_pricing.draft_service import normalize_draft_payload


class DraftRepositoryError(DatabaseOperationError):
    """Raised when a quote draft operation fails."""


class DuplicateDraftVersionError(DraftRepositoryError):
    """Raised when a duplicate draft version number is attempted."""


DRAFT_COLUMNS = (
    "id, draft_name, status, client_contact_id, internal_contact_id, "
    "latest_version_number, created_at, updated_at"
)
VERSION_COLUMNS = "id, draft_id, version_number, payload, saved_by_contact_id, version_note, created_at"


def _connection_context(conn: object | None):
    return nullcontext(conn) if conn is not None else get_connection()


def _json_payload(payload: dict[str, Any]) -> Any:
    normalized = normalize_draft_payload(payload)
    try:
        from psycopg.types.json import Jsonb

        return Jsonb(normalized)
    except Exception:
        return normalized


def _payload_from_db(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return normalize_draft_payload(value)
    if isinstance(value, str):
        return normalize_draft_payload(json.loads(value))
    return normalize_draft_payload(dict(value or {}))


def _is_unique_violation(exc: Exception) -> bool:
    sqlstate = getattr(exc, "sqlstate", "")
    return sqlstate == "23505" or "duplicate" in str(exc).lower() or "unique" in str(exc).lower()


def _translate_db_error(exc: Exception) -> Exception:
    if _is_unique_violation(exc):
        return DuplicateDraftVersionError("Draft version number already exists.")
    return DraftRepositoryError("Quote draft database operation failed.")


def _draft_from_row(row: tuple[Any, ...]) -> QuoteDraft:
    return QuoteDraft(
        id=str(row[0]),
        draft_name=str(row[1]),
        status=str(row[2]),
        client_contact_id=str(row[3]) if row[3] is not None else None,
        internal_contact_id=str(row[4]) if row[4] is not None else None,
        latest_version_number=int(row[5] or 0),
        created_at=row[6],
        updated_at=row[7],
    )


def _version_from_row(row: tuple[Any, ...]) -> QuoteDraftVersion:
    return QuoteDraftVersion(
        id=str(row[0]),
        draft_id=str(row[1]),
        version_number=int(row[2]),
        payload=_payload_from_db(row[3]),
        saved_by_contact_id=str(row[4]) if row[4] is not None else None,
        version_note=row[5],
        created_at=row[6],
    )


def _fetchone(conn: object, sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def _fetchall(conn: object, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def create_draft(
    *,
    draft_name: str,
    payload: dict[str, Any],
    client_contact_id: str | None = None,
    internal_contact_id: str | None = None,
    saved_by_contact_id: str | None = None,
    version_note: str | None = None,
    conn: object | None = None,
) -> tuple[QuoteDraft, QuoteDraftVersion]:
    with _connection_context(conn) as active_conn:
        try:
            draft_row = _fetchone(
                active_conn,
                f"""
                INSERT INTO quote_drafts (draft_name, status, client_contact_id, internal_contact_id)
                VALUES (%s, 'draft', %s, %s)
                RETURNING {DRAFT_COLUMNS}
                """,
                (draft_name.strip() or "Untitled Quote Draft", client_contact_id, internal_contact_id),
            )
            if draft_row is None:
                raise DraftRepositoryError("Quote draft insert returned no row.")
            draft = _draft_from_row(draft_row)
            version = _insert_version(
                active_conn,
                draft_id=draft.id,
                version_number=1,
                payload=payload,
                saved_by_contact_id=saved_by_contact_id,
                version_note=version_note,
            )
            updated = _fetchone(
                active_conn,
                f"""
                UPDATE quote_drafts
                SET latest_version_number = 1
                WHERE id = %s
                RETURNING {DRAFT_COLUMNS}
                """,
                (draft.id,),
            )
            if conn is None:
                active_conn.commit()
            return _draft_from_row(updated or draft_row), version
        except Exception as exc:
            if conn is None:
                active_conn.rollback()
            raise _translate_db_error(exc) from exc


def create_version(
    draft_id: str,
    *,
    payload: dict[str, Any],
    saved_by_contact_id: str | None = None,
    version_note: str | None = None,
    version_number: int | None = None,
    conn: object | None = None,
) -> QuoteDraftVersion:
    with _connection_context(conn) as active_conn:
        try:
            draft_row = _fetchone(
                active_conn,
                f"SELECT {DRAFT_COLUMNS} FROM quote_drafts WHERE id = %s FOR UPDATE",
                (draft_id,),
            )
            if draft_row is None:
                raise DraftRepositoryError("Quote draft was not found.")
            draft = _draft_from_row(draft_row)
            next_version = int(version_number or draft.latest_version_number + 1)
            version = _insert_version(
                active_conn,
                draft_id=draft.id,
                version_number=next_version,
                payload=payload,
                saved_by_contact_id=saved_by_contact_id,
                version_note=version_note,
            )
            _fetchone(
                active_conn,
                f"""
                UPDATE quote_drafts
                SET latest_version_number = %s
                WHERE id = %s
                RETURNING {DRAFT_COLUMNS}
                """,
                (next_version, draft.id),
            )
            if conn is None:
                active_conn.commit()
            return version
        except Exception as exc:
            if conn is None:
                active_conn.rollback()
            raise _translate_db_error(exc) from exc


def _insert_version(
    conn: object,
    *,
    draft_id: str,
    version_number: int,
    payload: dict[str, Any],
    saved_by_contact_id: str | None,
    version_note: str | None,
) -> QuoteDraftVersion:
    row = _fetchone(
        conn,
        f"""
        INSERT INTO quote_draft_versions
            (draft_id, version_number, payload, saved_by_contact_id, version_note)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING {VERSION_COLUMNS}
        """,
        (draft_id, version_number, _json_payload(payload), saved_by_contact_id, version_note),
    )
    if row is None:
        raise DraftRepositoryError("Quote draft version insert returned no row.")
    return _version_from_row(row)


def get_draft(draft_id: str, *, conn: object | None = None) -> QuoteDraft | None:
    with _connection_context(conn) as active_conn:
        row = _fetchone(active_conn, f"SELECT {DRAFT_COLUMNS} FROM quote_drafts WHERE id = %s", (draft_id,))
        return _draft_from_row(row) if row else None


def list_drafts(*, limit: int = 50, conn: object | None = None) -> list[QuoteDraft]:
    with _connection_context(conn) as active_conn:
        rows = _fetchall(
            active_conn,
            f"SELECT {DRAFT_COLUMNS} FROM quote_drafts ORDER BY updated_at DESC LIMIT %s",
            (int(limit),),
        )
        return [_draft_from_row(row) for row in rows]


def get_latest_version(draft_id: str, *, conn: object | None = None) -> QuoteDraftVersion | None:
    with _connection_context(conn) as active_conn:
        row = _fetchone(
            active_conn,
            f"""
            SELECT {VERSION_COLUMNS}
            FROM quote_draft_versions
            WHERE draft_id = %s
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (draft_id,),
        )
        return _version_from_row(row) if row else None


def list_versions(draft_id: str, *, conn: object | None = None) -> list[QuoteDraftVersion]:
    with _connection_context(conn) as active_conn:
        rows = _fetchall(
            active_conn,
            f"""
            SELECT {VERSION_COLUMNS}
            FROM quote_draft_versions
            WHERE draft_id = %s
            ORDER BY version_number DESC
            """,
            (draft_id,),
        )
        return [_version_from_row(row) for row in rows]


def get_version(draft_id: str, version_number: int, *, conn: object | None = None) -> QuoteDraftVersion | None:
    with _connection_context(conn) as active_conn:
        row = _fetchone(
            active_conn,
            f"""
            SELECT {VERSION_COLUMNS}
            FROM quote_draft_versions
            WHERE draft_id = %s AND version_number = %s
            """,
            (draft_id, int(version_number)),
        )
        return _version_from_row(row) if row else None


def restore_version_as_latest(
    draft_id: str,
    version_number: int,
    *,
    saved_by_contact_id: str | None = None,
    version_note: str | None = None,
    conn: object | None = None,
) -> QuoteDraftVersion:
    with _connection_context(conn) as active_conn:
        try:
            original = get_version(draft_id, version_number, conn=active_conn)
            if original is None:
                raise DraftRepositoryError("Quote draft version was not found.")
            restored = create_version(
                draft_id,
                payload=original.payload,
                saved_by_contact_id=saved_by_contact_id,
                version_note=version_note,
                conn=active_conn,
            )
            if conn is None:
                active_conn.commit()
            return restored
        except Exception as exc:
            if conn is None:
                active_conn.rollback()
            raise _translate_db_error(exc) from exc
