from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class QuoteDraft:
    id: str
    draft_name: str
    status: str
    client_contact_id: str | None
    internal_contact_id: str | None
    latest_version_number: int
    created_at: datetime | str | None
    updated_at: datetime | str | None


@dataclass(frozen=True)
class QuoteDraftVersion:
    id: str
    draft_id: str
    version_number: int
    payload: dict[str, Any]
    saved_by_contact_id: str | None
    version_note: str | None
    created_at: datetime | str | None
