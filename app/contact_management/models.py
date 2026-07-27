from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClientContact:
    id: str
    hubspot_record_id: Optional[str]
    company_name: str
    first_name: str
    last_name: str
    email: str
    active: bool = True

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def dropdown_label(self) -> str:
        return f"{self.company_name} — {self.full_name} — {self.email}"


@dataclass(frozen=True)
class InternalContact:
    id: str
    name: str
    title: str
    email: str
    active: bool = True

    @property
    def dropdown_label(self) -> str:
        return f"{self.name} — {self.title} — {self.email}"

