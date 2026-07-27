from __future__ import annotations

import calendar
import random
import string
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


REFERENCE_ALPHABET = string.ascii_uppercase + string.digits


@dataclass(frozen=True)
class QuoteMetadata:
    quote_title: str
    reference_number: str
    quote_created_date: date
    quote_expiration_date: date

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quote_created_date"] = self.quote_created_date.isoformat()
        payload["quote_expiration_date"] = self.quote_expiration_date.isoformat()
        return payload


def add_calendar_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def generate_reference_number(now: datetime | None = None, suffix: str | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix_value = suffix or "".join(random.choice(REFERENCE_ALPHABET) for _ in range(4))
    return f"{timestamp}-{suffix_value.upper()[:4]}"


def ensure_quote_metadata_state(state: dict[str, Any], *, today: date | None = None) -> None:
    current_date = today or date.today()
    state.setdefault("photo_pricing_reference_number", generate_reference_number())
    state.setdefault("photo_pricing_quote_created_date", current_date)
    state.setdefault("photo_pricing_expiration_overridden", False)

    created_date = state.get("photo_pricing_quote_created_date") or current_date
    state.setdefault("photo_pricing_quote_expiration_date", add_calendar_months(created_date, 3))
    state.setdefault("photo_pricing_previous_created_date", created_date)

    previous_created = state.get("photo_pricing_previous_created_date")
    if created_date != previous_created:
        if not state.get("photo_pricing_expiration_overridden", False):
            state["photo_pricing_quote_expiration_date"] = add_calendar_months(created_date, 3)
        state["photo_pricing_previous_created_date"] = created_date


def mark_quote_expiration_overridden(state: dict[str, Any]) -> None:
    state["photo_pricing_expiration_overridden"] = True


def build_quote_metadata(
    *,
    quote_title: Any,
    reference_number: Any,
    quote_created_date: date,
    quote_expiration_date: date,
) -> QuoteMetadata:
    return QuoteMetadata(
        quote_title=str(quote_title or "").strip(),
        reference_number=str(reference_number or "").strip(),
        quote_created_date=quote_created_date,
        quote_expiration_date=quote_expiration_date,
    )

