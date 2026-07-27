from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contact_management.database import ContactManagementError
from app.contact_management.import_service import seed_internal_contacts
from app.contact_management.repositories import ContactValidationError


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Usage: python scripts/seed_internal_contacts.py "private/internal_contacts.json"')
        return 2

    try:
        counts = seed_internal_contacts(argv[1])
    except (ContactManagementError, ContactValidationError, OSError):
        print("Internal contact seed failed.")
        return 1

    print(
        "Internal contact seed complete: "
        f"inserted={counts.inserted}, updated={counts.updated}, skipped={counts.skipped}, errors={counts.errors}"
    )
    return 0 if counts.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

