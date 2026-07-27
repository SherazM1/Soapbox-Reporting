# scripts/import_hubspot_contacts.py

from __future__ import annotations

import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contact_management.database import ContactManagementError
from app.contact_management.import_service import import_hubspot_contacts
from app.contact_management.repositories import ContactValidationError


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            'Usage: python scripts/import_hubspot_contacts.py '
            '"path/to/hubspot-export.xlsx"'
        )
        return 2

    try:
        counts = import_hubspot_contacts(argv[1])
    except (ContactManagementError, ContactValidationError, OSError) as exc:
        print(
            f"HubSpot contact import failed: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1
    except Exception as exc:
        print(
            f"HubSpot contact import failed unexpectedly: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1

    print(
        "HubSpot contact import complete: "
        f"inserted={counts.inserted}, "
        f"updated={counts.updated}, "
        f"skipped={counts.skipped}, "
        f"errors={counts.errors}"
    )
    return 0 if counts.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))