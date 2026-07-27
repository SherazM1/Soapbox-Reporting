from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contact_management.database import ContactManagementError, initialize_schema


def main() -> int:
    try:
        initialize_schema()
    except ContactManagementError:
        print("Contact database initialization failed.")
        return 1
    print("Contact database initialized successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

