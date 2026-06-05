#!/usr/bin/env python
"""Validate serious-game narrative V1 contract fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import (  # noqa: E402
    REQUIRED_SCENARIO_TYPES,
    get_contract_fixtures,
    validate_contract_fixtures,
)


def main() -> int:
    fixtures = get_contract_fixtures()
    errors = validate_contract_fixtures(fixtures)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    scenario_types = sorted({fixture["scenario_type"] for fixture in fixtures})
    print(f"Validated {len(fixtures)} narrative contract fixtures.")
    print(f"Required scenario types: {', '.join(sorted(REQUIRED_SCENARIO_TYPES))}")
    print(f"Fixture scenario types: {', '.join(scenario_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
