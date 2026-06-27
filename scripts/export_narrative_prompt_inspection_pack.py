#!/usr/bin/env python
"""Disabled legacy narrative prompt-inspection exporter."""

from __future__ import annotations

import sys


DISABLED_MESSAGE = (
    "scripts/export_narrative_prompt_inspection_pack.py is disabled. "
    "The previous exporter targeted a superseded narrative contract and should be rebuilt "
    "around the active Trial Score three-pass contract before use."
)


def main() -> int:
    print(DISABLED_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
