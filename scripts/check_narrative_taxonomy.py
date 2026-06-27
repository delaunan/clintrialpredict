#!/usr/bin/env python
"""Validate taxonomy metadata needed by narrative packets."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.packet_builder import STRUCTURED_FEATURE_KEYS, TEXT_CONTEXT_KEYS  # noqa: E402
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.prep.pipeline import export_pipeline_taxonomy  # noqa: E402


DEFERRED_TEXT_FIELDS = {"criteria_ui"}


def _validate_taxonomy(taxonomy: dict, *, label: str) -> list[str]:
    errors: list[str] = []
    fields = taxonomy.get("FIELDS") or {}

    required_fields = set(STRUCTURED_FEATURE_KEYS) | set(TEXT_CONTEXT_KEYS) | DEFERRED_TEXT_FIELDS
    for field_id in sorted(required_fields):
        ui = (fields.get(field_id) or {}).get("ui") or {}
        meaning = ui.get("meaning")
        if not isinstance(meaning, str) or not meaning.strip():
            errors.append(f"{label}: {field_id}: missing ui.meaning")

    option_keys_by_field = {}
    for field_id in STRUCTURED_FEATURE_KEYS:
        field = fields.get(field_id) or {}
        options = ((field.get("ui") or {}).get("options")) or []
        if options:
            option_keys_by_field[field_id] = {str(option[0]) for option in options}

    for fixture in get_contract_fixtures():
        fixture_id = fixture.get("fixture_id", "<unknown>")
        structured = ((fixture.get("input_packet") or {}).get("structured_features")) or {}
        for field_id, allowed_values in option_keys_by_field.items():
            value = structured.get(field_id)
            if value is None or str(value) in allowed_values:
                continue
            errors.append(f"{label}: {fixture_id}: {field_id} value {value!r} is not a taxonomy option key")

    if "criteria_ui" in TEXT_CONTEXT_KEYS:
        errors.append(f"{label}: criteria_ui should remain deferred from default V1 narrative text context")

    return errors


def _compare_meanings(committed: dict, regenerated: dict) -> list[str]:
    errors: list[str] = []
    committed_fields = committed.get("FIELDS") or {}
    regenerated_fields = regenerated.get("FIELDS") or {}
    required_fields = set(STRUCTURED_FEATURE_KEYS) | set(TEXT_CONTEXT_KEYS) | DEFERRED_TEXT_FIELDS
    for field_id in sorted(required_fields):
        committed_meaning = ((committed_fields.get(field_id) or {}).get("ui") or {}).get("meaning")
        regenerated_meaning = ((regenerated_fields.get(field_id) or {}).get("ui") or {}).get("meaning")
        if committed_meaning != regenerated_meaning:
            errors.append(f"{field_id}: committed taxonomy meaning differs from regenerated taxonomy meaning")
    return errors


def main() -> int:
    taxonomy_path = ROOT / "models" / "taxonomy_01.json"
    taxonomy = json.loads(taxonomy_path.read_text())
    errors = _validate_taxonomy(taxonomy, label="models/taxonomy_01.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        regenerated_path = Path(tmpdir) / "taxonomy_01.json"
        export_pipeline_taxonomy(regenerated_path)
        regenerated = json.loads(regenerated_path.read_text())
    errors.extend(_validate_taxonomy(regenerated, label="export_pipeline_taxonomy"))
    errors.extend(_compare_meanings(taxonomy, regenerated))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative taxonomy field meanings and default text-field policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
