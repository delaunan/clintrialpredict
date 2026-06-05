#!/usr/bin/env python
"""Validate deterministic narrative review packet assembly."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import (  # noqa: E402
    ACTIVE_OPERATIONAL_ASSUMPTION_KEYS,
    DIRECT_XGBOOST_SHAP_FIELDS,
    STRUCTURED_FEATURE_KEYS,
    build_review_packet_from_fixture,
    stable_packet_hash,
)


def _check_packet(fixture: dict, errors: list[str]) -> None:
    fixture_id = fixture["fixture_id"]
    source_packet = fixture["input_packet"]
    packet = build_review_packet_from_fixture(fixture)

    for key in (
        "prompt_version",
        "rubric_version",
        "mode",
        "trial_identity",
        "text_context",
        "structured_features",
        "operational_assumptions",
        "model_interpretation",
        "iteration_context",
        "input_hash",
    ):
        if key not in packet:
            errors.append(f"{fixture_id}: missing packet.{key}")

    if packet.get("prompt_version") != source_packet.get("prompt_version"):
        errors.append(f"{fixture_id}: prompt_version changed")
    if packet.get("rubric_version") != source_packet.get("rubric_version"):
        errors.append(f"{fixture_id}: rubric_version changed")
    if packet.get("mode") != source_packet.get("mode"):
        errors.append(f"{fixture_id}: mode changed")

    missing_features = set(STRUCTURED_FEATURE_KEYS).difference(packet.get("structured_features", {}))
    if missing_features:
        errors.append(f"{fixture_id}: missing structured features {sorted(missing_features)}")

    missing_operational = set(ACTIVE_OPERATIONAL_ASSUMPTION_KEYS).difference(packet.get("operational_assumptions", {}))
    if missing_operational:
        errors.append(f"{fixture_id}: missing operational assumptions {sorted(missing_operational)}")

    model = packet.get("model_interpretation", {})
    source_model = source_packet.get("model_interpretation", {})
    if model.get("completion_score") != source_model.get("completion_score"):
        errors.append(f"{fixture_id}: completion_score changed")
    if model.get("score_delta") != source_model.get("score_delta"):
        errors.append(f"{fixture_id}: score_delta changed")
    if model.get("direct_xgboost_shap_fields") != list(DIRECT_XGBOOST_SHAP_FIELDS):
        errors.append(f"{fixture_id}: direct_xgboost_shap_fields mismatch")

    iteration = packet.get("iteration_context", {})
    source_iteration = source_packet.get("iteration_context", {})
    if iteration.get("changed_fields") != source_iteration.get("changed_fields"):
        errors.append(f"{fixture_id}: changed_fields changed")
    if iteration.get("compact_storyline_memory") != source_iteration.get("compact_storyline_memory"):
        errors.append(f"{fixture_id}: compact_storyline_memory changed")

    packet_without_hash = dict(packet)
    input_hash = packet_without_hash.pop("input_hash", None)
    if stable_packet_hash(packet_without_hash) != input_hash:
        errors.append(f"{fixture_id}: input_hash is not stable over packet content")


def main() -> int:
    errors: list[str] = []
    fixtures = get_contract_fixtures()
    for fixture in fixtures:
        _check_packet(fixture, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Validated deterministic packet assembly for {len(fixtures)} narrative fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
