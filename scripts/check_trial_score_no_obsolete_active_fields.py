#!/usr/bin/env python
"""Guard active Trial Score paths against obsolete scenario-review fields."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.mock_reviewer import FAILURE_PROVIDER_ERROR, review_packet_with_mock  # noqa: E402
from src.narratives.packet_builder import build_review_packet  # noqa: E402
from src.narratives.provider import (  # noqa: E402
    PASS1_REPAIR_STAGE_JSON_SHAPE,
    _failure_result,
    _score_provider_review,
)


FORBIDDEN_KEYS = {
    "strategic_review",
    "strategic_review_assessment",
    "strategic_review_object",
    "strategic_review_analysis",
    "strategic_review_continuity",
    "design_confidence",
    "design_confidence_assessment",
    "design_confidence_subcategories",
    "design_confidence_analysis",
    "design_confidence_contributions",
    "design_confidence_continuity",
    "total_scenario_score",
    "quality_adjustment",
    "final_candidate_score",
    "quality_assessment",
}


def _walk_forbidden(value, *, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_KEYS:
                findings.append(child_path)
            findings.extend(_walk_forbidden(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_walk_forbidden(child, path=f"{path}[{index}]"))
    return findings


def _assert_no_forbidden(errors: list[str], label: str, value) -> None:
    findings = _walk_forbidden(value)
    if findings:
        errors.append(f"{label} contains obsolete fields: {findings[:8]!r}")


def _minimal_packet() -> dict:
    return build_review_packet(
        current_snapshot={
            "snapshot_id": "current",
            "source": "simulation_ptc",
            "score": 65.0,
            "structured_features": {
                "phase_ml": "PHASE2",
                "endpoint_rigor_ml": "SUBJECTIVE_PATIENT_REPORTED",
                "allocation_ml": "RANDOMIZED",
            },
            "operational_assumptions": {
                "planned_enrollment": {"value": 120},
                "planned_sites": {"value": 8},
                "planned_duration_months": {"value": 24.0},
            },
            "model_interpretation": {
                "completion_score": 65.0,
                "score_delta": 0.0,
                "pillar_impacts": [],
            },
            "changed_fields": ["operational_assumptions.planned_sites"],
        },
        previous_snapshot={"snapshot_id": "previous", "score": 65.0},
        baseline_snapshot={"snapshot_id": "baseline", "score": 65.0},
        previous_review_trace={
            "input_hash": "previous-review",
            "iteration_id": 1,
            "status": "reviewed",
            "validation_status": "valid",
            "trial_score": 65.0,
            "pre_reality_score": 65.0,
            "operational_fit_points": 0.0,
            "reality_check_points": 0.0,
            "reality_check_assessment": {"effect": "neutral", "strength": "none", "points": 0.0},
            "validated_review": {
                "completion_outlook_analysis": {"risk_pattern_summary": "Prior completion outlook."},
                "tension_question_options": [
                    {
                        "tension": {
                            "summary": "Prior current tension.",
                            "why_it_matters": "Prior reason.",
                            "supporting_evidence": ["phase_ml"],
                        },
                        "participant_wider_question": {
                            "question": "Prior question?",
                            "supporting_evidence": ["phase_ml"],
                        },
                    }
                ],
                "continuity_update": {"watch_next": "Prior next consideration."},
                "key_questions": {
                    "medical_clinical_development_question": "Prior medical question?",
                    "strategic_development_question": "Prior strategic question?",
                },
            },
            "storyline_state": {
                "active_tension": "Prior current tension.",
                "protected_gains": ["prior gain"],
                "regression_watch": ["prior watch"],
                "next_consideration": "Prior next consideration.",
            },
        },
    )


def main() -> int:
    errors: list[str] = []
    packet = _minimal_packet()

    scoring_source = (ROOT / "src/narratives/scoring.py").read_text(encoding="utf-8")
    for token in (
        "strategic_review",
        "design_confidence",
        "total_scenario_score",
        "validate_review_json",
        "score_validated_review",
    ):
        if token in scoring_source:
            errors.append(f"active scoring facade still contains obsolete token: {token}")

    failure = _failure_result(
        packet,
        provider="openai",
        model_name="test-model",
        status="provider_error",
        message="provider unavailable",
    )
    _assert_no_forbidden(errors, "provider failure scoring", failure.get("scoring"))

    malformed = _score_provider_review(
        packet,
        provider="openai",
        model_name="test-model",
        review={"reality_check": "malformed"},
        provider_metadata={},
    )
    _assert_no_forbidden(errors, "provider malformed scoring", malformed.get("scoring"))
    if (malformed.get("provider_metadata") or {}).get("pass1_validation_stage") != PASS1_REPAIR_STAGE_JSON_SHAPE:
        errors.append("provider malformed top-level Pass 1 response should classify as JSON shape")

    mock_failure = review_packet_with_mock(packet, failure_mode=FAILURE_PROVIDER_ERROR)
    _assert_no_forbidden(errors, "mock failure scoring", mock_failure.get("scoring"))

    mock_no_fixture = review_packet_with_mock(packet)
    _assert_no_forbidden(errors, "mock no-fixture scoring", mock_no_fixture.get("scoring"))

    _assert_no_forbidden(errors, "packet review_context", packet.get("review_context"))
    _assert_no_forbidden(errors, "packet iteration_context", packet.get("iteration_context"))
    if "trial_score_continuity" not in (packet.get("iteration_context") or {}):
        errors.append("packet should keep current trial_score_continuity")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated active Trial Score paths do not emit obsolete scenario-review fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
