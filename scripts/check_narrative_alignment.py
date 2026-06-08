#!/usr/bin/env python
"""Validate deterministic narrative structured/text alignment checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.alignment import unresolved_clarification_issues  # noqa: E402
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402


def _issue_ids(packet: dict) -> set[str]:
    return {str(item.get("issue_id")) for item in unresolved_clarification_issues(packet)}


def main() -> int:
    errors: list[str] = []
    fixtures = {item["fixture_id"]: item for item in get_contract_fixtures()}

    baseline_packet = build_review_packet_from_fixture(fixtures["baseline_hidden_review_v1"])
    if _issue_ids(baseline_packet):
        errors.append("hidden baseline packet should not require alignment clarification")

    endpoint_packet = build_review_packet_from_fixture(
        fixtures["endpoint_structure_text_alignment_requires_clarification_v1"]
    )
    if _issue_ids(endpoint_packet) != {"endpoint_structure_text_mismatch"}:
        errors.append("endpoint structure fixture should require endpoint_structure_text_mismatch")

    explained_packet = build_review_packet_from_fixture(
        fixtures["endpoint_structure_text_alignment_explained_v1"]
    )
    if _issue_ids(explained_packet):
        errors.append("explained endpoint structure fixture should not require clarification")

    placebo_packet = {
        **endpoint_packet,
        "structured_features": {"has_placebo_ml": "1"},
        "structured_feature_display_values": {"has_placebo_ml": "Yes"},
        "text_context": {"summary_ui": "This study uses standard care without placebo."},
        "clarification_context": {"user_clarifications": []},
    }
    if _issue_ids(placebo_packet) != {"placebo_text_structured_mismatch"}:
        errors.append("placebo mismatch should require placebo_text_structured_mismatch")

    placebo_added_text_packet = {
        **endpoint_packet,
        "structured_features": {"has_placebo_ml": "0"},
        "structured_feature_display_values": {"has_placebo_ml": "No"},
        "text_context": {"summary_ui": "The participant added a placebo-control design to improve comparison."},
        "clarification_context": {"user_clarifications": []},
    }
    if _issue_ids(placebo_added_text_packet) != {"placebo_text_structured_mismatch"}:
        errors.append("placebo-control text with structured No should require placebo_text_structured_mismatch")

    no_placebo_control_packet = {
        **endpoint_packet,
        "structured_features": {"has_placebo_ml": "0"},
        "structured_feature_display_values": {"has_placebo_ml": "No"},
        "text_context": {"summary_ui": "The participant clarified that there is no placebo-control design."},
        "clarification_context": {"user_clarifications": []},
    }
    if _issue_ids(no_placebo_control_packet):
        errors.append("no placebo-control text with structured No should not require clarification")

    endpoint_rigor_packet = {
        **endpoint_packet,
        "structured_features": {"endpoint_rigor_ml": "HARD_CLINICAL"},
        "structured_feature_display_values": {"endpoint_rigor_ml": "Hard Clinical (Survival/Death)"},
        "text_context": {
            "primary_outcomes_ui": "Primary endpoints are immunogenicity antibody titers and seroprotection rate."
        },
        "clarification_context": {"user_clarifications": []},
    }
    if _issue_ids(endpoint_rigor_packet) != {"endpoint_rigor_text_mismatch"}:
        errors.append("hard-clinical endpoint rigor with immunogenicity text should require endpoint_rigor_text_mismatch")

    endpoint_rigor_explained_packet = {
        **endpoint_rigor_packet,
        "clarification_context": {
            "user_clarifications": [
                {
                    "issue_id": "endpoint_rigor_text_mismatch",
                    "explanation": "The hard clinical value is intentional because later adjudicated clinical protection is the decision endpoint.",
                }
            ]
        },
    }
    if _issue_ids(endpoint_rigor_explained_packet):
        errors.append("explained endpoint-rigor mismatch should not require clarification")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated deterministic narrative alignment checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
