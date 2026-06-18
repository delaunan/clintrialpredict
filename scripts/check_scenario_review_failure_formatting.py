#!/usr/bin/env python
"""Validate participant-facing Trial Score review failure formatting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.utils.scenario_review_failure import participant_review_failure_reason  # noqa: E402


def main() -> int:
    cases = [
        (
            {"failure_reason": "Gemini provider call failed: ServerError"},
            "Trial Score review could not be generated (ServerError).",
        ),
        (
            {"failure_reason": "OpenAI provider call failed: TimeoutError"},
            "Trial Score review could not be generated (TimeoutError).",
        ),
        (
            {"failure_reason": "Gemini provider response was not a JSON object."},
            "Trial Score review could not be generated (InvalidResponse).",
        ),
        (
            {"failure_reason": "OpenAI provider response was incomplete: max_output_tokens"},
            "Trial Score review could not be generated (InvalidResponse).",
        ),
        (
            {"failure_reason": "gemini provider is missing an API key."},
            "Trial Score review could not be generated (ConfigurationError).",
        ),
        (
            {"failure_reason": "Narrative provider config is required for gemini."},
            "Trial Score review could not be generated (ConfigurationError).",
        ),
        (
            {"failure_reason": "Unsupported narrative provider: custom"},
            "Trial Score review could not be generated (ConfigurationError).",
        ),
        (
            {"validation_errors": ["completion_outlook_analysis.risk_pattern_summary must be a string"]},
            "completion_outlook_analysis.risk_pattern_summary must be a string",
        ),
        (
            {"repair_warning": "Provider review failed after the repair retry at Reality Check contract: allocation_target_id is required"},
            "Provider review failed after the repair retry at Reality Check contract: allocation_target_id is required",
        ),
        (
            {"provider_metadata": {"validation_retry_final_error": "Provider review failed after the repair retry at Operational Fit contract: combined_operational_fit.rating must be one of"}},
            "Provider review failed after the repair retry at Operational Fit contract: combined_operational_fit.rating must be one of",
        ),
        (
            {"participant_narrative_warning": "Participant narrative failed after the Pass 2 repair retry: trial_score_narrative is incomplete"},
            "Participant narrative failed after the Pass 2 repair retry: trial_score_narrative is incomplete",
        ),
    ]
    errors: list[str] = []
    for trace, expected in cases:
        actual = participant_review_failure_reason(trace)
        if actual != expected:
            errors.append(f"expected {expected!r}, got {actual!r} for {trace!r}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated Trial Score review participant-facing failure formatting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
