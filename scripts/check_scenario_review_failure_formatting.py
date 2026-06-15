#!/usr/bin/env python
"""Validate participant-facing Scenario Review failure formatting."""

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
            "Scenario Review could not be generated (ServerError).",
        ),
        (
            {"failure_reason": "OpenAI provider call failed: TimeoutError"},
            "Scenario Review could not be generated (TimeoutError).",
        ),
        (
            {"failure_reason": "Gemini provider response was not a JSON object."},
            "Scenario Review could not be generated (InvalidResponse).",
        ),
        (
            {"failure_reason": "OpenAI provider response was incomplete: max_output_tokens"},
            "Scenario Review could not be generated (InvalidResponse).",
        ),
        (
            {"failure_reason": "gemini provider is missing an API key."},
            "Scenario Review could not be generated (ConfigurationError).",
        ),
        (
            {"failure_reason": "Narrative provider config is required for gemini."},
            "Scenario Review could not be generated (ConfigurationError).",
        ),
        (
            {"failure_reason": "Unsupported narrative provider: custom"},
            "Scenario Review could not be generated (ConfigurationError).",
        ),
        (
            {"validation_errors": ["completion_outlook_analysis.risk_pattern_summary must be a string"]},
            "completion_outlook_analysis.risk_pattern_summary must be a string",
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
    print("Validated Scenario Review participant-facing failure formatting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
