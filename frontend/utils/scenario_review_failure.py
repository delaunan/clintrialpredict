"""Participant-facing Scenario Review failure message formatting."""

from __future__ import annotations

import re
from typing import Any


def participant_review_failure_reason(trace: dict[str, Any] | None) -> str:
    """Return a provider-neutral failure reason suitable for participant UI."""
    if not trace:
        return "Trial Score review did not return a usable response."

    metadata = trace.get("provider_metadata") or {}
    reason = (
        trace.get("failure_reason")
        or trace.get("repair_warning")
        or trace.get("participant_narrative_warning")
        or metadata.get("validation_retry_final_error")
        or metadata.get("pass2_retry_final_error")
        or "; ".join(trace.get("validation_errors") or [])
    )
    if not reason and str(trace.get("status") or "") == "no_fixture_match":
        reason = "No mock Trial Score review fixture matched this live scenario."
    if not reason:
        return "Validation did not produce Reality Check and Trial Score."

    reason_text = str(reason).strip()
    provider_call_error = re.match(
        r"^(?:Gemini|OpenAI) provider call failed:\s*([A-Za-z0-9_]+)",
        reason_text,
        flags=re.I,
    )
    if provider_call_error:
        return f"Trial Score review could not be generated ({provider_call_error.group(1)})."

    provider_response_error = re.match(
        r"^(?:Gemini|OpenAI) provider response was (?:not a JSON object|incomplete)(?::.*)?\.?$",
        reason_text,
        flags=re.I,
    )
    if provider_response_error:
        return "Trial Score review could not be generated (InvalidResponse)."

    provider_config_error = re.match(
        r"^(?:Narrative provider config is required for|(?:(?:Gemini|OpenAI|mock|configured)\s+)?(?:provider\s+)?(?:is missing an API key|config is required)|Unsupported narrative provider:).*$",
        reason_text,
        flags=re.I,
    )
    if provider_config_error:
        return "Trial Score review could not be generated (ConfigurationError)."

    return reason_text
