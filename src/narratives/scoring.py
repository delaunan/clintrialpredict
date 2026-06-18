"""Current Trial Score validation and deterministic scoring facade."""

from __future__ import annotations

from typing import Any

from src.narratives.trial_score_contract import score_pass1_review, validate_pass1_review


def validate_and_score_review(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Validate and score the current Trial Score Pass 1 contract only."""
    candidate_review = review if isinstance(review, dict) else {}
    validated_review = validate_pass1_review(packet, candidate_review)
    scoring = score_pass1_review(packet, candidate_review)
    return {
        "validated_review": validated_review,
        "scoring": scoring,
    }
