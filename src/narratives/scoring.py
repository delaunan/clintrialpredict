"""Current Trial Score validation and deterministic scoring facade."""

from __future__ import annotations

from typing import Any

from src.narratives.trial_score_contract import (
    score_pass1_review,
    validate_pass1_review,
    validate_scoring_adjudication,
)


def validate_and_score_review(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Validate the current Trial Score evidence pass only."""
    candidate_review = review if isinstance(review, dict) else {}
    validated_review = validate_pass1_review(packet, candidate_review)
    scoring = score_pass1_review(packet, candidate_review)
    return {
        "validated_review": validated_review,
        "scoring": scoring,
    }


def validate_and_score_adjudication(
    packet: dict[str, Any],
    pass1_review: dict[str, Any],
    scoring_review: dict[str, Any],
) -> dict[str, Any]:
    """Validate LLM-owned score adjudication and app arithmetic rails."""
    candidate_review = pass1_review if isinstance(pass1_review, dict) else {}
    candidate_scoring = scoring_review if isinstance(scoring_review, dict) else {}
    return validate_scoring_adjudication(packet, candidate_review, candidate_scoring)
