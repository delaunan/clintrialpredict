"""Structured Trial Feature incompatibility checks for simulator UI."""

from __future__ import annotations

from typing import Any


def _numeric_value(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def structured_incompatibility_attention_fields(values: dict[str, Any]) -> set[str]:
    """Return field ids involved in impossible structured-feature combinations."""
    intervention_model = values.get("intervention_model_ml")
    allocation = values.get("allocation_ml")
    masking = values.get("masking_ml")
    comparator = values.get("comparator_benchmark_ml")
    placebo = values.get("has_placebo_ml")
    arms_value = _numeric_value(values.get("number_of_arms_ml"))
    one_or_fewer_arms = arms_value is not None and arms_value <= 1
    active_comparator = comparator in {"ACTIVE_LEGACY_STANDARD", "ACTIVE_MODERN_STANDARD"}
    high_blinding = masking in {"DOUBLE", "TRIPLE", "QUADRUPLE"}
    no_control = comparator == "NO_CONTROL_GROUP"
    placebo_yes = placebo == "1"
    placebo_no = placebo == "0"

    fields: set[str] = set()

    def flag(*field_ids: str) -> None:
        fields.update(field_ids)

    if intervention_model == "PARALLEL" and one_or_fewer_arms:
        flag("intervention_model_ml", "number_of_arms_ml")
    if allocation == "RANDOMIZED" and intervention_model == "SINGLE_GROUP":
        flag("allocation_ml", "intervention_model_ml")
    if allocation == "RANDOMIZED" and one_or_fewer_arms:
        flag("allocation_ml", "number_of_arms_ml")
    if comparator == "PLACEBO" and placebo_no:
        flag("comparator_benchmark_ml", "has_placebo_ml")
    if placebo_yes and no_control:
        flag("has_placebo_ml", "comparator_benchmark_ml")
    if placebo_yes and one_or_fewer_arms:
        flag("has_placebo_ml", "number_of_arms_ml")
    if intervention_model == "SINGLE_GROUP" and placebo_yes:
        flag("intervention_model_ml", "has_placebo_ml")
    if active_comparator and intervention_model == "SINGLE_GROUP":
        flag("comparator_benchmark_ml", "intervention_model_ml")
    if active_comparator and one_or_fewer_arms:
        flag("comparator_benchmark_ml", "number_of_arms_ml")
    if high_blinding and intervention_model == "SINGLE_GROUP" and no_control and placebo_no:
        flag("masking_ml", "intervention_model_ml", "comparator_benchmark_ml", "has_placebo_ml")
    if high_blinding and one_or_fewer_arms and no_control and placebo_no:
        flag("masking_ml", "number_of_arms_ml", "comparator_benchmark_ml", "has_placebo_ml")
    if intervention_model == "FACTORIAL" and one_or_fewer_arms:
        flag("intervention_model_ml", "number_of_arms_ml")
    if intervention_model == "CROSSOVER" and one_or_fewer_arms:
        flag("intervention_model_ml", "number_of_arms_ml")
    if intervention_model == "SEQUENTIAL" and one_or_fewer_arms:
        flag("intervention_model_ml", "number_of_arms_ml")

    return fields
