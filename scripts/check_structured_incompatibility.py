"""Check structured Trial Feature red-flag combinations."""

from __future__ import annotations

from frontend.utils.structured_incompatibility import structured_incompatibility_attention_fields


BASE = {
    "intervention_model_ml": "PARALLEL",
    "allocation_ml": "RANDOMIZED",
    "masking_ml": "DOUBLE",
    "comparator_benchmark_ml": "PLACEBO",
    "has_placebo_ml": "1",
    "number_of_arms_ml": 2,
}


def _fields(**updates):
    values = dict(BASE)
    values.update(updates)
    return structured_incompatibility_attention_fields(values)


def main() -> int:
    cases = [
        (
            "parallel_one_arm",
            _fields(number_of_arms_ml=1),
            {"intervention_model_ml", "number_of_arms_ml"},
        ),
        (
            "randomized_single_group",
            _fields(intervention_model_ml="SINGLE_GROUP"),
            {"allocation_ml", "intervention_model_ml"},
        ),
        (
            "randomized_one_arm",
            _fields(number_of_arms_ml=1),
            {"allocation_ml", "number_of_arms_ml"},
        ),
        (
            "placebo_comparator_no_placebo",
            _fields(has_placebo_ml="0"),
            {"comparator_benchmark_ml", "has_placebo_ml"},
        ),
        (
            "placebo_with_no_control",
            _fields(comparator_benchmark_ml="NO_CONTROL_GROUP", has_placebo_ml="1"),
            {"comparator_benchmark_ml", "has_placebo_ml"},
        ),
        (
            "placebo_one_arm",
            _fields(number_of_arms_ml=1),
            {"has_placebo_ml", "number_of_arms_ml"},
        ),
        (
            "single_group_placebo",
            _fields(intervention_model_ml="SINGLE_GROUP"),
            {"intervention_model_ml", "has_placebo_ml"},
        ),
        (
            "active_comparator_single_group",
            _fields(intervention_model_ml="SINGLE_GROUP", comparator_benchmark_ml="ACTIVE_MODERN_STANDARD"),
            {"comparator_benchmark_ml", "intervention_model_ml"},
        ),
        (
            "active_comparator_one_arm",
            _fields(comparator_benchmark_ml="ACTIVE_LEGACY_STANDARD", number_of_arms_ml=1),
            {"comparator_benchmark_ml", "number_of_arms_ml"},
        ),
        (
            "double_blind_single_group_no_control_no_placebo",
            _fields(
                intervention_model_ml="SINGLE_GROUP",
                comparator_benchmark_ml="NO_CONTROL_GROUP",
                has_placebo_ml="0",
            ),
            {"masking_ml", "intervention_model_ml", "comparator_benchmark_ml", "has_placebo_ml"},
        ),
        (
            "double_blind_one_arm_no_control_no_placebo",
            _fields(
                number_of_arms_ml=1,
                comparator_benchmark_ml="NO_CONTROL_GROUP",
                has_placebo_ml="0",
            ),
            {"masking_ml", "number_of_arms_ml", "comparator_benchmark_ml", "has_placebo_ml"},
        ),
        (
            "factorial_one_arm",
            _fields(intervention_model_ml="FACTORIAL", number_of_arms_ml=1),
            {"intervention_model_ml", "number_of_arms_ml"},
        ),
        (
            "crossover_one_arm",
            _fields(intervention_model_ml="CROSSOVER", number_of_arms_ml=1),
            {"intervention_model_ml", "number_of_arms_ml"},
        ),
        (
            "sequential_one_arm",
            _fields(intervention_model_ml="SEQUENTIAL", number_of_arms_ml=1),
            {"intervention_model_ml", "number_of_arms_ml"},
        ),
    ]

    errors: list[str] = []
    for label, actual, expected_subset in cases:
        missing = expected_subset - actual
        if missing:
            errors.append(f"{label}: missing expected attention fields {sorted(missing)}; got {sorted(actual)}")

    coherent = _fields()
    if coherent:
        errors.append(f"coherent parallel randomized placebo-controlled two-arm case should not flag; got {sorted(coherent)}")

    if errors:
        print("\n".join(errors))
        return 1

    print("Structured incompatibility checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
