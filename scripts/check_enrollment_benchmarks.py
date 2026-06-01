from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.enrollment_benchmarks import (
    DEFAULT_ARTIFACT_PATH,
    REQUIRED_ARTIFACT_COLUMNS,
    classify_enrollment,
    load_enrollment_benchmarks,
    lookup_enrollment_benchmark,
    planned_enrollment_metadata,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    artifact = load_enrollment_benchmarks(DEFAULT_ARTIFACT_PATH)
    _assert(DEFAULT_ARTIFACT_PATH.exists(), f"Missing artifact: {DEFAULT_ARTIFACT_PATH}")
    _assert(not artifact.empty, "Artifact has no rows")
    _assert(REQUIRED_ARTIFACT_COLUMNS.issubset(artifact.columns), "Artifact is missing expected columns")

    strict_row = artifact[
        artifact["benchmark_level_used"].eq("phase_indication_rare")
        & artifact["low_confidence_flag"].eq(False)
    ].iloc[0]
    strict_snapshot = {
        "phase": strict_row["phase"],
        "gbd_cause_id_3_ml": int(strict_row["gbd_cause_id_3_ml"]),
        "therapeutic_area": strict_row.get("therapeutic_area"),
        "is_rare_disease_ml": int(strict_row["rare_disease_flag"]),
    }
    _assert(
        lookup_enrollment_benchmark(strict_snapshot, artifact)["benchmark_level_used"] == "phase_indication_rare",
        "Strict benchmark lookup failed",
    )

    fallback_row = artifact[
        artifact["benchmark_level_used"].eq("phase_ta")
        & artifact["low_confidence_flag"].eq(False)
    ].iloc[0]
    fallback_snapshot = {
        "phase": fallback_row["phase"],
        "gbd_cause_id_3_ml": 999999999,
        "therapeutic_area": fallback_row["therapeutic_area"],
        "is_rare_disease_ml": 1,
    }
    fallback = lookup_enrollment_benchmark(fallback_snapshot, artifact)
    _assert(fallback is not None, "Fallback lookup returned nothing")
    _assert(fallback["benchmark_level_used"] in {"phase_ta_rare", "phase_ta", "phase_only"}, "Fallback lookup used an invalid level")

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_metadata = planned_enrollment_metadata(
            strict_snapshot,
            100,
            artifact_path=Path(tmpdir) / "missing.csv",
        )
    _assert(
        missing_metadata["planned_enrollment"]["enrollment_status"] == "not_available",
        "Missing artifact should return not_available",
    )

    boundary = pd.Series({
        "benchmark_p25": 25,
        "benchmark_p75": 75,
        "benchmark_p90": 90,
    })
    _assert(classify_enrollment(24, boundary) == "below_benchmark", "P25 lower boundary failed")
    _assert(classify_enrollment(25, boundary) == "typical", "P25 inclusive boundary failed")
    _assert(classify_enrollment(75, boundary) == "typical", "P75 inclusive boundary failed")
    _assert(classify_enrollment(76, boundary) == "ambitious", "P75 upper boundary failed")
    _assert(classify_enrollment(90, boundary) == "ambitious", "P90 inclusive boundary failed")
    _assert(classify_enrollment(91, boundary) == "above_benchmark_high", "P90 upper boundary failed")

    metadata = planned_enrollment_metadata(strict_snapshot, strict_row["benchmark_p50"], artifact=artifact)
    _assert(metadata["planned_enrollment"]["enrollment_status"] == "typical", "Metadata classification failed")
    _assert(metadata["planned_enrollment"]["support_level"] == "not_evaluated", "Phase 1 support level changed")

    print("Enrollment benchmark checks passed.")


if __name__ == "__main__":
    main()
