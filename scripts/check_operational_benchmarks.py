from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.operational_benchmarks import (
    DEFAULT_ARTIFACT_PATH,
    REQUIRED_ARTIFACT_COLUMNS,
    classify_enrollment,
    classify_site_count,
    load_operational_benchmarks,
    lookup_operational_benchmark,
    planned_enrollment_metadata,
    planned_enrollment_default_from_operational_benchmark,
    planned_sites_metadata,
    planned_sites_default_from_operational_benchmark,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    artifact = load_operational_benchmarks(DEFAULT_ARTIFACT_PATH)
    _assert(DEFAULT_ARTIFACT_PATH.exists(), f"Missing artifact: {DEFAULT_ARTIFACT_PATH}")
    _assert(not artifact.empty, "Operational artifact has no rows")
    _assert(REQUIRED_ARTIFACT_COLUMNS.issubset(artifact.columns), "Operational artifact is missing expected columns")
    _assert(artifact["benchmark_key"].duplicated().sum() == 0, "Operational artifact has duplicate benchmark keys")

    for prefix in ("enrollment", "site_count", "patients_per_site"):
        for suffix in ("n", "p25", "p50", "p75", "p90"):
            column = f"{prefix}_{suffix}"
            _assert(column in artifact.columns, f"Missing column: {column}")
            _assert(pd.to_numeric(artifact[column], errors="coerce").notna().any(), f"No numeric values for {column}")

    strict_row = artifact[
        artifact["benchmark_level_used"].eq("phase_indication_rare")
        & artifact["patients_per_site_low_confidence_flag"].eq(False)
        & artifact["patients_per_site_n"].gt(0)
    ].iloc[0]
    strict_snapshot = {
        "phase": strict_row["phase"],
        "gbd_cause_id_3_ml": int(strict_row["gbd_cause_id_3_ml"]),
        "therapeutic_area": strict_row.get("therapeutic_area"),
        "is_rare_disease_ml": int(strict_row["rare_disease_flag"]),
    }
    strict_lookup = lookup_operational_benchmark(strict_snapshot, artifact, metric_prefix="patients_per_site")
    _assert(strict_lookup is not None, "Strict patients-per-site lookup failed")
    _assert(strict_lookup["benchmark_level_used"] == "phase_indication_rare", "Strict lookup used wrong level")

    mismatch = artifact[
        artifact["benchmark_level_used"].eq("phase_indication_rare")
        & artifact["enrollment_n"].eq(49)
        & artifact["site_count_n"].eq(51)
        & artifact["patients_per_site_n"].eq(49)
    ]
    _assert(not mismatch.empty, "Expected near-threshold mismatch row is missing")
    mismatch_row = mismatch.iloc[0]
    mismatch_snapshot = {
        "phase": mismatch_row["phase"],
        "gbd_cause_id_3_ml": int(mismatch_row["gbd_cause_id_3_ml"]),
        "is_rare_disease_ml": int(mismatch_row["rare_disease_flag"]),
    }
    mismatch_lookup = lookup_operational_benchmark(mismatch_snapshot, artifact, metric_prefix="patients_per_site")
    _assert(
        mismatch_lookup is not None and mismatch_lookup["benchmark_key"] == mismatch_row["benchmark_key"],
        "Near-threshold row should be used instead of falling back",
    )

    fallback_row = artifact[
        artifact["benchmark_level_used"].eq("phase_ta")
        & artifact["site_count_low_confidence_flag"].eq(False)
        & artifact["site_count_n"].gt(0)
    ].iloc[0]
    fallback_snapshot = {
        "phase": fallback_row["phase"],
        "gbd_cause_id_3_ml": 999999999,
        "therapeutic_area": fallback_row["therapeutic_area"],
        "is_rare_disease_ml": 1,
    }
    fallback_lookup = lookup_operational_benchmark(fallback_snapshot, artifact, metric_prefix="site_count")
    _assert(fallback_lookup is not None, "Fallback site-count lookup failed")
    _assert(
        fallback_lookup["benchmark_level_used"] in {"phase_ta_rare", "phase_ta", "phase_only"},
        "Fallback site-count lookup used invalid level",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_artifact = load_operational_benchmarks(Path(tmpdir) / "missing.csv")
    _assert(missing_artifact.empty, "Missing artifact should return schema-safe empty DataFrame")
    _assert(REQUIRED_ARTIFACT_COLUMNS.issubset(missing_artifact.columns), "Missing artifact schema is unsafe")

    completed_default = planned_sites_default_from_operational_benchmark(
        strict_snapshot,
        planned_enrollment=500,
        current_registry_facility_count_proxy=17,
        overall_status="COMPLETED",
        artifact=artifact,
    )
    _assert(completed_default["value"] == 17, "Completed trial should use completed registry facility-count proxy")
    _assert(
        completed_default["source"] == "completed_registry_facility_count",
        "Completed source label is incorrect",
    )

    non_completed_default = planned_sites_default_from_operational_benchmark(
        strict_snapshot,
        planned_enrollment=200,
        current_registry_facility_count_proxy=2,
        overall_status="RECRUITING",
        artifact=artifact,
    )
    _assert(non_completed_default["value"] >= 2, "Non-completed default should respect lower-bound current proxy")
    _assert(
        non_completed_default["source"]
        in {"current_registry_facility_count_proxy", "benchmark_default", "enrollment_coherent_benchmark_default"},
        "Non-completed default source is not conservative",
    )
    _assert(
        non_completed_default["site_default_basis"]
        in {"current_registry_facility_count_proxy", "benchmark_default", "enrollment_coherent_benchmark_default"},
        "Non-completed default basis is invalid",
    )

    registry = pd.read_csv("frontend/data/search_registry.csv", low_memory=False)
    small_planned = registry[registry["nct_id"].eq("NCT07543380")].iloc[0]
    small_snapshot = small_planned.to_dict()
    small_snapshot["phase"] = small_snapshot.get("phase_ml")
    small_snapshot["therapeutic_area"] = small_snapshot.get("therapeutic_area_ml")
    small_default = planned_sites_default_from_operational_benchmark(
        small_snapshot,
        planned_enrollment=small_planned["enrollment"],
        current_registry_facility_count_proxy=small_planned["number_of_facilities"],
        overall_status=small_planned["overall_status"],
        artifact=artifact,
    )
    _assert(small_default["value"] == 5, "Small planned trial should keep coherent current site proxy")
    _assert(
        small_default["source"] == "current_registry_facility_count_proxy",
        "Small planned trial source should be current registry proxy",
    )

    enrollment_p50 = float(strict_row["enrollment_p50"])
    lower_bound_wins = planned_enrollment_default_from_operational_benchmark(
        strict_snapshot,
        observed_lower_bound=enrollment_p50 + 100,
        artifact=artifact,
    )
    _assert(lower_bound_wins["source"] == "observed_lower_bound", "Observed lower-bound should win above P50")
    _assert(lower_bound_wins["value"] == int(round(enrollment_p50 + 100)), "Observed lower-bound value was not preserved")

    p50_wins = planned_enrollment_default_from_operational_benchmark(
        strict_snapshot,
        observed_lower_bound=max(1, enrollment_p50 - 100),
        artifact=artifact,
    )
    _assert(p50_wins["source"] == "model_default", "Benchmark P50 should win above observed lower-bound")

    enrollment_boundary = pd.Series({"enrollment_p25": 25, "enrollment_p75": 75, "enrollment_p90": 90})
    site_boundary = pd.Series({"site_count_p25": 25, "site_count_p75": 75, "site_count_p90": 90})
    _assert(classify_enrollment(24, enrollment_boundary) == "below_benchmark", "Enrollment P25 lower boundary failed")
    _assert(classify_enrollment(25, enrollment_boundary) == "typical", "Enrollment P25 inclusive boundary failed")
    _assert(classify_enrollment(75, enrollment_boundary) == "typical", "Enrollment P75 inclusive boundary failed")
    _assert(classify_enrollment(90, enrollment_boundary) == "ambitious", "Enrollment P90 inclusive boundary failed")
    _assert(classify_site_count(91, site_boundary) == "above_benchmark_high", "Site-count P90 upper boundary failed")

    enrollment_metadata = planned_enrollment_metadata(strict_snapshot, strict_row["enrollment_p50"], artifact=artifact)
    _assert("planned_enrollment" in enrollment_metadata, "Unified metadata should return planned_enrollment")
    _assert(
        enrollment_metadata["planned_enrollment"]["support_level"] == "not_evaluated",
        "Enrollment support level changed",
    )

    site_metadata = planned_sites_metadata(
        strict_snapshot,
        strict_row["site_count_p50"],
        artifact=artifact,
        planned_enrollment=strict_row["enrollment_p50"],
        current_registry_facility_count_proxy=2,
        overall_status="RECRUITING",
    )
    _assert("planned_sites" in site_metadata, "Unified metadata should return planned_sites")
    _assert(site_metadata["planned_sites"]["support_level"] == "not_evaluated", "Site support level changed")
    _assert(
        "patients_per_site_p50" in site_metadata["planned_sites"],
        "Site metadata should include patients-per-site context",
    )

    print("Operational benchmark checks passed.")


if __name__ == "__main__":
    main()
