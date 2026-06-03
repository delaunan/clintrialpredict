from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.operational_benchmarks import (
    DEFAULT_ARTIFACT_PATH,
    REQUIRED_ARTIFACT_COLUMNS,
    classify_duration_months,
    classify_enrollment,
    classify_primary_completion_months,
    classify_site_count,
    load_operational_benchmarks,
    lookup_operational_benchmark,
    planned_duration_default_from_operational_benchmark,
    planned_duration_months_metadata,
    planned_enrollment_metadata,
    planned_enrollment_default_from_operational_benchmark,
    planned_primary_completion_default_from_operational_benchmark,
    planned_sites_metadata,
    planned_sites_default_from_operational_benchmark,
)


SEARCH_REGISTRY_PATH = Path("frontend/data/search_registry.csv")
EDIT_TRIAL_PATH = Path("frontend/views/edit_trial.py")
REPORT_PATH = Path("frontend/data/operational_benchmarks_v1_report.json")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _positive_number(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    return float(numeric)


def _snapshot_from_registry_row(row: pd.Series) -> dict:
    snapshot = row.replace({pd.NA: None}).to_dict()
    snapshot["phase"] = snapshot.get("phase_ml")
    if not snapshot.get("therapeutic_area"):
        snapshot["therapeutic_area"] = snapshot.get("therapeutic_area_ml")
    snapshot["is_rare_disease"] = snapshot.get("is_rare_disease_ml")
    return snapshot


def _assert_registry_coverage_report(artifact: pd.DataFrame) -> None:
    _assert(REPORT_PATH.exists(), f"Missing benchmark report: {REPORT_PATH}")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    coverage = report.get("coverage_qa", {})
    for metric in ("enrollment", "site_count", "patients_per_site", "primary_completion_months", "duration_months"):
        metric_coverage = coverage.get(metric, {})
        _assert(
            metric_coverage.get("not_available") == 0,
            f"{metric} report coverage has {metric_coverage.get('not_available')} rows without a benchmark",
        )
        _assert(
            metric_coverage.get("low_confidence_matches") == 0,
            f"{metric} report coverage has {metric_coverage.get('low_confidence_matches')} low-confidence matches",
        )

    for metric in ("primary_completion_months", "duration_months"):
        metric_rows = artifact[pd.to_numeric(artifact[f"{metric}_n"], errors="coerce").fillna(0).gt(0)]
        _assert(
            metric_rows["benchmark_level_used"].str.contains("endpoint_bin", regex=False).any(),
            f"{metric} artifact should include endpoint-duration-bin benchmark rows",
        )


def _assert_site_defaulting(registry: pd.DataFrame, artifact: pd.DataFrame) -> None:
    completed_proxy_not_used = 0
    below_current_proxy = 0
    for _, row in registry.iterrows():
        snapshot = _snapshot_from_registry_row(row)
        status = str(row.get("overall_status") or "").strip().upper()
        current_proxy = _positive_number(row.get("number_of_facilities"))
        enrollment_default = planned_enrollment_default_from_operational_benchmark(
            snapshot,
            observed_lower_bound=row.get("enrollment"),
            artifact=artifact,
        )
        default = planned_sites_default_from_operational_benchmark(
            snapshot,
            planned_enrollment=enrollment_default.get("value") or row.get("enrollment"),
            current_registry_facility_count_proxy=current_proxy,
            overall_status=status,
            artifact=artifact,
        )
        value = _positive_number(default.get("value"))
        if status == "COMPLETED" and current_proxy is not None:
            if value is None or int(round(value)) != int(round(current_proxy)):
                completed_proxy_not_used += 1
        elif current_proxy is not None and value is not None and value < current_proxy:
            below_current_proxy += 1

    _assert(completed_proxy_not_used == 0, "Completed site proxy was not preserved")
    _assert(below_current_proxy == 0, "Non-completed site default fell below current registry proxy")


def _assert_model_boundary() -> None:
    text = EDIT_TRIAL_PATH.read_text(encoding="utf-8")
    try:
        start = text.index("SIMULATION_FEATURE_IDS")
        end = text.index("]", start)
        feature_block = text[start:end]
    except ValueError:
        feature_block = ""
    for field in ("planned_sites", "planned_enrollment", "planned_duration_months", "number_of_facilities"):
        _assert(field not in feature_block, f"{field} should not be model-facing in SIMULATION_FEATURE_IDS")


def main() -> None:
    artifact = load_operational_benchmarks(DEFAULT_ARTIFACT_PATH)
    _assert(DEFAULT_ARTIFACT_PATH.exists(), f"Missing artifact: {DEFAULT_ARTIFACT_PATH}")
    _assert(not artifact.empty, "Operational artifact has no rows")
    _assert(REQUIRED_ARTIFACT_COLUMNS.issubset(artifact.columns), "Operational artifact is missing expected columns")
    _assert(artifact["benchmark_key"].duplicated().sum() == 0, "Operational artifact has duplicate benchmark keys")
    _assert(
        artifact["benchmark_level_used"].eq("phase_only_modality").sum() == 0,
        "Artifact should not include phase-only modality rows",
    )
    indication_levels = artifact["benchmark_level_used"].str.startswith("phase_indication_rare")
    _assert(
        pd.to_numeric(artifact.loc[indication_levels, "gbd_cause_id_3_ml"], errors="coerce").fillna(0).gt(0).all(),
        "Indication-level benchmark rows should not use unclassified indication id 0",
    )
    ta_levels = artifact["benchmark_level_used"].str.startswith("phase_ta")
    invalid_tas = {"", "OTHER", "OTHER/UNCLASSIFIED", "UNKNOWN", "UNCLASSIFIED"}
    artifact_tas = artifact.loc[ta_levels, "therapeutic_area"].fillna("").astype(str).str.upper()
    _assert(
        ~artifact_tas.isin(invalid_tas).any(),
        "TA-level benchmark rows should not use unclassified therapeutic-area placeholders",
    )
    modality_levels = artifact["benchmark_level_used"].str.endswith("_modality")
    invalid_modalities = {"", "UNKNOWN", "UNCLASSIFIED"}
    artifact_modalities = artifact.loc[modality_levels, "therapeutic_modality"].fillna("").astype(str).str.upper()
    _assert(
        ~artifact_modalities.isin(invalid_modalities).any(),
        "Modality benchmark rows should not use unknown/unclassified modality placeholders",
    )
    _assert(
        artifact.loc[artifact["benchmark_level_used"].str.endswith("_modality"), "site_count_n"].fillna(0).eq(0).all(),
        "Raw site-count benchmarks should not use modality refinement rows",
    )
    _assert(
        artifact.loc[artifact["benchmark_level_used"].str.endswith("_modality"), "duration_months_n"].fillna(0).eq(0).all(),
        "Duration benchmarks should not use modality refinement rows",
    )
    _assert(
        artifact.loc[artifact["benchmark_level_used"].str.endswith("_non_vaccine_infections"), "duration_months_n"]
        .fillna(0)
        .eq(0)
        .all(),
        "Duration benchmarks should not use non-vaccine Infections fallback rows",
    )
    duration_bin_levels = artifact["benchmark_level_used"].str.contains("endpoint_bin", regex=False)
    _assert(
        artifact.loc[duration_bin_levels, "endpoint_duration_bin"].fillna("").astype(str).str.len().gt(0).all(),
        "Endpoint-bin duration rows should include endpoint_duration_bin",
    )

    for prefix in ("enrollment", "site_count", "patients_per_site", "primary_completion_months", "duration_months"):
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

    modality_snapshot = {
        "phase": "PHASE3",
        "gbd_cause_id_3_ml": 302,
        "therapeutic_area": "INFECTIONS",
        "is_rare_disease_ml": 0,
        "therapeutic_modality_ui": "Vaccine",
    }
    modality_enrollment = lookup_operational_benchmark(modality_snapshot, artifact, metric_prefix="enrollment")
    _assert(modality_enrollment is not None, "Modality enrollment lookup failed")
    _assert(
        modality_enrollment["benchmark_level_used"] == "phase_ta_rare_modality",
        "Enrollment should use same-level modality refinement when confident",
    )
    _assert(
        int(modality_enrollment["enrollment_n"]) >= 50,
        "Confident enrollment modality refinement should have n >= 50",
    )
    modality_pps = lookup_operational_benchmark(modality_snapshot, artifact, metric_prefix="patients_per_site")
    _assert(modality_pps is not None, "Modality patients-per-site lookup failed")
    _assert(
        modality_pps["benchmark_level_used"] == "phase_ta_rare_modality",
        "Patients-per-site should use same-level modality refinement when confident",
    )
    _assert(
        int(modality_pps["patients_per_site_n"]) >= 50,
        "Confident patients-per-site modality refinement should have n >= 50",
    )
    modality_site = lookup_operational_benchmark(modality_snapshot, artifact, metric_prefix="site_count")
    _assert(modality_site is not None, "Modality site-count lookup failed")
    _assert(
        modality_site["benchmark_level_used"] == "phase_ta_rare",
        "Raw site-count lookup should remain on the clinical cohort, not modality refinement",
    )
    modality_duration = lookup_operational_benchmark(modality_snapshot, artifact, metric_prefix="duration_months")
    _assert(modality_duration is not None, "Duration lookup failed for vaccine snapshot")
    _assert(
        not str(modality_duration["benchmark_level_used"]).endswith("_modality"),
        "Duration lookup should not use modality refinement",
    )

    registry = pd.read_csv("frontend/data/search_registry.csv", low_memory=False)
    non_vaccine_infections_trial = registry[registry["nct_id"].eq("NCT04938830")].iloc[0].to_dict()
    non_vaccine_infections_trial["phase"] = non_vaccine_infections_trial.get("phase_ml")
    if not non_vaccine_infections_trial.get("therapeutic_area"):
        non_vaccine_infections_trial["therapeutic_area"] = non_vaccine_infections_trial.get("therapeutic_area_ml")
    non_vaccine_enrollment = lookup_operational_benchmark(
        non_vaccine_infections_trial,
        artifact,
        metric_prefix="enrollment",
    )
    _assert(
        non_vaccine_enrollment is not None
        and non_vaccine_enrollment["benchmark_level_used"] == "phase_ta_rare_non_vaccine_infections",
        "Non-vaccine Infections enrollment should fall back to non-vaccine Infections cohort",
    )
    _assert(
        int(non_vaccine_enrollment["enrollment_n"]) >= 50,
        "Non-vaccine Infections enrollment fallback should require n >= 50",
    )
    non_vaccine_pps = lookup_operational_benchmark(
        non_vaccine_infections_trial,
        artifact,
        metric_prefix="patients_per_site",
    )
    _assert(
        non_vaccine_pps is not None
        and non_vaccine_pps["benchmark_level_used"] == "phase_ta_rare_non_vaccine_infections",
        "Non-vaccine Infections patients-per-site should fall back to non-vaccine Infections cohort",
    )
    _assert(
        int(non_vaccine_pps["patients_per_site_n"]) >= 50,
        "Non-vaccine Infections patients-per-site fallback should require n >= 50",
    )
    non_vaccine_site = lookup_operational_benchmark(
        non_vaccine_infections_trial,
        artifact,
        metric_prefix="site_count",
    )
    _assert(
        non_vaccine_site is not None and not str(non_vaccine_site["benchmark_level_used"]).endswith("_non_vaccine_infections"),
        "Raw site-count should not use non-vaccine Infections fallback",
    )
    non_vaccine_duration = lookup_operational_benchmark(
        non_vaccine_infections_trial,
        artifact,
        metric_prefix="duration_months",
    )
    _assert(
        non_vaccine_duration is not None
        and not str(non_vaccine_duration["benchmark_level_used"]).endswith("_non_vaccine_infections"),
        "Duration should not use non-vaccine Infections fallback",
    )

    duration_row = artifact[
        artifact["benchmark_level_used"].str.contains("endpoint_bin", regex=False)
        & artifact["duration_months_low_confidence_flag"].eq(False)
        & artifact["duration_months_n"].gt(0)
    ].iloc[0]
    endpoint_bin_midpoints = {
        "<=3": 3,
        "3-6": 4.5,
        "6-12": 9,
        "12-18": 15,
        "18-24": 21,
        "24-36": 30,
        "36-60": 48,
        ">60": 72,
    }
    duration_snapshot = {
        "phase": duration_row["phase"],
        "gbd_cause_id_3_ml": int(duration_row["gbd_cause_id_3_ml"]) if pd.notna(duration_row["gbd_cause_id_3_ml"]) else None,
        "therapeutic_area": duration_row.get("therapeutic_area"),
        "is_rare_disease_ml": int(duration_row["rare_disease_flag"]) if pd.notna(duration_row["rare_disease_flag"]) else None,
        "primary_duration_months_ml": endpoint_bin_midpoints[str(duration_row["endpoint_duration_bin"])],
    }
    duration_lookup = lookup_operational_benchmark(duration_snapshot, artifact, metric_prefix="duration_months")
    _assert(duration_lookup is not None, "Endpoint-bin duration lookup failed")
    _assert(
        "endpoint_bin" in str(duration_lookup["benchmark_level_used"]),
        "Duration lookup should prefer endpoint-bin cohorts when available",
    )
    _assert(
        int(duration_lookup["duration_months_n"]) >= 50,
        "Duration lookup should require duration_months_n >= 50",
    )

    synthetic_specific = artifact.iloc[0].copy()
    synthetic_specific["benchmark_key"] = (
        "phase_indication_rare_endpoint_bin|phase=PHASE3|indication=123456|rare=0|endpoint_bin=6-12"
    )
    synthetic_specific["benchmark_level_used"] = "phase_indication_rare_endpoint_bin"
    synthetic_specific["phase"] = "PHASE3"
    synthetic_specific["gbd_cause_id_3_ml"] = 123456
    synthetic_specific["therapeutic_area"] = ""
    synthetic_specific["rare_disease_flag"] = 0
    synthetic_specific["endpoint_duration_bin"] = "6-12"
    synthetic_specific["duration_months_n"] = 49
    synthetic_specific["duration_months_p50"] = 1
    synthetic_specific["duration_months_low_confidence_flag"] = True

    synthetic_broad = artifact.iloc[0].copy()
    synthetic_broad["benchmark_key"] = "phase_endpoint_bin|phase=PHASE3|endpoint_bin=6-12"
    synthetic_broad["benchmark_level_used"] = "phase_endpoint_bin"
    synthetic_broad["phase"] = "PHASE3"
    synthetic_broad["gbd_cause_id_3_ml"] = ""
    synthetic_broad["therapeutic_area"] = ""
    synthetic_broad["rare_disease_flag"] = ""
    synthetic_broad["endpoint_duration_bin"] = "6-12"
    synthetic_broad["duration_months_n"] = 50
    synthetic_broad["duration_months_p50"] = 99
    synthetic_broad["duration_months_low_confidence_flag"] = False

    synthetic_artifact = pd.DataFrame([synthetic_specific, synthetic_broad], columns=artifact.columns)
    strict_duration_lookup = lookup_operational_benchmark(
        {
            "phase": "PHASE3",
            "gbd_cause_id_3_ml": 123456,
            "is_rare_disease_ml": 0,
            "primary_duration_months_ml": 9,
        },
        synthetic_artifact,
        metric_prefix="duration_months",
        min_metric_n=50,
        cohort_metric_prefixes=("duration_months",),
    )
    _assert(
        strict_duration_lookup is not None
        and strict_duration_lookup["benchmark_level_used"] == "phase_endpoint_bin"
        and int(strict_duration_lookup["duration_months_n"]) == 50,
        "Duration lookup should skip n=49 granular rows and use the first n>=50 fallback",
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
    duration_boundary = pd.Series({"duration_months_p25": 25, "duration_months_p75": 75, "duration_months_p90": 90})
    primary_boundary = pd.Series(
        {
            "primary_completion_months_p25": 25,
            "primary_completion_months_p75": 75,
            "primary_completion_months_p90": 90,
        }
    )
    _assert(classify_duration_months(24, duration_boundary) == "below_benchmark", "Duration P25 lower boundary failed")
    _assert(classify_duration_months(75, duration_boundary) == "typical", "Duration P75 inclusive boundary failed")
    _assert(classify_primary_completion_months(91, primary_boundary) == "above_benchmark_high", "Primary completion P90 upper boundary failed")

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

    completed_duration_default = planned_duration_default_from_operational_benchmark(
        {
            **strict_snapshot,
            "completion_duration_months": 37,
            "completion_date_type": "ACTUAL",
            "primary_completion_duration_months": 20,
            "primary_completion_date_type": "ACTUAL",
            "primary_duration_months_ml": 18,
            "overall_status": "COMPLETED",
        },
        artifact=artifact,
    )
    _assert(
        completed_duration_default["source"] == "final_observed_total_duration",
        "Completed ACTUAL completion duration should be direct final observed duration",
    )
    _assert(completed_duration_default["value"] == 37, "Completed ACTUAL total duration was not preserved")

    active_estimated_duration_default = planned_duration_default_from_operational_benchmark(
        {
            **strict_snapshot,
            "completion_duration_months": 42,
            "completion_date_type": "ESTIMATED",
            "primary_completion_duration_months": 24,
            "primary_completion_date_type": "ESTIMATED",
            "primary_duration_months_ml": 30,
            "overall_status": "RECRUITING",
        },
        artifact=artifact,
    )
    _assert(
        active_estimated_duration_default["source"] == "estimated_planned_total_duration",
        "Active ESTIMATED completion duration should be direct planned total duration",
    )

    stopped_duration_default = planned_duration_default_from_operational_benchmark(
        {
            **strict_snapshot,
            "completion_duration_months": 50,
            "completion_date_type": "ACTUAL",
            "primary_completion_duration_months": 20,
            "primary_completion_date_type": "ACTUAL",
            "primary_duration_months_ml": 18,
            "overall_status": "TERMINATED",
        },
        artifact=artifact,
    )
    _assert(
        stopped_duration_default["source"] == "benchmark_default_with_floors",
        "Stopped ACTUAL completion duration should be a lower-bound floor, not a direct source",
    )
    _assert(stopped_duration_default["value"] >= 50, "Stopped duration lower-bound floor was not respected")

    primary_default = planned_primary_completion_default_from_operational_benchmark(
        {
            **strict_snapshot,
            "primary_completion_duration_months": 10,
            "primary_completion_date_type": "ESTIMATED",
            "primary_duration_months_ml": 18,
            "overall_status": "RECRUITING",
        },
        artifact=artifact,
    )
    _assert(
        primary_default["source"] == "estimated_primary_completion",
        "Trusted active ESTIMATED primary completion duration should be direct",
    )
    _assert(
        "primary_completion_shorter_than_primary_duration_ml" in primary_default["warnings"],
        "Shorter trusted primary completion should carry warning metadata",
    )

    duration_metadata = planned_duration_months_metadata(
        {
            **strict_snapshot,
            "completion_duration_months": 42,
            "completion_date_type": "ESTIMATED",
            "primary_completion_duration_months": 24,
            "primary_completion_date_type": "ESTIMATED",
            "primary_duration_months_ml": 30,
            "overall_status": "RECRUITING",
        },
        artifact=artifact,
    )
    _assert("planned_duration_months" in duration_metadata, "Unified metadata should return planned_duration_months")
    _assert(
        duration_metadata["planned_duration_months"]["support_level"] == "not_evaluated",
        "Duration support level changed",
    )
    _assert(
        "planned_primary_completion_months" in duration_metadata["planned_duration_months"],
        "Duration metadata should include primary completion context",
    )
    _assert(
        duration_metadata["planned_duration_months"]["primary_completion_source"] == "estimated_primary_completion",
        "Trusted active ESTIMATED primary completion should remain direct in duration metadata",
    )
    _assert(
        duration_metadata["planned_duration_months"]["planned_primary_completion_months"] == 24.0,
        "Trusted active ESTIMATED primary completion value should not be replaced by same-cohort benchmark P50",
    )
    _assert(
        duration_metadata["planned_duration_months"]["benchmark_n"] >= 50,
        "Duration metadata should use a full-duration cohort with n >= 50",
    )
    primary_n = duration_metadata["planned_duration_months"]["primary_completion_n"]
    if primary_n is not None:
        _assert(primary_n >= 50, "Primary-completion context should require same-cohort n >= 50")
        _assert(
            duration_metadata["planned_duration_months"]["primary_completion_benchmark_level_used"]
            == duration_metadata["planned_duration_months"]["benchmark_level_used"],
            "Primary-completion context should come from the selected full-duration cohort",
        )

    _assert_registry_coverage_report(artifact)
    _assert_site_defaulting(registry, artifact)
    _assert_model_boundary()

    print("Operational benchmark checks passed.")


if __name__ == "__main__":
    main()
