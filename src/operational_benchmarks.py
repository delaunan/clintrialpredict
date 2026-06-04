from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "frontend" / "data" / "operational_benchmarks_v1.csv"

LEVEL_ORDER = [
    "phase_indication_rare",
    "phase_ta_rare",
    "phase_ta",
    "phase_only",
]

DURATION_LEVEL_ORDER = [
    "phase_indication_rare_endpoint_bin",
    "phase_ta_rare_endpoint_bin",
    "phase_ta_endpoint_bin",
    "phase_endpoint_bin",
    "phase_indication_rare",
    "phase_ta_rare",
    "phase_ta",
    "phase_only",
]

MODALITY_REFINEMENT_LEVELS = {
    "phase_indication_rare": "phase_indication_rare_modality",
    "phase_ta_rare": "phase_ta_rare_modality",
    "phase_ta": "phase_ta_modality",
}

NON_VACCINE_INFECTIONS_LEVELS = {
    "phase_indication_rare": "phase_indication_rare_non_vaccine_infections",
    "phase_ta_rare": "phase_ta_rare_non_vaccine_infections",
    "phase_ta": "phase_ta_non_vaccine_infections",
}

MODALITY_REFINED_METRICS = {"enrollment", "patients_per_site"}
DURATION_METRICS = {"primary_completion_months", "duration_months"}

METRIC_PREFIXES = ("enrollment", "site_count", "patients_per_site", "primary_completion_months", "duration_months")
MIN_USABLE_METRIC_N = 30
MIN_DURATION_METRIC_N = 50
MIN_MODALITY_REFINEMENT_N = 50
MIN_NON_VACCINE_INFECTIONS_N = 50

INVALID_THERAPEUTIC_AREAS = {"", "OTHER", "OTHER/UNCLASSIFIED", "UNKNOWN", "UNCLASSIFIED"}
INVALID_MODALITIES = {"", "UNKNOWN", "UNCLASSIFIED"}
STOPPED_OR_INTERRUPTED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}

REQUIRED_ARTIFACT_COLUMNS = {
    "benchmark_version",
    "source_data_version",
    "benchmark_key",
    "phase",
    "gbd_cause_id_3_ml",
    "therapeutic_area",
    "rare_disease_flag",
    "therapeutic_modality",
    "endpoint_duration_bin",
    "benchmark_level_used",
    "created_at",
    "outlier_policy",
    "calibration_notes",
}

for _prefix in METRIC_PREFIXES:
    REQUIRED_ARTIFACT_COLUMNS.update(
        {
            f"{_prefix}_n",
            f"{_prefix}_p25",
            f"{_prefix}_p50",
            f"{_prefix}_p75",
            f"{_prefix}_p90",
            f"{_prefix}_low_confidence_flag",
        }
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_phase(value: Any) -> str | None:
    if _is_missing(value):
        return None
    numeric_phase_map = {
        "1": "PHASE1/PHASE2",
        "2": "PHASE2",
        "3": "PHASE2/PHASE3",
        "4": "PHASE3",
    }
    text = str(value).strip()
    if text in numeric_phase_map:
        return numeric_phase_map[text]
    return text.upper() if text else None


def _clean_ta(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _clean_modality(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text.upper() if text else None


def endpoint_duration_bin(value: Any) -> str | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    months = float(numeric)
    if months <= 3:
        return "<=3"
    if months <= 6:
        return "3-6"
    if months <= 12:
        return "6-12"
    if months <= 18:
        return "12-18"
    if months <= 24:
        return "18-24"
    if months <= 36:
        return "24-36"
    if months <= 60:
        return "36-60"
    return ">60"


def _is_valid_indication(value: Any) -> bool:
    indication = _clean_int(value)
    return indication is not None and indication > 0


def _is_valid_ta(value: Any) -> bool:
    therapeutic_area = _clean_ta(value)
    return therapeutic_area is not None and therapeutic_area not in INVALID_THERAPEUTIC_AREAS


def _is_valid_modality(value: Any) -> bool:
    modality = _clean_modality(value)
    return modality is not None and modality not in INVALID_MODALITIES


def _clean_int(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def _clean_boolish_int(value: Any) -> int | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y"}:
            return 1
        if text in {"false", "no", "n"}:
            return 0
    return _clean_int(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _positive_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    return float(numeric)


def _empty_artifact() -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(REQUIRED_ARTIFACT_COLUMNS))


def load_operational_benchmarks(path: str | Path = DEFAULT_ARTIFACT_PATH) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return _empty_artifact()

    try:
        artifact = pd.read_csv(artifact_path)
    except Exception:
        return _empty_artifact()

    missing_columns = REQUIRED_ARTIFACT_COLUMNS.difference(artifact.columns)
    if missing_columns:
        return _empty_artifact()

    artifact["phase"] = artifact["phase"].map(_clean_phase)
    artifact["therapeutic_area"] = artifact["therapeutic_area"].map(_clean_ta)
    artifact["therapeutic_modality"] = artifact["therapeutic_modality"].map(_clean_modality)
    artifact["endpoint_duration_bin"] = artifact["endpoint_duration_bin"].map(lambda value: None if _is_missing(value) else str(value).strip())
    artifact["gbd_cause_id_3_ml"] = pd.to_numeric(artifact["gbd_cause_id_3_ml"], errors="coerce")
    artifact["rare_disease_flag"] = pd.to_numeric(artifact["rare_disease_flag"], errors="coerce")
    for prefix in METRIC_PREFIXES:
        for suffix in ("n", "p25", "p50", "p75", "p90"):
            artifact[f"{prefix}_{suffix}"] = pd.to_numeric(artifact[f"{prefix}_{suffix}"], errors="coerce")
        artifact[f"{prefix}_low_confidence_flag"] = (
            artifact[f"{prefix}_low_confidence_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    return artifact


def _candidate_keys(snapshot: dict[str, Any], *, include_duration_bin: bool = False) -> list[tuple[str, dict[str, Any]]]:
    phase = _clean_phase(_first_present(snapshot.get("phase"), snapshot.get("phase_ui"), snapshot.get("phase_ml")))
    indication = _clean_int(snapshot.get("gbd_cause_id_3_ml"))
    therapeutic_area = _clean_ta(
        _first_present(snapshot.get("therapeutic_area"), snapshot.get("therapeutic_area_ui"), snapshot.get("therapeutic_area_ml"))
    )
    rare = _clean_boolish_int(_first_present(snapshot.get("is_rare_disease_ml"), snapshot.get("is_rare_disease")))

    if not phase:
        return []

    candidates: list[tuple[str, dict[str, Any]]] = []
    duration_bin = endpoint_duration_bin(snapshot.get("primary_duration_months_ml"))
    if include_duration_bin and duration_bin:
        if _is_valid_indication(indication) and rare is not None:
            candidates.append(
                (
                    "phase_indication_rare_endpoint_bin",
                    {
                        "phase": phase,
                        "gbd_cause_id_3_ml": indication,
                        "rare_disease_flag": rare,
                        "endpoint_duration_bin": duration_bin,
                    },
                )
            )
        if _is_valid_ta(therapeutic_area) and rare is not None:
            candidates.append(
                (
                    "phase_ta_rare_endpoint_bin",
                    {
                        "phase": phase,
                        "therapeutic_area": therapeutic_area,
                        "rare_disease_flag": rare,
                        "endpoint_duration_bin": duration_bin,
                    },
                )
            )
        if _is_valid_ta(therapeutic_area):
            candidates.append(
                (
                    "phase_ta_endpoint_bin",
                    {"phase": phase, "therapeutic_area": therapeutic_area, "endpoint_duration_bin": duration_bin},
                )
            )
        candidates.append(("phase_endpoint_bin", {"phase": phase, "endpoint_duration_bin": duration_bin}))
    if _is_valid_indication(indication) and rare is not None:
        candidates.append(("phase_indication_rare", {"phase": phase, "gbd_cause_id_3_ml": indication, "rare_disease_flag": rare}))
    if _is_valid_ta(therapeutic_area) and rare is not None:
        candidates.append(("phase_ta_rare", {"phase": phase, "therapeutic_area": therapeutic_area, "rare_disease_flag": rare}))
    if _is_valid_ta(therapeutic_area):
        candidates.append(("phase_ta", {"phase": phase, "therapeutic_area": therapeutic_area}))
    candidates.append(("phase_only", {"phase": phase}))
    return candidates


def _snapshot_modality(snapshot: dict[str, Any]) -> str | None:
    return _clean_modality(
        _first_present(
            snapshot.get("therapeutic_modality"),
            snapshot.get("therapeutic_modality_ui"),
            snapshot.get("therapeutic_modality_ml"),
        )
    )


def _same_level_modality_refinement(
    snapshot: dict[str, Any],
    benchmarks: pd.DataFrame,
    base_row: pd.Series,
    metric_prefix: str | None,
) -> pd.Series:
    if metric_prefix not in MODALITY_REFINED_METRICS:
        return base_row

    base_level = str(base_row.get("benchmark_level_used") or "")
    refined_level = MODALITY_REFINEMENT_LEVELS.get(base_level)
    modality = _snapshot_modality(snapshot)
    if not refined_level or not _is_valid_modality(modality):
        return base_row

    mask = benchmarks["benchmark_level_used"].eq(refined_level)
    mask &= benchmarks["phase"].eq(base_row.get("phase"))
    if base_level in {"phase_indication_rare", "phase_ta_rare"}:
        mask &= benchmarks["rare_disease_flag"].eq(base_row.get("rare_disease_flag"))
    if base_level == "phase_indication_rare":
        mask &= benchmarks["gbd_cause_id_3_ml"].eq(base_row.get("gbd_cause_id_3_ml"))
    if base_level in {"phase_ta_rare", "phase_ta"}:
        mask &= benchmarks["therapeutic_area"].eq(base_row.get("therapeutic_area"))
    mask &= benchmarks["therapeutic_modality"].eq(modality)
    mask &= pd.to_numeric(benchmarks[f"{metric_prefix}_n"], errors="coerce").fillna(0).ge(MIN_MODALITY_REFINEMENT_N)

    refined = benchmarks[mask].copy()
    if refined.empty:
        return base_row
    return refined.sort_values(f"{metric_prefix}_n", ascending=False).iloc[0]


def _is_non_vaccine_infections_snapshot(snapshot: dict[str, Any]) -> bool:
    therapeutic_area = _clean_ta(
        _first_present(snapshot.get("therapeutic_area"), snapshot.get("therapeutic_area_ui"), snapshot.get("therapeutic_area_ml"))
    )
    modality = _snapshot_modality(snapshot)
    return therapeutic_area == "INFECTIONS" and modality not in {None, "VACCINE"}


def _infection_non_vaccine_key_values(base_row: pd.Series, level: str) -> dict[str, Any]:
    values: dict[str, Any] = {"phase": base_row.get("phase")}
    if level == "phase_indication_rare":
        values["gbd_cause_id_3_ml"] = base_row.get("gbd_cause_id_3_ml")
        values["rare_disease_flag"] = base_row.get("rare_disease_flag")
    elif level == "phase_ta_rare":
        values["therapeutic_area"] = "INFECTIONS"
        values["rare_disease_flag"] = base_row.get("rare_disease_flag")
    elif level == "phase_ta":
        values["therapeutic_area"] = "INFECTIONS"
    return values


def _infection_non_vaccine_fallback(
    snapshot: dict[str, Any],
    benchmarks: pd.DataFrame,
    base_row: pd.Series,
    metric_prefix: str | None,
) -> pd.Series:
    if metric_prefix not in MODALITY_REFINED_METRICS or not _is_non_vaccine_infections_snapshot(snapshot):
        return base_row

    base_level = str(base_row.get("benchmark_level_used") or "")
    clinical_levels = ("phase_indication_rare", "phase_ta_rare", "phase_ta")
    if base_level not in clinical_levels:
        return base_row

    start_index = clinical_levels.index(base_level)
    for clinical_level in clinical_levels[start_index:]:
        non_vaccine_level = NON_VACCINE_INFECTIONS_LEVELS[clinical_level]
        values = _infection_non_vaccine_key_values(base_row, clinical_level)
        mask = benchmarks["benchmark_level_used"].eq(non_vaccine_level)
        for column, expected in values.items():
            if column in {"gbd_cause_id_3_ml", "rare_disease_flag"}:
                mask &= benchmarks[column].eq(float(expected))
            else:
                mask &= benchmarks[column].eq(expected)
        mask &= pd.to_numeric(benchmarks[f"{metric_prefix}_n"], errors="coerce").fillna(0).ge(MIN_NON_VACCINE_INFECTIONS_N)
        candidates = benchmarks[mask].copy()
        if not candidates.empty:
            return candidates.sort_values(f"{metric_prefix}_n", ascending=False).iloc[0]
    return base_row


def _refine_operational_row(
    snapshot: dict[str, Any],
    benchmarks: pd.DataFrame,
    base_row: pd.Series,
    metric_prefix: str | None,
) -> pd.Series:
    modality_row = _same_level_modality_refinement(snapshot, benchmarks, base_row, metric_prefix)
    if modality_row.get("benchmark_key") != base_row.get("benchmark_key"):
        return modality_row
    return _infection_non_vaccine_fallback(snapshot, benchmarks, base_row, metric_prefix)


def lookup_operational_benchmark(
    snapshot: dict[str, Any],
    artifact: pd.DataFrame | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    metric_prefix: str | None = None,
    cohort_metric_prefixes: tuple[str, ...] = METRIC_PREFIXES,
    min_metric_n: int = MIN_USABLE_METRIC_N,
    require_confident: bool = True,
) -> pd.Series | None:
    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    if benchmarks.empty:
        return None
    if metric_prefix is not None and metric_prefix not in METRIC_PREFIXES:
        raise ValueError(f"Unknown metric prefix: {metric_prefix}")
    for prefix in cohort_metric_prefixes:
        if prefix not in METRIC_PREFIXES:
            raise ValueError(f"Unknown cohort metric prefix: {prefix}")

    include_duration_bin = metric_prefix in DURATION_METRICS
    for level, values in _candidate_keys(snapshot, include_duration_bin=include_duration_bin):
        mask = benchmarks["benchmark_level_used"].eq(level)
        for column, expected in values.items():
            if column in {"gbd_cause_id_3_ml", "rare_disease_flag"}:
                mask &= benchmarks[column].eq(float(expected))
            else:
                mask &= benchmarks[column].eq(expected)
        candidates = benchmarks[mask].copy()
        if metric_prefix is not None:
            candidates = candidates[
                pd.to_numeric(candidates[f"{metric_prefix}_n"], errors="coerce").fillna(0).ge(min_metric_n)
            ]
        if candidates.empty:
            continue
        if require_confident and metric_prefix is not None:
            cohort_confident_mask = pd.Series(False, index=candidates.index)
            for prefix in cohort_metric_prefixes:
                cohort_confident_mask |= ~candidates[f"{prefix}_low_confidence_flag"]
            confident = candidates[cohort_confident_mask]
            if not confident.empty:
                base_row = confident.sort_values(f"{metric_prefix}_n", ascending=False).iloc[0]
                return _refine_operational_row(snapshot, benchmarks, base_row, metric_prefix)
        elif require_confident:
            confidence_columns = [f"{prefix}_low_confidence_flag" for prefix in METRIC_PREFIXES]
            confident = candidates[~candidates[confidence_columns].all(axis=1)]
            if not confident.empty:
                return confident.iloc[0]
        else:
            sort_columns = [f"{metric_prefix}_low_confidence_flag", f"{metric_prefix}_n"] if metric_prefix else ["benchmark_key"]
            ascending = [True, False] if metric_prefix else [True]
            base_row = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
            return _refine_operational_row(snapshot, benchmarks, base_row, metric_prefix)

    if require_confident:
        return lookup_operational_benchmark(
            snapshot,
            benchmarks,
            artifact_path=artifact_path,
            metric_prefix=metric_prefix,
            cohort_metric_prefixes=cohort_metric_prefixes,
            min_metric_n=min_metric_n,
            require_confident=False,
        )
    return None


def planned_sites_default_from_operational_benchmark(
    snapshot: dict[str, Any],
    *,
    planned_enrollment: Any,
    current_registry_facility_count_proxy: Any = None,
    overall_status: Any = None,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    current_proxy = _positive_float(current_registry_facility_count_proxy)
    completed = str(overall_status or snapshot.get("overall_status") or "").strip().upper() == "COMPLETED"
    if completed and current_proxy is not None:
        return {
            "value": int(round(current_proxy)),
            "source": "completed_registry_facility_count",
            "site_default_basis": "completed_registry_facility_count",
            "current_registry_facility_count_proxy": current_proxy,
            "site_count_benchmark_p50": None,
            "patients_per_site_p50": None,
            "patients_per_site_benchmark_level_used": None,
            "patients_per_site_n": None,
            "patients_per_site_low_confidence_flag": None,
            "enrollment_coherent_site_candidate": None,
            "operational_benchmark_snapshot_id": None,
        }

    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    site_row = lookup_operational_benchmark(snapshot, benchmarks, metric_prefix="site_count")
    pps_row = lookup_operational_benchmark(snapshot, benchmarks, metric_prefix="patients_per_site")

    site_p50 = _positive_float(site_row.get("site_count_p50")) if site_row is not None else None
    pps_p50 = _positive_float(pps_row.get("patients_per_site_p50")) if pps_row is not None else None
    enrollment_value = _positive_float(planned_enrollment)
    enrollment_candidate = None
    if enrollment_value is not None and pps_p50 is not None:
        enrollment_candidate = enrollment_value / pps_p50

    candidates = [
        ("current_registry_facility_count_proxy", current_proxy),
        ("enrollment_coherent_benchmark_default", enrollment_candidate),
    ]
    if enrollment_candidate is None:
        candidates.append(("benchmark_default", site_p50))
    available = [(basis, value) for basis, value in candidates if value is not None and value > 0]
    if not available:
        return {
            "value": None,
            "source": "registry_facility_count_proxy",
            "site_default_basis": "not_available",
            "current_registry_facility_count_proxy": current_proxy,
            "site_count_benchmark_p50": site_p50,
            "patients_per_site_p50": pps_p50,
            "patients_per_site_benchmark_level_used": pps_row.get("benchmark_level_used") if pps_row is not None else None,
            "patients_per_site_n": _metric_n(pps_row, "patients_per_site") if pps_row is not None else None,
            "patients_per_site_low_confidence_flag": (
                bool(pps_row.get("patients_per_site_low_confidence_flag", True)) if pps_row is not None else None
            ),
            "enrollment_coherent_site_candidate": enrollment_candidate,
            "operational_benchmark_snapshot_id": None,
        }

    selected_basis, selected_value = max(available, key=lambda item: item[1])
    if selected_basis == "current_registry_facility_count_proxy":
        source = "current_registry_facility_count_proxy"
    else:
        source = selected_basis

    snapshot_row = pps_row if pps_row is not None else site_row
    snapshot_id = None
    if snapshot_row is not None:
        snapshot_id = (
            f"{snapshot_row.get('benchmark_version')}:"
            f"{snapshot_row.get('source_data_version')}:"
            f"{snapshot_row.get('benchmark_key')}"
        )

    return {
        "value": int(math.ceil(float(selected_value))),
        "source": source,
        "site_default_basis": selected_basis,
        "current_registry_facility_count_proxy": current_proxy,
        "site_count_benchmark_p50": site_p50,
        "patients_per_site_p50": pps_p50,
        "patients_per_site_benchmark_level_used": pps_row.get("benchmark_level_used") if pps_row is not None else None,
        "patients_per_site_n": _metric_n(pps_row, "patients_per_site") if pps_row is not None else None,
        "patients_per_site_low_confidence_flag": (
            bool(pps_row.get("patients_per_site_low_confidence_flag", True)) if pps_row is not None else None
        ),
        "enrollment_coherent_site_candidate": enrollment_candidate,
        "operational_benchmark_snapshot_id": snapshot_id,
    }


def planned_enrollment_default_from_operational_benchmark(
    snapshot: dict[str, Any],
    *,
    observed_lower_bound: Any = None,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    row = lookup_operational_benchmark(snapshot, benchmarks, metric_prefix="enrollment")
    p50 = _positive_float(row.get("enrollment_p50")) if row is not None else None
    lower_bound = _positive_float(observed_lower_bound)
    candidates = [
        ("observed_lower_bound", lower_bound),
        ("model_default", p50),
    ]
    available = [(basis, value) for basis, value in candidates if value is not None and value > 0]
    if not available:
        return {
            "value": None,
            "source": "planned_value",
            "observed_lower_bound": lower_bound,
            "enrollment_benchmark_p50": p50,
            "operational_benchmark_snapshot_id": None,
        }

    selected_basis, selected_value = max(available, key=lambda item: item[1])
    return {
        "value": int(round(float(selected_value))),
        "source": selected_basis,
        "observed_lower_bound": lower_bound,
        "enrollment_benchmark_p50": p50,
        "operational_benchmark_snapshot_id": _benchmark_snapshot_id(row),
    }


def _status_text(snapshot: dict[str, Any], explicit_status: Any = None) -> str:
    return str(_first_present(explicit_status, snapshot.get("overall_status")) or "").strip().upper()


def _date_type_text(value: Any) -> str:
    return str(value or "").strip().upper()


def _duration_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_completion_duration_months_context": _positive_float(
            snapshot.get("primary_completion_duration_months")
        ),
        "total_duration_months_observed": _positive_float(snapshot.get("completion_duration_months")),
        "endpoint_duration_months_context": _positive_float(snapshot.get("primary_duration_months_ml")),
        "primary_completion_date_type": _date_type_text(snapshot.get("primary_completion_date_type")),
        "completion_date_type": _date_type_text(snapshot.get("completion_date_type")),
    }


def planned_primary_completion_default_from_operational_benchmark(
    snapshot: dict[str, Any],
    *,
    overall_status: Any = None,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    benchmark_row: pd.Series | dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    row = pd.Series(benchmark_row) if benchmark_row is not None else None
    if row is None:
        row = lookup_operational_benchmark(
            snapshot,
            benchmarks,
            metric_prefix="primary_completion_months",
            min_metric_n=MIN_DURATION_METRIC_N,
            cohort_metric_prefixes=("primary_completion_months",),
        )
    primary_n = _metric_n(row, "primary_completion_months") if row is not None else None
    benchmark_p50 = None
    if row is not None and primary_n is not None and primary_n >= MIN_DURATION_METRIC_N:
        benchmark_p50 = _positive_float(row.get("primary_completion_months_p50"))
    context = _duration_context(snapshot)
    primary_duration = context["primary_completion_duration_months_context"]
    endpoint_duration = context["endpoint_duration_months_context"]
    date_type = context["primary_completion_date_type"]
    status = _status_text(snapshot, overall_status)
    completed = status == "COMPLETED"
    stopped = status in STOPPED_OR_INTERRUPTED_STATUSES
    active_non_stopped = bool(status and not completed and not stopped)
    warnings: list[str] = []

    if primary_duration is not None:
        if active_non_stopped and date_type == "ACTUAL":
            source = "actual_primary_completion"
        elif active_non_stopped and date_type == "ESTIMATED":
            source = "estimated_primary_completion"
        elif completed and date_type == "ACTUAL":
            source = "completed_actual_primary_completion"
        elif completed and not date_type:
            source = "completed_missing_primary_date_type_duration"
            warnings.append("missing_primary_completion_date_type")
        else:
            source = ""
        if source:
            if endpoint_duration is not None and primary_duration < endpoint_duration:
                warnings.append("primary_completion_shorter_than_primary_duration_ml")
            return {
                "value": round(float(primary_duration), 2),
                "source": source,
                "trusted_direct_value": True,
                "benchmark_primary_completion_p50": benchmark_p50,
                "actual_primary_completion_lower_bound": None,
                "estimated_primary_completion_candidate": primary_duration if source == "estimated_primary_completion" else None,
                "endpoint_duration_months_context": endpoint_duration,
                "benchmark_row": row,
                "warnings": warnings,
            }

    if benchmark_row is not None:
        if benchmark_p50 is None:
            return {
                "value": None,
                "source": "not_available",
                "trusted_direct_value": False,
                "benchmark_primary_completion_p50": None,
                "actual_primary_completion_lower_bound": None,
                "estimated_primary_completion_candidate": None,
                "endpoint_duration_months_context": endpoint_duration,
                "benchmark_row": row,
                "warnings": warnings,
            }
        return {
            "value": round(float(benchmark_p50), 2),
            "source": "same_cohort_benchmark",
            "primary_completion_default_basis": "same_duration_cohort_benchmark",
            "trusted_direct_value": False,
            "benchmark_primary_completion_p50": benchmark_p50,
            "actual_primary_completion_lower_bound": None,
            "estimated_primary_completion_candidate": None,
            "endpoint_duration_months_context": endpoint_duration,
            "benchmark_row": row,
            "warnings": warnings,
        }

    lower_bound = None
    estimated_candidate = None
    if stopped and primary_duration is not None:
        if date_type == "ESTIMATED":
            estimated_candidate = primary_duration
        else:
            lower_bound = primary_duration

    candidates = [
        ("benchmark_default", benchmark_p50),
        ("endpoint_duration_floor", endpoint_duration),
        ("actual_primary_completion_lower_bound", lower_bound),
        ("estimated_primary_completion_floor", estimated_candidate),
    ]
    available = [(basis, value) for basis, value in candidates if value is not None and value > 0]
    if not available:
        return {
            "value": None,
            "source": "not_available",
            "trusted_direct_value": False,
            "benchmark_primary_completion_p50": benchmark_p50,
            "actual_primary_completion_lower_bound": lower_bound,
            "estimated_primary_completion_candidate": estimated_candidate,
            "endpoint_duration_months_context": endpoint_duration,
            "benchmark_row": row,
            "warnings": warnings,
        }

    selected_basis, selected_value = max(available, key=lambda item: item[1])
    return {
        "value": round(float(selected_value), 2),
        "source": "benchmark_default_with_floors",
        "primary_completion_default_basis": selected_basis,
        "trusted_direct_value": False,
        "benchmark_primary_completion_p50": benchmark_p50,
        "actual_primary_completion_lower_bound": lower_bound,
        "estimated_primary_completion_candidate": estimated_candidate,
        "endpoint_duration_months_context": endpoint_duration,
        "benchmark_row": row,
        "warnings": warnings,
    }


def planned_duration_default_from_operational_benchmark(
    snapshot: dict[str, Any],
    *,
    overall_status: Any = None,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    row = lookup_operational_benchmark(
        snapshot,
        benchmarks,
        metric_prefix="duration_months",
        min_metric_n=MIN_DURATION_METRIC_N,
        cohort_metric_prefixes=("duration_months",),
    )
    benchmark_p50 = _positive_float(row.get("duration_months_p50")) if row is not None else None
    context = _duration_context(snapshot)
    total_duration = context["total_duration_months_observed"]
    endpoint_duration = context["endpoint_duration_months_context"]
    date_type = context["completion_date_type"]
    status = _status_text(snapshot, overall_status)
    completed = status == "COMPLETED"
    stopped = status in STOPPED_OR_INTERRUPTED_STATUSES
    active_non_stopped = bool(status and not completed and not stopped)
    primary_default = planned_primary_completion_default_from_operational_benchmark(
        snapshot,
        overall_status=status,
        artifact=benchmarks,
        benchmark_row=row,
    )
    planned_primary = _positive_float(primary_default.get("value"))
    trusted_primary = bool(primary_default.get("trusted_direct_value"))
    warnings = list(primary_default.get("warnings") or [])

    if total_duration is not None:
        if completed and date_type == "ACTUAL":
            source = "final_observed_total_duration"
        elif completed and not date_type:
            source = "completed_missing_completion_date_type_duration"
            warnings.append("missing_completion_date_type")
        elif active_non_stopped and date_type == "ACTUAL":
            source = "actual_completion_noncompleted_status_lag"
            warnings.append("actual_completion_date_on_non_completed_status")
        elif active_non_stopped and date_type == "ESTIMATED":
            source = "estimated_planned_total_duration"
        else:
            source = ""
        if source:
            if planned_primary is not None and total_duration < planned_primary:
                warnings.append("total_duration_shorter_than_primary_completion")
            return {
                "value": round(float(total_duration), 2),
                "source": source,
                "trusted_direct_value": True,
                "planned_primary_completion_months": planned_primary,
                "primary_completion_source": primary_default.get("source"),
                "benchmark_total_duration_p50": benchmark_p50,
                "actual_total_duration_lower_bound": None,
                "actual_primary_completion_lower_bound": primary_default.get("actual_primary_completion_lower_bound"),
                "estimated_total_duration_candidate": total_duration if source == "estimated_planned_total_duration" else None,
                "estimated_primary_completion_candidate": primary_default.get("estimated_primary_completion_candidate"),
                "endpoint_duration_months_context": endpoint_duration,
                "benchmark_primary_completion_p50": primary_default.get("benchmark_primary_completion_p50"),
                "benchmark_row": row,
                "primary_benchmark_row": primary_default.get("benchmark_row"),
                "warnings": warnings,
            }

    lower_bound = None
    estimated_candidate = None
    if stopped and total_duration is not None:
        if date_type == "ESTIMATED":
            estimated_candidate = total_duration
        else:
            lower_bound = total_duration

    candidates = [
        ("benchmark_default", benchmark_p50),
        ("actual_total_completion_lower_bound", lower_bound),
        ("estimated_total_completion_floor", estimated_candidate),
    ]
    if planned_primary is not None and primary_default.get("source") != "not_available":
        candidates.append(("planned_primary_completion_months_same_cohort", planned_primary))
    available = [(basis, value) for basis, value in candidates if value is not None and value > 0]
    if not available:
        return {
            "value": None,
            "source": "not_available",
            "trusted_direct_value": False,
            "planned_primary_completion_months": planned_primary,
            "primary_completion_source": primary_default.get("source"),
            "benchmark_total_duration_p50": benchmark_p50,
            "actual_total_duration_lower_bound": lower_bound,
            "actual_primary_completion_lower_bound": primary_default.get("actual_primary_completion_lower_bound"),
            "estimated_total_duration_candidate": estimated_candidate,
            "estimated_primary_completion_candidate": primary_default.get("estimated_primary_completion_candidate"),
            "endpoint_duration_months_context": endpoint_duration,
            "benchmark_primary_completion_p50": primary_default.get("benchmark_primary_completion_p50"),
            "benchmark_row": row,
            "primary_benchmark_row": primary_default.get("benchmark_row"),
            "warnings": warnings,
        }

    selected_basis, selected_value = max(available, key=lambda item: item[1])
    return {
        "value": round(float(selected_value), 2),
        "source": "benchmark_default_with_floors",
        "duration_default_basis": selected_basis,
        "trusted_direct_value": False,
        "planned_primary_completion_months": planned_primary,
        "primary_completion_source": primary_default.get("source"),
        "benchmark_total_duration_p50": benchmark_p50,
        "actual_total_duration_lower_bound": lower_bound,
        "actual_primary_completion_lower_bound": primary_default.get("actual_primary_completion_lower_bound"),
        "estimated_total_duration_candidate": estimated_candidate,
        "estimated_primary_completion_candidate": primary_default.get("estimated_primary_completion_candidate"),
        "endpoint_duration_months_context": endpoint_duration,
        "benchmark_primary_completion_p50": primary_default.get("benchmark_primary_completion_p50"),
        "benchmark_row": row,
        "primary_benchmark_row": primary_default.get("benchmark_row"),
        "warnings": warnings,
    }


def classify_enrollment(planned_enrollment: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    return _classify_against_percentiles(planned_enrollment, benchmark_row, "enrollment")


def classify_site_count(planned_sites: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    return _classify_against_percentiles(planned_sites, benchmark_row, "site_count")


def classify_duration_months(planned_duration_months: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    return _classify_against_percentiles(planned_duration_months, benchmark_row, "duration_months")


def classify_primary_completion_months(
    planned_primary_completion_months: Any,
    benchmark_row: pd.Series | dict[str, Any],
) -> str:
    return _classify_against_percentiles(planned_primary_completion_months, benchmark_row, "primary_completion_months")


def _classify_against_percentiles(value: Any, benchmark_row: pd.Series | dict[str, Any], prefix: str) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or float(numeric_value) <= 0:
        return "not_available"

    row = pd.Series(benchmark_row)
    p25 = pd.to_numeric(row.get(f"{prefix}_p25"), errors="coerce")
    p75 = pd.to_numeric(row.get(f"{prefix}_p75"), errors="coerce")
    p90 = pd.to_numeric(row.get(f"{prefix}_p90"), errors="coerce")
    if pd.isna(p25) or pd.isna(p75) or pd.isna(p90):
        return "not_available"

    numeric_value = float(numeric_value)
    if numeric_value < float(p25):
        return "below_benchmark"
    if numeric_value <= float(p75):
        return "typical"
    if numeric_value <= float(p90):
        return "ambitious"
    return "above_benchmark_high"


def _benchmark_snapshot_id(row: pd.Series | dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    series = pd.Series(row)
    return f"{series.get('benchmark_version')}:{series.get('source_data_version')}:{series.get('benchmark_key')}"


def _metric_value(row: pd.Series | dict[str, Any], column: str) -> float | None:
    value = pd.to_numeric(pd.Series(row).get(column), errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _metric_n(row: pd.Series | dict[str, Any], prefix: str) -> int | None:
    value = pd.to_numeric(pd.Series(row).get(f"{prefix}_n"), errors="coerce")
    if pd.isna(value):
        return None
    return int(value)


def _empty_enrollment_metadata(
    value: Any = None,
    source: str = "planned_value",
    hint: str | None = None,
    is_benchmark_stale: bool = False,
) -> dict[str, Any]:
    return {
        "planned_enrollment": {
            "value": value,
            "source": source,
            "benchmark_level_used": "not_available",
            "benchmark_n": None,
            "benchmark_p25": None,
            "benchmark_p50": None,
            "benchmark_p75": None,
            "benchmark_p90": None,
            "enrollment_status": "not_available",
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": None,
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": True,
            "interpretation_hint": hint or "Enrollment benchmark is not available for this snapshot.",
        }
    }


def _empty_site_metadata(
    value: Any = None,
    source: str = "registry_facility_count_proxy",
    hint: str | None = None,
    is_benchmark_stale: bool = False,
) -> dict[str, Any]:
    return {
        "planned_sites": {
            "value": value,
            "source": source,
            "benchmark_level_used": "not_available",
            "benchmark_n": None,
            "benchmark_p25": None,
            "benchmark_p50": None,
            "benchmark_p75": None,
            "benchmark_p90": None,
            "site_count_status": "not_available",
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": None,
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": True,
            "interpretation_hint": hint or "Site-count benchmark is not available for this snapshot.",
        }
    }


def _empty_duration_metadata(
    value: Any = None,
    source: str = "not_available",
    hint: str | None = None,
    is_benchmark_stale: bool = False,
) -> dict[str, Any]:
    return {
        "planned_duration_months": {
            "value": value,
            "source": source,
            "duration_definition": "start_date_to_completion_date_months",
            "benchmark_level_used": "not_available",
            "benchmark_n": None,
            "benchmark_p25": None,
            "benchmark_p50": None,
            "benchmark_p75": None,
            "benchmark_p90": None,
            "duration_status": "not_available",
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": None,
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": True,
            "planned_primary_completion_months": None,
            "primary_completion_source": "not_available",
            "primary_completion_duration_months_context": None,
            "endpoint_duration_months_context": None,
            "actual_total_duration_lower_bound": None,
            "actual_primary_completion_lower_bound": None,
            "estimated_total_duration_candidate": None,
            "estimated_primary_completion_candidate": None,
            "benchmark_total_duration_p50": None,
            "benchmark_primary_completion_p50": None,
            "warnings": [],
            "interpretation_hint": hint or "Duration benchmark is not available for this snapshot.",
        }
    }


def planned_enrollment_metadata(
    snapshot: dict[str, Any],
    planned_enrollment: Any,
    *,
    source: str = "planned_value",
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    is_benchmark_stale: bool = False,
) -> dict[str, Any]:
    numeric_value = _positive_float(planned_enrollment)
    if numeric_value is None:
        return _empty_enrollment_metadata(planned_enrollment, source, "Planned enrollment is missing or invalid.", is_benchmark_stale)

    row = lookup_operational_benchmark(
        snapshot,
        artifact=artifact,
        artifact_path=artifact_path,
        metric_prefix="enrollment",
    )
    if row is None:
        return _empty_enrollment_metadata(float(numeric_value), source, is_benchmark_stale=is_benchmark_stale)

    status = classify_enrollment(float(numeric_value), row)
    if status == "not_available":
        return _empty_enrollment_metadata(
            float(numeric_value),
            source,
            "Benchmark percentiles are incomplete for this snapshot.",
            is_benchmark_stale,
        )

    hint_map = {
        "below_benchmark": "Enrollment is below the usual historical benchmark for the matched cohort.",
        "typical": "Enrollment is within the usual historical benchmark range for the matched cohort.",
        "ambitious": "Enrollment is above the usual range but not beyond the high historical benchmark.",
        "above_benchmark_high": "Enrollment is above the high historical benchmark for the matched cohort.",
    }
    return {
        "planned_enrollment": {
            "value": float(numeric_value),
            "source": source,
            "benchmark_level_used": row.get("benchmark_level_used"),
            "benchmark_n": _metric_n(row, "enrollment"),
            "benchmark_p25": _metric_value(row, "enrollment_p25"),
            "benchmark_p50": _metric_value(row, "enrollment_p50"),
            "benchmark_p75": _metric_value(row, "enrollment_p75"),
            "benchmark_p90": _metric_value(row, "enrollment_p90"),
            "enrollment_status": status,
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": _benchmark_snapshot_id(row),
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": bool(row.get("enrollment_low_confidence_flag", True)),
            "interpretation_hint": hint_map[status],
        }
    }


def planned_sites_metadata(
    snapshot: dict[str, Any],
    planned_sites: Any,
    *,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    source: str = "registry_facility_count_proxy",
    is_benchmark_stale: bool = False,
    planned_enrollment: Any = None,
    current_registry_facility_count_proxy: Any = None,
    overall_status: Any = None,
) -> dict[str, Any]:
    numeric_value = _positive_float(planned_sites)
    if numeric_value is None:
        return _empty_site_metadata(planned_sites, source, "Planned sites value is missing or invalid.", is_benchmark_stale)

    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    row = lookup_operational_benchmark(snapshot, artifact=benchmarks, metric_prefix="site_count")
    if row is None:
        return _empty_site_metadata(float(numeric_value), source, is_benchmark_stale=is_benchmark_stale)

    status = classify_site_count(float(numeric_value), row)
    if status == "not_available":
        return _empty_site_metadata(
            float(numeric_value),
            source,
            "Benchmark percentiles are incomplete for this snapshot.",
            is_benchmark_stale,
        )

    default_context = planned_sites_default_from_operational_benchmark(
        snapshot,
        planned_enrollment=planned_enrollment,
        current_registry_facility_count_proxy=current_registry_facility_count_proxy,
        overall_status=overall_status,
        artifact=benchmarks,
    )
    hint_map = {
        "below_benchmark": "Site count is below the usual completed registry facility-count proxy benchmark for the matched cohort.",
        "typical": "Site count is within the usual completed registry facility-count proxy benchmark range for the matched cohort.",
        "ambitious": "Site count is above the usual range but not beyond the high completed registry facility-count proxy benchmark.",
        "above_benchmark_high": "Site count is above the high completed registry facility-count proxy benchmark for the matched cohort.",
    }
    return {
        "planned_sites": {
            "value": float(numeric_value),
            "source": source,
            "benchmark_level_used": row.get("benchmark_level_used"),
            "benchmark_n": _metric_n(row, "site_count"),
            "benchmark_p25": _metric_value(row, "site_count_p25"),
            "benchmark_p50": _metric_value(row, "site_count_p50"),
            "benchmark_p75": _metric_value(row, "site_count_p75"),
            "benchmark_p90": _metric_value(row, "site_count_p90"),
            "site_count_status": status,
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": _benchmark_snapshot_id(row),
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": bool(row.get("site_count_low_confidence_flag", True)),
            "patients_per_site_benchmark_level_used": default_context.get("patients_per_site_benchmark_level_used"),
            "patients_per_site_n": default_context.get("patients_per_site_n"),
            "patients_per_site_low_confidence_flag": default_context.get("patients_per_site_low_confidence_flag"),
            "current_registry_facility_count_proxy": default_context.get("current_registry_facility_count_proxy"),
            "site_default_basis": default_context.get("site_default_basis"),
            "site_count_benchmark_p50": default_context.get("site_count_benchmark_p50"),
            "patients_per_site_p50": default_context.get("patients_per_site_p50"),
            "enrollment_coherent_site_candidate": default_context.get("enrollment_coherent_site_candidate"),
            "operational_benchmark_snapshot_id": default_context.get("operational_benchmark_snapshot_id"),
            "interpretation_hint": hint_map[status],
        }
    }


def planned_duration_months_metadata(
    snapshot: dict[str, Any],
    planned_duration_months: Any = None,
    *,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    source: str | None = None,
    is_benchmark_stale: bool = False,
    overall_status: Any = None,
) -> dict[str, Any]:
    benchmarks = load_operational_benchmarks(artifact_path) if artifact is None else artifact
    default_context = planned_duration_default_from_operational_benchmark(
        snapshot,
        overall_status=overall_status,
        artifact=benchmarks,
    )
    numeric_value = _positive_float(planned_duration_months)
    if numeric_value is None:
        numeric_value = _positive_float(default_context.get("value"))
    if numeric_value is None:
        return _empty_duration_metadata(
            planned_duration_months,
            default_context.get("source", "not_available"),
            "Planned duration is missing or invalid.",
            is_benchmark_stale,
        )

    row = default_context.get("benchmark_row")
    if row is None:
        return _empty_duration_metadata(float(numeric_value), default_context.get("source", "not_available"), is_benchmark_stale=is_benchmark_stale)

    status = classify_duration_months(float(numeric_value), row)
    if status == "not_available":
        return _empty_duration_metadata(
            float(numeric_value),
            default_context.get("source", "not_available"),
            "Benchmark percentiles are incomplete for this snapshot.",
            is_benchmark_stale,
        )

    primary_row = default_context.get("primary_benchmark_row")
    hint_map = {
        "below_benchmark": "Duration is below the usual historical total-duration benchmark for the matched cohort.",
        "typical": "Duration is within the usual historical total-duration benchmark range for the matched cohort.",
        "ambitious": "Duration is above the usual range but not beyond the high historical benchmark.",
        "above_benchmark_high": "Duration is above the high historical total-duration benchmark for the matched cohort.",
    }
    return {
        "planned_duration_months": {
            "value": float(numeric_value),
            "source": source or default_context.get("source", "not_available"),
            "duration_definition": "start_date_to_completion_date_months",
            "benchmark_level_used": row.get("benchmark_level_used"),
            "benchmark_n": _metric_n(row, "duration_months"),
            "benchmark_p25": _metric_value(row, "duration_months_p25"),
            "benchmark_p50": _metric_value(row, "duration_months_p50"),
            "benchmark_p75": _metric_value(row, "duration_months_p75"),
            "benchmark_p90": _metric_value(row, "duration_months_p90"),
            "duration_status": status,
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": _benchmark_snapshot_id(row),
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": bool(row.get("duration_months_low_confidence_flag", True)),
            "planned_primary_completion_months": default_context.get("planned_primary_completion_months"),
            "primary_completion_source": default_context.get("primary_completion_source"),
            "primary_completion_benchmark_level_used": (
                primary_row.get("benchmark_level_used") if primary_row is not None else None
            ),
            "primary_completion_n": _metric_n(primary_row, "primary_completion_months") if primary_row is not None else None,
            "primary_completion_low_confidence_flag": (
                bool(primary_row.get("primary_completion_months_low_confidence_flag", True))
                if primary_row is not None
                else None
            ),
            "primary_completion_duration_months_context": _duration_context(snapshot)[
                "primary_completion_duration_months_context"
            ],
            "endpoint_duration_months_context": default_context.get("endpoint_duration_months_context"),
            "actual_total_duration_lower_bound": default_context.get("actual_total_duration_lower_bound"),
            "actual_primary_completion_lower_bound": default_context.get("actual_primary_completion_lower_bound"),
            "estimated_total_duration_candidate": default_context.get("estimated_total_duration_candidate"),
            "estimated_primary_completion_candidate": default_context.get("estimated_primary_completion_candidate"),
            "benchmark_total_duration_p50": default_context.get("benchmark_total_duration_p50"),
            "benchmark_primary_completion_p50": default_context.get("benchmark_primary_completion_p50"),
            "duration_default_basis": default_context.get("duration_default_basis"),
            "warnings": default_context.get("warnings") or [],
            "interpretation_hint": hint_map[status],
        }
    }
