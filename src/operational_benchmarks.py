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

METRIC_PREFIXES = ("enrollment", "site_count", "patients_per_site")
MIN_USABLE_METRIC_N = 20

REQUIRED_ARTIFACT_COLUMNS = {
    "benchmark_version",
    "source_data_version",
    "benchmark_key",
    "phase",
    "gbd_cause_id_3_ml",
    "therapeutic_area",
    "rare_disease_flag",
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
    artifact["gbd_cause_id_3_ml"] = pd.to_numeric(artifact["gbd_cause_id_3_ml"], errors="coerce")
    artifact["rare_disease_flag"] = pd.to_numeric(artifact["rare_disease_flag"], errors="coerce")
    for prefix in METRIC_PREFIXES:
        for suffix in ("n", "p25", "p50", "p75", "p90"):
            artifact[f"{prefix}_{suffix}"] = pd.to_numeric(artifact[f"{prefix}_{suffix}"], errors="coerce")
        artifact[f"{prefix}_low_confidence_flag"] = (
            artifact[f"{prefix}_low_confidence_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    return artifact


def _candidate_keys(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    phase = _clean_phase(_first_present(snapshot.get("phase"), snapshot.get("phase_ui"), snapshot.get("phase_ml")))
    indication = _clean_int(snapshot.get("gbd_cause_id_3_ml"))
    therapeutic_area = _clean_ta(
        _first_present(snapshot.get("therapeutic_area"), snapshot.get("therapeutic_area_ui"), snapshot.get("therapeutic_area_ml"))
    )
    rare = _clean_boolish_int(_first_present(snapshot.get("is_rare_disease_ml"), snapshot.get("is_rare_disease")))

    if not phase:
        return []

    candidates: list[tuple[str, dict[str, Any]]] = []
    if indication is not None and rare is not None:
        candidates.append(("phase_indication_rare", {"phase": phase, "gbd_cause_id_3_ml": indication, "rare_disease_flag": rare}))
    if therapeutic_area and rare is not None:
        candidates.append(("phase_ta_rare", {"phase": phase, "therapeutic_area": therapeutic_area, "rare_disease_flag": rare}))
    if therapeutic_area:
        candidates.append(("phase_ta", {"phase": phase, "therapeutic_area": therapeutic_area}))
    candidates.append(("phase_only", {"phase": phase}))
    return candidates


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

    for level, values in _candidate_keys(snapshot):
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
                return confident.sort_values(f"{metric_prefix}_n", ascending=False).iloc[0]
        elif require_confident:
            confidence_columns = [f"{prefix}_low_confidence_flag" for prefix in METRIC_PREFIXES]
            confident = candidates[~candidates[confidence_columns].all(axis=1)]
            if not confident.empty:
                return confident.iloc[0]
        else:
            sort_columns = [f"{metric_prefix}_low_confidence_flag", f"{metric_prefix}_n"] if metric_prefix else ["benchmark_key"]
            ascending = [True, False] if metric_prefix else [True]
            return candidates.sort_values(sort_columns, ascending=ascending).iloc[0]

    if require_confident:
        return lookup_operational_benchmark(
            snapshot,
            benchmarks,
            artifact_path=artifact_path,
            metric_prefix=metric_prefix,
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


def classify_enrollment(planned_enrollment: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    return _classify_against_percentiles(planned_enrollment, benchmark_row, "enrollment")


def classify_site_count(planned_sites: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    return _classify_against_percentiles(planned_sites, benchmark_row, "site_count")


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
            "patients_per_site_n": _metric_n(row, "patients_per_site"),
            "patients_per_site_low_confidence_flag": bool(row.get("patients_per_site_low_confidence_flag", True)),
            "current_registry_facility_count_proxy": default_context.get("current_registry_facility_count_proxy"),
            "site_default_basis": default_context.get("site_default_basis"),
            "site_count_benchmark_p50": default_context.get("site_count_benchmark_p50"),
            "patients_per_site_p50": default_context.get("patients_per_site_p50"),
            "enrollment_coherent_site_candidate": default_context.get("enrollment_coherent_site_candidate"),
            "operational_benchmark_snapshot_id": default_context.get("operational_benchmark_snapshot_id"),
            "interpretation_hint": hint_map[status],
        }
    }
