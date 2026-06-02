from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.operational_benchmarks import (
    DEFAULT_ARTIFACT_PATH,
    REQUIRED_ARTIFACT_COLUMNS,
    load_operational_benchmarks,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEARCH_REGISTRY_PATH = PROJECT_ROOT / "frontend" / "data" / "search_registry.csv"
REPORT_PATH = PROJECT_ROOT / "frontend" / "data" / "operational_benchmarks_v1_audit.json"

ACTIVE_OPERATIONAL_FEATURE_BOUNDARY_PATH = PROJECT_ROOT / "frontend" / "views" / "edit_trial.py"

EXPECTED_GENERAL_LEVELS = {
    "phase_indication_rare",
    "phase_ta_rare",
    "phase_ta",
    "phase_only",
}
EXPECTED_MODALITY_LEVELS = {
    "phase_indication_rare_modality",
    "phase_ta_rare_modality",
    "phase_ta_modality",
}
EXPECTED_NON_VACCINE_INFECTIONS_LEVELS = {
    "phase_indication_rare_non_vaccine_infections",
    "phase_ta_rare_non_vaccine_infections",
    "phase_ta_non_vaccine_infections",
}
MIN_MODALITY_REFINEMENT_N = 50
MIN_NON_VACCINE_INFECTIONS_N = 50


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _positive_number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    return float(numeric)


def _snapshot_from_registry_row(row: pd.Series) -> dict[str, Any]:
    snapshot = row.replace({pd.NA: None}).to_dict()
    snapshot["phase"] = snapshot.get("phase_ml")
    if not snapshot.get("therapeutic_area"):
        snapshot["therapeutic_area"] = snapshot.get("therapeutic_area_ml")
    snapshot["is_rare_disease"] = snapshot.get("is_rare_disease_ml")
    return snapshot


def _clean_phase(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    phase_map = {
        "1": "PHASE1/PHASE2",
        "2": "PHASE2",
        "3": "PHASE2/PHASE3",
        "4": "PHASE3",
    }
    return phase_map.get(text, text.upper() if text else None)


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _clean_int(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def _artifact_index(artifact: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
    index: dict[tuple[Any, ...], pd.Series] = {}
    for _, row in artifact.iterrows():
        level = str(row.get("benchmark_level_used"))
        phase = row.get("phase")
        indication = _clean_int(row.get("gbd_cause_id_3_ml"))
        ta = row.get("therapeutic_area")
        rare = _clean_int(row.get("rare_disease_flag"))
        modality = row.get("therapeutic_modality")
        if level == "phase_indication_rare":
            key = (level, phase, indication, rare)
        elif level == "phase_ta_rare":
            key = (level, phase, ta, rare)
        elif level == "phase_ta":
            key = (level, phase, ta)
        elif level == "phase_only":
            key = (level, phase)
        elif level == "phase_indication_rare_modality":
            key = (level, phase, indication, rare, modality)
        elif level == "phase_ta_rare_modality":
            key = (level, phase, ta, rare, modality)
        elif level == "phase_ta_modality":
            key = (level, phase, ta, modality)
        elif level == "phase_indication_rare_non_vaccine_infections":
            key = (level, phase, indication, rare)
        elif level == "phase_ta_rare_non_vaccine_infections":
            key = (level, phase, ta, rare)
        elif level == "phase_ta_non_vaccine_infections":
            key = (level, phase, ta)
        else:
            continue
        index[key] = row
    return index


def _clinical_candidates(row: pd.Series) -> list[tuple[Any, ...]]:
    phase = _clean_phase(row.get("phase_ml") if "phase_ml" in row else row.get("phase"))
    indication = _clean_int(row.get("gbd_cause_id_3_ml"))
    ta = _clean_text(row.get("therapeutic_area"))
    rare = _clean_int(row.get("is_rare_disease_ml"))
    if not phase:
        return []
    candidates: list[tuple[Any, ...]] = []
    if indication is not None and rare is not None:
        candidates.append(("phase_indication_rare", phase, indication, rare))
    if ta and rare is not None:
        candidates.append(("phase_ta_rare", phase, ta, rare))
    if ta:
        candidates.append(("phase_ta", phase, ta))
    candidates.append(("phase_only", phase))
    return candidates


def _modality_key_for_base(row: pd.Series, base: pd.Series) -> tuple[Any, ...] | None:
    modality = _clean_text(row.get("therapeutic_modality_ui"))
    if not modality:
        return None
    level = str(base.get("benchmark_level_used"))
    phase = base.get("phase")
    indication = _clean_int(base.get("gbd_cause_id_3_ml"))
    ta = base.get("therapeutic_area")
    rare = _clean_int(base.get("rare_disease_flag"))
    if level == "phase_indication_rare":
        return ("phase_indication_rare_modality", phase, indication, rare, modality)
    if level == "phase_ta_rare":
        return ("phase_ta_rare_modality", phase, ta, rare, modality)
    if level == "phase_ta":
        return ("phase_ta_modality", phase, ta, modality)
    return None


def _is_non_vaccine_infections_row(row: pd.Series) -> bool:
    return _clean_text(row.get("therapeutic_area")) == "INFECTIONS" and _clean_text(row.get("therapeutic_modality_ui")) != "VACCINE"


def _non_vaccine_infections_keys_for_base(base: pd.Series) -> list[tuple[Any, ...]]:
    level = str(base.get("benchmark_level_used"))
    phase = base.get("phase")
    indication = _clean_int(base.get("gbd_cause_id_3_ml"))
    rare = _clean_int(base.get("rare_disease_flag"))
    if level == "phase_indication_rare":
        return [
            ("phase_indication_rare_non_vaccine_infections", phase, indication, rare),
            ("phase_ta_rare_non_vaccine_infections", phase, "INFECTIONS", rare),
            ("phase_ta_non_vaccine_infections", phase, "INFECTIONS"),
        ]
    if level == "phase_ta_rare":
        return [
            ("phase_ta_rare_non_vaccine_infections", phase, "INFECTIONS", rare),
            ("phase_ta_non_vaccine_infections", phase, "INFECTIONS"),
        ]
    if level == "phase_ta":
        return [("phase_ta_non_vaccine_infections", phase, "INFECTIONS")]
    return []


def _lookup_fast(row: pd.Series, index: dict[tuple[Any, ...], pd.Series], metric: str) -> pd.Series | None:
    first_low: pd.Series | None = None
    for key in _clinical_candidates(row):
        base = index.get(key)
        if base is None:
            continue
        metric_n = pd.to_numeric(base.get(f"{metric}_n"), errors="coerce")
        if pd.isna(metric_n) or int(metric_n) < 20:
            continue
        if first_low is None:
            first_low = base
        if bool(base.get(f"{metric}_low_confidence_flag", True)):
            continue
        if metric in {"enrollment", "patients_per_site"}:
            modality_key = _modality_key_for_base(row, base)
            refined = index.get(modality_key) if modality_key else None
            if refined is not None:
                refined_n = pd.to_numeric(refined.get(f"{metric}_n"), errors="coerce")
                if not pd.isna(refined_n) and int(refined_n) >= MIN_MODALITY_REFINEMENT_N:
                    return refined
            if _is_non_vaccine_infections_row(row):
                for non_vaccine_key in _non_vaccine_infections_keys_for_base(base):
                    non_vaccine = index.get(non_vaccine_key)
                    if non_vaccine is None:
                        continue
                    non_vaccine_n = pd.to_numeric(non_vaccine.get(f"{metric}_n"), errors="coerce")
                    if not pd.isna(non_vaccine_n) and int(non_vaccine_n) >= MIN_NON_VACCINE_INFECTIONS_N:
                        return non_vaccine
        return base
    return first_low


def _level_counts(rows: list[pd.Series | None]) -> dict[str, int]:
    levels = [
        "not_available" if row is None else str(row.get("benchmark_level_used") or "not_available")
        for row in rows
    ]
    return {str(k): int(v) for k, v in Counter(levels).items()}


def _modality_assignment_summary(registry: pd.DataFrame) -> dict[str, Any]:
    modality = registry.get("therapeutic_modality_ui", pd.Series(dtype=object))
    assigned = (
        modality.notna()
        & modality.astype(str).str.strip().ne("")
        & ~modality.astype(str).str.strip().str.upper().isin({"UNKNOWN", "NOT AVAILABLE", "NAN", "NONE"})
    )
    return {
        "rows": int(registry.shape[0]),
        "assigned": int(assigned.sum()),
        "not_assigned": int((~assigned).sum()),
        "assigned_pct": round(float(assigned.mean() * 100), 2) if registry.shape[0] else 0.0,
        "counts": {str(k): int(v) for k, v in modality[assigned].value_counts().items()},
    }


def _safe_lookup_rows(
    registry: pd.DataFrame,
    index: dict[tuple[Any, ...], pd.Series],
    metric: str,
) -> list[pd.Series | None]:
    return [_lookup_fast(row, index, metric) for _, row in registry.iterrows()]


def _summarize_lookup_coverage(
    registry: pd.DataFrame,
    index: dict[tuple[Any, ...], pd.Series],
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for metric in ("enrollment", "site_count", "patients_per_site"):
        rows = _safe_lookup_rows(registry, index, metric)
        not_available = sum(row is None for row in rows)
        modality_rows = [
            row
            for row in rows
            if row is not None and str(row.get("benchmark_level_used") or "").endswith("_modality")
        ]
        non_vaccine_infections_rows = [
            row
            for row in rows
            if row is not None and str(row.get("benchmark_level_used") or "").endswith("_non_vaccine_infections")
        ]
        low_confidence = [
            row
            for row in rows
            if row is not None and bool(row.get(f"{metric}_low_confidence_flag", True))
        ]
        too_sparse = [
            row
            for row in rows
            if row is not None and (pd.to_numeric(row.get(f"{metric}_n"), errors="coerce") < 20)
        ]
        coverage[metric] = {
            "not_available": int(not_available),
            "level_counts": _level_counts(rows),
            "modality_refined_matches": int(len(modality_rows)),
            "non_vaccine_infections_matches": int(len(non_vaccine_infections_rows)),
            "low_confidence_matches": int(len(low_confidence)),
            "too_sparse_matches": int(len(too_sparse)),
        }
    return coverage


def _summarize_defaulting(
    registry: pd.DataFrame,
    index: dict[tuple[Any, ...], pd.Series],
) -> dict[str, Any]:
    enrollment_sources: Counter[str] = Counter()
    site_sources: Counter[str] = Counter()
    site_basis: Counter[str] = Counter()
    site_below_current_proxy = 0
    non_completed_with_positive_proxy = 0
    completed_site_proxy_not_used = 0
    examples: list[dict[str, Any]] = []

    for _, row in registry.iterrows():
        status = str(row.get("overall_status") or "").strip().upper()
        current_proxy = _positive_number(row.get("number_of_facilities"))
        observed_enrollment = _positive_number(row.get("enrollment"))

        enrollment_row = _lookup_fast(row, index, "enrollment")
        enrollment_p50 = _positive_number(enrollment_row.get("enrollment_p50")) if enrollment_row is not None else None
        observed_lower_bound = observed_enrollment if status != "COMPLETED" else None
        enrollment_candidates = [
            ("observed_lower_bound", observed_lower_bound),
            ("model_default", enrollment_p50),
        ]
        enrollment_available = [(source, value) for source, value in enrollment_candidates if value is not None]
        if enrollment_available:
            enrollment_source, planned_enrollment = max(enrollment_available, key=lambda item: item[1])
        else:
            enrollment_source, planned_enrollment = "planned_value", observed_enrollment
        enrollment_sources[enrollment_source] += 1

        if status == "COMPLETED" and current_proxy is not None:
            site_value = round(current_proxy)
            site_source = "completed_registry_facility_count"
            site_default_basis = "completed_registry_facility_count"
            pps_p50 = None
            pps_level = None
        else:
            site_row = _lookup_fast(row, index, "site_count")
            pps_row = _lookup_fast(row, index, "patients_per_site")
            site_p50 = _positive_number(site_row.get("site_count_p50")) if site_row is not None else None
            pps_p50 = _positive_number(pps_row.get("patients_per_site_p50")) if pps_row is not None else None
            pps_level = pps_row.get("benchmark_level_used") if pps_row is not None else None
            enrollment_candidate = (
                planned_enrollment / pps_p50
                if planned_enrollment is not None and pps_p50 is not None
                else None
            )
            candidates = [
                ("current_registry_facility_count_proxy", current_proxy),
                ("enrollment_coherent_benchmark_default", enrollment_candidate),
            ]
            if enrollment_candidate is None:
                candidates.append(("benchmark_default", site_p50))
            available = [(basis, value) for basis, value in candidates if value is not None and value > 0]
            if available:
                site_default_basis, selected_value = max(available, key=lambda item: item[1])
                site_value = int(selected_value) if float(selected_value).is_integer() else int(selected_value) + 1
                site_source = site_default_basis
            else:
                site_default_basis = "not_available"
                site_value = None
                site_source = "registry_facility_count_proxy"

        site_sources[site_source] += 1
        site_basis[site_default_basis] += 1

        if status == "COMPLETED" and current_proxy is not None and site_value != round(current_proxy):
            completed_site_proxy_not_used += 1
        if status != "COMPLETED" and current_proxy is not None:
            non_completed_with_positive_proxy += 1
            if site_value is not None and site_value < current_proxy:
                site_below_current_proxy += 1

        if len(examples) < 8 and site_source in {
            "current_registry_facility_count_proxy",
            "enrollment_coherent_benchmark_default",
            "benchmark_default",
        }:
            examples.append(
                {
                    "nct_id": row.get("nct_id"),
                    "status": status,
                    "modality": row.get("therapeutic_modality_ui"),
                    "enrollment": observed_enrollment,
                    "current_registry_facility_count_proxy": current_proxy,
                    "site_default_value": site_value,
                    "site_default_basis": site_default_basis,
                    "patients_per_site_p50": pps_p50,
                    "patients_per_site_level": pps_level,
                }
            )

    return {
        "enrollment_default_sources": {str(k): int(v) for k, v in enrollment_sources.items()},
        "site_default_sources": {str(k): int(v) for k, v in site_sources.items()},
        "site_default_basis": {str(k): int(v) for k, v in site_basis.items()},
        "non_completed_with_positive_current_proxy": int(non_completed_with_positive_proxy),
        "non_completed_site_defaults_below_current_proxy": int(site_below_current_proxy),
        "completed_site_proxy_not_used": int(completed_site_proxy_not_used),
        "examples": examples,
    }


def _artifact_summary(artifact: pd.DataFrame) -> dict[str, Any]:
    modality_rows = artifact["benchmark_level_used"].astype(str).str.endswith("_modality")
    return {
        "path": str(DEFAULT_ARTIFACT_PATH),
        "rows": int(artifact.shape[0]),
        "columns": list(artifact.columns),
        "duplicate_benchmark_keys": int(artifact["benchmark_key"].duplicated().sum()),
        "rows_by_level": {
            str(k): int(v) for k, v in artifact["benchmark_level_used"].value_counts().sort_index().items()
        },
        "modality_rows": int(modality_rows.sum()),
        "phase_only_modality_rows": int(artifact["benchmark_level_used"].eq("phase_only_modality").sum()),
        "site_count_modality_rows_with_values": int(artifact.loc[modality_rows, "site_count_n"].fillna(0).gt(0).sum()),
        "site_count_non_vaccine_infections_rows_with_values": int(
            artifact.loc[
                artifact["benchmark_level_used"].astype(str).str.endswith("_non_vaccine_infections"),
                "site_count_n",
            ].fillna(0).gt(0).sum()
        ),
        "low_confidence_rows_by_metric": {
            metric: int((artifact[f"{metric}_n"].fillna(0).gt(0) & artifact[f"{metric}_low_confidence_flag"]).sum())
            for metric in ("enrollment", "site_count", "patients_per_site")
        },
    }


def _boundary_summary() -> dict[str, Any]:
    text = ACTIVE_OPERATIONAL_FEATURE_BOUNDARY_PATH.read_text(encoding="utf-8")
    try:
        start = text.index("SIMULATION_FEATURE_IDS")
        end = text.index("]", start)
        feature_block = text[start:end]
    except ValueError:
        feature_block = ""
    return {
        "planned_sites_in_simulation_feature_ids": "planned_sites" in feature_block,
        "planned_enrollment_in_simulation_feature_ids": "planned_enrollment" in feature_block,
        "number_of_facilities_in_simulation_feature_ids": "number_of_facilities" in feature_block,
    }


def build_audit_report() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    artifact = load_operational_benchmarks(DEFAULT_ARTIFACT_PATH)
    registry = pd.read_csv(SEARCH_REGISTRY_PATH, low_memory=False)
    index = _artifact_index(artifact)

    _assert(DEFAULT_ARTIFACT_PATH.exists(), f"Missing artifact: {DEFAULT_ARTIFACT_PATH}", failures)
    _assert(not artifact.empty, "Artifact is empty or malformed", failures)
    _assert(REQUIRED_ARTIFACT_COLUMNS.issubset(artifact.columns), "Artifact missing required columns", failures)
    _assert(artifact["benchmark_key"].duplicated().sum() == 0, "Artifact has duplicate benchmark keys", failures)

    levels = set(artifact["benchmark_level_used"].dropna().astype(str))
    _assert(EXPECTED_GENERAL_LEVELS.issubset(levels), "Artifact missing general clinical levels", failures)
    _assert(EXPECTED_MODALITY_LEVELS.issubset(levels), "Artifact missing modality refinement levels", failures)
    _assert(
        EXPECTED_NON_VACCINE_INFECTIONS_LEVELS.issubset(levels),
        "Artifact missing non-vaccine Infections fallback levels",
        failures,
    )
    _assert("phase_only_modality" not in levels, "Artifact should not include phase_only_modality", failures)
    modality_rows = artifact["benchmark_level_used"].astype(str).str.endswith("_modality")
    _assert(
        artifact.loc[modality_rows, "site_count_n"].fillna(0).eq(0).all(),
        "Raw site-count rows should not use modality refinement",
        failures,
    )

    coverage = _summarize_lookup_coverage(registry, index)
    for metric, summary in coverage.items():
        _assert(summary["not_available"] == 0, f"{metric} lookup has not_available matches", failures)
        _assert(summary["too_sparse_matches"] == 0, f"{metric} lookup has n < 20 matches", failures)
    _assert(
        coverage["site_count"]["modality_refined_matches"] == 0,
        "Site-count lookup should not use modality-refined rows",
        failures,
    )
    _assert(
        coverage["site_count"]["non_vaccine_infections_matches"] == 0,
        "Site-count lookup should not use non-vaccine Infections fallback rows",
        failures,
    )
    _assert(
        coverage["enrollment"]["modality_refined_matches"] > 0,
        "Enrollment lookup should have modality-refined matches",
        failures,
    )
    _assert(
        coverage["patients_per_site"]["modality_refined_matches"] > 0,
        "Patients-per-site lookup should have modality-refined matches",
        failures,
    )

    defaulting = _summarize_defaulting(registry, index)
    _assert(defaulting["non_completed_site_defaults_below_current_proxy"] == 0, "Site default below current proxy", failures)
    _assert(defaulting["completed_site_proxy_not_used"] == 0, "Completed site proxy was not preserved", failures)

    boundary = _boundary_summary()
    for key, value in boundary.items():
        _assert(value is False, f"Model-facing boundary failed: {key}", failures)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": _artifact_summary(artifact),
        "registry_modality_assignment": _modality_assignment_summary(registry),
        "lookup_coverage": coverage,
        "defaulting": defaulting,
        "model_boundary": boundary,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    return report, failures


def main() -> None:
    report, failures = build_audit_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Operational benchmark audit status: {report['status']}")
    print(f"Artifact rows: {report['artifact']['rows']}")
    print(f"Duplicate benchmark keys: {report['artifact']['duplicate_benchmark_keys']}")
    print(f"Modality assignment: {report['registry_modality_assignment']['assigned']} assigned / {report['registry_modality_assignment']['not_assigned']} not assigned")
    print("Lookup coverage:")
    for metric, summary in report["lookup_coverage"].items():
        print(
            f"  {metric}: not_available={summary['not_available']}, "
            f"modality_refined={summary['modality_refined_matches']}, "
            f"non_vaccine_infections={summary['non_vaccine_infections_matches']}, "
            f"low_confidence={summary['low_confidence_matches']}"
        )
    print("Defaulting:")
    print(f"  enrollment sources: {report['defaulting']['enrollment_default_sources']}")
    print(f"  site sources: {report['defaulting']['site_default_sources']}")
    print(f"  site defaults below current proxy: {report['defaulting']['non_completed_site_defaults_below_current_proxy']}")
    print(f"  completed site proxy not used: {report['defaulting']['completed_site_proxy_not_used']}")
    print(f"Wrote report: {REPORT_PATH}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
