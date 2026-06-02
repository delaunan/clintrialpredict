from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "frontend" / "data" / "site_benchmarks_v1.csv"

LEVEL_ORDER = [
    "phase_indication_rare",
    "phase_ta_rare",
    "phase_ta",
    "phase_only",
]

REQUIRED_ARTIFACT_COLUMNS = {
    "benchmark_version",
    "source_data_version",
    "benchmark_key",
    "phase",
    "gbd_cause_id_3_ml",
    "therapeutic_area",
    "rare_disease_flag",
    "benchmark_level_used",
    "benchmark_n",
    "benchmark_p25",
    "benchmark_p50",
    "benchmark_p75",
    "benchmark_p90",
    "low_confidence_flag",
}


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


def _empty_metadata(
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


def _empty_artifact() -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(REQUIRED_ARTIFACT_COLUMNS))


def load_site_benchmarks(path: str | Path = DEFAULT_ARTIFACT_PATH) -> pd.DataFrame:
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
    for column in ["benchmark_n", "benchmark_p25", "benchmark_p50", "benchmark_p75", "benchmark_p90"]:
        artifact[column] = pd.to_numeric(artifact[column], errors="coerce")
    artifact["low_confidence_flag"] = artifact["low_confidence_flag"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    return artifact


def classify_site_count(planned_sites: Any, benchmark_row: pd.Series | dict[str, Any]) -> str:
    value = pd.to_numeric(planned_sites, errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return "not_available"

    row = pd.Series(benchmark_row)
    p25 = pd.to_numeric(row.get("benchmark_p25"), errors="coerce")
    p75 = pd.to_numeric(row.get("benchmark_p75"), errors="coerce")
    p90 = pd.to_numeric(row.get("benchmark_p90"), errors="coerce")
    if pd.isna(p25) or pd.isna(p75) or pd.isna(p90):
        return "not_available"

    value = float(value)
    if value < float(p25):
        return "below_benchmark"
    if value <= float(p75):
        return "typical"
    if value <= float(p90):
        return "ambitious"
    return "above_benchmark_high"


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


def lookup_site_benchmark(
    snapshot: dict[str, Any],
    artifact: pd.DataFrame | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    require_confident: bool = True,
) -> pd.Series | None:
    benchmarks = load_site_benchmarks(artifact_path) if artifact is None else artifact
    if benchmarks.empty:
        return None

    for level, values in _candidate_keys(snapshot):
        mask = benchmarks["benchmark_level_used"].eq(level)
        for column, expected in values.items():
            if column in {"gbd_cause_id_3_ml", "rare_disease_flag"}:
                mask &= benchmarks[column].eq(float(expected))
            else:
                mask &= benchmarks[column].eq(expected)
        candidates = benchmarks[mask].copy()
        if require_confident:
            confident = candidates[~candidates["low_confidence_flag"]]
            if not confident.empty:
                return confident.sort_values("benchmark_n", ascending=False).iloc[0]
        elif not candidates.empty:
            return candidates.sort_values(["low_confidence_flag", "benchmark_n"], ascending=[True, False]).iloc[0]

    if require_confident:
        return lookup_site_benchmark(snapshot, benchmarks, artifact_path=artifact_path, require_confident=False)
    return None


def planned_sites_metadata(
    snapshot: dict[str, Any],
    planned_sites: Any,
    *,
    artifact: pd.DataFrame | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    source: str = "registry_facility_count_proxy",
    is_benchmark_stale: bool = False,
) -> dict[str, Any]:
    numeric_value = pd.to_numeric(planned_sites, errors="coerce")
    if pd.isna(numeric_value) or float(numeric_value) <= 0:
        return _empty_metadata(planned_sites, source, "Planned sites value is missing or invalid.", is_benchmark_stale)

    row = lookup_site_benchmark(snapshot, artifact=artifact, artifact_path=artifact_path)
    if row is None:
        return _empty_metadata(float(numeric_value), source, is_benchmark_stale=is_benchmark_stale)

    status = classify_site_count(float(numeric_value), row)
    if status == "not_available":
        return _empty_metadata(
            float(numeric_value),
            source,
            "Benchmark percentiles are incomplete for this snapshot.",
            is_benchmark_stale,
        )

    snapshot_id = f"{row.get('benchmark_version')}:{row.get('source_data_version')}:{row.get('benchmark_key')}"
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
            "benchmark_n": int(row["benchmark_n"]) if pd.notna(row.get("benchmark_n")) else None,
            "benchmark_p25": float(row["benchmark_p25"]) if pd.notna(row.get("benchmark_p25")) else None,
            "benchmark_p50": float(row["benchmark_p50"]) if pd.notna(row.get("benchmark_p50")) else None,
            "benchmark_p75": float(row["benchmark_p75"]) if pd.notna(row.get("benchmark_p75")) else None,
            "benchmark_p90": float(row["benchmark_p90"]) if pd.notna(row.get("benchmark_p90")) else None,
            "site_count_status": status,
            "support_level": "not_evaluated",
            "supporting_signals": [],
            "conflicting_signals": [],
            "benchmark_snapshot_id": snapshot_id,
            "is_benchmark_stale": bool(is_benchmark_stale),
            "low_confidence_flag": bool(row.get("low_confidence_flag", True)),
            "interpretation_hint": hint_map[status],
        }
    }
