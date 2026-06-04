from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CLINPRED_PATH = PROJECT_ROOT / "data" / "data_clinpred.csv"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "data" / "operational_benchmarks_v1.csv"
REPORT_PATH = PROJECT_ROOT / "frontend" / "data" / "operational_benchmarks_v1_report.json"
EXCEL_PATH = PROJECT_ROOT / "frontend" / "data" / "operational_benchmarks_v1.xlsx"

BENCHMARK_VERSION = "operational_benchmarks_v1"
OUTLIER_POLICY = (
    "positive completed ACTUAL enrollment, positive completed registry-derived facility-count proxy, "
    "completed ACTUAL enrollment per positive facility-count proxy, completed ACTUAL total duration, "
    "and completed ACTUAL primary completion duration; percentiles used without winsorization"
)
CALIBRATION_NOTES = (
    "Deterministic combined operational benchmark for planned enrollment, site-count proxy, "
    "enrollment-coherent patients-per-site defaults, and duration defaults; same-level therapeutic modality "
    "refinement is used for enrollment and patients-per-site only; duration uses endpoint-duration-bin "
    "clinical fallback; no operational ML model."
)

MIN_N_DEFAULT = 50

INVALID_THERAPEUTIC_AREAS = {"", "OTHER", "OTHER/UNCLASSIFIED", "UNKNOWN", "UNCLASSIFIED"}
INVALID_MODALITIES = {"", "UNKNOWN", "UNCLASSIFIED"}

REQUIRED_COLUMNS = [
    "nct_id",
    "enrollment",
    "enrollment_type",
    "number_of_facilities",
    "overall_status",
    "primary_completion_date_type",
    "completion_date_type",
    "primary_completion_duration_months",
    "completion_duration_months",
    "primary_duration_months_ml",
    "phase_ml",
    "phase",
    "therapeutic_area",
    "therapeutic_area_ml",
    "gbd_cause_id_3_ml",
    "gbd_indication_name_3",
    "is_rare_disease_ml",
    "is_rare_disease",
    "therapeutic_modality_ui",
]

GENERAL_LEVELS = [
    {
        "name": "phase_indication_rare",
        "group_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta_rare",
        "group_cols": ["phase", "therapeutic_area", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta",
        "group_cols": ["phase", "therapeutic_area"],
    },
    {
        "name": "phase_only",
        "group_cols": ["phase"],
    },
]

DURATION_BIN_LEVELS = [
    {
        "name": "phase_indication_rare_endpoint_bin",
        "group_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml", "endpoint_duration_bin"],
    },
    {
        "name": "phase_ta_rare_endpoint_bin",
        "group_cols": ["phase", "therapeutic_area", "is_rare_disease_ml", "endpoint_duration_bin"],
    },
    {
        "name": "phase_ta_endpoint_bin",
        "group_cols": ["phase", "therapeutic_area", "endpoint_duration_bin"],
    },
    {
        "name": "phase_endpoint_bin",
        "group_cols": ["phase", "endpoint_duration_bin"],
    },
]

MODALITY_REFINEMENT_LEVELS = [
    {
        "name": "phase_indication_rare_modality",
        "group_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml", "therapeutic_modality"],
    },
    {
        "name": "phase_ta_rare_modality",
        "group_cols": ["phase", "therapeutic_area", "is_rare_disease_ml", "therapeutic_modality"],
    },
    {
        "name": "phase_ta_modality",
        "group_cols": ["phase", "therapeutic_area", "therapeutic_modality"],
    },
]

NON_VACCINE_INFECTIONS_LEVELS = [
    {
        "name": "phase_indication_rare_non_vaccine_infections",
        "group_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta_rare_non_vaccine_infections",
        "group_cols": ["phase", "therapeutic_area", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta_non_vaccine_infections",
        "group_cols": ["phase", "therapeutic_area"],
    },
]

LEVELS = GENERAL_LEVELS + MODALITY_REFINEMENT_LEVELS + NON_VACCINE_INFECTIONS_LEVELS + DURATION_BIN_LEVELS

PERCENTILES = [0.25, 0.5, 0.75, 0.9]
PERCENTILE_SUFFIXES = {
    0.25: "p25",
    0.5: "p50",
    0.75: "p75",
    0.9: "p90",
}


def _source_data_version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text.upper() if text else "UNKNOWN"


def _as_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


def _is_valid_indication(value: object) -> bool:
    return _as_int(value) > 0


def _is_valid_ta(value: object) -> bool:
    return _clean_text(value) not in INVALID_THERAPEUTIC_AREAS


def _is_valid_modality(value: object) -> bool:
    return _clean_text(value) not in INVALID_MODALITIES


def _is_valid_endpoint_bin(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def endpoint_duration_bin(value: object) -> str | None:
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


def _is_valid_level_key(level_name: str, key_data: dict[str, object]) -> bool:
    if "phase_indication_rare" in level_name and not _is_valid_indication(key_data.get("gbd_cause_id_3_ml")):
        return False
    if "phase_ta" in level_name and not _is_valid_ta(key_data.get("therapeutic_area")):
        return False
    if level_name.endswith("_modality") and not _is_valid_modality(key_data.get("therapeutic_modality")):
        return False
    if "endpoint_bin" in level_name and not _is_valid_endpoint_bin(key_data.get("endpoint_duration_bin")):
        return False
    return True


def _available_usecols(path: Path, requested_columns: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    return [column for column in requested_columns if column in header]


def _benchmark_key(level: str, row: pd.Series) -> str:
    endpoint_suffix = ""
    if "endpoint_bin" in level:
        endpoint_suffix = f"|endpoint_bin={row['endpoint_duration_bin']}"
    if level in {"phase_indication_rare", "phase_indication_rare_endpoint_bin"}:
        return (
            f"{level}|phase={row['phase']}|indication={_as_int(row['gbd_cause_id_3_ml'])}"
            f"|rare={_as_int(row['rare_disease_flag'])}{endpoint_suffix}"
        )
    if level == "phase_indication_rare_modality":
        return (
            f"{level}|phase={row['phase']}|indication={_as_int(row['gbd_cause_id_3_ml'])}"
            f"|rare={_as_int(row['rare_disease_flag'])}|modality={row['therapeutic_modality']}"
        )
    if level == "phase_indication_rare_non_vaccine_infections":
        return (
            f"{level}|phase={row['phase']}|indication={_as_int(row['gbd_cause_id_3_ml'])}"
            f"|rare={_as_int(row['rare_disease_flag'])}|modality=NON_VACCINE"
        )
    if level in {"phase_ta_rare", "phase_ta_rare_endpoint_bin"}:
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}|rare={_as_int(row['rare_disease_flag'])}{endpoint_suffix}"
    if level == "phase_ta_rare_modality":
        return (
            f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}"
            f"|rare={_as_int(row['rare_disease_flag'])}|modality={row['therapeutic_modality']}"
        )
    if level == "phase_ta_rare_non_vaccine_infections":
        return (
            f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}"
            f"|rare={_as_int(row['rare_disease_flag'])}|modality=NON_VACCINE"
        )
    if level in {"phase_ta", "phase_ta_endpoint_bin"}:
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}{endpoint_suffix}"
    if level == "phase_ta_modality":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}|modality={row['therapeutic_modality']}"
    if level == "phase_ta_non_vaccine_infections":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}|modality=NON_VACCINE"
    if level == "phase_endpoint_bin":
        return f"{level}|phase={row['phase']}{endpoint_suffix}"
    return f"{level}|phase={row['phase']}"


def load_source(path: Path = DATA_CLINPRED_PATH) -> pd.DataFrame:
    usecols = _available_usecols(path, REQUIRED_COLUMNS)
    missing_required = {
        "nct_id",
        "enrollment",
        "enrollment_type",
        "number_of_facilities",
        "overall_status",
        "phase",
        "therapeutic_area",
        "gbd_cause_id_3_ml",
        "is_rare_disease_ml",
    }.difference(usecols)
    if missing_required:
        raise ValueError(f"Missing required source columns: {sorted(missing_required)}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["number_of_facilities"] = pd.to_numeric(df["number_of_facilities"], errors="coerce")
    for column in ("primary_completion_duration_months", "completion_duration_months", "primary_duration_months_ml"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = pd.NA
    df["enrollment_type"] = df["enrollment_type"].map(_clean_text)
    df["overall_status"] = df["overall_status"].map(_clean_text)
    df["primary_completion_date_type"] = df.get("primary_completion_date_type", pd.Series(pd.NA, index=df.index)).map(_clean_text)
    df["completion_date_type"] = df.get("completion_date_type", pd.Series(pd.NA, index=df.index)).map(_clean_text)
    df["phase"] = df["phase"].map(_clean_text)
    df["therapeutic_area"] = df["therapeutic_area"].map(_clean_text)
    if "therapeutic_modality_ui" in df.columns:
        df["therapeutic_modality"] = df["therapeutic_modality_ui"].map(_clean_text)
    else:
        df["therapeutic_modality"] = "UNKNOWN"
    df["gbd_cause_id_3_ml"] = pd.to_numeric(df["gbd_cause_id_3_ml"], errors="coerce").fillna(0).astype(int)
    df["is_rare_disease_ml"] = pd.to_numeric(df["is_rare_disease_ml"], errors="coerce").fillna(0).astype(int)
    return df


def add_source_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    completed = flagged["overall_status"].eq("COMPLETED")
    actual_enrollment = flagged["enrollment_type"].eq("ACTUAL")
    actual_primary_completion = flagged["primary_completion_date_type"].eq("ACTUAL")
    actual_completion = flagged["completion_date_type"].eq("ACTUAL")
    positive_enrollment = flagged["enrollment"].gt(0)
    positive_site_count = flagged["number_of_facilities"].gt(0)
    positive_primary_completion_duration = flagged["primary_completion_duration_months"].gt(0)
    positive_completion_duration = flagged["completion_duration_months"].gt(0)

    flagged["is_completed_actual_enrollment_target"] = completed & actual_enrollment & positive_enrollment
    flagged["is_completed_positive_site_count_target"] = completed & positive_site_count
    flagged["is_completed_patients_per_site_target"] = (
        completed & actual_enrollment & positive_enrollment & positive_site_count
    )
    flagged["is_completed_actual_primary_completion_target"] = (
        completed & actual_primary_completion & positive_primary_completion_duration
    )
    flagged["is_completed_actual_total_duration_target"] = (
        completed & actual_completion & positive_completion_duration
    )
    flagged["endpoint_duration_bin"] = flagged["primary_duration_months_ml"].map(endpoint_duration_bin)
    flagged["patients_per_site"] = pd.NA
    target_mask = flagged["is_completed_patients_per_site_target"]
    flagged.loc[target_mask, "patients_per_site"] = (
        flagged.loc[target_mask, "enrollment"] / flagged.loc[target_mask, "number_of_facilities"]
    )
    flagged["patients_per_site"] = pd.to_numeric(flagged["patients_per_site"], errors="coerce")
    return flagged


def _metric_summary(series: pd.Series, prefix: str, min_n: int) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    summary: dict[str, object] = {
        f"{prefix}_n": int(clean.shape[0]),
        f"{prefix}_low_confidence_flag": bool(clean.shape[0] < min_n),
    }
    if clean.empty:
        for suffix in PERCENTILE_SUFFIXES.values():
            summary[f"{prefix}_{suffix}"] = pd.NA
        return summary

    quantiles = clean.quantile(PERCENTILES)
    for percentile, suffix in PERCENTILE_SUFFIXES.items():
        summary[f"{prefix}_{suffix}"] = round(float(quantiles.loc[percentile]), 2)
    return summary


def _summaries_by_level(
    target: pd.DataFrame,
    value_column: str,
    prefix: str,
    min_n: int,
    levels: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for level in levels:
        for keys, group in target.groupby(level["group_cols"], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_data = dict(zip(level["group_cols"], keys, strict=True))
            if not _is_valid_level_key(level["name"], key_data):
                continue
            base = {
                "benchmark_level_used": level["name"],
                "phase": key_data.get("phase", "UNKNOWN"),
                "gbd_cause_id_3_ml": key_data.get("gbd_cause_id_3_ml", ""),
                "therapeutic_area": key_data.get("therapeutic_area", ""),
                "rare_disease_flag": key_data.get("is_rare_disease_ml", ""),
                "therapeutic_modality": key_data.get("therapeutic_modality", ""),
                "endpoint_duration_bin": key_data.get("endpoint_duration_bin", ""),
            }
            benchmark_key = _benchmark_key(level["name"], pd.Series(base))
            summaries[benchmark_key] = _metric_summary(group[value_column], prefix, min_n)
    return summaries


def build_benchmarks(
    df: pd.DataFrame,
    *,
    min_n: int = MIN_N_DEFAULT,
    created_at: str | None = None,
    source_data_version: str = "unknown",
) -> pd.DataFrame:
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    enrollment_target = df[df["is_completed_actual_enrollment_target"]].copy()
    site_target = df[df["is_completed_positive_site_count_target"]].copy()
    pps_target = df[df["is_completed_patients_per_site_target"]].copy()
    primary_completion_target = df[df["is_completed_actual_primary_completion_target"]].copy()
    total_duration_target = df[df["is_completed_actual_total_duration_target"]].copy()

    infection_non_vaccine = (
        df["therapeutic_area"].eq("INFECTIONS") & ~df["therapeutic_modality"].eq("VACCINE")
    )

    enrollment = _summaries_by_level(enrollment_target, "enrollment", "enrollment", min_n, GENERAL_LEVELS + MODALITY_REFINEMENT_LEVELS)
    site_count = _summaries_by_level(site_target, "number_of_facilities", "site_count", min_n, GENERAL_LEVELS)
    patients_per_site = _summaries_by_level(pps_target, "patients_per_site", "patients_per_site", min_n, GENERAL_LEVELS + MODALITY_REFINEMENT_LEVELS)
    primary_completion = _summaries_by_level(
        primary_completion_target,
        "primary_completion_duration_months",
        "primary_completion_months",
        min_n,
        DURATION_BIN_LEVELS + GENERAL_LEVELS,
    )
    total_duration = _summaries_by_level(
        total_duration_target,
        "completion_duration_months",
        "duration_months",
        min_n,
        DURATION_BIN_LEVELS + GENERAL_LEVELS,
    )

    enrollment.update(
        _summaries_by_level(
            enrollment_target[infection_non_vaccine.loc[enrollment_target.index]],
            "enrollment",
            "enrollment",
            min_n,
            NON_VACCINE_INFECTIONS_LEVELS,
        )
    )
    patients_per_site.update(
        _summaries_by_level(
            pps_target[infection_non_vaccine.loc[pps_target.index]],
            "patients_per_site",
            "patients_per_site",
            min_n,
            NON_VACCINE_INFECTIONS_LEVELS,
        )
    )

    rows: list[dict[str, object]] = []
    for level in LEVELS:
        level_keys: set[str] = set()
        for summaries in (enrollment, site_count, patients_per_site, primary_completion, total_duration):
            level_keys.update(
                key for key in summaries if key.startswith(f"{level['name']}|")
            )

        for benchmark_key in sorted(level_keys):
            parts = dict(part.split("=", 1) for part in benchmark_key.split("|")[1:])
            base = {
                "benchmark_version": BENCHMARK_VERSION,
                "source_data_version": source_data_version,
                "benchmark_key": benchmark_key,
                "phase": parts.get("phase", "UNKNOWN"),
                "gbd_cause_id_3_ml": _as_int(parts["indication"]) if "indication" in parts else "",
                "therapeutic_area": parts.get("ta", ""),
                "rare_disease_flag": _as_int(parts["rare"]) if "rare" in parts else "",
                "therapeutic_modality": parts.get("modality", ""),
                "endpoint_duration_bin": parts.get("endpoint_bin", ""),
                "benchmark_level_used": level["name"],
                "created_at": created_at,
                "outlier_policy": OUTLIER_POLICY,
                "calibration_notes": CALIBRATION_NOTES,
            }
            row = dict(base)
            for prefix, summaries in (
                ("enrollment", enrollment),
                ("site_count", site_count),
                ("patients_per_site", patients_per_site),
                ("primary_completion_months", primary_completion),
                ("duration_months", total_duration),
            ):
                row.update(summaries.get(benchmark_key, _metric_summary(pd.Series(dtype=float), prefix, min_n)))
            rows.append(row)

    columns = [
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
        "enrollment_n",
        "enrollment_p25",
        "enrollment_p50",
        "enrollment_p75",
        "enrollment_p90",
        "enrollment_low_confidence_flag",
        "site_count_n",
        "site_count_p25",
        "site_count_p50",
        "site_count_p75",
        "site_count_p90",
        "site_count_low_confidence_flag",
        "patients_per_site_n",
        "patients_per_site_p25",
        "patients_per_site_p50",
        "patients_per_site_p75",
        "patients_per_site_p90",
        "patients_per_site_low_confidence_flag",
        "primary_completion_months_n",
        "primary_completion_months_p25",
        "primary_completion_months_p50",
        "primary_completion_months_p75",
        "primary_completion_months_p90",
        "primary_completion_months_low_confidence_flag",
        "duration_months_n",
        "duration_months_p25",
        "duration_months_p50",
        "duration_months_p75",
        "duration_months_p90",
        "duration_months_low_confidence_flag",
        "created_at",
        "outlier_policy",
        "calibration_notes",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "benchmark_level_used",
            "phase",
            "gbd_cause_id_3_ml",
            "therapeutic_area",
            "rare_disease_flag",
            "endpoint_duration_bin",
        ]
    )


def _lookup_coverage(df: pd.DataFrame, artifact: pd.DataFrame, metric_prefix: str) -> tuple[dict[str, int], int, int]:
    confident: dict[tuple[object, ...], dict[str, object]] = {}
    low_confidence: dict[tuple[object, ...], dict[str, object]] = {}
    if metric_prefix in {"primary_completion_months", "duration_months"}:
        level_sequence = DURATION_BIN_LEVELS + GENERAL_LEVELS
    else:
        level_sequence = LEVELS
    level_cols = {level["name"]: level["group_cols"] for level in level_sequence}
    n_column = f"{metric_prefix}_n"
    low_column = f"{metric_prefix}_low_confidence_flag"

    for _, row in artifact.iterrows():
        metric_n = pd.to_numeric(row.get(n_column), errors="coerce")
        if pd.isna(metric_n) or int(metric_n) <= 0:
            continue
        level_name = row["benchmark_level_used"]
        if level_name not in level_cols:
            continue
        cols = level_cols[level_name]
        values = []
        for column in cols:
            source_column = "rare_disease_flag" if column == "is_rare_disease_ml" else column
            values.append(row[source_column])
        key = tuple([level_name] + values)
        target = low_confidence if bool(row[low_column]) else confident
        target[key] = row.to_dict()

    matched_levels: list[str] = []
    low_confidence_matches = 0
    for _, row in df.iterrows():
        first_low_confidence: str | None = None
        matched_level = "not_available"
        for level in level_sequence:
            key = tuple([level["name"]] + [row[column] for column in level["group_cols"]])
            if key in confident:
                matched_level = level["name"]
                break
            if first_low_confidence is None and key in low_confidence:
                first_low_confidence = level["name"]
        else:
            if first_low_confidence is not None:
                matched_level = first_low_confidence
                low_confidence_matches += 1
        matched_levels.append(matched_level)

    counts = {str(k): int(v) for k, v in Counter(matched_levels).items()}
    return counts, int(counts.get("not_available", 0)), int(low_confidence_matches)


def build_report(
    df: pd.DataFrame,
    artifact: pd.DataFrame,
    *,
    min_n: int,
    created_at: str,
    source_data_version: str,
) -> dict[str, object]:
    enrollment_target = df[df["is_completed_actual_enrollment_target"]].copy()
    site_target = df[df["is_completed_positive_site_count_target"]].copy()
    pps_target = df[df["is_completed_patients_per_site_target"]].copy()
    primary_completion_target = df[df["is_completed_actual_primary_completion_target"]].copy()
    total_duration_target = df[df["is_completed_actual_total_duration_target"]].copy()

    coverage = {}
    for prefix in ("enrollment", "site_count", "patients_per_site", "primary_completion_months", "duration_months"):
        counts, not_available, low_confidence = _lookup_coverage(df, artifact, prefix)
        coverage[prefix] = {
            "match_counts": counts,
            "not_available": not_available,
            "low_confidence_matches": low_confidence,
        }

    return {
        "source_records_loaded": int(len(df)),
        "completed_actual_enrollment_targets": int(enrollment_target.shape[0]),
        "completed_positive_site_count_targets": int(site_target.shape[0]),
        "completed_patients_per_site_targets": int(pps_target.shape[0]),
        "completed_actual_primary_completion_targets": int(primary_completion_target.shape[0]),
        "completed_actual_total_duration_targets": int(total_duration_target.shape[0]),
        "artifact_rows": int(artifact.shape[0]),
        "minimum_confident_cohort_threshold": int(min_n),
        "duplicate_benchmark_keys": int(artifact["benchmark_key"].duplicated().sum()),
        "benchmark_rows_by_level": {
            str(k): int(v) for k, v in artifact["benchmark_level_used"].value_counts().sort_index().items()
        },
        "low_confidence_rows_by_metric": {
            "enrollment": int((artifact["enrollment_n"].gt(0) & artifact["enrollment_low_confidence_flag"]).sum()),
            "site_count": int((artifact["site_count_n"].gt(0) & artifact["site_count_low_confidence_flag"]).sum()),
            "patients_per_site": int(
                (artifact["patients_per_site_n"].gt(0) & artifact["patients_per_site_low_confidence_flag"]).sum()
            ),
            "primary_completion_months": int(
                (
                    artifact["primary_completion_months_n"].gt(0)
                    & artifact["primary_completion_months_low_confidence_flag"]
                ).sum()
            ),
            "duration_months": int(
                (artifact["duration_months_n"].gt(0) & artifact["duration_months_low_confidence_flag"]).sum()
            ),
        },
        "coverage_qa": coverage,
        "source_data_version": source_data_version,
        "created_at": created_at,
        "outlier_policy": OUTLIER_POLICY,
        "calibration_notes": CALIBRATION_NOTES,
    }


def write_outputs(
    artifact: pd.DataFrame,
    report: dict[str, object],
    output_path: Path,
    report_path: Path,
    excel_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        artifact.to_excel(writer, sheet_name="operational_benchmarks", index=False)
        worksheet = writer.book["operational_benchmarks"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            header = str(column_cells[0].value or "")
            width = min(max(len(header) + 2, 12), 42)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined deterministic operational benchmark artifact.")
    parser.add_argument("--input", type=Path, default=DATA_CLINPRED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH)
    parser.add_argument("--min-n", type=int, default=MIN_N_DEFAULT)
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = add_source_flags(load_source(args.input))
    source_version = _source_data_version(args.input)
    artifact = build_benchmarks(df, min_n=args.min_n, created_at=created_at, source_data_version=source_version)
    report = build_report(df, artifact, min_n=args.min_n, created_at=created_at, source_data_version=source_version)
    write_outputs(artifact, report, args.output, args.report, args.excel)

    print(f"Loaded {report['source_records_loaded']} records")
    print(f"Completed ACTUAL enrollment targets: {report['completed_actual_enrollment_targets']}")
    print(f"Completed positive site-count proxy targets: {report['completed_positive_site_count_targets']}")
    print(f"Completed patients-per-site targets: {report['completed_patients_per_site_targets']}")
    print(f"Completed ACTUAL primary completion targets: {report['completed_actual_primary_completion_targets']}")
    print(f"Completed ACTUAL total duration targets: {report['completed_actual_total_duration_targets']}")
    print(f"Wrote operational benchmark artifact: {args.output} ({len(artifact)} rows)")
    print(f"Wrote operational benchmark report: {args.report}")
    print(f"Wrote operational benchmark Excel export: {args.excel}")
    print("Benchmark rows by level:")
    for level, count in report["benchmark_rows_by_level"].items():
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
