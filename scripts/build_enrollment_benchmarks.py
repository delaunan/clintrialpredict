from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CLINPRED_PATH = PROJECT_ROOT / "data" / "data_clinpred.csv"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "data" / "enrollment_benchmarks_v1.csv"
REPORT_PATH = PROJECT_ROOT / "frontend" / "data" / "enrollment_benchmarks_v1_report.json"

BENCHMARK_VERSION = "enrollment_benchmarks_v1"
OUTLIER_POLICY = "positive completed ACTUAL enrollment; percentiles used without winsorization"
CALIBRATION_NOTES = "Deterministic v1 planned-enrollment benchmark; no enrollment ML model."

REQUIRED_COLUMNS = [
    "nct_id",
    "enrollment",
    "enrollment_type",
    "overall_status",
    "phase_ml",
    "phase",
    "therapeutic_area",
    "therapeutic_area_ml",
    "gbd_cause_id_3_ml",
    "gbd_indication_name_3",
    "is_rare_disease_ml",
    "is_rare_disease",
]

LEVELS = [
    {
        "name": "phase_indication_rare",
        "group_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml"],
        "key_cols": ["phase", "gbd_cause_id_3_ml", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta_rare",
        "group_cols": ["phase", "therapeutic_area", "is_rare_disease_ml"],
        "key_cols": ["phase", "therapeutic_area", "is_rare_disease_ml"],
    },
    {
        "name": "phase_ta",
        "group_cols": ["phase", "therapeutic_area"],
        "key_cols": ["phase", "therapeutic_area"],
    },
    {
        "name": "phase_only",
        "group_cols": ["phase"],
        "key_cols": ["phase"],
    },
]


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


def _benchmark_key(level: str, row: pd.Series) -> str:
    if level == "phase_indication_rare":
        return f"{level}|phase={row['phase']}|indication={_as_int(row['gbd_cause_id_3_ml'])}|rare={_as_int(row['rare_disease_flag'])}"
    if level == "phase_ta_rare":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}|rare={_as_int(row['rare_disease_flag'])}"
    if level == "phase_ta":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}"
    return f"{level}|phase={row['phase']}"


def load_source(path: Path = DATA_CLINPRED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, low_memory=False)
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["enrollment_type"] = df["enrollment_type"].map(_clean_text)
    df["overall_status"] = df["overall_status"].map(_clean_text)
    df["phase"] = df["phase"].map(_clean_text)
    df["therapeutic_area"] = df["therapeutic_area"].map(_clean_text)
    df["phase_ml"] = pd.to_numeric(df["phase_ml"], errors="coerce")
    df["therapeutic_area_ml"] = pd.to_numeric(df["therapeutic_area_ml"], errors="coerce")
    df["gbd_cause_id_3_ml"] = pd.to_numeric(df["gbd_cause_id_3_ml"], errors="coerce").fillna(0).astype(int)
    df["is_rare_disease_ml"] = pd.to_numeric(df["is_rare_disease_ml"], errors="coerce").fillna(0).astype(int)
    return df


def add_source_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    positive_enrollment = flagged["enrollment"].gt(0)
    completed = flagged["overall_status"].eq("COMPLETED")
    actual = flagged["enrollment_type"].eq("ACTUAL")
    estimated = flagged["enrollment_type"].eq("ESTIMATED")
    ongoing = flagged["overall_status"].isin(
        {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"}
    )

    flagged["is_completed_actual_enrollment_target"] = completed & actual & positive_enrollment
    flagged["is_estimated_planned_enrollment"] = estimated & positive_enrollment
    flagged["is_ongoing_actual_enrollment_lower_bound"] = ongoing & actual & positive_enrollment
    return flagged


def _percentiles(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.25, 0.5, 0.75, 0.9])
    return {
        "benchmark_p25": round(float(quantiles.loc[0.25]), 2),
        "benchmark_p50": round(float(quantiles.loc[0.5]), 2),
        "benchmark_p75": round(float(quantiles.loc[0.75]), 2),
        "benchmark_p90": round(float(quantiles.loc[0.9]), 2),
    }


def build_benchmarks(
    df: pd.DataFrame,
    *,
    min_n: int = 50,
    created_at: str | None = None,
    source_data_version: str = "unknown",
) -> pd.DataFrame:
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    target = df[df["is_completed_actual_enrollment_target"]].copy()

    rows: list[dict[str, object]] = []
    for level in LEVELS:
        for keys, group in target.groupby(level["group_cols"], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_data = dict(zip(level["group_cols"], keys, strict=True))
            summary = _percentiles(group["enrollment"])
            row = {
                "benchmark_version": BENCHMARK_VERSION,
                "source_data_version": source_data_version,
                "benchmark_level_used": level["name"],
                "phase": key_data.get("phase", "UNKNOWN"),
                "indication_or_therapeutic_area": (
                    key_data.get("gbd_cause_id_3_ml")
                    if level["name"] == "phase_indication_rare"
                    else key_data.get("therapeutic_area", "ALL")
                ),
                "gbd_cause_id_3_ml": key_data.get("gbd_cause_id_3_ml", ""),
                "therapeutic_area": key_data.get("therapeutic_area", ""),
                "rare_disease_flag": key_data.get("is_rare_disease_ml", ""),
                "benchmark_n": int(len(group)),
                "low_confidence_flag": bool(len(group) < min_n),
                "created_at": created_at,
                "outlier_policy": OUTLIER_POLICY,
                "calibration_notes": CALIBRATION_NOTES,
            }
            row.update(summary)
            row["benchmark_key"] = _benchmark_key(level["name"], pd.Series(row))
            rows.append(row)

    columns = [
        "benchmark_version",
        "source_data_version",
        "benchmark_key",
        "phase",
        "indication_or_therapeutic_area",
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
        "created_at",
        "outlier_policy",
        "calibration_notes",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["benchmark_level_used", "phase", "indication_or_therapeutic_area", "rare_disease_flag"]
    )


def build_report(df: pd.DataFrame, artifact: pd.DataFrame, *, min_n: int) -> dict[str, object]:
    target = df[df["is_completed_actual_enrollment_target"]].copy()
    very_large_threshold = float(target["enrollment"].quantile(0.99)) if not target.empty else 0.0
    sparse_rows = artifact[artifact["benchmark_n"].lt(min_n)]

    def counts(column: str, limit: int | None = None) -> dict[str, int]:
        values = df[column].value_counts(dropna=False)
        if limit is not None:
            values = values.head(limit)
        return {
            str(k): int(v)
            for k, v in values.sort_index().items()
        }

    sanity = {}
    if not artifact.empty:
        for column in ["benchmark_p25", "benchmark_p50", "benchmark_p75", "benchmark_p90"]:
            sanity[column] = {
                "min": float(artifact[column].min()),
                "median": float(artifact[column].median()),
                "max": float(artifact[column].max()),
            }

    return {
        "records_loaded": int(len(df)),
        "min_n_threshold": int(min_n),
        "enrollment_field_availability_by_enrollment_type": {
            str(k): {
                "rows": int(len(g)),
                "positive_enrollment": int(g["enrollment"].gt(0).sum()),
                "missing_enrollment": int(g["enrollment"].isna().sum()),
                "zero_or_negative_enrollment": int(g["enrollment"].le(0).fillna(False).sum()),
            }
            for k, g in df.groupby("enrollment_type", dropna=False)
        },
        "completed_actual_enrollment_targets": int(target.shape[0]),
        "counts_by_phase": counts("phase"),
        "counts_by_therapeutic_area": counts("therapeutic_area"),
        "counts_by_indication": counts("gbd_cause_id_3_ml"),
        "counts_by_rare_disease_flag": counts("is_rare_disease_ml"),
        "benchmark_rows_by_level": {
            str(k): int(v) for k, v in artifact["benchmark_level_used"].value_counts().sort_index().items()
        },
        "sparse_cohort_count": int(sparse_rows.shape[0]),
        "low_confidence_benchmark_count": int(artifact["low_confidence_flag"].sum()),
        "outlier_summary": {
            "very_large_threshold_p99": round(very_large_threshold, 2),
            "very_large_completed_actual_count": int(target["enrollment"].gt(very_large_threshold).sum()),
            "max_completed_actual_enrollment": float(target["enrollment"].max()) if not target.empty else None,
        },
        "percentile_sanity_summary": sanity,
    }


def write_outputs(artifact: pd.DataFrame, report: dict[str, object], output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic planned-enrollment benchmark artifact.")
    parser.add_argument("--input", type=Path, default=DATA_CLINPRED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--min-n", type=int, default=50)
    args = parser.parse_args()

    df = add_source_flags(load_source(args.input))
    source_version = _source_data_version(args.input)
    artifact = build_benchmarks(df, min_n=args.min_n, source_data_version=source_version)
    report = build_report(df, artifact, min_n=args.min_n)
    write_outputs(artifact, report, args.output, args.report)

    print(f"Loaded {report['records_loaded']} records")
    print(f"Completed ACTUAL benchmark targets: {report['completed_actual_enrollment_targets']}")
    print(f"Wrote benchmark artifact: {args.output} ({len(artifact)} rows)")
    print(f"Wrote calibration report: {args.report}")
    print("Benchmark rows by level:")
    for level, count in report["benchmark_rows_by_level"].items():
        print(f"  {level}: {count}")
    print(f"Low-confidence benchmark rows: {report['low_confidence_benchmark_count']}")


if __name__ == "__main__":
    main()
