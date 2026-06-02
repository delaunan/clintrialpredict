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
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "data" / "site_benchmarks_v1.csv"
REPORT_PATH = PROJECT_ROOT / "frontend" / "data" / "site_benchmarks_v1_report.json"

BENCHMARK_VERSION = "site_benchmarks_v1"
OUTLIER_POLICY = "positive completed registry-derived facility-count proxy; percentiles used without winsorization"
CALIBRATION_NOTES = "Deterministic S2 planned-sites benchmark based on number_of_facilities proxy; no site-count ML model."

REQUIRED_COLUMNS = [
    "nct_id",
    "number_of_facilities",
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


def _available_usecols(path: Path, requested_columns: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    return [column for column in requested_columns if column in header]


def _benchmark_key(level: str, row: pd.Series) -> str:
    if level == "phase_indication_rare":
        return f"{level}|phase={row['phase']}|indication={_as_int(row['gbd_cause_id_3_ml'])}|rare={_as_int(row['rare_disease_flag'])}"
    if level == "phase_ta_rare":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}|rare={_as_int(row['rare_disease_flag'])}"
    if level == "phase_ta":
        return f"{level}|phase={row['phase']}|ta={row['therapeutic_area']}"
    return f"{level}|phase={row['phase']}"


def load_source(path: Path = DATA_CLINPRED_PATH) -> pd.DataFrame:
    usecols = _available_usecols(path, REQUIRED_COLUMNS)
    missing_required = {"nct_id", "number_of_facilities", "overall_status", "phase", "therapeutic_area", "gbd_cause_id_3_ml", "is_rare_disease_ml"}.difference(usecols)
    if missing_required:
        raise ValueError(f"Missing required source columns: {sorted(missing_required)}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["number_of_facilities"] = pd.to_numeric(df["number_of_facilities"], errors="coerce")
    df["overall_status"] = df["overall_status"].map(_clean_text)
    df["phase"] = df["phase"].map(_clean_text)
    df["therapeutic_area"] = df["therapeutic_area"].map(_clean_text)
    if "phase_ml" in df.columns:
        df["phase_ml"] = pd.to_numeric(df["phase_ml"], errors="coerce")
    if "therapeutic_area_ml" in df.columns:
        df["therapeutic_area_ml"] = pd.to_numeric(df["therapeutic_area_ml"], errors="coerce")
    df["gbd_cause_id_3_ml"] = pd.to_numeric(df["gbd_cause_id_3_ml"], errors="coerce").fillna(0).astype(int)
    df["is_rare_disease_ml"] = pd.to_numeric(df["is_rare_disease_ml"], errors="coerce").fillna(0).astype(int)
    if "is_rare_disease" in df.columns:
        df["is_rare_disease"] = pd.to_numeric(df["is_rare_disease"], errors="coerce")
    return df


def add_source_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    positive_site_count = flagged["number_of_facilities"].gt(0)
    completed = flagged["overall_status"].eq("COMPLETED")
    ongoing = flagged["overall_status"].isin(
        {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"}
    )

    flagged["is_completed_positive_site_count_target"] = completed & positive_site_count
    flagged["is_current_registry_facility_count_proxy"] = ongoing & positive_site_count
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
    target = df[df["is_completed_positive_site_count_target"]].copy()

    rows: list[dict[str, object]] = []
    for level in LEVELS:
        for keys, group in target.groupby(level["group_cols"], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_data = dict(zip(level["group_cols"], keys, strict=True))
            summary = _percentiles(group["number_of_facilities"])
            row = {
                "benchmark_version": BENCHMARK_VERSION,
                "source_data_version": source_data_version,
                "benchmark_level_used": level["name"],
                "phase": key_data.get("phase", "UNKNOWN"),
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
        ["benchmark_level_used", "phase", "gbd_cause_id_3_ml", "therapeutic_area", "rare_disease_flag"]
    )


def _lookup_coverage(df: pd.DataFrame, artifact: pd.DataFrame) -> tuple[dict[str, int], int, int]:
    confident: dict[tuple[object, ...], dict[str, object]] = {}
    low_confidence: dict[tuple[object, ...], dict[str, object]] = {}
    level_cols = {level["name"]: level["group_cols"] for level in LEVELS}
    for _, row in artifact.iterrows():
        level_name = row["benchmark_level_used"]
        cols = level_cols[level_name]
        values = []
        for column in cols:
            source_column = "rare_disease_flag" if column == "is_rare_disease_ml" else column
            values.append(row[source_column])
        key = tuple([level_name] + values)
        target = low_confidence if bool(row["low_confidence_flag"]) else confident
        target[key] = row.to_dict()

    matched_levels: list[str] = []
    low_confidence_matches = 0
    for _, row in df.iterrows():
        first_low_confidence: str | None = None
        matched_level = "not_available"
        for level in LEVELS:
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


def _numeric_quality(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce")
    return {
        "total_rows": int(len(clean)),
        "present": int(clean.notna().sum()),
        "missing": int(clean.isna().sum()),
        "zero": int(clean.eq(0).sum()),
        "negative": int(clean.lt(0).sum()),
        "positive": int(clean.gt(0).sum()),
        "min": float(clean.min()) if clean.notna().any() else 0.0,
        "p25": float(clean.quantile(0.25)) if clean.notna().any() else 0.0,
        "median": float(clean.quantile(0.5)) if clean.notna().any() else 0.0,
        "p75": float(clean.quantile(0.75)) if clean.notna().any() else 0.0,
        "p90": float(clean.quantile(0.9)) if clean.notna().any() else 0.0,
        "p95": float(clean.quantile(0.95)) if clean.notna().any() else 0.0,
        "p99": float(clean.quantile(0.99)) if clean.notna().any() else 0.0,
        "max": float(clean.max()) if clean.notna().any() else 0.0,
    }


def build_report(
    df: pd.DataFrame,
    artifact: pd.DataFrame,
    *,
    min_n: int,
    created_at: str,
    source_data_version: str,
) -> dict[str, object]:
    target = df[df["is_completed_positive_site_count_target"]].copy()
    coverage_counts, coverage_not_available, coverage_low_confidence = _lookup_coverage(df, artifact)

    def counts(column: str, limit: int | None = None) -> dict[str, int]:
        if column not in df.columns:
            return {}
        values = df[column].value_counts(dropna=False)
        if limit is not None:
            values = values.head(limit)
        return {str(k): int(v) for k, v in values.sort_index().items()}

    return {
        "source_records_loaded": int(len(df)),
        "completed_positive_site_count_targets": int(target.shape[0]),
        "artifact_rows": int(artifact.shape[0]),
        "minimum_confident_cohort_threshold": int(min_n),
        "low_confidence_benchmark_rows": int(artifact["low_confidence_flag"].sum()),
        "duplicate_benchmark_keys": int(artifact["benchmark_key"].duplicated().sum()),
        "benchmark_rows_by_level": {
            str(k): int(v) for k, v in artifact["benchmark_level_used"].value_counts().sort_index().items()
        },
        "counts_by_phase": counts("phase"),
        "counts_by_therapeutic_area": counts("therapeutic_area"),
        "counts_by_indication": counts("gbd_cause_id_3_ml"),
        "counts_by_rare_flag": counts("is_rare_disease_ml"),
        "phase_only_fallback_rows": int(artifact["benchmark_level_used"].eq("phase_only").sum()),
        "coverage_qa_match_counts": coverage_counts,
        "coverage_qa_not_available": coverage_not_available,
        "coverage_qa_low_confidence_matches": coverage_low_confidence,
        "quality_stats": _numeric_quality(df["number_of_facilities"]),
        "outlier_policy": OUTLIER_POLICY,
        "calibration_notes": CALIBRATION_NOTES,
        "created_at": created_at,
        "source_data_version": source_data_version,
    }


def write_outputs(artifact: pd.DataFrame, report: dict[str, object], output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic planned-sites benchmark artifact.")
    parser.add_argument("--input", type=Path, default=DATA_CLINPRED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--min-n", type=int, default=50)
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = add_source_flags(load_source(args.input))
    source_version = _source_data_version(args.input)
    artifact = build_benchmarks(df, min_n=args.min_n, created_at=created_at, source_data_version=source_version)
    report = build_report(df, artifact, min_n=args.min_n, created_at=created_at, source_data_version=source_version)
    write_outputs(artifact, report, args.output, args.report)

    print(f"Loaded {report['source_records_loaded']} records")
    print(f"Completed positive registry facility-count proxy targets: {report['completed_positive_site_count_targets']}")
    print(f"Wrote benchmark artifact: {args.output} ({len(artifact)} rows)")
    print(f"Wrote benchmark report: {args.report}")
    print("Benchmark rows by level:")
    for level, count in report["benchmark_rows_by_level"].items():
        print(f"  {level}: {count}")
    print(f"Low-confidence benchmark rows: {report['low_confidence_benchmark_rows']}")
    print(f"Duplicate benchmark keys: {report['duplicate_benchmark_keys']}")
    print(f"Coverage QA not available: {report['coverage_qa_not_available']}")


if __name__ == "__main__":
    main()
