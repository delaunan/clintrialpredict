#!/usr/bin/env python
"""Compare narrative eval reports across generation-control settings and duplicate runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "narrative_evals"


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_labeled_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--report must use label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--report label must not be empty")
    return label, Path(path).expanduser()


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _iter_iterations(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in report.get("trials") or []:
        nct_id = ((trial.get("trial") or {}).get("nct_id") or "").strip()
        for iteration in trial.get("iterations") or []:
            rows.append({
                "key": (nct_id, iteration.get("step_id")),
                "trial": trial.get("trial") or {},
                "iteration": iteration,
            })
    return rows


def _finding_counts(report: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in _iter_iterations(report):
        for finding in row["iteration"].get("findings") or []:
            counts[f"{finding.get('severity')}:{finding.get('check')}"] += 1
    return dict(sorted(counts.items()))


def _design_score_values(report: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for row in _iter_iterations(report):
        value = row["iteration"].get("design_confidence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _report_summary(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    design_values = _design_score_values(report)
    provider_config = report.get("provider_config") or {}
    return {
        "label": label,
        "path": str(path),
        "run_id": report.get("run_id"),
        "provider": report.get("provider"),
        "model": (provider_config.get("models") or {}).get(str(report.get("provider") or "")),
        "temperature": provider_config.get("temperature"),
        "gemini_thinking_level": provider_config.get("gemini_thinking_level"),
        "seed": provider_config.get("seed"),
        "summary": report.get("summary") or {},
        "finding_counts": _finding_counts(report),
        "design_confidence_min": min(design_values) if design_values else None,
        "design_confidence_max": max(design_values) if design_values else None,
        "design_confidence_mean": round(sum(design_values) / len(design_values), 3) if design_values else None,
    }


def _narrative_signature(iteration: dict[str, Any]) -> dict[str, str]:
    narrative = iteration.get("narrative") or {}
    return {
        key: str(narrative.get(key) or "").strip()
        for key in (
            "consistency_note",
            "completion",
            "design",
            "medical_question",
            "operations_question",
            "strategic_question",
        )
    }


def _subcategory_signature(iteration: dict[str, Any]) -> dict[str, Any]:
    subcategories = iteration.get("design_confidence_subcategories") or {}
    return {
        key: {
            "rating": value.get("rating"),
            "score_materiality": value.get("score_materiality"),
            "points": value.get("points"),
        }
        for key, value in sorted(subcategories.items())
    }


def _numeric_delta(left: object, right: object) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(right) - float(left), 3)


def _subcategory_drift(left_iteration: dict[str, Any], right_iteration: dict[str, Any]) -> dict[str, Any]:
    left_subcategories = left_iteration.get("design_confidence_subcategories") or {}
    right_subcategories = right_iteration.get("design_confidence_subcategories") or {}
    drift: dict[str, Any] = {}
    for key in sorted(set(left_subcategories) | set(right_subcategories)):
        left_value = left_subcategories.get(key) or {}
        right_value = right_subcategories.get(key) or {}
        points_delta = _numeric_delta(left_value.get("points"), right_value.get("points"))
        changed = (
            left_value.get("rating") != right_value.get("rating")
            or left_value.get("score_materiality") != right_value.get("score_materiality")
            or points_delta not in (None, 0.0)
        )
        if changed:
            drift[key] = {
                "left_points": left_value.get("points"),
                "right_points": right_value.get("points"),
                "points_delta": points_delta,
                "left_rating": left_value.get("rating"),
                "right_rating": right_value.get("rating"),
                "left_score_materiality": left_value.get("score_materiality"),
                "right_score_materiality": right_value.get("score_materiality"),
            }
    return drift


def _compare_duplicate_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["key"]: row["iteration"] for row in _iter_iterations(left)}
    right_rows = {row["key"]: row["iteration"] for row in _iter_iterations(right)}
    keys = sorted(set(left_rows) | set(right_rows))
    comparisons = []
    totals = Counter()
    for key in keys:
        left_iteration = left_rows.get(key)
        right_iteration = right_rows.get(key)
        if left_iteration is None or right_iteration is None:
            comparisons.append({"key": key, "status": "missing_iteration"})
            totals["missing_iteration"] += 1
            continue

        left_narrative = _narrative_signature(left_iteration)
        right_narrative = _narrative_signature(right_iteration)
        exact_narrative_matches = {
            name: left_narrative[name] == right_narrative[name]
            for name in left_narrative
        }
        normalized_narrative_matches = {
            name: _normalize_text(left_narrative[name]) == _normalize_text(right_narrative[name])
            for name in left_narrative
        }
        score_match = {
            "status": left_iteration.get("status") == right_iteration.get("status"),
            "validation_status": left_iteration.get("validation_status") == right_iteration.get("validation_status"),
            "design_confidence": left_iteration.get("design_confidence") == right_iteration.get("design_confidence"),
            "total_scenario_score": left_iteration.get("total_scenario_score") == right_iteration.get("total_scenario_score"),
            "subcategories": _subcategory_signature(left_iteration) == _subcategory_signature(right_iteration),
        }
        design_confidence_delta = _numeric_delta(
            left_iteration.get("design_confidence"),
            right_iteration.get("design_confidence"),
        )
        total_scenario_score_delta = _numeric_delta(
            left_iteration.get("total_scenario_score"),
            right_iteration.get("total_scenario_score"),
        )
        subcategory_drift = _subcategory_drift(left_iteration, right_iteration)
        all_exact = all(score_match.values()) and all(exact_narrative_matches.values())
        all_normalized = all(score_match.values()) and all(normalized_narrative_matches.values())
        totals["iterations_compared"] += 1
        totals["exact_iteration_matches"] += int(all_exact)
        totals["normalized_iteration_matches"] += int(all_normalized)
        totals["score_matches"] += int(all(score_match.values()))
        if design_confidence_delta not in (None, 0.0):
            totals["design_confidence_score_drifts"] += 1
        if subcategory_drift:
            totals["subcategory_drifts"] += 1
        comparisons.append({
            "key": {"nct_id": key[0], "step_id": key[1]},
            "all_exact_match": all_exact,
            "all_normalized_match": all_normalized,
            "score_match": score_match,
            "design_confidence_delta": design_confidence_delta,
            "total_scenario_score_delta": total_scenario_score_delta,
            "subcategory_drift": subcategory_drift,
            "exact_narrative_matches": exact_narrative_matches,
            "normalized_narrative_matches": normalized_narrative_matches,
        })
    return {
        "summary": dict(totals),
        "comparisons": comparisons,
    }


def _write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [f"# Narrative Generation-Control Comparison: {comparison['comparison_id']}", ""]
    lines.append(f"- Generated: `{comparison['generated_at']}`")
    lines.append("")
    lines.append("## Reports")
    for report in comparison.get("reports") or []:
        lines.append("")
        lines.append(f"### {report['label']}")
        lines.append(f"- Path: `{report['path']}`")
        lines.append(f"- Run ID: `{report['run_id']}`")
        lines.append(f"- Provider/model: `{report['provider']}` / `{report.get('model')}`")
        lines.append(f"- Temperature: `{report.get('temperature')}`")
        lines.append(f"- Gemini thinking level: `{report.get('gemini_thinking_level')}`")
        lines.append(f"- Seed: `{report.get('seed')}`")
        for key, value in (report.get("summary") or {}).items():
            lines.append(f"- {key}: `{value}`")
        if report.get("finding_counts"):
            lines.append("- Finding counts:")
            for key, value in report["finding_counts"].items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append(
            "- Design Confidence range/mean: "
            f"`{report.get('design_confidence_min')}` / `{report.get('design_confidence_max')}` / "
            f"`{report.get('design_confidence_mean')}`"
        )

    duplicate = comparison.get("duplicate_reproducibility")
    if duplicate:
        lines.append("")
        lines.append("## Duplicate Reproducibility")
        summary = duplicate.get("summary") or {}
        for key, value in summary.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
        lines.append("Iteration-level drift:")
        for item in duplicate.get("comparisons") or []:
            key = item.get("key") or {}
            lines.append(
                f"- `{key.get('nct_id')}` / `{key.get('step_id')}`: "
                f"exact `{item.get('all_exact_match')}`, normalized `{item.get('all_normalized_match')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=_parse_labeled_report, default=[], help="Temperature report as label=path.")
    parser.add_argument("--repro-a", type=Path, default=None, help="First duplicate-run report for reproducibility comparison.")
    parser.add_argument("--repro-b", type=Path, default=None, help="Second duplicate-run report for reproducibility comparison.")
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    comparison_id = args.comparison_id or datetime.now(timezone.utc).strftime("generation_control_comparison_%Y%m%d_%H%M%S")
    report_summaries = []
    for label, path in args.report:
        report_summaries.append(_report_summary(label, path, _load_report(path)))

    duplicate = None
    if args.repro_a or args.repro_b:
        if not (args.repro_a and args.repro_b):
            parser.error("--repro-a and --repro-b must be provided together")
        duplicate = _compare_duplicate_reports(_load_report(args.repro_a), _load_report(args.repro_b))
        duplicate["left_path"] = str(args.repro_a)
        duplicate["right_path"] = str(args.repro_b)

    output = {
        "comparison_id": comparison_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": report_summaries,
        "duplicate_reproducibility": duplicate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{comparison_id}.json"
    md_path = args.out_dir / f"{comparison_id}.md"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_markdown(md_path, output)
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
