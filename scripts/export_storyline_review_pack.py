#!/usr/bin/env python
"""Export a storyline eval report into a human-review Markdown pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("reports/narrative_evals/final_validation_storyline_candidates_12_1.json")
DEFAULT_OUT = Path("reports/narrative_evals/final_validation_storyline_review_pack.md")
LEGACY_EXPORT_DISABLED_MESSAGE = (
    "scripts/export_storyline_review_pack.py is disabled for the Strategic Review migration. "
    "It formats superseded Design Confidence / Total Scenario Score eval reports. "
    "Rebuild the export around Strategic Review and Trial Score before use."
)

SUBCATEGORY_ORDER = (
    "phase_intent_alignment",
    "endpoint_evidence_strength",
    "target_population_alignment",
    "operational_burden_balance",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _bullet_lines(items: list[Any]) -> list[str]:
    if not items:
        return ["- n/a"]
    return [f"- {_text(item)}" for item in items]


def _findings(item: dict[str, Any]) -> list[str]:
    findings = item.get("findings") or []
    if not findings:
        return ["- None"]
    return [
        f"- `{_fmt(finding.get('severity'))}` `{_fmt(finding.get('check'))}`: {_fmt(finding.get('detail'))}"
        for finding in findings
    ]


def _narrative(item: dict[str, Any], key: str) -> str:
    return _fmt((item.get("narrative") or {}).get(key))


def _subcategory_rows(item: dict[str, Any]) -> list[str]:
    subcategories = item.get("design_confidence_subcategories") or {}
    rows = [
        "| Subcategory | Rating | Points | Materiality | Short rationale | Rationale | Evidence fields |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for key in SUBCATEGORY_ORDER:
        value = subcategories.get(key) or {}
        evidence = ", ".join(_text(field) for field in value.get("evidence_fields") or [])
        rows.append(
            "| "
            + " | ".join(
                [
                    key,
                    _fmt(value.get("rating")),
                    _fmt(value.get("points")),
                    _fmt(value.get("score_materiality")),
                    _fmt(value.get("short_rationale")).replace("|", "\\|"),
                    _fmt(value.get("rationale")).replace("|", "\\|"),
                    _fmt(evidence).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return rows


def _score_line(item: dict[str, Any]) -> str:
    return (
        f"Completion Outlook: `{_fmt(item.get('completion_score'))}` "
        f"(delta `{_fmt(item.get('score_delta'))}`), "
        f"Design Confidence: `{_fmt(item.get('design_confidence'))}`, "
        f"Total Scenario Score: `{_fmt(item.get('total_scenario_score'))}`"
    )


def _item_block(item: dict[str, Any], *, baseline: bool = False) -> list[str]:
    title = "Hidden Baseline Narratives And Scores" if baseline else f"{_fmt(item.get('step_id'))}: {_fmt(item.get('title'))}"
    lines = [f"### {title}", ""]
    lines.append(_score_line(item))
    lines.append("")
    if not baseline:
        lines.append("Changed fields:")
        lines.extend(_bullet_lines(item.get("changes") or []))
        lines.append("")
    consistency_note = _narrative(item, "consistency_note")
    if consistency_note != "n/a":
        lines.extend(["Consistency note:", "", consistency_note, ""])
    lines.extend(["Completion Outlook narrative:", "", _narrative(item, "completion"), ""])
    lines.extend(["Design Confidence narrative:", "", _narrative(item, "design"), ""])
    lines.append("Design Confidence subcategory ratings and rationales:")
    lines.extend(_subcategory_rows(item))
    lines.append("")
    lines.extend(
        [
            "Questions:",
            f"- Medical / clinical development: {_narrative(item, 'medical_question')}",
            f"- Strategic development: {_narrative(item, 'strategic_question')}",
            "",
            "Findings:",
        ]
    )
    lines.extend(_findings(item))
    lines.append("")
    return lines


def _trial_block(trial_entry: dict[str, Any], index: int) -> list[str]:
    trial = trial_entry.get("trial") or {}
    title = _fmt(trial.get("trial_label"))
    nct = _fmt(trial.get("nct_id"))
    sponsor = _fmt(trial.get("lead_sponsor_canonical"))
    ta = _fmt(trial.get("therapeutic_area"))
    lines = [
        f"## {index}. {nct} - {title}",
        "",
        f"- Sponsor: `{sponsor}`",
        f"- Therapeutic Area: `{ta}`",
        f"- Baseline Completion Score: `{_fmt(trial.get('baseline_completion_score'))}`",
        f"- Score band: `{_fmt(trial.get('score_band'))}`",
        f"- One-shot candidate flag: `{_fmt(trial.get('one_shot_candidate'))}`",
        "",
        "Reviewer classification: `[ ] presentation_ready` `[ ] good_after_light_edit` `[ ] useful_for_stress_test_only` `[ ] discard`",
        "",
        "Reviewer notes:",
        "",
        "> ",
        "",
    ]
    baseline = trial_entry.get("baseline_review")
    if isinstance(baseline, dict):
        lines.extend(_item_block(baseline, baseline=True))
    for item in trial_entry.get("iterations") or []:
        lines.extend(_item_block(item))
    return lines


def export_review_pack(report_path: Path, out_path: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {
        key: data.get(key)
        for key in ("visible_iterations", "reviewed_iterations", "failed_checks", "warning_checks")
    }
    plan_value = data.get("scenario_plan") or {}
    if isinstance(plan_value, dict):
        plan_name = plan_value.get("name")
        plan_description = plan_value.get("description")
    else:
        plan_name = str(plan_value)
        plan_description = data.get("scenario_description") or ""
    visible_iterations = _fmt(summary.get("visible_iterations"))
    trial_count = len(data.get("trials") or [])
    iteration_count_text = visible_iterations
    try:
        if trial_count:
            iteration_count_text = str(int(summary.get("visible_iterations") or 0) // trial_count)
    except (TypeError, ValueError, ZeroDivisionError):
        iteration_count_text = visible_iterations
    lines = [
        f"# Storyline Review Pack: {_fmt(data.get('run_id') or report_path.stem)}",
        "",
        f"- Source report: `{report_path}`",
        f"- Scenario plan: `{_fmt(plan_name)}` - {_fmt(plan_description)}",
        f"- Generated: `{_fmt(data.get('generated_at'))}`",
        f"- Visible iterations: `{_fmt(summary.get('visible_iterations'))}`",
        f"- Reviewed iterations: `{_fmt(summary.get('reviewed_iterations'))}`",
        f"- Failed checks: `{_fmt(summary.get('failed_checks'))}`",
        f"- Warning checks: `{_fmt(summary.get('warning_checks'))}`",
        "",
        "Review each full storyline before selecting one-shot examples. Each trial includes hidden baseline narratives and "
        f"scores first, followed by the {iteration_count_text} visible iterations. Focus on the credibility of the full arc, Completion Outlook / "
        "Design Confidence separation, subcategory scoring quality, narrative usefulness, and question quality.",
        "",
    ]
    for index, trial_entry in enumerate(data.get("trials") or [], start=1):
        lines.extend(_trial_block(trial_entry, index))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    print(LEGACY_EXPORT_DISABLED_MESSAGE)
    return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export_review_pack(args.report, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
