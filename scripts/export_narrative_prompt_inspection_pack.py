#!/usr/bin/env python
"""Export a structured Scenario Review prompt-inspection pack.

This helper does not call an LLM provider. It exports representative prompt
cases for human review before the next live validation wave.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_storyline_review_docx import convert_markdown_to_docx  # noqa: E402
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet, build_review_packet_from_fixture  # noqa: E402
from src.narratives.prompt_builder import (  # noqa: E402
    build_provider_prompt,
    infer_prompt_mode,
    provider_response_contract,
)

DEFAULT_OUT_DIR = Path("reports/narrative_evals/prompt_inspection_phase3")
DEFAULT_CASES = (
    "baseline_hidden_review_v2",
    "score_improves_evidence_weakens_v2",
    "score_declines_design_improves_v2",
    "operational_only_ambitious_enrollment_v2",
    "material_text_only_endpoint_conflict_v2",
    "biomarker_population_mismatch_v2",
)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _text_dump(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")


def _fixture_map() -> dict[str, dict[str, Any]]:
    return {str(fixture.get("fixture_id")): fixture for fixture in get_contract_fixtures()}


def _case_packet(fixture_id: str, fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    try:
        return build_review_packet_from_fixture(fixtures[fixture_id])
    except KeyError as exc:
        available = "\n".join(f"  - {name}" for name in sorted(fixtures))
        raise SystemExit(f"Unknown fixture_id: {fixture_id}\n\nAvailable fixtures:\n{available}") from exc


def _synthetic_later_visible_packet(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixture = fixtures["score_improves_evidence_weakens_v2"]
    packet = fixture["input_packet"]
    baseline_trace = {
        "input_hash": "inspection-baseline-input-hash",
        "iteration_id": 0,
        "status": "reviewed",
        "validation_status": "valid",
        "changed_fields": [],
        "score_delta": 0,
        "central_tension": "Baseline central tension.",
        "validated_review": {
            "review_metadata": {"review_mode": "hidden_baseline", "visible": False},
            "main_tension": "Baseline main tension from dedicated field.",
            "completion_outlook_analysis": {
                "risk_pattern_summary": "Baseline score reflects an acceptable original design profile.",
            },
            "design_confidence_subcategories": {
                "phase_intent_alignment": {
                    "rating": "supportive",
                    "score_materiality": "low",
                    "rationale": "Phase and development intent are broadly aligned.",
                    "evidence_fields": ["phase_ml", "strategic_ambition_ml"],
                },
                "endpoint_evidence_strength": {
                    "rating": "supportive",
                    "score_materiality": "low",
                    "rationale": "Baseline endpoint and allocation preserve conventional rigor.",
                    "evidence_fields": ["endpoint_rigor_ml", "allocation_ml"],
                },
                "target_population_alignment": {
                    "rating": "balanced",
                    "score_materiality": "minimal",
                    "rationale": "Population choices are interpretable but not materially changed.",
                    "evidence_fields": ["patient_severity_ml", "line_of_therapy_ml"],
                },
                "operational_burden_balance": {
                    "rating": "balanced",
                    "score_materiality": "minimal",
                    "rationale": "Baseline execution burden appears proportionate.",
                    "evidence_fields": ["operational_assumptions.planned_enrollment"],
                },
            },
            "design_confidence_analysis": {
                "summary": "Baseline design comment.",
                "confidence_rationale": "Baseline central tension.",
                "supporting_evidence": [],
                "limiting_evidence": [],
            },
            "key_questions": {
                "medical_clinical_development_question": "What evidence standard matters most for the intended decision?",
                "strategic_development_question": "What broader development tension does the original scenario expose?",
            },
            "continuity": {"new_concerns": [], "storyline_update": "Baseline memory."},
        },
        "compact_storyline_memory": "Baseline memory.",
    }
    previous_trace = {
        **baseline_trace,
        "input_hash": "inspection-previous-visible-input-hash",
        "iteration_id": 1,
        "design_confidence": -2,
        "total_scenario_score": 66,
        "changed_fields": ["endpoint_rigor_ml"],
        "score_delta": 4,
        "central_tension": "Previous visible iteration tension.",
        "design_confidence_assessment": {
            "subcategories": {
                "phase_intent_alignment": {"points": 1, "raw_points": 1},
                "endpoint_evidence_strength": {"points": -2, "raw_points": -2},
                "target_population_alignment": {"points": 0, "raw_points": 0},
                "operational_burden_balance": {"points": 0, "raw_points": 0},
            },
        },
        "validated_review": {
            **baseline_trace["validated_review"],
            "main_tension": "Previous visible main tension.",
        },
        "compact_storyline_memory": "Previous visible memory.",
    }
    return build_review_packet(
        current_snapshot={
            "snapshot_id": "inspection-later-visible-current",
            "structured_features": packet.get("structured_features", {}),
            "display_values": packet.get("structured_feature_display_values", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": [],
            "changed_operational_assumptions": ["planned_enrollment"],
        },
        previous_snapshot={
            "snapshot_id": "inspection-previous-visible",
            "score": 68,
            "iteration_context": {"iteration_number": 1},
        },
        baseline_snapshot={"snapshot_id": "inspection-baseline"},
        baseline_review_trace=baseline_trace,
        previous_review_trace=previous_trace,
        compact_storyline_memory="Previous visible memory.",
    )


def _case_metrics(packet: dict[str, Any], prompt: str) -> dict[str, Any]:
    iteration = packet.get("iteration_context") or {}
    model = packet.get("model_interpretation") or {}
    continuity = iteration.get("design_confidence_continuity") or {}
    return {
        "prompt_mode": infer_prompt_mode(packet),
        "prompt_characters": len(prompt),
        "approx_prompt_tokens": round(len(prompt) / 4),
        "input_hash": packet.get("input_hash"),
        "completion_score": model.get("completion_score"),
        "score_delta": model.get("score_delta"),
        "changed_fields": iteration.get("changed_fields") or [],
        "field_change_count": len(iteration.get("field_changes") or []),
        "reference_packs": [
            pack.get("pack_id")
            for pack in packet.get("reference_packs") or []
            if isinstance(pack, dict) and pack.get("pack_id")
        ],
        "continuity_available": continuity.get("available"),
        "continuity_subcategories": sorted((continuity.get("subcategories") or {}).keys()),
    }


def _continuity_summary(packet: dict[str, Any]) -> list[str]:
    continuity = ((packet.get("iteration_context") or {}).get("design_confidence_continuity") or {})
    if not continuity.get("available"):
        return ["- Continuity anchors: unavailable for this case."]
    lines = [
        f"- Source iteration: `{continuity.get('source_iteration_id')}`",
        f"- Changed fields: `{', '.join(continuity.get('changed_fields') or []) or 'none'}`",
    ]
    for name, subcategory in sorted((continuity.get("subcategories") or {}).items()):
        lines.append(
            "- "
            f"`{name}`: previous `{subcategory.get('previous_rating')}` / "
            f"`{subcategory.get('previous_points')}` pts; relevant current changes "
            f"`{', '.join(subcategory.get('current_relevant_changed_fields') or []) or 'none'}`"
        )
    return lines


def _inspection_markdown(case_summaries: list[dict[str, Any]], out_dir: Path) -> str:
    lines: list[str] = [
        "# Scenario Review Prompt Inspection Pack",
        "",
        "## Purpose",
        "",
        "This pack is for Point 3 / Phase 3 of the active prompt implementation plan: prompt simplification. It checks whether the simplified Scenario Review prompt, response contract, packet evidence, and `design_confidence_continuity` anchors are understandable before live validation.",
        "",
        "No LLM provider was called. Fixture case folders contain the exact prompt and packet that would be sent. The synthetic later-visible continuity case is schema-only and should not be used to judge clinical realism, score behavior, or final narrative quality.",
        "",
        "## Inspection Checklist",
        "",
        "- Prompt instructions are concise enough to follow and not overloaded by old style-control language.",
        "- Visible output target is clear: Completion Outlook Analysis, Design Confidence Analysis, Main Tension, and two Key Questions.",
        "- Completion Outlook is separated from Design Confidence and uses score-input / score-pattern wording.",
        "- Planned enrollment, planned sites, and Planned Total Timeline stay out of Completion Outlook movement explanations.",
        "- `design_confidence_continuity` is absent or unavailable for hidden baseline / first visible cases.",
        "- `design_confidence_continuity` is visible and readable for the schema-only synthetic later-visible case.",
        "- Continuity anchors preserve previous subcategory direction unless current relevant changed fields justify movement.",
        "- Response contract still leaves all point calculations to the application.",
        "- The packet contains enough evidence for high-quality narratives without relying on one-shot examples.",
        "",
        "## Files",
        "",
        f"- Output folder: `{out_dir}`",
        "- Top-level `prompt_inspection_pack.md`: this review guide.",
        "- Top-level `prompt_inspection_pack.docx`: Word version of this guide.",
        "- Each `case_*` folder contains `01_prompt.txt`, `02_packet.json`, `03_response_contract.json`, and `04_metrics.json`.",
        "",
        "## Case Summary",
        "",
        "| Case | Mode | Prompt chars | Approx tokens | Score delta | Continuity | Changed fields |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for summary in case_summaries:
        metrics = summary["metrics"]
        changed = ", ".join(metrics.get("changed_fields") or []) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["case_id"],
                    str(metrics.get("prompt_mode")),
                    str(metrics.get("prompt_characters")),
                    str(metrics.get("approx_prompt_tokens")),
                    str(metrics.get("score_delta")),
                    str(metrics.get("continuity_available")),
                    changed.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")

    for summary in case_summaries:
        metrics = summary["metrics"]
        lines.extend([
            f"## {summary['case_id']}",
            "",
            f"- Fixture/source: `{summary['source']}`",
            f"- Folder: `{summary['folder']}`",
            f"- Prompt mode: `{metrics.get('prompt_mode')}`",
            f"- Input hash: `{metrics.get('input_hash')}`",
            f"- Completion Score: `{metrics.get('completion_score')}`",
            f"- Score delta: `{metrics.get('score_delta')}`",
            f"- Prompt characters: `{metrics.get('prompt_characters')}`",
            f"- Approx prompt tokens: `{metrics.get('approx_prompt_tokens')}`",
            f"- Reference packs: `{', '.join(metrics.get('reference_packs') or []) or 'none'}`",
            "",
            "Continuity Review:",
        ])
        lines.extend(summary["continuity_lines"])
        lines.extend([
            "",
            "Inspection Questions:",
            "",
            "- Is the prompt mode correct for this case?",
            "- Are changed fields and field changes enough to understand the latest scenario?",
            "- Is the continuity object either correctly unavailable or readable and useful?",
            "- Would a reviewer know which evidence fields can support each Design Confidence subcategory?",
            "- Is anything too verbose or distracting before adding one-shot examples?",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _manual_review_markdown(case_summaries: list[dict[str, Any]], out_dir: Path) -> str:
    lines: list[str] = [
        "# Scenario Review Manual Triage",
        "",
        "## Review Goal",
        "",
        "Use this compact file for the human Phase 3 prompt-simplification check. Open the full prompt files only when this summary points to a concern.",
        "",
        "Phase 3 here means point 3 of the active prompt implementation plan: prompt simplification.",
        "",
        "## Decision Checklist",
        "",
        "- The prompt still clearly separates Completion Outlook from Design Confidence.",
        "- Planning assumptions remain Design Confidence context, not Completion Outlook drivers.",
        "- The visible target is still four concise sections: Completion Outlook, Design Confidence, Main Tension, and two Key Questions.",
        "- Continuity anchors are understandable for later-visible reviews.",
        "- The prompt feels concise enough to test live without adding a one-shot example yet.",
        "- The schema-only synthetic case is not used to judge clinical realism or narrative quality.",
        "",
        "## Triage Table",
        "",
        "| Case | Mode | Approx tokens | Changed fields | Continuity | Manual action |",
        "|---|---|---:|---|---|---|",
    ]
    for summary in case_summaries:
        metrics = summary["metrics"]
        changed = ", ".join(metrics.get("changed_fields") or []) or "none"
        action = "Check mode and boundary wording"
        if metrics.get("continuity_available"):
            action = "Check continuity anchors only"
        if "synthetic" in summary["case_id"]:
            action = "Schema-only: do not judge narrative realism"
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["case_id"],
                    str(metrics.get("prompt_mode")),
                    str(metrics.get("approx_prompt_tokens")),
                    changed.replace("|", "\\|"),
                    str(metrics.get("continuity_available")),
                    action,
                ]
            )
            + " |"
        )
    lines.extend([
        "",
        "## Case Notes",
        "",
    ])
    for summary in case_summaries:
        metrics = summary["metrics"]
        lines.extend([
            f"### {summary['case_id']}",
            "",
            f"- Source: `{summary['source']}`",
            f"- Full files: `{summary['folder']}`",
            f"- Prompt mode: `{metrics.get('prompt_mode')}`",
            f"- Approx prompt tokens: `{metrics.get('approx_prompt_tokens')}`",
            f"- Completion Score / delta: `{metrics.get('completion_score')}` / `{metrics.get('score_delta')}`",
            f"- Changed fields: `{', '.join(metrics.get('changed_fields') or []) or 'none'}`",
            f"- Reference packs: `{', '.join(metrics.get('reference_packs') or []) or 'none'}`",
            "",
            "Continuity:",
        ])
        lines.extend(summary["continuity_lines"])
        if "synthetic" in summary["case_id"]:
            lines.extend([
                "",
                "Manual caution: this is a schema-only continuity packet. It is useful for checking whether the prompt exposes prior Design Confidence anchors clearly, but it is not a realistic trial storyline.",
            ])
        lines.append("")
    lines.extend([
        "## Stop Rule",
        "",
        "If this compact review finds no blocker, proceed to a 1-2 trial live validation. Do not read every full prompt manually before the small live run.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def export_inspection_pack(out_dir: Path, fixture_ids: tuple[str, ...]) -> tuple[Path, Path, Path, Path]:
    fixtures = _fixture_map()
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = provider_response_contract()
    cases: list[tuple[str, str, dict[str, Any]]] = []
    for fixture_id in fixture_ids:
        cases.append((fixture_id, fixture_id, _case_packet(fixture_id, fixtures)))
    cases.append((
        "synthetic_later_visible_continuity",
        "synthetic later-visible packet with previous visible Design Confidence context",
        _synthetic_later_visible_packet(fixtures),
    ))

    summaries: list[dict[str, Any]] = []
    for index, (case_id, source, packet) in enumerate(cases, start=1):
        prompt = build_provider_prompt(packet)
        folder = out_dir / f"case_{index:02d}_{_slug(case_id)}"
        folder.mkdir(parents=True, exist_ok=True)
        metrics = _case_metrics(packet, prompt)
        _text_dump(folder / "01_prompt.txt", prompt)
        _json_dump(folder / "02_packet.json", packet)
        _json_dump(folder / "03_response_contract.json", contract)
        _json_dump(folder / "04_metrics.json", metrics)
        summaries.append({
            "case_id": case_id,
            "source": source,
            "folder": str(folder),
            "metrics": metrics,
            "continuity_lines": _continuity_summary(packet),
        })

    markdown_path = out_dir / "prompt_inspection_pack.md"
    docx_path = out_dir / "prompt_inspection_pack.docx"
    manual_markdown_path = out_dir / "manual_triage_summary.md"
    manual_docx_path = out_dir / "manual_triage_summary.docx"
    _text_dump(markdown_path, _inspection_markdown(summaries, out_dir))
    convert_markdown_to_docx(markdown_path, docx_path)
    _text_dump(manual_markdown_path, _manual_review_markdown(summaries, out_dir))
    convert_markdown_to_docx(manual_markdown_path, manual_docx_path)
    _json_dump(out_dir / "prompt_inspection_index.json", summaries)
    return markdown_path, docx_path, manual_markdown_path, manual_docx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Fixture IDs to include before the synthetic later-visible continuity case.",
    )
    args = parser.parse_args()
    markdown_path, docx_path, manual_markdown_path, manual_docx_path = export_inspection_pack(args.out, tuple(args.fixtures))
    print(f"Wrote {markdown_path}")
    print(f"Wrote {docx_path}")
    print(f"Wrote {manual_markdown_path}")
    print(f"Wrote {manual_docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
