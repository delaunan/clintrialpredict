#!/usr/bin/env python
"""Run first-wave Scenario Review narrative evals without Streamlit UI clicks.

The suite builds deterministic narrative-review packets from real registry
trials, applies predefined scenario edits, optionally calls a live provider,
and writes Markdown/JSON reports for human prompt-quality review.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.packet_builder import (  # noqa: E402
    ACTIVE_OPERATIONAL_ASSUMPTION_KEYS,
    STRUCTURED_FEATURE_KEYS,
    TEXT_CONTEXT_KEYS,
    build_review_packet,
    build_review_packet_from_fixture,
    stable_packet_hash,
)
from src.narratives.provider import (  # noqa: E402
    PROVIDER_GEMINI,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    review_packet_with_provider,
    review_packet_with_provider_chain,
)
from src.narratives.provider_config import (  # noqa: E402
    load_narrative_provider_config,
    provider_config_cache_namespace,
)
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.prompt_builder import build_provider_prompt  # noqa: E402
from src.narratives.review_store import (  # noqa: E402
    compact_storyline_from_trace,
    store_review_trace,
)
from src.narratives.scoring import validate_and_score_review  # noqa: E402

REGISTRY_PATH = ROOT / "frontend" / "data" / "search_registry.csv"
TAXONOMY_PATH = ROOT / "models" / "taxonomy_01.json"
DEFAULT_REPORT_DIR = ROOT / "reports" / "narrative_evals"

PILLAR_COLUMNS = (
    "Therapeutic Context",
    "Scientific Challenge",
    "Patient Profile",
    "Execution Framework",
)

OPERATIONAL_ASSUMPTION_LABELS = {
    "planned_enrollment": "Planned Enrollment",
    "planned_sites": "Planned Sites",
    "planned_duration_months": "Duration (months)",
}

STRUCTURED_TEXT_CONFLICT_WARNING = (
    "Some scenario details are not fully aligned across free-text fields and selected fields. In this case the value in "
    "the selected fields drive the analysis, while the text is used as supporting context."
)

OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY = (
    "The Completion Outlook is essentially unchanged because planning assumptions such as enrollment, site count, "
    "and total duration do not directly feed the score. They still matter for whether the scenario feels operationally "
    "proportionate and executable, so they are reflected in Design Confidence instead."
)

FIRST_WAVE_TARGETS = (
    ("Oncology", "borderline"),
    ("Hematology", "borderline"),
    ("Neurology", "moderate"),
    ("Musculoskeletal", "moderate"),
    ("Infections", "high"),
    ("Cardiovascular", "low"),
    ("Immunology", "moderate"),
    ("Respiratory", "borderline"),
)

SCORE_BANDS = {
    "low": (0.0, 35.0),
    "borderline": (35.0, 55.0),
    "moderate": (55.0, 75.0),
    "high": (75.0, 100.0),
}


@dataclass(frozen=True)
class ScenarioStep:
    step_id: str
    title: str
    completion_delta: float
    pillar_for_delta: str
    structured_edits: dict[str, Any] = field(default_factory=dict)
    text_edits: dict[str, str] = field(default_factory=dict)
    operational_multipliers: dict[str, float] = field(default_factory=dict)
    operational_additions: dict[str, float] = field(default_factory=dict)
    expectations: dict[str, Any] = field(default_factory=dict)


SCENARIO_STEPS = (
    ScenarioStep(
        step_id="shortcut_endpoint_simplification",
        title="Model-favorable simplification with weaker evidence value",
        completion_delta=5.0,
        pillar_for_delta="Scientific Challenge",
        structured_edits={
            "strategic_ambition_ml": "PIVOTAL_INTENT",
            "intervention_model_ml": "SINGLE_GROUP",
            "allocation_ml": "NON-RANDOMIZED",
            "masking_ml": "UNKNOWN",
            "endpoint_rigor_ml": "SURROGATE",
            "endpoint_structure_ml": "SINGLE_GOAL",
        },
        expectations={
            "design_confidence_max": 0.0,
            "must_challenge_completion_gain": True,
            "forbid_completion_operational_drivers": True,
            "expected_quality": (
                "Completion Outlook may improve, but Design Confidence should be neutral or negative "
                "because the scenario weakens comparative rigor or endpoint interpretability."
            ),
        },
    ),
    ScenarioStep(
        step_id="patient_relevance_with_added_rigor",
        title="Harder but more clinically focused scenario",
        completion_delta=-4.0,
        pillar_for_delta="Patient Profile",
        structured_edits={
            "patient_severity_ml": "ADVANCED_METASTATIC",
            "line_of_therapy_ml": "REFRACTORY_RELAPSED",
            "is_rare_disease_ml": "1",
            "has_dmc_ml": "1",
            "endpoint_structure_ml": "MULTI_COMPOSITE",
        },
        expectations={
            "target_population_alignment_min": 0.5,
            "must_allow_design_gain_despite_completion_decline": True,
            "expected_quality": (
                "Completion Outlook may decline, while Target Population Alignment should improve because the "
                "scenario is clinically more focused. Total Design Confidence may remain negative if cumulative "
                "pivotal-design weaknesses from prior iterations still dominate."
            ),
        },
    ),
    ScenarioStep(
        step_id="operational_burden_without_matching_evidence_gain",
        title="Operational burden increase without matching evidence gain",
        completion_delta=0.0,
        pillar_for_delta="Execution Framework",
        operational_multipliers={
            "planned_enrollment": 4.0,
            "planned_sites": 2.5,
        },
        operational_additions={
            "planned_duration_months": 18.0,
        },
        expectations={
            "operational_burden_balance_max": 0.0,
            "must_not_move_completion_from_operational_only": True,
            "forbid_completion_operational_drivers": True,
            "expected_quality": (
                f"{OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY} Design Confidence should flag proportionality "
                "if burden is not matched by evidence gain. Total Design Confidence may remain positive if inherited "
                "patient or evidence strengths remain, but Operational Burden Balance should not improve."
            ),
        },
    ),
    ScenarioStep(
        step_id="structured_text_intervention_conflict",
        title="Structured/free-text contradiction",
        completion_delta=-0.3,
        pillar_for_delta="Scientific Challenge",
        structured_edits={
            "target_pathway_class_ml": "GPCR_TARGET",
            "therapeutic_modality_ml": "SMALL_MOLECULE",
            "administration_complexity_ml": "SIMPLE_ORAL",
        },
        text_edits={
            "interventions_ui": (
                "Scenario note: the intervention is described as a cell-based immunotherapy requiring "
                "individualized manufacturing, chain-of-identity controls, and infusion-site coordination."
            ),
        },
        expectations={
            "requires_consistency_note": True,
            "requires_exact_consistency_warning": True,
            "expected_consistency_fields": ("Intervention text", "Therapeutic Modality"),
            "structured_fields_prevail": True,
            "forbid_completion_operational_drivers": True,
            "expected_quality": (
                "The selected structured fields should drive analysis for Completion Outlook and Design Confidence. "
                "Contradictory intervention text must trigger a clear warning, but should not override selected fields "
                "or become the main Design Confidence score driver."
            ),
        },
    ),
)


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _merged_env(load_dotenv: bool) -> dict[str, str]:
    env = dict(os.environ)
    if load_dotenv:
        for key, value in _load_env_file(ROOT / ".env").items():
            env.setdefault(key, value)
    return env


def _load_taxonomy() -> dict[str, Any]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["FIELDS"]


def _option_label(taxonomy: dict[str, Any], field_id: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    options = ((taxonomy.get(field_id) or {}).get("ui") or {}).get("options") or []
    for option_key, label in options:
        if text == str(option_key) or text == str(label):
            return str(label)
    mapping = (taxonomy.get(field_id) or {}).get("mapping") or {}
    mapped = mapping.get(text) or mapping.get(text.upper())
    if isinstance(mapped, list) and len(mapped) >= 2:
        return str(mapped[1])
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        for mapped in mapping.values():
            if (
                isinstance(mapped, list)
                and len(mapped) >= 2
                and pd.notna(pd.to_numeric(mapped[0], errors="coerce"))
                and float(mapped[0]) == float(numeric)
            ):
                return str(mapped[1])
    return text


def _field_label(taxonomy: dict[str, Any], field_id: str) -> str:
    return str(((taxonomy.get(field_id) or {}).get("ui") or {}).get("label") or field_id)


def _operational_label(assumption_key: str) -> str:
    return OPERATIONAL_ASSUMPTION_LABELS.get(assumption_key, assumption_key)


def _field_value(row: pd.Series, field_id: str) -> Any:
    if field_id in row and pd.notna(row[field_id]):
        return row[field_id].item() if hasattr(row[field_id], "item") else row[field_id]
    ui_field = field_id.replace("_ml", "_ui")
    if ui_field in row and pd.notna(row[ui_field]):
        return row[ui_field].item() if hasattr(row[ui_field], "item") else row[ui_field]
    return None


def _normalize_field_value(taxonomy: dict[str, Any], field_id: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value).strip()
    options = ((taxonomy.get(field_id) or {}).get("ui") or {}).get("options") or []
    for option_key, label in options:
        if text == str(option_key) or text == str(label):
            return str(option_key)
    mapping = (taxonomy.get(field_id) or {}).get("mapping") or {}
    if text in mapping:
        return text
    if text.upper() in mapping:
        return text.upper()
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        for option_key, mapped in mapping.items():
            if (
                isinstance(mapped, list)
                and mapped
                and pd.notna(pd.to_numeric(mapped[0], errors="coerce"))
                and float(mapped[0]) == float(numeric)
            ):
                return str(option_key)
    return value


def _baseline_structured_features(row: pd.Series, taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        field_id: _normalize_field_value(taxonomy, field_id, _field_value(row, field_id))
        for field_id in STRUCTURED_FEATURE_KEYS
    }


def _display_values(taxonomy: dict[str, Any], structured: dict[str, Any], row: pd.Series) -> dict[str, str]:
    display = {}
    for field_id, value in structured.items():
        if field_id == "gbd_cause_id_3_ml":
            display[field_id] = str(row.get("gbd_indication_name_3") or value or "")
        else:
            display[field_id] = _option_label(taxonomy, field_id, value)
    return display


def _text_context(row: pd.Series) -> dict[str, str]:
    return {
        key: "" if key not in row or pd.isna(row[key]) else str(row[key])
        for key in TEXT_CONTEXT_KEYS
    }


def _trial_identity(row: pd.Series) -> dict[str, Any]:
    return {
        "nct_id": str(row.get("nct_id")),
        "trial_label": str(row.get("brief_title") or row.get("title") or row.get("nct_id")),
        "lead_sponsor_canonical": str(row.get("lead_sponsor_canonical") or row.get("lead_sponsor") or ""),
        "start_year": int(row.get("start_year")) if pd.notna(row.get("start_year")) else None,
    }


def _score(row: pd.Series) -> float:
    value = pd.to_numeric(row.get("Clinical_Score"), errors="coerce")
    return 50.0 if pd.isna(value) else round(float(value), 1)


def _pillar_impacts_from_row(row: pd.Series) -> list[dict[str, Any]]:
    impacts = []
    for pillar in PILLAR_COLUMNS:
        value = pd.to_numeric(row.get(pillar), errors="coerce")
        impacts.append({"Pillar": pillar, "Impact": 0.0 if pd.isna(value) else round(float(value), 1)})
    return impacts


def _apply_pillar_delta(pillars: list[dict[str, Any]], pillar_name: str, delta: float) -> list[dict[str, Any]]:
    updated = deepcopy(pillars)
    for item in updated:
        if item.get("Pillar") == pillar_name:
            item["Impact"] = round(float(item.get("Impact") or 0.0) + float(delta), 1)
            return updated
    updated.append({"Pillar": pillar_name, "Impact": round(float(delta), 1)})
    return updated


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


def _baseline_operational_assumptions(row: pd.Series) -> dict[str, dict[str, Any]]:
    enrollment = pd.to_numeric(row.get("enrollment"), errors="coerce")
    sites = pd.to_numeric(row.get("number_of_facilities"), errors="coerce")
    duration = pd.to_numeric(row.get("completion_duration_months"), errors="coerce")
    if pd.isna(duration):
        duration = pd.to_numeric(row.get("primary_duration_months_ml"), errors="coerce")
    values = {
        "planned_enrollment": 100 if pd.isna(enrollment) or enrollment <= 0 else int(round(float(enrollment))),
        "planned_sites": 10 if pd.isna(sites) or sites <= 0 else int(round(float(sites))),
        "planned_duration_months": 24.0 if pd.isna(duration) or duration <= 0 else round(float(duration), 1),
    }
    return {
        key: {
            "value": value,
            "source": "eval_baseline_registry",
            "status": "eval_reference",
        }
        for key, value in values.items()
    }


def _apply_operational_edits(
    operational: dict[str, dict[str, Any]],
    multipliers: dict[str, float],
    additions: dict[str, float],
) -> dict[str, dict[str, Any]]:
    updated = deepcopy(operational)
    for key in ACTIVE_OPERATIONAL_ASSUMPTION_KEYS:
        current = pd.to_numeric((updated.get(key) or {}).get("value"), errors="coerce")
        if pd.isna(current):
            current = 0
        value = float(current)
        if key in multipliers:
            value *= float(multipliers[key])
        if key in additions:
            value += float(additions[key])
        if key == "planned_duration_months":
            final_value: float | int = round(value, 1)
        else:
            final_value = int(round(value))
        updated[key] = {
            **dict(updated.get(key) or {}),
            "value": final_value,
            "source": "eval_user_scenario",
            "status": "eval_modified",
        }
    return updated


def _snapshot(
    *,
    row: pd.Series,
    taxonomy: dict[str, Any],
    snapshot_id: str,
    source: str,
    structured: dict[str, Any],
    text_context: dict[str, str],
    operational: dict[str, dict[str, Any]],
    score: float,
    pillar_impacts: list[dict[str, Any]],
    previous_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    previous_score = (previous_snapshot or {}).get("score")
    score_delta = None
    if previous_score is not None:
        score_delta = round(float(score) - float(previous_score), 1)
    display = _display_values(taxonomy, structured, row)
    previous_values = (previous_snapshot or {}).get("structured_features") or {}
    changed_fields = [
        field_id
        for field_id, value in structured.items()
        if str(value) != str(previous_values.get(field_id))
    ] if previous_snapshot else []
    changed_text = []
    if previous_snapshot:
        previous_text = previous_snapshot.get("text_context") or {}
        changed_text = [
            key
            for key, value in text_context.items()
            if re.sub(r"\W+", "", str(value).lower()) != re.sub(r"\W+", "", str(previous_text.get(key, "")).lower())
        ]
    changed_operational = []
    if previous_snapshot:
        previous_operational = previous_snapshot.get("operational_assumptions") or {}
        for key in ACTIVE_OPERATIONAL_ASSUMPTION_KEYS:
            if (operational.get(key) or {}).get("value") != (previous_operational.get(key) or {}).get("value"):
                changed_operational.append(key)
    return {
        "snapshot_id": snapshot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nct_id": str(row.get("nct_id")),
        "source": source,
        "trial_identity": _trial_identity(row),
        "structured_features": deepcopy(structured),
        "submitted_values": deepcopy(structured),
        "compare_values": deepcopy(structured),
        "display_values": display,
        "text_context": deepcopy(text_context),
        "operational_assumptions": deepcopy(operational),
        "score": score,
        "previous_score": previous_score,
        "score_delta_points": score_delta,
        "pillar_impacts": deepcopy(pillar_impacts),
        "previous_pillar_impacts": deepcopy((previous_snapshot or {}).get("pillar_impacts") or []),
        "result": {
            "score": score,
            "pillar_impacts": deepcopy(pillar_impacts),
            "subcat_impacts": [],
        },
        "changed_fields": changed_fields,
        "changed_text_context_fields": changed_text,
        "changed_operational_assumptions": changed_operational,
        "iteration_context": {
            "iteration_number": 0 if source == "prerecorded_baseline" else ((previous_snapshot or {}).get("iteration_context", {}).get("iteration_number", 0) + 1),
        },
    }


def _initial_step_edit_count(row: pd.Series, taxonomy: dict[str, Any]) -> int:
    baseline = _baseline_structured_features(row, taxonomy)
    return sum(
        1
        for field_id, target_value in SCENARIO_STEPS[0].structured_edits.items()
        if str(baseline.get(field_id)) != str(target_value)
    )


def _select_trials(registry: pd.DataFrame, taxonomy: dict[str, Any], max_trials: int) -> list[pd.Series]:
    selected: list[pd.Series] = []
    used_ncts: set[str] = set()
    for therapeutic_area, band in FIRST_WAVE_TARGETS:
        low, high = SCORE_BANDS[band]
        candidates = registry[
            (registry["therapeutic_area_ui"].astype(str) == therapeutic_area)
            & (pd.to_numeric(registry["Clinical_Score"], errors="coerce") >= low)
            & (pd.to_numeric(registry["Clinical_Score"], errors="coerce") < high)
        ].copy()
        if candidates.empty:
            candidates = registry[registry["therapeutic_area_ui"].astype(str) == therapeutic_area].copy()
        if candidates.empty:
            continue
        candidates["_initial_edit_count"] = candidates.apply(
            lambda item: _initial_step_edit_count(item, taxonomy),
            axis=1,
        )
        editable_candidates = candidates[candidates["_initial_edit_count"] >= 3].copy()
        if not editable_candidates.empty:
            candidates = editable_candidates
        midpoint = (low + high) / 2.0
        candidates["_distance"] = (pd.to_numeric(candidates["Clinical_Score"], errors="coerce") - midpoint).abs()
        for _, row in candidates.sort_values(["_distance", "_initial_edit_count", "nct_id"], ascending=[True, False, True]).iterrows():
            nct_id = str(row["nct_id"])
            if nct_id not in used_ncts:
                selected.append(row)
                used_ncts.add(nct_id)
                break
        if len(selected) >= max_trials:
            break
    return selected[:max_trials]


def _provider_result(
    packet: dict[str, Any],
    *,
    provider: str,
    config: Any,
) -> dict[str, Any]:
    if provider == "configured":
        return review_packet_with_provider_chain(packet, config=config)
    return review_packet_with_provider(packet, provider=provider, config=config)


def _review_controls_for_step(step: ScenarioStep) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "latest_change_step_id": step.step_id,
        "latest_change_title": step.title,
        "completion_outlook_mode": "movement_explanation",
        "question_controls": {
            "forbid_repeating_prior_questions": True,
            "medical_question_focus": "newest_evidence_or_population_tension",
            "operations_question_focus": "latest_operational_or_coherence_tension",
        },
    }
    if step.step_id == "operational_burden_without_matching_evidence_gain":
        controls.update({
            "completion_outlook_mode": "fixed_planning_assumption_boundary",
            "required_completion_outlook_sentence": OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY,
            "completion_outlook_forbidden_latest_fields": [
                "operational_assumptions.planned_enrollment",
                "operational_assumptions.planned_sites",
                "operational_assumptions.planned_duration_months",
            ],
            "latest_change_focus": "planning_assumptions_only",
        })
        controls["question_controls"].update({
            "medical_question_focus": "evidence_standard_only_if_reframed_around_planning_burden",
            "operations_question_focus": "operational_proportionality_or_executability",
            "at_least_one_question_must_address": "planning_assumption_proportionality",
        })
    elif step.step_id == "structured_text_intervention_conflict":
        controls.update({
            "completion_outlook_mode": "consistency_note_only",
            "latest_change_focus": "structured_free_text_conflict",
        })
        controls["question_controls"].update({
            "medical_question_focus": "evidence_implications_of_resolving_the_modality_mismatch",
            "operations_question_focus": "resolve_structured_free_text_contradiction",
            "at_least_one_question_must_address": "structured_free_text_contradiction",
        })
    elif step.step_id == "patient_relevance_with_added_rigor":
        controls.update({"latest_change_focus": "population_relevance_and_endpoint_complexity"})
        controls["question_controls"].update({
            "medical_question_focus": "population_relevance_versus_evidence_standard",
            "operations_question_focus": "recruitment_and_data_reliability_for_latest_population_change",
        })
    elif step.step_id == "shortcut_endpoint_simplification":
        controls.update({"latest_change_focus": "evidence_shortcut_and_bias_control"})
        controls["question_controls"].update({
            "medical_question_focus": "evidence_standard_and_bias_control",
            "operations_question_focus": "data_reliability_under_simplified_design",
        })
    return controls


def _attach_review_controls(packet: dict[str, Any], step: ScenarioStep) -> dict[str, Any]:
    controlled = deepcopy(packet)
    controlled["review_controls"] = _review_controls_for_step(step)
    controlled["input_hash"] = stable_packet_hash({key: value for key, value in controlled.items() if key != "input_hash"})
    return controlled


def _apply_review_control_overrides(packet: dict[str, Any], review_result: dict[str, Any]) -> dict[str, Any]:
    controls = packet.get("review_controls") or {}
    required_sentence = controls.get("required_completion_outlook_sentence")
    if (
        review_result.get("status") != "reviewed"
        or controls.get("completion_outlook_mode") != "fixed_planning_assumption_boundary"
        or not isinstance(required_sentence, str)
        or not required_sentence.strip()
    ):
        return review_result

    pre_control_review = deepcopy(review_result.get("review") or {})
    pre_control_validated_review = deepcopy(review_result.get("validated_review") or {})
    pre_control_scoring = deepcopy(review_result.get("scoring") or {})
    review = deepcopy(pre_control_review)
    completion = review.setdefault("completion_outlook_analysis", {})
    completion["risk_pattern_summary"] = required_sentence
    completion["driver_summary"] = "No Completion Outlook score input changed in this planning-assumption-only iteration."
    completion["movement_explanation"] = required_sentence
    review.setdefault("trace", {})["completion_outlook_control_applied"] = "fixed_planning_assumption_boundary"

    rescored = validate_and_score_review(packet, review)
    scoring = rescored["scoring"]
    return {
        **deepcopy(review_result),
        "review": review,
        "validated_review": rescored["validated_review"],
        "scoring": scoring,
        "provider_metadata": {
            **deepcopy(review_result.get("provider_metadata") or {}),
            "review_control_override": "fixed_planning_assumption_boundary",
            "pre_control_review": pre_control_review,
            "pre_control_validated_review": pre_control_validated_review,
            "pre_control_scoring": pre_control_scoring,
        },
    }


def _words(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _similarity(a: str, b: str) -> float:
    left = _words(a)
    right = _words(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _review_texts(trace: dict[str, Any]) -> dict[str, str]:
    validated = trace.get("validated_review") or {}
    completion = validated.get("completion_outlook_analysis") or {}
    design = validated.get("design_confidence_analysis") or {}
    questions = validated.get("key_questions") or {}
    consistency = validated.get("scenario_consistency_note") or {}
    return {
        "completion": str(completion.get("risk_pattern_summary") or ""),
        "design": str(design.get("summary") or ""),
        "medical_question": str(questions.get("medical_development_question") or ""),
        "operations_question": str(questions.get("clinical_operations_question") or ""),
        "consistency_note": str(consistency.get("message") or ""),
    }


def _is_operational_only_zero_delta(trace: dict[str, Any]) -> bool:
    packet = trace.get("input_packet") or {}
    iteration = packet.get("iteration_context") or {}
    changed_fields = [str(field or "") for field in iteration.get("changed_fields") or []]
    score_delta = pd.to_numeric(trace.get("score_delta"), errors="coerce")
    return (
        changed_fields
        and pd.notna(score_delta)
        and abs(float(score_delta)) < 0.05
        and all(field.startswith("operational_assumptions.") for field in changed_fields)
    )


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has_completion_movement_language(text: str) -> bool:
    lower = str(text or "").lower()
    movement_terms = (
        "increased early-termination risk",
        "lower early-termination risk",
        "higher early-termination risk",
        "completion outlook appears to decline",
        "completion outlook appears to improve",
        "completion outlook appears more favorable",
        "completion outlook appears less favorable",
        "completion outlook improves",
        "completion outlook declines",
        "lower risk profile",
        "increased risk profile",
        "harder to sustain",
    )
    return any(term in lower for term in movement_terms)


def _has_operational_boundary_language(text: str) -> bool:
    lower = str(text or "").lower()
    boundary_terms = (
        "outside the model",
        "does not explain completion outlook movement",
        "do not explain completion outlook movement",
        "planning assumptions such as enrollment, site count, and total duration do not directly feed the score",
        "reflected in design confidence instead",
    )
    return any(term in lower for term in boundary_terms)


def _completion_text_without_operational_boundary(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized.replace(OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY, "")


def _has_expected_structured_text_warning(message: str, expected_fields: tuple[str, ...] = ()) -> bool:
    text = str(message or "").strip()
    if not text.startswith(STRUCTURED_TEXT_CONFLICT_WARNING):
        return False
    match = re.search(r"\(([^)]*)\)\s*$", text)
    if not match:
        return False
    field_text = match.group(1).lower()
    return all(field.lower() in field_text for field in expected_fields)


def _design_confidence_subcategories(trace: dict[str, Any]) -> dict[str, Any]:
    assessment = trace.get("design_confidence_assessment") or {}
    subcategories = assessment.get("subcategories") or trace.get("design_confidence_subcategories")
    if subcategories:
        return subcategories
    validated = trace.get("validated_review") or {}
    return validated.get("design_confidence_subcategories") or {}


def _grade_trace(
    trace: dict[str, Any],
    step: ScenarioStep | None,
    previous_visible_trace: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    findings: list[dict[str, str]] = []
    comments: list[str] = []
    status = trace.get("status")
    if status != "reviewed":
        findings.append({"severity": "fail", "check": "provider_status", "detail": str(trace.get("failure_reason") or status)})
        comments.append("Provider did not return a validated Scenario Review, so narrative quality cannot be assessed.")
        return findings, comments

    scoring_status = trace.get("validation_status")
    if scoring_status != "valid":
        findings.append({"severity": "fail", "check": "validation_status", "detail": str(trace.get("validation_errors") or scoring_status)})

    texts = _review_texts(trace)
    completion_lower = texts["completion"].lower()
    operational_terms = (
        "planned enrollment",
        "planned site",
        "planned total duration",
        "operational benchmark",
    )
    if any(term in completion_lower for term in operational_terms) and not _has_operational_boundary_language(texts["completion"]):
        findings.append({"severity": "fail", "check": "completion_outlook_operational_driver", "detail": "Completion Outlook appears to cite operational-only evidence."})

    if step and step.expectations.get("must_not_move_completion_from_operational_only") and _is_operational_only_zero_delta(trace):
        unchanged_terms = (
            "essentially unchanged",
            "unchanged",
            "outside the model",
        )
        if _has_completion_movement_language(texts["completion"]):
            findings.append({
                "severity": "fail",
                "check": "completion_outlook_operational_zero_delta",
                "detail": "Operational-only zero-delta scenario uses Completion Outlook movement/risk-change language.",
            })
        if not any(term in completion_lower for term in unchanged_terms):
            findings.append({
                "severity": "fail",
                "check": "completion_outlook_operational_zero_delta",
                "detail": "Operational-only zero-delta scenario should use participant-friendly unchanged Completion Outlook boundary wording.",
            })
        extra_completion_text = _completion_text_without_operational_boundary(texts["completion"]).lower()
        extra_operational_terms = (
            "planned enrollment",
            "planned site",
            "site count",
            "total duration",
            "duration reflects",
            "operational footprint",
            "operational scale",
            "site footprint",
            "recruitment footprint",
            "operationally more extensive",
        )
        if any(term in extra_completion_text for term in extra_operational_terms):
            findings.append({
                "severity": "fail",
                "check": "completion_outlook_operational_extra_detail",
                "detail": "Operational-only zero-delta Completion Outlook includes extra operational detail beyond the agreed boundary sentence.",
            })

    if step and step.step_id != "operational_burden_without_matching_evidence_gain":
        stale_planning_terms = (
            "specific operational assumptions adjusted in this iteration",
            "planning assumptions adjusted in this iteration",
            "increase in planned enrollment",
            "increase in site count",
            "increase in planned duration",
            "planned enrollment, site count, and duration",
        )
        if any(term in completion_lower for term in stale_planning_terms):
            findings.append({
                "severity": "fail",
                "check": "completion_outlook_stale_prior_change",
                "detail": "Completion Outlook describes prior planning-assumption changes as if they were the latest iteration.",
            })

    causal_terms = ("causes completion", "caused completion", "will complete", "chance of completion")
    if any(term in completion_lower for term in causal_terms):
        findings.append({"severity": "warn", "check": "completion_outlook_overclaim", "detail": "Completion Outlook may be too causal or promise-like."})

    participant_text = " ".join(
        texts[key]
        for key in ("completion", "design", "medical_question", "operations_question")
    ).lower()
    internal_model_terms = (
        "model-facing",
        "model supported",
        "model-supported",
        "model signals",
        "model suggests",
        "model indicates",
        "model predicts",
        "model implies",
        "model-driven",
        "according to the model",
        "in the model",
        "the model says",
        "model reflects",
    )
    if any(term in participant_text for term in internal_model_terms):
        findings.append({
            "severity": "fail",
            "check": "participant_model_language",
            "detail": "Participant-facing narrative uses internal model vocabulary instead of score-pattern language.",
        })

    design_confidence = pd.to_numeric(trace.get("design_confidence"), errors="coerce")
    if step:
        expectations = step.expectations
        if "design_confidence_min" in expectations and pd.notna(design_confidence) and design_confidence < expectations["design_confidence_min"]:
            findings.append({"severity": "fail", "check": "design_confidence_direction", "detail": f"Expected Design Confidence >= {expectations['design_confidence_min']}, got {design_confidence}."})
        if "design_confidence_max" in expectations and pd.notna(design_confidence) and design_confidence > expectations["design_confidence_max"]:
            findings.append({"severity": "fail", "check": "design_confidence_direction", "detail": f"Expected Design Confidence <= {expectations['design_confidence_max']}, got {design_confidence}."})
        if "target_population_alignment_min" in expectations:
            subcategory_points = _design_confidence_subcategories(trace)
            target_population = pd.to_numeric(
                (subcategory_points.get("target_population_alignment") or {}).get("points"),
                errors="coerce",
            )
            if pd.isna(target_population) or target_population < expectations["target_population_alignment_min"]:
                findings.append({
                    "severity": "fail",
                    "check": "target_population_alignment_direction",
                    "detail": (
                        "Expected Target Population Alignment >= "
                        f"{expectations['target_population_alignment_min']}, got {target_population}."
                    ),
                })
        if "operational_burden_balance_max" in expectations:
            subcategory_points = _design_confidence_subcategories(trace)
            operational_burden = pd.to_numeric(
                (subcategory_points.get("operational_burden_balance") or {}).get("points"),
                errors="coerce",
            )
            if pd.isna(operational_burden) or operational_burden > expectations["operational_burden_balance_max"]:
                findings.append({
                    "severity": "fail",
                    "check": "operational_burden_balance_direction",
                    "detail": (
                        "Expected Operational Burden Balance <= "
                        f"{expectations['operational_burden_balance_max']}, got {operational_burden}."
                    ),
                })
        consistency = (trace.get("scenario_consistency_note") or {})
        if expectations.get("requires_consistency_note") and not consistency.get("has_clear_mismatch"):
            findings.append({"severity": "fail", "check": "missing_consistency_note", "detail": "Expected visible scenario_consistency_note for structured/free-text conflict."})
        if expectations.get("requires_exact_consistency_warning") and not _has_expected_structured_text_warning(
            consistency.get("message") or "",
            tuple(expectations.get("expected_consistency_fields") or ()),
        ):
            findings.append({
                "severity": "fail",
                "check": "scenario_consistency_warning",
                "detail": "Expected structured/free-text warning followed by participant-readable field labels in parentheses.",
            })
        if expectations.get("structured_fields_prevail"):
            design_lower = texts["design"].lower()
            subcategory_points = _design_confidence_subcategories(trace)
            if "cell" in completion_lower and "not" not in completion_lower and "contradict" not in completion_lower:
                findings.append({"severity": "warn", "check": "structured_fields_prevail", "detail": "Completion Outlook may treat contradictory cell-therapy text as scenario truth."})
            if "small molecule" not in (completion_lower + " " + design_lower):
                findings.append({"severity": "warn", "check": "structured_fields_context", "detail": "Review may not clearly acknowledge the selected Small Molecule field."})
            endpoint = subcategory_points.get("endpoint_evidence_strength") or {}
            endpoint_points = pd.to_numeric(endpoint.get("points"), errors="coerce")
            endpoint_evidence = " ".join(str(field).lower() for field in endpoint.get("evidence_fields") or [])
            endpoint_refs = (
                "endpoint_rigor_ml",
                "endpoint_structure_ml",
                "primary_duration_months_ml",
                "primary_outcomes_ui",
                "text_context.primary_outcomes_ui",
            )
            if pd.notna(endpoint_points) and endpoint_points <= -3 and not any(ref in endpoint_evidence for ref in endpoint_refs):
                findings.append({
                    "severity": "warn",
                    "check": "structured_text_bounded_impact",
                    "detail": "Structured/free-text mismatch strongly penalizes Endpoint Evidence Strength without endpoint evidence support.",
                })
            severe_subcategories = [
                name
                for name, value in subcategory_points.items()
                if pd.notna(pd.to_numeric((value or {}).get("points"), errors="coerce"))
                and pd.to_numeric((value or {}).get("points"), errors="coerce") <= -3
            ]
            mismatch_terms = ("contradiction", "mismatch", "misalignment", "conflict")
            if (
                pd.notna(design_confidence)
                and design_confidence <= -6
                and len(severe_subcategories) >= 2
                and any(term in design_lower for term in mismatch_terms)
            ):
                findings.append({
                    "severity": "warn",
                    "check": "structured_text_bounded_impact",
                    "detail": (
                        "Structured/free-text mismatch appears to dominate multiple Design Confidence subcategories; "
                        "human review should confirm selected fields still prevail."
                    ),
                })
    for label, question in (
        ("medical_question", texts["medical_question"]),
        ("operations_question", texts["operations_question"]),
    ):
        if not question.strip().endswith("?"):
            findings.append({"severity": "warn", "check": f"{label}_form", "detail": "Question does not end with a question mark."})
        if re.match(r"^\s*(is|are|can|could|should|would|will|does|do|did)\b", question, re.I):
            findings.append({"severity": "warn", "check": f"{label}_yes_no", "detail": "Question may be answerable yes/no."})

    if previous_visible_trace:
        previous_texts = _review_texts(previous_visible_trace)
        for label in ("medical_question", "operations_question"):
            if _normalize_question(texts[label]) == _normalize_question(previous_texts[label]):
                findings.append({
                    "severity": "fail",
                    "check": f"{label}_verbatim_repetition",
                    "detail": "Question repeats the prior visible question verbatim.",
                })
                continue
            similarity = _similarity(texts[label], previous_texts[label])
            if similarity >= 0.55:
                findings.append({
                    "severity": "warn",
                    "check": f"{label}_freshness",
                    "detail": (
                        f"Question similarity to previous iteration is {similarity:.2f}; check whether one question "
                        "is anchored to the newest material change and the other uses the trial as an example of a broader development-design tension."
                    ),
                })

    combined_questions = f"{texts['medical_question']} {texts['operations_question']}".lower()
    if step and step.step_id == "operational_burden_without_matching_evidence_gain":
        # Accepted terms for detecting latest-change focus in the two key questions.
        # This is not a Completion Outlook forbidden-term list.
        question_focus_terms = (
            "proportionate",
            "proportionality",
            "executable",
            "executability",
            "enrollment",
            "duration",
            "scale",
            "scaled",
            "planned enrollment",
            "site count",
            "expanded site",
            "expanded network",
            "network",
            "oversight",
            "data quality",
            "site-level",
            "governance",
            "larger patient",
            "total duration",
        )
        if not any(term in combined_questions for term in question_focus_terms):
            findings.append({
                "severity": "fail",
                "check": "question_latest_change_focus",
                "detail": "Operational-only iteration should include a question about planning-assumption proportionality or executability.",
            })
    if step and step.step_id == "structured_text_intervention_conflict":
        focus_terms = (
            "contradiction",
            "mismatch",
            "discrepancy",
            "conflict",
            "reconcile",
            "resolve",
            "intervention text",
            "therapeutic modality",
        )
        if not any(term in combined_questions for term in focus_terms):
            findings.append({
                "severity": "fail",
                "check": "question_latest_change_focus",
                "detail": "Structured/free-text contradiction iteration should include a question about resolving the mismatch.",
            })

    if not findings:
        comments.append("No deterministic quality gaps found; review is ready for human narrative-quality assessment.")
    else:
        comments.append("Review needs human attention on the flagged checks before treating this prompt behavior as stable.")
    return findings, comments


def _compact_changes(taxonomy: dict[str, Any], previous_snapshot: dict[str, Any], current_snapshot: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    prev_structured = previous_snapshot.get("structured_features") or {}
    curr_structured = current_snapshot.get("structured_features") or {}
    for field_id in current_snapshot.get("changed_fields") or []:
        label = _field_label(taxonomy, field_id)
        before = _option_label(taxonomy, field_id, prev_structured.get(field_id))
        after = _option_label(taxonomy, field_id, curr_structured.get(field_id))
        changes.append(f"{label}: {before or prev_structured.get(field_id)} -> {after or curr_structured.get(field_id)}")
    for key in current_snapshot.get("changed_operational_assumptions") or []:
        before = ((previous_snapshot.get("operational_assumptions") or {}).get(key) or {}).get("value")
        after = ((current_snapshot.get("operational_assumptions") or {}).get(key) or {}).get("value")
        changes.append(f"{_operational_label(key)}: {before} -> {after}")
    for key in current_snapshot.get("changed_text_context_fields") or []:
        changes.append(f"{_field_label(taxonomy, key)}: text changed")
    return changes


def _run_trial(
    row: pd.Series,
    *,
    taxonomy: dict[str, Any],
    provider: str,
    config: Any,
    cache_namespace: str | None,
    include_baseline_review: bool,
) -> dict[str, Any]:
    nct_id = str(row["nct_id"])
    state: dict[str, Any] = {}
    session_id = f"eval:{nct_id}"
    baseline_structured = _baseline_structured_features(row, taxonomy)
    baseline_text = _text_context(row)
    baseline_operational = _baseline_operational_assumptions(row)
    baseline_score = _score(row)
    baseline_pillars = _pillar_impacts_from_row(row)
    baseline_snapshot = _snapshot(
        row=row,
        taxonomy=taxonomy,
        snapshot_id=f"{nct_id}:baseline",
        source="prerecorded_baseline",
        structured=baseline_structured,
        text_context=baseline_text,
        operational=baseline_operational,
        score=baseline_score,
        pillar_impacts=baseline_pillars,
        previous_snapshot=None,
    )

    baseline_trace = None
    trial_result = {
        "trial": {
            **_trial_identity(row),
            "therapeutic_area": str(row.get("therapeutic_area_ui") or ""),
            "baseline_completion_score": baseline_score,
            "score_band": _score_band_label(baseline_score),
        },
        "baseline_review": None,
        "iterations": [],
    }

    if include_baseline_review:
        baseline_packet = build_review_packet(
            current_snapshot=baseline_snapshot,
            previous_snapshot=None,
            baseline_snapshot=baseline_snapshot,
            trial_identity=_trial_identity(row),
            text_context=baseline_text,
        )
        baseline_review = _provider_result(baseline_packet, provider=provider, config=config)
        baseline_trace = store_review_trace(
            state,
            packet=baseline_packet,
            review_result=baseline_review,
            session_id=session_id,
            cache_namespace=cache_namespace,
        )
        baseline_trace["hidden_baseline"] = True
        trial_result["baseline_review"] = _trace_summary(baseline_trace, None, [], [])

    previous_snapshot = baseline_snapshot
    previous_visible_trace = None
    structured = deepcopy(baseline_structured)
    text = deepcopy(baseline_text)
    operational = deepcopy(baseline_operational)
    score = baseline_score
    pillars = deepcopy(baseline_pillars)

    for index, step in enumerate(SCENARIO_STEPS, start=1):
        structured.update(step.structured_edits)
        text.update(step.text_edits)
        operational = _apply_operational_edits(
            operational,
            step.operational_multipliers,
            step.operational_additions,
        )
        score = _clamp_score(score + step.completion_delta)
        pillars = _apply_pillar_delta(pillars, step.pillar_for_delta, step.completion_delta)
        current_snapshot = _snapshot(
            row=row,
            taxonomy=taxonomy,
            snapshot_id=f"{nct_id}:iter{index}:{step.step_id}",
            source="eval_simulation",
            structured=structured,
            text_context=text,
            operational=operational,
            score=score,
            pillar_impacts=pillars,
            previous_snapshot=previous_snapshot,
        )
        packet = build_review_packet(
            current_snapshot=current_snapshot,
            previous_snapshot=previous_snapshot,
            baseline_snapshot=baseline_snapshot,
            baseline_review_trace=baseline_trace,
            previous_review_trace=previous_visible_trace,
            trial_identity=_trial_identity(row),
            text_context=text,
            compact_storyline_memory=compact_storyline_from_trace(previous_visible_trace),
        )
        packet = _attach_review_controls(packet, step)
        review = _provider_result(packet, provider=provider, config=config)
        review = _apply_review_control_overrides(packet, review)
        trace = store_review_trace(
            state,
            packet=packet,
            review_result=review,
            session_id=session_id,
            baseline_id=baseline_snapshot["snapshot_id"],
            cache_namespace=cache_namespace,
        )
        findings, comments = _grade_trace(trace, step, previous_visible_trace)
        changes = _compact_changes(taxonomy, previous_snapshot, current_snapshot)
        trial_result["iterations"].append(_trace_summary(trace, step, changes, findings, comments))
        previous_snapshot = current_snapshot
        if trace.get("status") == "reviewed":
            previous_visible_trace = trace

    return trial_result


def _score_band_label(score: float) -> str:
    for name, (low, high) in SCORE_BANDS.items():
        if low <= score < high:
            return name
    return "high" if score >= 75 else "unknown"


def _trace_summary(
    trace: dict[str, Any],
    step: ScenarioStep | None,
    changes: list[str] | None,
    findings: list[dict[str, str]] | None = None,
    comments: list[str] | None = None,
) -> dict[str, Any]:
    texts = _review_texts(trace)
    input_packet = deepcopy(trace.get("input_packet") or {})
    provider_metadata = deepcopy(trace.get("provider_metadata") or {})
    return {
        "step_id": step.step_id if step else "hidden_baseline",
        "title": step.title if step else "Hidden baseline review",
        "expectations": step.expectations if step else {"expected_quality": "Baseline qualitative context only; no participant-visible Design Confidence comparison."},
        "deterministic_checks": _deterministic_check_labels(step),
        "human_review_expectations": _human_review_expectations(step),
        "changes": changes or [],
        "provider": trace.get("provider"),
        "model_name": trace.get("model_name"),
        "status": trace.get("status"),
        "validation_status": trace.get("validation_status"),
        "failure_reason": trace.get("failure_reason"),
        "completion_score": ((trace.get("input_packet") or {}).get("model_interpretation") or {}).get("completion_score"),
        "score_delta": trace.get("score_delta"),
        "design_confidence": trace.get("design_confidence"),
        "total_scenario_score": trace.get("total_scenario_score"),
        "scenario_consistency_note": trace.get("scenario_consistency_note") or {},
        "design_confidence_subcategories": _subcategory_summary(trace),
        "narrative": texts,
        "findings": findings or [],
        "codex_comments": comments or [],
        "input_hash": trace.get("input_hash"),
        "provider_metadata": provider_metadata,
        "pre_control_output_json": provider_metadata.get("pre_control_review"),
        "pre_control_validated_review": provider_metadata.get("pre_control_validated_review"),
        "pre_control_scoring": provider_metadata.get("pre_control_scoring"),
        "input_packet": input_packet,
        "provider_prompt": build_provider_prompt(input_packet) if input_packet else "",
        "raw_output_json": deepcopy(trace.get("output_json")),
        "validated_review": deepcopy(trace.get("validated_review")),
    }


def _deterministic_check_labels(step: ScenarioStep | None) -> list[str]:
    if not step:
        return ["provider_status", "validation_status", "participant_model_language"]
    labels = ["provider_status", "validation_status", "participant_model_language", "question_form", "question_freshness"]
    expectations = step.expectations
    if "design_confidence_min" in expectations or "design_confidence_max" in expectations:
        labels.append("design_confidence_direction")
    if "target_population_alignment_min" in expectations:
        labels.append("target_population_alignment_direction")
    if "operational_burden_balance_max" in expectations:
        labels.append("operational_burden_balance_direction")
    if expectations.get("requires_consistency_note"):
        labels.append("scenario_consistency_note")
    if expectations.get("requires_exact_consistency_warning"):
        labels.append("scenario_consistency_warning")
    if expectations.get("forbid_completion_operational_drivers"):
        labels.append("completion_outlook_operational_driver")
    if expectations.get("must_not_move_completion_from_operational_only"):
        labels.append("completion_outlook_operational_zero_delta")
        labels.append("completion_outlook_operational_extra_detail")
    if step.step_id != "operational_burden_without_matching_evidence_gain":
        labels.append("completion_outlook_stale_prior_change")
    if expectations.get("structured_fields_prevail"):
        labels.append("structured_fields_prevail")
        labels.append("structured_text_bounded_impact")
    if step.step_id in {"operational_burden_without_matching_evidence_gain", "structured_text_intervention_conflict"}:
        labels.append("question_latest_change_focus")
    return labels


def _human_review_expectations(step: ScenarioStep | None) -> list[str]:
    if not step:
        return ["Baseline should create qualitative context without participant-visible baseline scoring language."]
    expectations = step.expectations
    notes = [str(expectations.get("expected_quality") or "").strip()]
    flag_text = {
        "must_challenge_completion_gain": "Review whether Design Confidence challenges a model-favorable shortcut instead of rewarding it.",
        "must_allow_design_gain_despite_completion_decline": "Review whether the narrative allows lower Completion Outlook to coexist with stronger design defensibility.",
        "must_not_move_completion_from_operational_only": "Review whether Completion Outlook stays separate from operational-only assumptions.",
        "structured_fields_prevail": "Review whether selected structured fields clearly prevail over contradictory free text, while the mismatch remains a warning/context item rather than the main Design Confidence score driver.",
    }
    for key, text in flag_text.items():
        if expectations.get(key):
            notes.append(text)
    return [note for note in notes if note]


def _subcategory_summary(trace: dict[str, Any]) -> dict[str, Any]:
    assessment = trace.get("design_confidence_assessment") or {}
    subcategory_points = assessment.get("subcategories") or {}
    validated = trace.get("validated_review") or {}
    subcategories = validated.get("design_confidence_subcategories") or {}
    summary = {}
    for key, value in subcategories.items():
        summary[key] = {
            "rating": value.get("rating"),
            "score_materiality": value.get("score_materiality"),
            "points": (subcategory_points.get(key) or {}).get("points"),
            "evidence_fields": value.get("evidence_fields") or [],
            "short_rationale": value.get("short_rationale"),
            "rationale": value.get("rationale"),
        }
    return summary


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Narrative Eval Report: {data['run_id']}")
    lines.append("")
    lines.append(f"- Provider: `{data['provider']}`")
    lines.append(f"- Trials: `{len(data['trials'])}`")
    lines.append(f"- Generated: `{data['generated_at']}`")
    lines.append("")
    lines.append("## Summary")
    totals = data.get("summary") or {}
    for key, value in totals.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    for trial in data["trials"]:
        info = trial["trial"]
        lines.append(f"## {info['nct_id']} - {info['trial_label']}")
        lines.append("")
        lines.append(f"- Therapeutic Area: `{info['therapeutic_area']}`")
        lines.append(f"- Baseline Completion Score: `{info['baseline_completion_score']}` ({info['score_band']})")
        lines.append(f"- Sponsor: `{info.get('lead_sponsor_canonical') or ''}`")
        if trial.get("baseline_review"):
            baseline = trial["baseline_review"]
            lines.append("")
            lines.append("### Hidden Baseline")
            lines.extend(_markdown_trace_block(baseline))
        for iteration in trial["iterations"]:
            lines.append("")
            lines.append(f"### {iteration['step_id']}: {iteration['title']}")
            lines.extend(_markdown_trace_block(iteration))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_trace_block(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append("Expected:")
    lines.append(f"- {item.get('expectations', {}).get('expected_quality', '')}")
    human_expectations = item.get("human_review_expectations") or []
    if human_expectations:
        lines.append("")
        lines.append("Human Review Focus:")
        for expectation in human_expectations:
            lines.append(f"- {expectation}")
    deterministic = item.get("deterministic_checks") or []
    if deterministic:
        lines.append("")
        lines.append("Deterministic Checks:")
        for check in deterministic:
            lines.append(f"- `{check}`")
    if item.get("changes"):
        lines.append("")
        lines.append("Changes:")
        for change in item["changes"]:
            lines.append(f"- {change}")
    lines.append("")
    lines.append(
        f"Scores: Completion `{item.get('completion_score')}`, delta `{item.get('score_delta')}`, "
        f"Design Confidence `{item.get('design_confidence')}`, Total `{item.get('total_scenario_score')}`"
    )
    lines.append(f"Provider: `{item.get('provider')}` / `{item.get('model_name')}`, status `{item.get('status')}`")
    if item.get("failure_reason"):
        lines.append(f"Failure: {item['failure_reason']}")
    findings = item.get("findings") or []
    lines.append("")
    lines.append("Gap Analysis:")
    if findings:
        for finding in findings:
            lines.append(f"- `{finding['severity']}` {finding['check']}: {finding['detail']}")
    else:
        lines.append("- No deterministic quality gaps found.")
    comments = item.get("codex_comments") or []
    if comments:
        lines.append("")
        lines.append("Codex Comments:")
        for comment in comments:
            lines.append(f"- {comment}")
    narrative = item.get("narrative") or {}
    provider_metadata = item.get("provider_metadata") or {}
    if provider_metadata.get("review_control_override"):
        lines.append("")
        lines.append(f"Control Override: `{provider_metadata.get('review_control_override')}`")
        pre_control = item.get("pre_control_validated_review") or item.get("pre_control_output_json") or {}
        pre_completion = (pre_control.get("completion_outlook_analysis") or {}).get("risk_pattern_summary")
        if pre_completion:
            lines.append(f"- Pre-control Completion Outlook: {pre_completion}")
    lines.append("")
    lines.append("Narrative:")
    for label, key in (
        ("Consistency Note", "consistency_note"),
        ("Completion Outlook", "completion"),
        ("Design Confidence", "design"),
        ("Medical Development Question", "medical_question"),
        ("Clinical Operations Question", "operations_question"),
    ):
        value = str(narrative.get(key) or "").strip()
        if value:
            lines.append(f"- **{label}:** {value}")
    subcategories = item.get("design_confidence_subcategories") or {}
    if subcategories:
        lines.append("")
        lines.append("Design Confidence Subcategories:")
        for key, value in subcategories.items():
            lines.append(
                f"- `{key}`: {value.get('rating')} / {value.get('points')} pts - "
                f"{value.get('short_rationale') or value.get('rationale') or ''}"
            )
    return lines


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_iterations = 0
    reviewed = 0
    failures = 0
    warnings = 0
    for trial in results:
        for item in trial.get("iterations") or []:
            total_iterations += 1
            if item.get("status") == "reviewed":
                reviewed += 1
            for finding in item.get("findings") or []:
                if finding.get("severity") == "fail":
                    failures += 1
                elif finding.get("severity") == "warn":
                    warnings += 1
    return {
        "visible_iterations": total_iterations,
        "reviewed_iterations": reviewed,
        "failed_checks": failures,
        "warning_checks": warnings,
    }


def _run_success_smoke() -> int:
    fixture = next(item for item in get_contract_fixtures() if item.get("mock_review"))
    packet = build_review_packet_from_fixture(fixture)
    review = review_packet_with_provider(packet, provider=PROVIDER_MOCK)
    state: dict[str, Any] = {}
    trace = store_review_trace(
        state,
        packet=packet,
        review_result=review,
        session_id="eval-success-smoke",
    )
    findings, comments = _grade_trace(trace, None, None)
    summary = _trace_summary(trace, None, [], findings, comments)
    errors = []
    if summary.get("status") != "reviewed":
        errors.append(f"expected reviewed status, got {summary.get('status')}")
    if summary.get("design_confidence") is None:
        errors.append("expected design_confidence in success summary")
    if summary.get("raw_output_json") is None:
        errors.append("expected raw_output_json in success summary")
    if not summary.get("validated_review"):
        errors.append("expected validated_review in success summary")
    if not summary.get("input_packet"):
        errors.append("expected input_packet in success summary")
    if "Scenario Review response contract" not in str(summary.get("provider_prompt") or ""):
        errors.append("expected provider_prompt in success summary")

    controlled_packet = _attach_review_controls(packet, SCENARIO_STEPS[2])
    controlled_review = _apply_review_control_overrides(controlled_packet, review)
    controlled_state: dict[str, Any] = {}
    controlled_trace = store_review_trace(
        controlled_state,
        packet=controlled_packet,
        review_result=controlled_review,
        session_id="eval-success-smoke-controls",
    )
    controlled_summary = _trace_summary(controlled_trace, SCENARIO_STEPS[2], [], [], [])
    controlled_completion = (
        (controlled_summary.get("validated_review") or {})
        .get("completion_outlook_analysis", {})
        .get("risk_pattern_summary")
    )
    pre_control_completion = (
        (controlled_summary.get("pre_control_validated_review") or controlled_summary.get("pre_control_output_json") or {})
        .get("completion_outlook_analysis", {})
        .get("risk_pattern_summary")
    )
    if controlled_packet.get("input_hash") == packet.get("input_hash"):
        errors.append("expected review_controls to change the packet input_hash")
    if (controlled_packet.get("review_controls") or {}).get("completion_outlook_mode") != "fixed_planning_assumption_boundary":
        errors.append("expected fixed planning-assumption review control in controlled packet")
    if controlled_completion != OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY:
        errors.append("expected control override to normalize Completion Outlook boundary wording")
    if not pre_control_completion:
        errors.append("expected pre-control Completion Outlook to be preserved")
    if (controlled_summary.get("provider_metadata") or {}).get("review_control_override") != "fixed_planning_assumption_boundary":
        errors.append("expected provider_metadata review_control_override marker")
    if errors:
        for error in errors:
            print(f"success smoke failed: {error}", file=sys.stderr)
        return 1
    print("Validated eval harness success path and review_controls override path with fixture-backed mock review.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["configured", PROVIDER_MOCK, PROVIDER_GEMINI, PROVIDER_OPENAI], default="configured")
    parser.add_argument("--max-trials", type=int, default=2, help="Number of first-wave trials to run. Use 8 for the full first wave.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-dotenv", action="store_true", help="Do not load local .env values.")
    parser.add_argument("--skip-baseline-review", action="store_true", help="Skip hidden baseline provider calls.")
    parser.add_argument("--success-smoke", action="store_true", help="Run a fixture-backed valid-review smoke without external provider calls.")
    args = parser.parse_args()

    if args.success_smoke:
        return _run_success_smoke()

    taxonomy = _load_taxonomy()
    registry = pd.read_csv(REGISTRY_PATH)
    selected_trials = _select_trials(registry, taxonomy, max(1, args.max_trials))
    env = _merged_env(load_dotenv=not args.no_dotenv)
    config = load_narrative_provider_config(env)
    cache_namespace = provider_config_cache_namespace(config) if args.provider == "configured" else None
    run_id = args.run_id or datetime.now(timezone.utc).strftime("first_wave_%Y%m%d_%H%M%S")

    if args.provider == "configured" and not (config.provider_available() or config.fallback_available()):
        print(
            f"Configured provider `{config.provider}` and fallback `{config.fallback_provider}` are not available. "
            "Set GEMINI_API_KEY/GOOGLE_API_KEY or OPENAI_API_KEY, or run with --provider mock.",
            file=sys.stderr,
        )
        return 2

    results = [
        _run_trial(
            row,
            taxonomy=taxonomy,
            provider=args.provider,
            config=config,
            cache_namespace=cache_namespace,
            include_baseline_review=not args.skip_baseline_review,
        )
        for row in selected_trials
    ]
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "provider_config": config.sanitized_trace_metadata(),
        "scenario_steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "expectations": step.expectations,
            }
            for step in SCENARIO_STEPS
        ],
        "summary": _summary(results),
        "trials": results,
    }
    json_path = args.out_dir / f"{run_id}.json"
    md_path = args.out_dir / f"{run_id}.md"
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
