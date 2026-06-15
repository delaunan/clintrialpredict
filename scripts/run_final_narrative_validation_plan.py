#!/usr/bin/env python
"""Run or print the final narrative validation wave.

The wave separates three review goals:

- boundary behavior checks for unusual latest-change patterns;
- credible storyline candidates for later one-shot example selection;
- duplicate runs for reproducibility/drift inspection under identical inputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _command_to_text(command: list[str]) -> str:
    return " ".join(command)


def _run(command: list[str], *, execute: bool) -> int:
    print(_command_to_text(command))
    if not execute:
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gemini", choices=("configured", "mock", "gemini", "openai"))
    parser.add_argument("--temperature", default="omit")
    parser.add_argument("--gemini-thinking-level", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--boundary-trials", type=int, default=10)
    parser.add_argument("--storyline-trials", type=int, default=12)
    parser.add_argument("--repro-trials", type=int, default=3)
    parser.add_argument("--boundary-run-id", default="final_validation_boundary_10_1")
    parser.add_argument("--storyline-run-id", default="final_validation_storyline_candidates_12_1")
    parser.add_argument("--repro-a-run-id", default="final_validation_repro_3_a")
    parser.add_argument("--repro-b-run-id", default="final_validation_repro_3_b")
    parser.add_argument("--comparison-id", default="final_validation_repro_3_comparison")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the commands. Without this flag, only print them.",
    )
    args = parser.parse_args()

    base = [
        sys.executable,
        "scripts/run_narrative_eval_suite.py",
        "--provider",
        args.provider,
        "--temperature",
        args.temperature,
        "--gemini-thinking-level",
        args.gemini_thinking_level,
    ]

    boundary = base + [
        "--scenario-plan",
        "boundary",
        "--max-trials",
        str(args.boundary_trials),
        "--run-id",
        args.boundary_run_id,
    ]
    storyline = base + [
        "--scenario-plan",
        "storyline",
        "--max-trials",
        str(args.storyline_trials),
        "--run-id",
        args.storyline_run_id,
    ]
    repro_a = base + [
        "--scenario-plan",
        "storyline",
        "--max-trials",
        str(args.repro_trials),
        "--run-id",
        args.repro_a_run_id,
    ]
    repro_b = base + [
        "--scenario-plan",
        "storyline",
        "--max-trials",
        str(args.repro_trials),
        "--run-id",
        args.repro_b_run_id,
    ]
    compare = [
        sys.executable,
        "scripts/compare_narrative_temperature_reports.py",
        "--report",
        f"boundary=reports/narrative_evals/{args.boundary_run_id}.json",
        "--report",
        f"storyline=reports/narrative_evals/{args.storyline_run_id}.json",
        "--repro-a",
        f"reports/narrative_evals/{args.repro_a_run_id}.json",
        "--repro-b",
        f"reports/narrative_evals/{args.repro_b_run_id}.json",
        "--comparison-id",
        args.comparison_id,
    ]

    if not args.execute:
        print("# Dry run. Add --execute to run these commands.")
    for command in (boundary, storyline, repro_a, repro_b, compare):
        return_code = _run(command, execute=args.execute)
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
