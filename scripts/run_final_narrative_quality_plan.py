#!/usr/bin/env python
"""Run or print the final-settings narrative quality/reproducibility plan."""

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
    parser.add_argument("--temperature", default="omit", help="Temperature override passed to run_narrative_eval_suite.py.")
    parser.add_argument("--quality-trials", type=int, default=10)
    parser.add_argument("--repro-trials", type=int, default=3)
    parser.add_argument("--quality-run-id", default="final_settings_quality_10_1")
    parser.add_argument("--repro-a-run-id", default="final_settings_repro_3_a")
    parser.add_argument("--repro-b-run-id", default="final_settings_repro_3_b")
    parser.add_argument("--comparison-id", default="final_settings_quality_repro_comparison")
    parser.add_argument(
        "--gemini-thinking-level",
        choices=("low", "medium", "high"),
        default="high",
        help="Gemini primary thinking level for this plan. Defaults to the final-settings candidate.",
    )
    parser.add_argument("--execute", action="store_true", help="Run the commands. Without this flag, only print them.")
    args = parser.parse_args()

    base = [
        sys.executable,
        "scripts/run_narrative_eval_suite.py",
        "--provider",
        args.provider,
        "--temperature",
        args.temperature,
    ]
    base.extend(["--gemini-thinking-level", args.gemini_thinking_level])

    quality = base + ["--max-trials", str(args.quality_trials), "--run-id", args.quality_run_id]
    repro_a = base + ["--max-trials", str(args.repro_trials), "--run-id", args.repro_a_run_id]
    repro_b = base + ["--max-trials", str(args.repro_trials), "--run-id", args.repro_b_run_id]
    compare = [
        sys.executable,
        "scripts/compare_narrative_temperature_reports.py",
        "--report",
        f"quality=reports/narrative_evals/{args.quality_run_id}.json",
        "--repro-a",
        f"reports/narrative_evals/{args.repro_a_run_id}.json",
        "--repro-b",
        f"reports/narrative_evals/{args.repro_b_run_id}.json",
        "--comparison-id",
        args.comparison_id,
    ]

    if not args.execute:
        print("# Dry run. Add --execute to run these commands.")
    for command in (quality, repro_a, repro_b, compare):
        return_code = _run(command, execute=args.execute)
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
