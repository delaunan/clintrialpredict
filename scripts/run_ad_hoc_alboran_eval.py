#!/usr/bin/env python
"""Run the requested ad hoc ALBORAN narrative review scenario."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

import scripts.run_narrative_eval_suite as runner


RUN_ID = "ad_hoc_alboran_control_upgrade_1"


def main() -> int:
    taxonomy = runner._load_taxonomy()
    registry = pd.read_csv(runner.REGISTRY_PATH)
    matches = registry[registry["nct_id"].astype(str) == "NCT06992609"]
    if matches.empty:
        raise SystemExit("NCT06992609 not found in search registry.")
    row = matches.iloc[0]

    step = runner.ScenarioStep(
        step_id="alboran_factorial_randomized_control_upgrade",
        title="ALBORAN factorial randomized active-control upgrade",
        completion_delta=0.0,
        pillar_for_delta="Scientific Challenge",
        structured_edits={
            "number_of_arms_ml": 2,
            "intervention_model_ml": "FACTORIAL",
            "allocation_ml": "RANDOMIZED",
            "masking_ml": "DOUBLE",
            "comparator_benchmark_ml": "ACTIVE_LEGACY_STANDARD",
        },
        expectations={
            "expected_quality": (
                "Ad hoc ALBORAN scenario: review whether adding a second arm, "
                "factorial structure, randomization, double blinding, and active "
                "legacy comparator improves evidence controls while adding execution "
                "and governance complexity."
            )
        },
    )

    env = runner._merged_env(load_dotenv=True)
    config = runner.load_narrative_provider_config(env)
    cache_namespace = runner.provider_config_cache_namespace(config)

    trial_result = runner._run_trial(
        row,
        taxonomy=taxonomy,
        provider="configured",
        config=config,
        cache_namespace=cache_namespace,
        include_baseline_review=True,
        scenario_steps=(step,),
        scenario_plan="ad_hoc_alboran_control_upgrade",
        cumulative=True,
    )

    data = {
        "run_id": RUN_ID,
        "provider": "configured",
        "scenario_plan": {
            "name": "ad_hoc_alboran_control_upgrade",
            "description": "Ad hoc single-iteration ALBORAN control/evidence upgrade.",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "visible_iterations": sum(1 for it in trial_result["iterations"] if it.get("status") == "reviewed"),
        "reviewed_iterations": sum(1 for it in trial_result["iterations"] if it.get("status") == "reviewed"),
        "failed_checks": sum(
            1
            for it in trial_result["iterations"]
            for finding in it.get("findings", [])
            if finding.get("severity") == "fail"
        ),
        "warning_checks": sum(
            1
            for it in trial_result["iterations"]
            for finding in it.get("findings", [])
            if finding.get("severity") == "warn"
        ),
        "trials": [trial_result],
    }

    out_json = runner.DEFAULT_REPORT_DIR / f"{RUN_ID}.json"
    out_md = runner.DEFAULT_REPORT_DIR / f"{RUN_ID}.md"
    runner._write_json(out_json, data)
    runner._write_markdown(out_md, data)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(json.dumps({
        key: data[key]
        for key in ("failed_checks", "reviewed_iterations", "visible_iterations", "warning_checks")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
