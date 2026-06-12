#!/usr/bin/env python
"""Export the exact Scenario Review prompt context for inspection.

This script does not call an LLM provider. It reconstructs the prompt, packet,
response contract, selected reference packs, and sanitized provider settings
from an existing narrative contract fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - export still works with process env only.
    load_dotenv = None
else:
    load_dotenv(ROOT / ".env")

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.prompt_builder import (  # noqa: E402
    build_provider_prompt,
    gemini_response_schema,
    infer_prompt_mode,
    provider_response_contract,
)
from src.narratives.provider import (  # noqa: E402
    GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS,
    GEMINI_MIN_SCHEMA_OUTPUT_TOKENS,
    GEMINI_PRIMARY_THINKING_LEVEL,
    GEMINI_RETRY_OUTPUT_TOKENS,
    GEMINI_RETRY_THINKING_LEVEL,
    PROVIDER_VALIDATION_RETRY_ATTEMPTS,
)
from src.narratives.provider_config import (  # noqa: E402
    load_narrative_provider_config,
    provider_config_cache_namespace,
)

DEFAULT_FIXTURE = "operational_only_ambitious_enrollment_v2"


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _text_dump(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fixture_by_id(fixture_id: str) -> dict[str, Any]:
    fixtures = get_contract_fixtures()
    for fixture in fixtures:
        if fixture.get("fixture_id") == fixture_id:
            return fixture
    available = "\n".join(f"  - {item.get('fixture_id')}" for item in fixtures)
    raise SystemExit(f"Unknown fixture_id: {fixture_id}\n\nAvailable fixtures:\n{available}")


def _safe_provider_settings() -> dict[str, Any]:
    config = load_narrative_provider_config(os.environ)
    metadata = config.sanitized_trace_metadata()
    return {
        "sanitized_config": metadata,
        "cache_namespace": provider_config_cache_namespace(config),
        "gemini_runtime_overrides": {
            "minimum_schema_output_tokens": GEMINI_MIN_SCHEMA_OUTPUT_TOKENS,
            "primary_thinking_level": GEMINI_PRIMARY_THINKING_LEVEL,
            "retry_thinking_level": GEMINI_RETRY_THINKING_LEVEL,
            "retry_output_tokens": GEMINI_RETRY_OUTPUT_TOKENS,
            "malformed_json_retry_attempts": GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS,
        },
        "validation_retry_attempts": PROVIDER_VALIDATION_RETRY_ATTEMPTS,
        "notes": [
            "API keys and raw secret values are intentionally omitted.",
            "This export does not call OpenAI, Gemini, or the mock reviewer.",
            "Live-provider routing still depends on NARRATIVE_LIVE_REVIEW_ENABLED in the simulator UI.",
        ],
    }


def _walkthrough(
    *,
    fixture_id: str,
    prompt_mode: str,
    packet: dict[str, Any],
    prompt: str,
    out_dir: Path,
) -> str:
    reference_packs = [
        pack.get("pack_id")
        for pack in packet.get("reference_packs", [])
        if isinstance(pack, dict) and pack.get("pack_id")
    ]
    changed_fields = (packet.get("iteration_context") or {}).get("changed_fields") or []
    model = packet.get("model_interpretation") or {}
    structured = packet.get("structured_features") or {}
    text_context = packet.get("text_context") or {}
    operational = packet.get("operational_assumptions") or {}
    return f"""# Narrative Prompt Export Walkthrough

## Export

- Fixture: `{fixture_id}`
- Prompt mode: `{prompt_mode}`
- Input hash: `{packet.get("input_hash")}`
- Output folder: `{out_dir}`
- Prompt characters: `{len(prompt)}`

## Files

- `01_prompt.txt`: exact prompt text sent to a live provider.
- `02_packet.json`: deterministic scenario evidence packet embedded at the end of the prompt.
- `03_response_contract.json`: app-owned response contract included in the prompt.
- `04_selected_reference_packs.json`: prompt-safe reference-pack summaries included in the packet.
- `05_provider_settings.json`: sanitized provider/model/settings metadata and retry controls.
- `06_gemini_response_schema.json`: strict Gemini SDK response schema.
- `07_walkthrough.md`: this explanation.

## What To Read First

1. Start with `01_prompt.txt` to see the complete provider request.
2. Then inspect `02_packet.json` to understand the evidence available to the LLM.
3. Review `04_selected_reference_packs.json` to see the knowledge substrate actually attached.
4. Review `03_response_contract.json` and `06_gemini_response_schema.json` to understand output constraints.
5. Use `05_provider_settings.json` to inspect model, generation controls, retry policy, and cache namespace without exposing secrets.

## Current Scenario Evidence

- Completion Score: `{model.get("completion_score")}`
- Previous Completion Score: `{model.get("previous_completion_score")}`
- Score delta: `{model.get("score_delta")}`
- Changed fields: `{", ".join(changed_fields) if changed_fields else "none"}`
- Reference packs selected: `{", ".join(reference_packs) if reference_packs else "none"}`
- Structured feature count: `{len(structured)}`
- Text context fields: `{", ".join(sorted(text_context)) if text_context else "none"}`
- Operational assumptions: `{", ".join(sorted(operational)) if operational else "none"}`

## Architecture Lesson

The prompt is not hand-written free text alone. It is a composed artifact:

1. `packet_builder.py` determines what evidence exists.
2. `prompt_builder.py` determines how the LLM is instructed to reason and respond.
3. Provider code sends the prompt and normalizes JSON.
4. `scoring.py` validates evidence fields and calculates Design Confidence and Total Scenario Score.
5. `review_store.py` stores trace, cache, provider metadata, and compact continuity context.

The LLM should only produce structured ratings, evidence fields, rationale, narrative, and continuity fields. The application owns all point calculations.

## Prompt-Engineering Questions

- Is the role specific enough for the output you want?
- Are the reference-pack summaries sufficiently rich, or too generic?
- Are taxonomy meanings clear enough for non-obvious fields?
- Does the packet include all evidence needed for the desired narrative?
- Are output length limits too long or too short for live serious-game use?
- Does the response contract force helpful structure, or does it make the narrative too rigid?
- Are provider settings supporting the desired quality/cost/latency trade-off?
"""


def export_prompt_context(fixture_id: str, out_dir: Path) -> None:
    fixture = _fixture_by_id(fixture_id)
    packet = build_review_packet_from_fixture(fixture)
    prompt_mode = infer_prompt_mode(packet)
    prompt = build_provider_prompt(packet, prompt_mode=prompt_mode)
    contract = provider_response_contract()
    reference_packs = packet.get("reference_packs") or []
    provider_settings = _safe_provider_settings()
    gemini_schema = gemini_response_schema()
    walkthrough = _walkthrough(
        fixture_id=fixture_id,
        prompt_mode=prompt_mode,
        packet=packet,
        prompt=prompt,
        out_dir=out_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _text_dump(out_dir / "01_prompt.txt", prompt)
    _json_dump(out_dir / "02_packet.json", packet)
    _json_dump(out_dir / "03_response_contract.json", contract)
    _json_dump(out_dir / "04_selected_reference_packs.json", reference_packs)
    _json_dump(out_dir / "05_provider_settings.json", provider_settings)
    _json_dump(out_dir / "06_gemini_response_schema.json", gemini_schema)
    _text_dump(out_dir / "07_walkthrough.md", walkthrough)

    print(f"Exported narrative prompt context for fixture '{fixture_id}' to {out_dir}")
    print(f"Prompt mode: {prompt_mode}")
    print(f"Input hash: {packet.get('input_hash')}")
    print("Files:")
    for path in sorted(out_dir.iterdir()):
        print(f"  - {path}")


def list_fixtures() -> None:
    for fixture in get_contract_fixtures():
        print(fixture.get("fixture_id"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the exact Scenario Review prompt context for a narrative fixture.",
    )
    parser.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        help=f"Fixture ID to export. Default: {DEFAULT_FIXTURE}",
    )
    parser.add_argument(
        "--out",
        default="/tmp/narrative_prompt_export",
        help="Output directory. Default: /tmp/narrative_prompt_export",
    )
    parser.add_argument(
        "--list-fixtures",
        action="store_true",
        help="List available fixture IDs and exit.",
    )
    args = parser.parse_args()

    if args.list_fixtures:
        list_fixtures()
        return 0

    export_prompt_context(args.fixture, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
