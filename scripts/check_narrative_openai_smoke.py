#!/usr/bin/env python
"""Optional OpenAI smoke test for narrative provider configuration.

This script is opt-in because it uses network access and may spend API credits.
Set RUN_NARRATIVE_OPENAI_SMOKE=1 to execute the API call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.provider_config import PROVIDER_OPENAI  # noqa: E402
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _response_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def main() -> int:
    load_dotenv()
    config = load_narrative_provider_config(os.environ)
    openai_settings = config.provider_settings(PROVIDER_OPENAI)

    if os.getenv("RUN_NARRATIVE_OPENAI_SMOKE") != "1":
        print("Skipped OpenAI smoke test. Set RUN_NARRATIVE_OPENAI_SMOKE=1 to call the API.")
        return 0

    if not openai_settings or not openai_settings.has_api_key:
        print("ERROR: OPENAI_API_KEY is required for OpenAI smoke test.")
        return 1

    max_output_tokens = min(config.max_output_tokens, 200)
    payload = {
        "model": openai_settings.model,
        "input": (
            "Return only valid compact JSON with exactly this shape: "
            "{\"status\":\"ok\",\"purpose\":\"narrative_provider_smoke_test\"}"
        ),
        "max_output_tokens": max_output_tokens,
    }

    try:
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {openai_settings.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        print(f"ERROR: OpenAI smoke test request failed: {exc.__class__.__name__}")
        return 1

    if response.status_code != 200:
        detail = response.text[:600]
        print(f"ERROR: OpenAI smoke test returned HTTP {response.status_code}: {detail}")
        return 1

    try:
        response_payload = response.json()
    except json.JSONDecodeError:
        print("ERROR: OpenAI smoke test response was not JSON.")
        return 1

    text = _response_text(response_payload).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"ERROR: OpenAI smoke test model output was not JSON: {text[:300]}")
        return 1

    if parsed.get("status") != "ok":
        print(f"ERROR: OpenAI smoke test returned unexpected payload: {parsed}")
        return 1

    print(f"Validated OpenAI smoke test with model {openai_settings.model}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
