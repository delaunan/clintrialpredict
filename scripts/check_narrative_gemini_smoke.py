#!/usr/bin/env python
"""Optional Gemini smoke test for narrative provider configuration.

This script is opt-in because it uses network access and may spend API credits.
Set RUN_NARRATIVE_GEMINI_SMOKE=1 to execute the API call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.provider_config import PROVIDER_GEMINI  # noqa: E402
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "status": {"type": "STRING"},
        "purpose": {"type": "STRING"},
    },
    "required": ["status", "purpose"],
}


def _parse_json_text(text: str) -> dict | None:
    text = str(text or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_response_payload(response: object) -> dict | None:
    parsed_payload = getattr(response, "parsed", None)
    if isinstance(parsed_payload, dict):
        return parsed_payload

    text = str(getattr(response, "text", "") or "").strip()
    return _parse_json_text(text)


def main() -> int:
    load_dotenv()
    config = load_narrative_provider_config(os.environ)
    gemini_settings = config.provider_settings(PROVIDER_GEMINI)

    if os.getenv("RUN_NARRATIVE_GEMINI_SMOKE") != "1":
        print("Skipped Gemini smoke test. Set RUN_NARRATIVE_GEMINI_SMOKE=1 to call the API.")
        return 0

    if not gemini_settings or not gemini_settings.has_api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini smoke test.")
        return 1

    generation_config = types.GenerateContentConfig(
        temperature=config.temperature,
        max_output_tokens=min(config.max_output_tokens, 1000),
        seed=config.seed,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )
    client = genai.Client(api_key=gemini_settings.api_key)

    try:
        response = client.models.generate_content(
            model=gemini_settings.model,
            contents=(
                "Return exactly one valid compact JSON object, with no markdown and no prose. Shape: "
                "{\"status\":\"ok\",\"purpose\":\"narrative_provider_smoke_test\"}"
            ),
            config=generation_config,
        )
    except Exception as exc:
        print(f"ERROR: Gemini smoke test request failed: {exc.__class__.__name__}: {str(exc)[:400]}")
        return 1

    parsed = _parse_response_payload(response)
    if parsed is None:
        text = str(getattr(response, "text", "") or "").strip()
        print(f"ERROR: Gemini smoke test model output was not JSON: {text[:300]}")
        return 1

    if parsed.get("status") != "ok":
        print(f"ERROR: Gemini smoke test returned unexpected payload: {parsed}")
        return 1

    print(f"Validated Gemini smoke test with model {gemini_settings.model}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
