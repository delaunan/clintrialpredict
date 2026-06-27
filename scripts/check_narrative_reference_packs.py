#!/usr/bin/env python3
from pathlib import Path
import json
import sys


BASE = Path("frontend/data/docs/narrative_reference_packs")
MANIFEST_PATH = BASE / "pack_manifest_v1.json"

REQUIRED_SECTIONS = [
    "## Source",
    "## When To Use",
    "## Key Principles",
    "## Relevance To Simulator Pillars",
    "## Do Not Infer",
    "## Prompt-Safe Summary",
]

ALLOWED_RUNTIME_SUFFIXES = {".md", ".json"}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    if not BASE.exists():
        fail(f"Folder not found: {BASE}")

    if not MANIFEST_PATH.exists():
        fail(f"Manifest not found: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    pack_ids = set()

    for pack in manifest.get("packs", []):
        pack_id = pack.get("pack_id")
        filename = pack.get("filename")

        if not pack_id:
            fail("Manifest pack entry missing pack_id")

        if not filename:
            fail(f"Manifest pack entry missing filename for {pack_id}")

        if pack_id in pack_ids:
            fail(f"Duplicate pack_id in manifest: {pack_id}")

        pack_ids.add(pack_id)

        path = BASE / filename

        if not path.exists():
            fail(f"Manifest references missing file: {filename}")

        text = path.read_text(encoding="utf-8")

        expected_title = f"# Pack ID: {pack_id}"
        if expected_title not in text:
            fail(f"{filename} missing expected title: {expected_title}")

        for section in REQUIRED_SECTIONS:
            if section not in text:
                fail(f"{filename} missing required section: {section}")

        source_section = text.split("## When To Use", 1)[0]
        if "URL:" not in source_section and "URLs:" not in source_section:
            fail(f"{filename} Source section missing URL or URLs field")

    for path in BASE.iterdir():
        if path.is_file() and path.suffix not in ALLOWED_RUNTIME_SUFFIXES:
            fail(f"Unexpected runtime file type in reference pack folder: {path.name}")

    print("Narrative reference pack check passed")
    print(f"Checked {len(pack_ids)} manifest packs")


if __name__ == "__main__":
    main()
