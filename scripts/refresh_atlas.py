#!/usr/bin/env python
"""Regenerate test_cases/atlas_subset.json from the official MITRE ATLAS data.

The repository vendors a small subset of ATLAS rather than the full 450 KB file,
so mapping validation works offline and the exact data every result was checked
against is version-controlled alongside the results.

Vendoring is only defensible if it is reproducible, which is what this script
is for: it re-derives the subset from upstream and records the ATLAS version and
retrieval date in the file's ``_provenance`` block.

    python scripts/refresh_atlas.py

If upstream has changed a technique's name or tactic, the refresh will surface
it the next time the suite loads - existing cases will fail validation rather
than silently carry a stale mapping.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "test_cases" / "atlas_subset.json"
SOURCE = "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml"
SOURCE_HUMAN = "https://github.com/mitre-atlas/atlas-data/blob/main/dist/ATLAS.yaml"

# The techniques this project maps to, plus the near neighbours cited in the
# mapping rationales. Add an id here when a new scenario needs it.
KEEP = {
    "AML.T0051",  # LLM Prompt Injection            - scenario 1
    "AML.T0054",  # LLM Jailbreak                   - cited to contrast with T0051
    "AML.T0056",  # Extract LLM System Prompt       - scenario 3 (planned)
    "AML.T0057",  # LLM Data Leakage                - scenario 4 (planned)
    "AML.T0068",  # LLM Prompt Obfuscation          - scenario 2
    "AML.T0069",  # Discover LLM System Information - cited alongside T0056
    "AML.T0086",  # Exfiltration via AI Agent Tool Invocation - scenario 5 (planned)
}

DESCRIPTION_CHARS = 400


def main() -> int:
    print(f"fetching {SOURCE}")
    try:
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"FAIL  could not fetch ATLAS data: {exc}", file=sys.stderr)
        return 1

    data = yaml.safe_load(response.text)
    matrix = data["matrices"][0]
    tactics = {t["id"]: t["name"] for t in matrix["tactics"]}

    techniques: dict[str, dict] = {}
    for technique in matrix["techniques"]:
        if technique["id"] not in KEEP:
            continue
        techniques[technique["id"]] = {
            "name": technique["name"],
            "tactics": technique.get("tactics", []),
            "description": " ".join(
                technique.get("description", "").split()
            )[:DESCRIPTION_CHARS],
        }

    missing = KEEP - set(techniques)
    if missing:
        print(
            f"FAIL  upstream no longer contains: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        return 1

    subset = {
        "_provenance": {
            "source": SOURCE_HUMAN,
            "atlas_version": data.get("version"),
            "retrieved": datetime.now(timezone.utc).date().isoformat(),
            "note": (
                "Subset covering only the techniques this project maps to. "
                "Regenerate with scripts/refresh_atlas.py."
            ),
        },
        "tactics": tactics,
        "techniques": techniques,
    }

    previous_version = None
    if OUTPUT.exists():
        previous_version = json.loads(OUTPUT.read_text(encoding="utf-8")).get(
            "_provenance", {}
        ).get("atlas_version")

    OUTPUT.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  ATLAS version : {data.get('version')}"
          + (f"  (was {previous_version})" if previous_version else ""))
    print(f"  techniques    : {len(techniques)}")
    if previous_version and previous_version != data.get("version"):
        print("\nATLAS version changed. Re-run the test suite: a renamed technique or a")
        print("reassigned tactic will now fail mapping validation rather than pass silently.")
        print("    python -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
