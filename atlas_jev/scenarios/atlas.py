"""ATLAS mapping validation.

A stated technique id is only worth something if it is real and if the tactic
actually belongs to it. This module checks every mapping in the suite against a
vendored subset of the official MITRE ATLAS data, so a typo or a wishful
mapping fails at load time rather than quietly appearing in a results table.

The subset in ``test_cases/atlas_subset.json`` records the upstream URL and the
ATLAS version it was taken from. Regenerate it with ``scripts/refresh_atlas.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from atlas_jev.config import TEST_CASES_DIR


@dataclass(frozen=True)
class AtlasTechnique:
    id: str
    name: str
    tactic_ids: tuple[str, ...]
    tactic_names: tuple[str, ...]
    description: str


@lru_cache(maxsize=1)
def load_atlas(path: Path | None = None) -> tuple[dict[str, AtlasTechnique], dict[str, str]]:
    """Return ``(techniques_by_id, tactic_name_by_id)`` plus provenance side effects."""
    path = path or TEST_CASES_DIR / "atlas_subset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tactics: dict[str, str] = data["tactics"]
    techniques = {
        tid: AtlasTechnique(
            id=tid,
            name=rec["name"],
            tactic_ids=tuple(rec.get("tactics", [])),
            tactic_names=tuple(tactics[t] for t in rec.get("tactics", []) if t in tactics),
            description=rec.get("description", ""),
        )
        for tid, rec in data["techniques"].items()
    }
    return techniques, tactics


def atlas_version(path: Path | None = None) -> str:
    path = path or TEST_CASES_DIR / "atlas_subset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("_provenance", {}).get("atlas_version", "unknown"))


def validate_mapping(technique_id: str, technique: str, tactic: str) -> list[str]:
    """Return a list of problems with one mapping. Empty means it checks out."""
    techniques, _ = load_atlas()
    problems: list[str] = []

    known = techniques.get(technique_id)
    if known is None:
        problems.append(
            f"technique_id {technique_id!r} is not in the vendored ATLAS subset "
            f"(known: {', '.join(sorted(techniques))})"
        )
        return problems

    if technique.strip().lower() != known.name.lower():
        problems.append(
            f"{technique_id} is named {known.name!r} in ATLAS, but the case says {technique!r}"
        )

    if tactic.strip().lower() not in {t.lower() for t in known.tactic_names}:
        problems.append(
            f"{technique_id} ({known.name}) belongs to tactic(s) "
            f"{', '.join(known.tactic_names)}, but the case claims {tactic!r}"
        )

    return problems
