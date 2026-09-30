"""ATLAS mappings must be real, not plausible-looking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_jev.scenarios.atlas import atlas_version, load_atlas, validate_mapping
from atlas_jev.scenarios.schema import AtlasMapping


def test_vendored_atlas_has_provenance():
    techniques, _tactics = load_atlas()
    assert techniques, "vendored ATLAS subset is empty"
    assert atlas_version() != "unknown", "ATLAS version must be recorded for reproducibility"


def test_known_good_mapping_passes():
    assert validate_mapping("AML.T0051", "LLM Prompt Injection", "Execution") == []


def test_unknown_technique_id_is_rejected():
    problems = validate_mapping("AML.T9999", "Invented Technique", "Execution")
    assert problems and "not in the vendored ATLAS subset" in problems[0]


def test_wrong_tactic_is_rejected():
    """AML.T0051 is Execution, not Initial Access.

    This is the specific error that prompted the validator: prompt injection is
    widely described as an initial-access technique in informal writing, but
    ATLAS assigns it to Execution.
    """
    problems = validate_mapping("AML.T0051", "LLM Prompt Injection", "Initial Access")
    assert any("Initial Access" in p and "Execution" in p for p in problems)


def test_wrong_technique_name_is_rejected():
    problems = validate_mapping("AML.T0056", "LLM Meta Prompt Extraction", "Exfiltration")
    assert any("Extract LLM System Prompt" in p for p in problems)


def test_schema_rejects_invalid_mapping_at_construction():
    with pytest.raises(ValidationError):
        AtlasMapping(
            tactic="Initial Access",
            technique="Prompt Injection",
            technique_id="AML.T0051",
            rationale="wrong on both name and tactic",
        )


def test_every_shipped_case_has_a_valid_mapping():
    """The suite loading at all proves this, but assert it explicitly."""
    from atlas_jev.scenarios import load_suite

    suite = load_suite()
    assert suite.cases, "no cases loaded"
    for case in suite.cases:
        assert validate_mapping(
            case.atlas.technique_id, case.atlas.technique, case.atlas.tactic
        ) == [], f"{case.id} has an invalid mapping"
        assert case.atlas.rationale.strip(), f"{case.id} has no mapping rationale"
