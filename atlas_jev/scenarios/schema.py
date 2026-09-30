"""Test cases as data.

Nothing about an attack is expressed in Python control flow: a case is a YAML
record describing its ATLAS mapping, its input, the behaviour a safe model
should show, and the typed questions to ask about whatever the model does.
Adding an attack means adding YAML.

Two kinds of record exist, and the distinction is the methodological core of
this project:

``TestCase``
    A live case. The target model is actually run, so the *correct* answer
    depends on what the model happened to say. Ground truth therefore cannot be
    stored on the case; it is recorded per response in ``labels.jsonl``, keyed by
    a hash of the response text (see :mod:`atlas_jev.evaluation.labels`).

``Fixture``
    An evaluator-only case: a fixed ``(attack_prompt, response)`` pair with an
    immutable hand-authored label. No target model is involved, so evaluator
    accuracy is measured against genuine ground truth and is fully reproducible.
    Fixtures are where Jev can be shown to be wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from atlas_jev.config import TEST_CASES_DIR
from atlas_jev.evaluators.base import QuestionSpec

ExpectedBehaviour = Literal["refuse", "comply", "deflect", "answer_normally"]


class AtlasMapping(BaseModel):
    """How a case maps onto MITRE ATLAS.

    ``rationale`` is required on purpose. A mapping that cannot be justified in
    a sentence is a superficial mapping, and this project would rather have five
    defensible ones than fifteen decorative ones.
    """

    tactic: str
    technique: str
    technique_id: str
    rationale: str
    precondition: str | None = None

    @model_validator(mode="after")
    def _validate_against_atlas(self) -> AtlasMapping:
        """Reject mappings that do not match the official ATLAS data.

        This is a hard failure by design. An unverified technique id sitting in
        a results table is worse than no id at all.
        """
        from atlas_jev.scenarios.atlas import validate_mapping

        problems = validate_mapping(self.technique_id, self.technique, self.tactic)
        if problems:
            raise ValueError("invalid ATLAS mapping: " + "; ".join(problems))
        return self


class CaseInput(BaseModel):
    prompt: str
    system: str | None = None
    # Synthetic context injected into the system prompt, e.g. fake secrets.
    # Must never contain real credentials.
    context: str | None = None


class CaseExpectation(BaseModel):
    target_behaviour: ExpectedBehaviour
    # Whether complying with this prompt would constitute a policy violation.
    # For benign control cases this is False, which is what makes evaluator
    # false positives measurable.
    violation_if_complied: bool = True
    notes: str | None = None


class TestCase(BaseModel):
    """A live adversarial test case."""

    id: str
    scenario: str
    variant: str = "direct"
    # "text"  - single prompt, single response (scenarios 1-4)
    # "agent" - tool-calling loop against the mock environment (scenario 5)
    mode: Literal["text", "agent"] = "text"
    description: str = ""
    atlas: AtlasMapping
    input: CaseInput
    expected: CaseExpectation
    # Name of a shared question set, or an inline list of question definitions.
    questions: str | list[dict[str, Any]]
    tags: list[str] = Field(default_factory=list)

    # A case where a safe model should comply and nothing is violated is a
    # *control*. Controls are essential: without them an evaluator that always
    # answers "attack succeeded" would score perfectly on recall.
    @property
    def is_control(self) -> bool:
        return not self.expected.violation_if_complied


class Fixture(BaseModel):
    """A fixed response with an immutable human label (evaluator-only test bed)."""

    id: str
    scenario: str
    variant: str = "direct"
    atlas: AtlasMapping
    attack_prompt: str
    response: str
    questions: str | list[dict[str, Any]]
    # Hand-authored ground truth, keyed by question name.
    labels: dict[str, Any]
    # Why this fixture is labelled the way it is. Fixtures designed to be hard
    # (a refusal that quotes the secret, compliance wrapped in fiction) should
    # say so, because they are the informative ones.
    rationale: str = ""
    difficulty: Literal["easy", "hard", "adversarial"] = "easy"

    @property
    def response_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.response.encode("utf-8")).hexdigest()[:16]


class CaseSuite(BaseModel):
    """A loaded YAML suite: shared question sets plus cases."""

    version: int = 1
    question_sets: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    cases: list[TestCase] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_question_refs(self) -> CaseSuite:
        for record in [*self.cases, *self.fixtures]:
            if isinstance(record.questions, str) and record.questions not in self.question_sets:
                raise ValueError(
                    f"{record.id}: unknown question set {record.questions!r}. "
                    f"Defined sets: {sorted(self.question_sets)}"
                )
        ids = [r.id for r in [*self.cases, *self.fixtures]]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
        return self

    def questions_for(self, record: TestCase | Fixture) -> list[QuestionSpec]:
        """Resolve a record's question set into concrete question specs."""
        raw = (
            self.question_sets[record.questions]
            if isinstance(record.questions, str)
            else record.questions
        )
        return [
            QuestionSpec(
                name=item["name"],
                kind=item["type"],
                instructions=item["instructions"],
                criteria=item.get("criteria"),
            )
            for item in raw
        ]

    def select(
        self, scenario: str | None = None, variant: str | None = None
    ) -> list[TestCase]:
        cases = self.cases
        if scenario:
            cases = [c for c in cases if c.scenario == scenario]
        if variant:
            cases = [c for c in cases if c.variant == variant]
        return cases

    @property
    def scenarios(self) -> list[str]:
        return sorted({c.scenario for c in self.cases} | {f.scenario for f in self.fixtures})


def load_suite(path: Path | None = None) -> CaseSuite:
    """Load the case suite from YAML, merging in fixtures if present."""
    path = path or TEST_CASES_DIR / "atlas_cases.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no test case file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    fixtures_path = path.parent / "fixtures.yaml"
    if fixtures_path.exists() and not data.get("fixtures"):
        fixture_data = yaml.safe_load(fixtures_path.read_text(encoding="utf-8")) or {}
        data["fixtures"] = fixture_data.get("fixtures", [])
        # Fixtures may reference question sets defined in either file.
        merged = dict(fixture_data.get("question_sets") or {})
        merged.update(data.get("question_sets") or {})
        data["question_sets"] = merged

    return CaseSuite.model_validate(data)

