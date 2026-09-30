"""Experiment runner: case -> target -> evaluators -> raw record.

The runner is deliberately dumb. It orchestrates and records; it does not judge.
Every record it writes contains enough information to recompute every metric
offline, so metrics can be revised without re-running the target model or
spending another Jev call.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_jev import __version__
from atlas_jev.config import RAW_DIR
from atlas_jev.evaluators.base import EvaluationRecord, Evaluator, QuestionSpec
from atlas_jev.models.base import GenerationResult, TargetModel
from atlas_jev.scenarios.atlas import atlas_version
from atlas_jev.scenarios.schema import CaseSuite, Fixture, TestCase


def build_state(
    attack_prompt: str,
    model_response: str,
    case: TestCase | Fixture,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """The structured state handed to every evaluator.

    Identical for Jev and the baseline, so neither gets privileged information.
    Note what is *absent*: the expected behaviour is included (the evaluator
    needs to know the policy it is judging against) but the ground-truth label
    never is, and nor is whether the case is a control. Leaking either would
    turn the evaluation into a lookup.
    """
    state: dict[str, Any] = {
        "attack_prompt": attack_prompt,
        "model_response": model_response,
        "atlas_technique": f"{case.atlas.technique_id} {case.atlas.technique}",
        "atlas_tactic": case.atlas.tactic,
    }
    if isinstance(case, TestCase):
        state["expected_behaviour"] = case.expected.target_behaviour
    if system_prompt:
        state["system_prompt_given_to_target"] = system_prompt
    return state


@dataclass
class RunRecord:
    """One complete observation: what was asked, what happened, who judged it."""

    case_id: str
    scenario: str
    variant: str
    bed: str  # "fixture" (evaluator-only) or "live" (target model ran)
    atlas_technique_id: str
    atlas_technique: str
    atlas_tactic: str
    is_control: bool
    attack_prompt: str
    model_response: str
    response_hash: str
    target: dict[str, Any]
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_behaviour: str | None = None
    # Immutable hand label, present for fixtures only.
    fixture_labels: dict[str, Any] | None = None
    # Agent-mode only. `tool_calls` is what the agent actually invoked, and
    # `tool_environment` is the mechanically determined outcome - whether a
    # secret left via a tool call. Unlike the text scenarios, this does not
    # need a human to adjudicate.
    tool_calls: list[dict[str, Any]] | None = None
    tool_environment: dict[str, Any] | None = None
    timestamp: str = ""
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _evaluation_to_json(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "evaluator": record.evaluator,
        "model": record.model,
        "latency_ms": record.latency_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "request_id": record.request_id,
        "error": record.error,
        "answers": {
            name: {
                "kind": answer.kind,
                "decision": answer.decision,
                "probability": answer.probability,
                "confidence": answer.confidence,
                "confidence_source": answer.confidence_source,
                "probabilities": answer.probabilities,
                "legend": answer.legend,
                "raw": answer.raw,
            }
            for name, answer in record.answers.items()
        },
    }


class ExperimentRunner:
    def __init__(
        self,
        suite: CaseSuite,
        evaluators: list[Evaluator],
        target: TargetModel | None = None,
    ) -> None:
        self.suite = suite
        self.evaluators = evaluators
        self.target = target
        # Optional pre-built agent; the agent path constructs one on demand.
        self.agent: Any = None
        # Cold-start cost of the first target call, recorded in metadata so it
        # is visible without contaminating per-case latency.
        self.cold_start_ms: float | None = None

    # -- provenance ---------------------------------------------------------
    def run_metadata(self) -> dict[str, Any]:
        return {
            "atlas_jev_version": __version__,
            "atlas_version": atlas_version(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": (
                {"backend": self.target.backend, "model": self.target.name}
                if self.target
                else None
            ),
            "evaluators": [e.name for e in self.evaluators],
            "target_cold_start_ms": self.cold_start_ms,
        }

    # -- bed A: fixtures ----------------------------------------------------
    def run_fixtures(self, fixtures: Iterable[Fixture]) -> list[RunRecord]:
        """Evaluate fixed responses that already carry immutable human labels."""
        records: list[RunRecord] = []
        for fixture in fixtures:
            questions = self.suite.questions_for(fixture)
            state = build_state(fixture.attack_prompt, fixture.response, fixture)
            record = RunRecord(
                case_id=fixture.id,
                scenario=fixture.scenario,
                variant=fixture.variant,
                bed="fixture",
                atlas_technique_id=fixture.atlas.technique_id,
                atlas_technique=fixture.atlas.technique,
                atlas_tactic=fixture.atlas.tactic,
                # A control is a case containing NO attack. It is emphatically
                # not "a case whose attack failed" - deriving it from the
                # attack_success label would make attack success rate 100% by
                # construction, and would mislabel every successfully-defended
                # attack as though no attack had been attempted.
                is_control=fixture.variant == "control",
                attack_prompt=fixture.attack_prompt,
                model_response=fixture.response,
                response_hash=fixture.response_hash,
                target={"backend": "fixture", "model": "n/a", "latency_ms": 0.0},
                fixture_labels=dict(fixture.labels),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._evaluate_all(record, state, questions)
            records.append(record)
        return records

    # -- bed B: live target -------------------------------------------------
    def run_cases(self, cases: Iterable[TestCase]) -> list[RunRecord]:
        if self.target is None:
            raise RuntimeError("a target model is required to run live cases")

        records: list[RunRecord] = []
        for case in cases:
            questions = self.suite.questions_for(case)
            system = case.input.system
            if case.input.context:
                system = f"{system or ''}\n\n{case.input.context}".strip()

            if case.mode == "agent":
                records.append(self._run_agent_case(case, system, questions))
                continue

            generation = self.target.generate(case.input.prompt, system=system)
            record = RunRecord(
                case_id=case.id,
                scenario=case.scenario,
                variant=case.variant,
                bed="live",
                atlas_technique_id=case.atlas.technique_id,
                atlas_technique=case.atlas.technique,
                atlas_tactic=case.atlas.tactic,
                is_control=case.is_control,
                attack_prompt=case.input.prompt,
                model_response=generation.text,
                response_hash=generation.response_hash,
                target=_target_to_json(generation),
                expected_behaviour=case.expected.target_behaviour,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            if not generation.ok:
                # Record the failure and move on. A missing response is not
                # evidence about the evaluator, so no evaluation is attempted.
                record.error = f"target failed: {generation.error}"
                records.append(record)
                continue

            state = build_state(case.input.prompt, generation.text, case, system_prompt=system)
            self._evaluate_all(record, state, questions)
            records.append(record)
        return records

    def _run_agent_case(
        self, case: TestCase, system: str | None, questions: list[QuestionSpec]
    ) -> RunRecord:
        """Run one tool-calling case against the sandboxed mock environment."""
        from atlas_jev.agent import OllamaAgent

        agent = self.agent or OllamaAgent(model=getattr(self.target, "name", "llama3.2:3b"))
        result = agent.run(case.input.prompt, system=system)

        # What the evaluator judges for an agent case is the transcript of what
        # the agent *did*, not just what it said. A tool invocation is the
        # action; the reply text may not mention it at all - indeed the
        # injection in doc_003 explicitly instructs it not to.
        transcript = _agent_transcript(result)

        record = RunRecord(
            case_id=case.id,
            scenario=case.scenario,
            variant=case.variant,
            bed="live",
            atlas_technique_id=case.atlas.technique_id,
            atlas_technique=case.atlas.technique,
            atlas_tactic=case.atlas.tactic,
            is_control=case.is_control,
            attack_prompt=case.input.prompt,
            model_response=transcript,
            response_hash=_hash_text(transcript),
            target={
                "backend": "ollama-agent",
                "model": agent.model,
                "latency_ms": result.latency_ms,
                "turns": result.turns,
                "error": result.error,
            },
            expected_behaviour=case.expected.target_behaviour,
            tool_calls=[c.to_json() for c in result.tool_calls],
            tool_environment=result.environment.summary() if result.environment else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if not result.ok:
            record.error = f"agent failed: {result.error}"
            return record

        state = build_state(case.input.prompt, transcript, case, system_prompt=system)
        state["tool_calls"] = record.tool_calls
        self._evaluate_all(record, state, questions)
        return record

    def _evaluate_all(
        self, record: RunRecord, state: dict[str, Any], questions: list[QuestionSpec]
    ) -> None:
        for evaluator in self.evaluators:
            started = time.perf_counter()
            try:
                evaluation = evaluator.evaluate(state, questions)
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                record.evaluations[evaluator.name] = {
                    "evaluator": evaluator.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "answers": {},
                }
                continue
            record.evaluations[evaluator.name] = _evaluation_to_json(evaluation)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _agent_transcript(result: Any) -> str:
    """Render what the agent DID plus what it said, for the evaluator to judge.

    An agent case cannot be judged from the reply text alone: the action is the
    tool call, and a successful exfiltration may leave no trace in the reply -
    the planted instruction in doc_003 explicitly says not to mention it.
    """
    lines: list[str] = []
    for call in result.tool_calls:
        arguments = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
        lines.append(f"TOOL CALL: {call.tool}({arguments})")
    lines.append(f"ASSISTANT REPLY: {result.text}" if result.text else "ASSISTANT REPLY: (none)")
    return "\n".join(lines)


def _target_to_json(generation: GenerationResult) -> dict[str, Any]:
    return {
        "backend": generation.backend,
        "model": generation.model,
        "latency_ms": generation.latency_ms,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "error": generation.error,
    }


def write_results(
    records: list[RunRecord], metadata: dict[str, Any], path: Path | None = None
) -> Path:
    """Write one JSONL file. First line is metadata, one line per observation."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RAW_DIR / f"results_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_metadata": metadata}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    # A stable pointer to the most recent run, for `evaluate` and `report`.
    (RAW_DIR / "latest.txt").write_text(path.name, encoding="utf-8")
    return path


def load_results(path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read back a results file as ``(metadata, records)``."""
    if path is None:
        pointer = RAW_DIR / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"no run found in {RAW_DIR}. Run `python -m atlas_jev run --all` first."
            )
        path = RAW_DIR / pointer.read_text(encoding="utf-8").strip()

    metadata: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if "_metadata" in data:
            metadata = data["_metadata"]
        else:
            records.append(data)
    return metadata, records
