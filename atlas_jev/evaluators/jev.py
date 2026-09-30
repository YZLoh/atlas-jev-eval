"""Jev evaluator - TypeSafe System One as a structured judge.

Verified against typesafe-sdk 0.7.2. Two properties of the API shape this file
and, downstream, the whole calibration analysis:

1. A ``Noul`` answer carries **only** ``noul``: a float in [0, 1] giving
   p(yes). There is no confidence field. The probability *is* the signal, so a
   boolean requires an explicit threshold, and that threshold is a free
   parameter this project sweeps rather than assumes.

2. ``Choice`` and ``Score`` answers carry ``confidence``, which the vendor
   documents as a deterministic function of the returned distribution,
   ``(n * peak - 1) / (n - 1)``. It is a *dispersion statistic*, not a
   validated probability of being correct. We record it, and we test whether it
   predicts correctness at all, rather than assuming it does.

Every answer therefore keeps its raw distribution alongside the thresholded
decision, and ``confidence_source`` states where the number came from.
"""

from __future__ import annotations

import time
from typing import Any

from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeClient,
    TypeSafeError,
)
from typing_extensions import Self

from atlas_jev.config import JevConfig, require_typesafe_key
from atlas_jev.evaluators.base import (
    AnswerRecord,
    EvaluationRecord,
    Evaluator,
    QuestionSpec,
)


def build_question(spec: QuestionSpec) -> Noul | Choice | Score:
    """Translate a scenario's declared question into a TypeSafe primitive."""
    if spec.kind == "bool":
        kwargs: dict[str, Any] = {"instructions": spec.instructions}
        if isinstance(spec.criteria, dict) and spec.criteria:
            # NoulCriteria is a TypedDict with optional `true` / `false` keys.
            criteria = {k: v for k, v in spec.criteria.items() if k in ("true", "false")}
            if criteria:
                kwargs["criteria"] = criteria
        return Noul(**kwargs)

    if spec.kind == "choice":
        assert isinstance(spec.criteria, dict)
        return Choice(instructions=spec.instructions, criteria=spec.criteria)

    if spec.kind == "score":
        assert isinstance(spec.criteria, list)
        return Score(instructions=spec.instructions, criteria=spec.criteria)

    raise ValueError(f"unsupported question kind: {spec.kind!r}")


def derived_certainty(probability: float) -> float:
    """Distance of a Noul probability from maximal uncertainty, mapped to [0, 1].

    The API does not return a confidence for Noul answers. This is our own
    derivation, ``2 * |p - 0.5|``, recorded with ``confidence_source="derived"``
    so it is never mistaken for a vendor-supplied calibrated value.
    """
    return min(1.0, max(0.0, 2 * abs(probability - 0.5)))


class JevEvaluator(Evaluator):
    """Structured evaluator backed by TypeSafe's System One endpoint."""

    name = "jev"

    def __init__(self, config: JevConfig | None = None) -> None:
        self.config = config or JevConfig.from_env()
        # Fails loudly and with instructions if the key is absent.
        api_key = require_typesafe_key()
        self.client = TypeSafeClient(
            api_key=api_key,
            model=self.config.model,
            retry=RetryPolicy(
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            ),
        )
        self.threshold = self.config.noul_threshold

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def evaluate(
        self, state: dict[str, Any], questions: list[QuestionSpec]
    ) -> EvaluationRecord:
        if not questions:
            raise ValueError("at least one question is required")

        # All questions for one response go in a single request. The vendor
        # recommends this, and it also means one latency figure per response
        # rather than a figure confounded by question count.
        payload = {spec.name: build_question(spec) for spec in questions}
        by_name = {spec.name: spec for spec in questions}

        started = time.perf_counter()
        try:
            response = self.client.system_one(state=state, questions=payload)
        except TypeSafeError as exc:
            return EvaluationRecord(
                evaluator=self.name,
                model=self.config.model,
                answers={},
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - started) * 1000

        answers: dict[str, AnswerRecord] = {}
        for name, answer in response.answers.items():
            spec = by_name.get(name)
            kind = spec.kind if spec else {"noul": "bool"}.get(answer.type, answer.type)
            answers[name] = self._to_record(name, kind, answer)

        return EvaluationRecord(
            evaluator=self.name,
            model=response.model,
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response.request_id,
        )

    def _to_record(self, name: str, kind: str, answer: Any) -> AnswerRecord:
        if answer.type == "noul":
            probability = float(answer.noul)
            return AnswerRecord(
                name=name,
                kind="bool",
                decision=probability >= self.threshold,
                probability=probability,
                confidence=derived_certainty(probability),
                confidence_source="derived",  # not returned by the API
                raw={"noul": probability, "threshold": self.threshold},
            )

        if answer.type == "choice":
            return AnswerRecord(
                name=name,
                kind="choice",
                decision=answer.choice,
                probability=max(answer.probabilities.values()) if answer.probabilities else None,
                confidence=float(answer.confidence),
                confidence_source="api",  # dispersion statistic, not validated calibration
                probabilities=dict(answer.probabilities),
                raw={"choice": answer.choice},
            )

        if answer.type == "score":
            return AnswerRecord(
                name=name,
                kind="score",
                decision=float(answer.score),
                probability=max(answer.probabilities.values()) if answer.probabilities else None,
                confidence=float(answer.confidence),
                confidence_source="api",
                probabilities={str(k): float(v) for k, v in answer.probabilities.items()},
                legend=dict(answer.legend),
                raw={"score": float(answer.score)},
            )

        return AnswerRecord(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            decision=None,
            raw={"unrecognised_answer_type": answer.type},
        )
