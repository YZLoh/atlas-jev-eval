"""Evaluator interface shared by Jev and the baseline detectors.

A deliberate design point: questions are *data*. A scenario declares the
questions it wants answered, and every evaluator answers the same questions, so
Jev and the keyword baseline are compared on identical ground.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionKind = Literal["bool", "choice", "score"]


@dataclass
class QuestionSpec:
    """One typed question to ask about a target-model response.

    ``criteria`` mirrors the TypeSafe primitives:

    * ``bool``   -> optional ``{"true": ..., "false": ...}`` disambiguation
    * ``choice`` -> ``{option_name: description}``
    * ``score``  -> ordered list of level descriptions, lowest first
    """

    name: str
    kind: QuestionKind
    instructions: str
    criteria: dict[str, Any] | list[Any] | None = None

    def __post_init__(self) -> None:
        if self.kind == "choice" and not isinstance(self.criteria, dict):
            raise ValueError(f"choice question {self.name!r} requires a dict of options")
        if self.kind == "score" and not (isinstance(self.criteria, list) and self.criteria):
            raise ValueError(f"score question {self.name!r} requires a non-empty list of levels")


@dataclass
class AnswerRecord:
    """An evaluator's answer to one question, with the raw signal preserved.

    ``decision`` is the thresholded interpretation used for metrics.
    ``probability`` keeps the underlying continuous signal, because the
    calibration analysis needs the raw value, not a boolean.
    """

    name: str
    kind: QuestionKind
    decision: bool | str | float | None
    probability: float | None = None
    confidence: float | None = None
    confidence_source: str = "none"
    probabilities: dict[str, float] | None = None
    legend: dict[int, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    """The full outcome of evaluating one target response."""

    evaluator: str
    model: str | None
    answers: dict[str, AnswerRecord]
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Evaluator(ABC):
    """Something that judges what a target model did."""

    name: str = "unknown"

    @abstractmethod
    def evaluate(
        self, state: dict[str, Any], questions: list[QuestionSpec]
    ) -> EvaluationRecord:
        """Answer ``questions`` about ``state``.

        Must not raise for ordinary API failures - return a record with
        ``error`` set so a partial experiment is recorded honestly.
        """
