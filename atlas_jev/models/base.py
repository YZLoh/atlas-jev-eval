"""The target-model interface.

Every backend implements :class:`TargetModel`, so the evaluation harness never
knows which provider produced a response. Adding vLLM, Hugging Face, or a
hosted API means adding one file here, not touching the evaluation code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """One response from the target model, with the provenance needed to reproduce it."""

    text: str
    model: str
    backend: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def response_hash(self) -> str:
        """Stable id for this response text.

        Human ground-truth labels are keyed by this hash, so a label stays
        attached to the exact text it was assigned to. If the target is re-run
        and produces different text, the label is correctly treated as missing
        rather than silently reused.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


class TargetModel(ABC):
    """An LLM under test."""

    name: str = "unknown"
    backend: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> GenerationResult:
        """Return the model's response to ``prompt``.

        Implementations must not raise for ordinary API failures. They return a
        ``GenerationResult`` with ``error`` set, so that a partial experiment is
        recorded honestly instead of aborting or, worse, silently substituting
        placeholder text.
        """

    def health_check(self) -> tuple[bool, str]:
        """Return ``(reachable, message)`` without consuming significant quota."""
        return True, "no health check implemented for this backend"

    def warmup(self) -> float | None:
        """Issue one throwaway request so the first measured call is not a cold start.

        Without this, a local backend reports the model-load time as the latency
        of whichever case happened to run first - observed at ~112 s against
        ~2.5 s for every subsequent call on the same model. That is a property
        of the measurement, not of the model, and leaving it in would corrupt
        every latency figure the experiment reports.

        Returns the cold-start cost in ms, which is recorded in run metadata
        rather than discarded - it is a real number, just not a per-case one.
        """
        result = self.generate("Reply with the single word: ready", system=None)
        return result.latency_ms if result.ok else None
