"""MOCK TARGET - FOR AUTOMATED TESTS ONLY.

This backend fabricates target-model responses. It exists so the harness,
metrics, and plotting code can be tested without a GPU or a network call.

It must never be used to produce results that are reported as findings, and
the runner stamps ``backend="mock"`` into every record so that any results file
built from it is self-evidently not a real experiment.

There is deliberately no mock Jev evaluator anywhere in this project: fabricated
evaluator answers would silently corrupt the exact quantity under study.
"""

from __future__ import annotations

import time
from typing import Any

from atlas_jev.models.base import GenerationResult, TargetModel

# Canned behaviours keyed by a marker substring in the prompt, so tests can
# force a compliant or refusing target deterministically.
_REFUSAL = (
    "I can't help with that. It would involve bypassing safety guidance "
    "I'm meant to follow."
)
_COMPLIANCE = "Sure. Step 1: disable the logging subsystem. Step 2: escalate to root."


class MockTargetModel(TargetModel):
    backend = "mock"

    def __init__(self, model: str = "mock-deterministic", default: str = "refuse") -> None:
        self.name = model
        self.default = default

    def health_check(self) -> tuple[bool, str]:
        return True, "mock backend - fabricated responses, tests only"

    def generate(
        self, prompt: str, *, system: str | None = None, **kwargs: Any
    ) -> GenerationResult:
        started = time.perf_counter()
        lowered = prompt.lower()
        if "[[mock:comply]]" in lowered:
            text = _COMPLIANCE
        elif "[[mock:refuse]]" in lowered:
            text = _REFUSAL
        elif "[[mock:error]]" in lowered:
            return GenerationResult(
                text="",
                model=self.name,
                backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000,
                error="simulated backend failure",
            )
        else:
            text = _REFUSAL if self.default == "refuse" else _COMPLIANCE
        return GenerationResult(
            text=text,
            model=self.name,
            backend=self.backend,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )
