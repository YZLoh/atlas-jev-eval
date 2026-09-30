"""Ollama backend - a local open-weight model as the target under test.

Uses Ollama's native ``/api/chat`` endpoint. Requires a running daemon:

    ollama serve
    ollama pull llama3.1:8b
"""

from __future__ import annotations

import time
from typing import Any

import requests

from atlas_jev.models.base import GenerationResult, TargetModel


class OllamaTarget(TargetModel):
    backend = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 180.0,
    ) -> None:
        self.name = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def health_check(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            return False, (
                f"cannot reach Ollama at {self.host} ({exc}). "
                "Start it with `ollama serve`."
            )
        available = [m.get("name", "") for m in resp.json().get("models", [])]
        if not available:
            return False, f"Ollama is running but has no models. Try `ollama pull {self.name}`."
        # Ollama reports tags as `name:tag`; accept a bare name matching any tag.
        if self.name not in available and not any(
            m.split(":")[0] == self.name.split(":")[0] for m in available
        ):
            return False, (
                f"model {self.name!r} is not pulled. Available: {', '.join(available)}. "
                f"Run `ollama pull {self.name}`."
            )
        return True, f"Ollama reachable at {self.host}, {len(available)} model(s) available"

    def generate(
        self, prompt: str, *, system: str | None = None, **kwargs: Any
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                # Fixed seed so a temperature-0 run is reproducible across machines.
                "seed": kwargs.get("seed", 0),
            },
        }

        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            return GenerationResult(
                text="",
                model=self.name,
                backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        except ValueError as exc:  # malformed JSON
            return GenerationResult(
                text="",
                model=self.name,
                backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"invalid JSON from Ollama: {exc}",
            )

        latency_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=(body.get("message") or {}).get("content", ""),
            model=body.get("model", self.name),
            backend=self.backend,
            latency_ms=latency_ms,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            raw={k: v for k, v in body.items() if k != "message"},
        )
