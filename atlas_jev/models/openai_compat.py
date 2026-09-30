"""OpenAI-compatible backend.

One implementation covers vLLM, Groq, Together, OpenRouter, llama.cpp's server,
and Ollama's own ``/v1`` shim, because they all speak ``/chat/completions``.
Point ``TARGET_BASE_URL`` at the host and set ``TARGET_MODEL``.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from atlas_jev.models.base import GenerationResult, TargetModel


class OpenAICompatTarget(TargetModel):
    backend = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 180.0,
    ) -> None:
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health_check(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            return False, f"cannot reach {self.base_url}/models ({exc})"
        return True, f"endpoint reachable at {self.base_url}"

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
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "seed": kwargs.get("seed", 0),
        }

        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            detail = ""
            response = getattr(exc, "response", None)
            if response is not None:
                detail = f" body={response.text[:300]}"
            return GenerationResult(
                text="",
                model=self.name,
                backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}{detail}",
            )

        latency_ms = (time.perf_counter() - started) * 1000
        choices = body.get("choices") or [{}]
        usage = body.get("usage") or {}
        return GenerationResult(
            text=(choices[0].get("message") or {}).get("content") or "",
            model=body.get("model", self.name),
            backend=self.backend,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw={"finish_reason": choices[0].get("finish_reason"), "id": body.get("id")},
        )
