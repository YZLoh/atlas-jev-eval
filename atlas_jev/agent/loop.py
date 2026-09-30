"""A minimal tool-calling agent over the mock environment.

Deliberately small: enough turns to let an agent search, read, and act, and no
more. It exists to answer one question - can an adversarial prompt cause a tool
to be invoked in a way the operator did not intend - not to be a general agent
framework.

Model-generated content is **never executed**. Tool names are matched against a
fixed dispatch table and unknown names are recorded and refused; arguments are
passed as data to functions that only append to a list.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from atlas_jev.agent.tools import TOOL_SCHEMAS, ToolCall, ToolEnvironment

# 5xx bodies that indicate the MODEL failed, not the server. These are
# reproducible under fixed decoding and must not be retried away.
DEGENERATION_MARKERS = ("token repeat limit", "prediction aborted")


@dataclass
class AgentResult:
    """What the agent said, what it invoked, and how long it took."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    environment: ToolEnvironment | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_json() for c in self.tool_calls],
            "turns": self.turns,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "environment": self.environment.summary() if self.environment else None,
        }


class OllamaAgent:
    """Runs a tool-calling loop against a local Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_turns: int = 6,
        timeout: float = 180.0,
        max_retries: int = 2,
        backoff: float = 1.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_turns = max_turns
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def _chat(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
        """One /api/chat call, retrying transient failures.

        Ollama intermittently returns 500 on a long tool-calling exchange. A
        single dropped turn would otherwise be recorded as a target error and
        silently shrink the sample, so transient 5xx and connection failures are
        retried with backoff. A 4xx is a real problem with the request and is
        returned immediately rather than hammered.
        """
        last_error = "unknown failure"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": TOOL_SCHEMAS,
                        "stream": False,
                        "options": {"temperature": self.temperature, "seed": 0},
                    },
                    timeout=self.timeout,
                )
                if 400 <= response.status_code < 500:
                    return None, f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code >= 500:
                    detail = response.text[:300]
                    # Distinguish the model degenerating from infrastructure
                    # failing. "token repeat limit reached" means greedy
                    # decoding fell into a loop - at temperature 0 with a fixed
                    # seed that is perfectly deterministic, so retrying just
                    # reproduces it. Retrying with a different seed would hide
                    # a real property of the target behind non-reproducibility.
                    if any(marker in detail.lower() for marker in DEGENERATION_MARKERS):
                        return None, (
                            f"model degeneration (not retried - deterministic at "
                            f"temperature={self.temperature}, seed=0): {detail}"
                        )
                    last_error = f"HTTP {response.status_code}: {detail}"
                    if attempt < self.max_retries:
                        time.sleep(self.backoff * (2**attempt))
                        continue
                    return None, f"{last_error} (after {self.max_retries} retries)"
                return response.json(), None
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
        return None, f"{last_error} (after {self.max_retries} retries)"

    def run(self, prompt: str, system: str | None = None) -> AgentResult:
        environment = ToolEnvironment()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        text = ""
        turns = 0

        for turn in range(self.max_turns):
            turns = turn + 1
            body, failure = self._chat(messages)
            if body is None:
                return AgentResult(
                    text=text,
                    tool_calls=environment.calls,
                    turns=turns,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=failure,
                    environment=environment,
                )

            message = body.get("message") or {}
            content = message.get("content") or ""
            if content:
                text = content
            requested = message.get("tool_calls") or []

            if not requested:
                break

            messages.append(message)
            for call in requested:
                function = call.get("function") or {}
                name = function.get("name", "")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    # Some backends return the argument object as a JSON string.
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw": arguments}
                result = environment.invoke(name, arguments)
                messages.append(
                    {"role": "tool", "content": result, "name": name}
                )

        return AgentResult(
            text=text,
            tool_calls=environment.calls,
            turns=turns,
            latency_ms=(time.perf_counter() - started) * 1000,
            environment=environment,
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"cannot reach Ollama at {self.host} ({exc})"
        return True, f"agent backend ready ({self.model})"
