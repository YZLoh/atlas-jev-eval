"""Environment-driven configuration.

Credentials come from the environment or a local ``.env``. Nothing is
hard-coded, and a missing key is a hard failure rather than a silent
fallback to a mock evaluator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
FIGURES_DIR = RESULTS_DIR / "figures"
TEST_CASES_DIR = REPO_ROOT / "test_cases"


class ConfigError(RuntimeError):
    """Raised when required configuration is absent or unusable."""


def load_dotenv_if_present() -> None:
    """Load ``.env`` from the repo root if python-dotenv is installed."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    load_dotenv(env_path, override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is not None:
        value = value.strip()
    return value or None


def require_typesafe_key() -> str:
    """Return the TypeSafe API key, or explain precisely how to supply one."""
    load_dotenv_if_present()
    key = _env("TYPESAFE_API_KEY")
    if not key:
        raise ConfigError(
            "TYPESAFE_API_KEY is not set.\n"
            "  1. cp .env.example .env\n"
            "  2. add your key from https://typesafe.ai\n"
            "Alternatively export it for this shell:\n"
            "  PowerShell:  $env:TYPESAFE_API_KEY = 'sk-...'\n"
            "  bash:        export TYPESAFE_API_KEY=sk-...\n"
            "This project deliberately does not fall back to a mock evaluator, "
            "because fabricated Jev answers would invalidate every metric it reports."
        )
    return key


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for the LLM under test."""

    backend: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 512
    ollama_host: str = "http://localhost:11434"
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> TargetConfig:
        load_dotenv_if_present()
        return cls(
            backend=(_env("TARGET_BACKEND") or "ollama").lower(),
            model=_env("TARGET_MODEL") or "llama3.1:8b",
            temperature=float(_env("TARGET_TEMPERATURE") or 0.0),
            max_tokens=int(_env("TARGET_MAX_TOKENS") or 512),
            ollama_host=_env("OLLAMA_HOST") or "http://localhost:11434",
            base_url=_env("TARGET_BASE_URL"),
            api_key=_env("TARGET_API_KEY"),
        )


@dataclass(frozen=True)
class JevConfig:
    """Configuration for the Jev evaluator."""

    model: str | None = None
    timeout: float = 30.0
    max_retries: int = 2
    # Decision threshold applied to a Noul probability to obtain a boolean.
    # 0.5 is the neutral default; calibration.py sweeps this rather than
    # assuming any single value is correct.
    noul_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> JevConfig:
        load_dotenv_if_present()
        return cls(
            model=_env("TYPESAFE_DEFAULT_MODEL"),
            timeout=float(_env("JEV_TIMEOUT") or 30.0),
            max_retries=int(_env("JEV_MAX_RETRIES") or 2),
            noul_threshold=float(_env("JEV_NOUL_THRESHOLD") or 0.5),
        )

    def threshold_repr(self) -> str:
        return f"{self.noul_threshold:.2f} (applied to Noul p(yes) to get a boolean)"
