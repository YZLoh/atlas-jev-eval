"""Backend factory. Selects a :class:`TargetModel` from configuration."""

from __future__ import annotations

from atlas_jev.config import ConfigError, TargetConfig
from atlas_jev.models.base import TargetModel

KNOWN_BACKENDS = ("ollama", "openai_compat", "mock")


def build_target(config: TargetConfig | None = None) -> TargetModel:
    """Construct the configured target model."""
    config = config or TargetConfig.from_env()

    if config.backend == "ollama":
        from atlas_jev.models.ollama import OllamaTarget

        return OllamaTarget(
            model=config.model,
            host=config.ollama_host,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.backend == "openai_compat":
        if not config.base_url:
            raise ConfigError(
                "TARGET_BACKEND=openai_compat requires TARGET_BASE_URL "
                "(for example http://localhost:8000/v1)."
            )
        from atlas_jev.models.openai_compat import OpenAICompatTarget

        return OpenAICompatTarget(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.backend == "mock":
        from atlas_jev.models.mock import MockTargetModel

        return MockTargetModel(model=config.model or "mock-deterministic")

    raise ConfigError(
        f"unknown TARGET_BACKEND={config.backend!r}. "
        f"Expected one of: {', '.join(KNOWN_BACKENDS)}."
    )
