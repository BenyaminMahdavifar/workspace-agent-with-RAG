from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class AgentConfig:
    base_url: str
    api_key: str
    model: str
    hf_api: str | None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        for name in ("base_url", "api_key", "model", "embedding_model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
