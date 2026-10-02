import pytest

from agent_cli.config import AgentConfig


def test_config_accepts_chat_and_embedding_settings() -> None:
    config = AgentConfig(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        hf_api=None,
        embedding_model="test/embeddings",
    )

    assert config.base_url == "https://example.invalid/v1"
    assert config.embedding_model == "test/embeddings"


@pytest.mark.parametrize("field", ["base_url", "api_key", "model", "embedding_model"])
def test_config_rejects_empty_required_setting(field: str) -> None:
    values = {
        "base_url": "https://example.invalid/v1",
        "api_key": "test-key",
        "model": "test-model",
        "hf_api": None,
        "embedding_model": "test/embeddings",
    }
    values[field] = " "

    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        AgentConfig(**values)
