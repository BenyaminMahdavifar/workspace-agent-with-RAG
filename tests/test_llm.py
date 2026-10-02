import sys
from types import ModuleType

from agent_cli.config import AgentConfig
from agent_cli.llm import ChatCompletionClient


def test_chat_client_uses_openai_compatible_configuration(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeCompletions:
        def create(self, *, model, messages):
            calls["request"] = (model, messages)
            message = type("Message", (), {"content": "hello"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            calls["client"] = (base_url, api_key)
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    openai = ModuleType("openai")
    openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )
    client = ChatCompletionClient(config)

    assert client.complete([{"role": "user", "content": "hello"}]) == "hello"
    assert calls["client"] == ("https://provider.example/v1", "provider-secret")
    assert calls["request"] == (
        "provider-model",
        [{"role": "user", "content": "hello"}],
    )
