import sys
from types import ModuleType

from agent_cli.config import AgentConfig
from agent_cli.llm import (
    CHAT_API_MAX_ATTEMPTS,
    ChatCompletionClient,
    ChatToolCall,
    ToolCallingUnsupported,
)


def test_chat_client_uses_openai_compatible_configuration(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeCompletions:
        def create(self, *, model, messages):
            calls["request"] = (model, messages)
            message = type("Message", (), {"content": "hello"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeOpenAI:
        def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            max_retries: int,
        ) -> None:
            calls["client"] = (base_url, api_key, max_retries)
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

    assert client.complete([{"role": "user", "content": "hello"}]).content == "hello"
    assert calls["client"] == (
        "https://provider.example/v1",
        "provider-secret",
        CHAT_API_MAX_ATTEMPTS - 1,
    )
    assert calls["request"] == (
        "provider-model",
        [{"role": "user", "content": "hello"}],
    )


def test_chat_client_retries_transient_api_failures_three_times(monkeypatch) -> None:
    import httpx
    import openai
    from openai import OpenAI as RealOpenAI

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) < CHAT_API_MAX_ATTEMPTS:
            return httpx.Response(
                503,
                headers={"retry-after": "0"},
                json={"error": {"message": "temporarily unavailable"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "provider-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "recovered"},
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(respond))

    def mock_openai(
        *,
        base_url: str,
        api_key: str,
        max_retries: int,
    ):
        return RealOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            http_client=http_client,
        )

    monkeypatch.setattr(openai, "OpenAI", mock_openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )

    client = ChatCompletionClient(config)
    result = client.complete([{"role": "user", "content": "hello"}])
    http_client.close()

    assert result.content == "recovered"
    assert len(requests) == CHAT_API_MAX_ATTEMPTS


def test_chat_client_does_not_retry_non_transient_bad_request(monkeypatch) -> None:
    import httpx
    import openai
    from openai import BadRequestError, OpenAI as RealOpenAI

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            json={"error": {"message": "invalid request", "type": "invalid_request_error"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(respond))

    def mock_openai(
        *,
        base_url: str,
        api_key: str,
        max_retries: int,
    ):
        return RealOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            http_client=http_client,
        )

    monkeypatch.setattr(openai, "OpenAI", mock_openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )
    client = ChatCompletionClient(config)

    try:
        client.complete([{"role": "user", "content": "hello"}])
    except BadRequestError:
        pass
    else:
        raise AssertionError("A 400 response should raise BadRequestError.")
    finally:
        http_client.close()

    assert len(requests) == 1


def test_chat_client_stops_after_three_transient_failures(monkeypatch) -> None:
    import httpx
    import openai
    from openai import InternalServerError, OpenAI as RealOpenAI

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            503,
            headers={"retry-after": "0"},
            json={"error": {"message": "still unavailable"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(respond))

    def mock_openai(*, base_url: str, api_key: str, max_retries: int):
        return RealOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            http_client=http_client,
        )

    monkeypatch.setattr(openai, "OpenAI", mock_openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )
    client = ChatCompletionClient(config)

    try:
        client.complete([{"role": "user", "content": "hello"}])
    except InternalServerError:
        pass
    else:
        raise AssertionError("Persistent 503 responses should raise InternalServerError.")
    finally:
        http_client.close()

    assert len(requests) == CHAT_API_MAX_ATTEMPTS


def test_chat_client_preserves_native_tool_call_payload(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **request):
            calls["request"] = request
            function = type(
                "Function",
                (),
                {
                    "name": "list_files",
                    "arguments": '{"path":".","recursive":true}',
                },
            )()
            tool_call = type(
                "ToolCall",
                (),
                {"id": "call-1", "function": function},
            )()
            message = type(
                "Message",
                (),
                {"content": "Inspecting.", "tool_calls": [tool_call]},
            )()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )
    tool_definitions: list[dict[str, object]] = [
        {"type": "function", "function": {"name": "list_files"}}
    ]

    result = ChatCompletionClient(config).complete(
        [{"role": "user", "content": "inspect"}],
        tools=tool_definitions,
    )

    assert result.content == "Inspecting."
    assert result.tool_calls == (
        ChatToolCall("call-1", "list_files", '{"path":".","recursive":true}'),
    )
    assert calls["request"]["tools"] == tool_definitions
    assert calls["request"]["tool_choice"] == "auto"


def test_chat_client_identifies_provider_tool_call_rejection(monkeypatch) -> None:
    import httpx
    import openai
    from openai import OpenAI as RealOpenAI

    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model does not support tools",
                        "type": "invalid_request_error",
                    }
                },
            )
        )
    )

    def mock_openai(*, base_url: str, api_key: str, max_retries: int):
        return RealOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            http_client=http_client,
        )

    monkeypatch.setattr(openai, "OpenAI", mock_openai)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="provider-secret",
        model="provider-model",
        hf_api=None,
    )
    client = ChatCompletionClient(config)

    try:
        try:
            client.complete(
                [{"role": "user", "content": "inspect"}],
                tools=[{"type": "function", "function": {"name": "list_files"}}],
            )
        except ToolCallingUnsupported:
            pass
        else:
            raise AssertionError("Tool-calling rejection should be classified explicitly.")
    finally:
        http_client.close()
