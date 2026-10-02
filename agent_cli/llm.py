from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import AgentConfig


CHAT_API_MAX_ATTEMPTS = 3


class ToolCallingUnsupported(RuntimeError):
    """Raised when a provider explicitly rejects Chat Completions tool calls."""


@dataclass(frozen=True)
class ChatToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str | None
    tool_calls: tuple[ChatToolCall, ...] = ()

    def as_assistant_message(self) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant"}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ChatCompletionClient:
    def __init__(self, config: AgentConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI client is missing. Install the project dependencies first."
            ) from exc

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            max_retries=CHAT_API_MAX_ATTEMPTS - 1,
        )
        self._model = config.model

    def complete(
        self,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, object]] | None = None,
    ) -> ChatCompletionResult:
        request: dict[str, object] = {
            "model": self._model,
            "messages": list(messages),
        }
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            if tools and self._tool_calls_not_supported(exc):
                raise ToolCallingUnsupported(
                    "The selected provider or model rejected native tool calling."
                ) from exc
            raise

        message = response.choices[0].message
        content = message.content
        normalized_content = content.strip() if isinstance(content, str) else None
        tool_calls: list[ChatToolCall] = []
        for call in getattr(message, "tool_calls", None) or ():
            function = call.function
            tool_calls.append(
                ChatToolCall(
                    id=call.id,
                    name=function.name,
                    arguments=function.arguments,
                )
            )
        if normalized_content is None and not tool_calls:
            raise RuntimeError("The chat completion returned neither text nor tool calls.")
        return ChatCompletionResult(normalized_content, tuple(tool_calls))

    @staticmethod
    def _tool_calls_not_supported(error: Exception) -> bool:
        try:
            from openai import BadRequestError
        except ImportError:
            return False
        if not isinstance(error, BadRequestError):
            return False
        message = str(error).lower()
        mentions_tools = "tool" in message or "function" in message
        indicates_unsupported = any(
            marker in message
            for marker in (
                "not support",
                "unsupported",
                "does not support",
                "unknown parameter",
                "unrecognized request argument",
            )
        )
        return mentions_tools and indicates_unsupported
