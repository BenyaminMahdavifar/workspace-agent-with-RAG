from __future__ import annotations

from collections.abc import Sequence

from .config import AgentConfig


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
        )
        self._model = config.model

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The chat completion returned an empty response.")
        return content.strip()
