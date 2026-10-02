from pathlib import Path

import pytest

from agent_cli import cli
from agent_cli.agent import Agent
from agent_cli.config import AgentConfig
from agent_cli.tools import ToolManager
from agent_cli.workspace import WorkspaceTools


class FakeEmbedder:
    def __init__(self, model_name: str, token: str | None) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeKnowledge:
    def __init__(self, root: Path, embedder: FakeEmbedder) -> None:
        self.root = root

    def search(self, query: str):
        return []

    def record_error(self, error: str, context: str) -> str:
        raise AssertionError("No error should be recorded in CLI command tests.")


class FakeChatClient:
    def __init__(self, config: AgentConfig) -> None:
        pass

    def complete(self, messages):
        return "response"


def test_cli_uses_current_directory_as_default_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert cli._parse_args([]).workspace == tmp_path


def test_cli_help_exposes_workspace_option(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli._parse_args(["--help"])

    assert exit_info.value.code == 0
    assert "--workspace" in capsys.readouterr().out


def test_cli_parses_workspace_override(tmp_path: Path) -> None:
    args = cli._parse_args(["--workspace", str(tmp_path)])

    assert args.workspace == tmp_path


def test_main_supports_help_status_clear_and_exit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="test-chat",
        hf_api=None,
        embedding_model="test-embedding",
    )
    monkeypatch.setattr(cli, "_read_config", lambda: config)
    monkeypatch.setattr(cli, "HuggingFaceEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "ErrorKnowledgeBase", FakeKnowledge)
    monkeypatch.setattr(cli, "ChatCompletionClient", FakeChatClient)
    monkeypatch.setenv("AGENT_HOME", str(tmp_path / "agent-data"))
    inputs = iter(["/help", "/status", "/clear", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = cli.main(["--workspace", str(tmp_path)])
    output = capsys.readouterr().out

    assert result == 0
    assert "Commands:" in output
    assert f"Workspace: {tmp_path}" in output
    assert "Chat model: test-chat" in output
    assert "Generated tools: 0" in output
    assert "Conversation history cleared" in output
    assert "Goodbye." in output


def test_clear_command_only_resets_conversation(tmp_path: Path, capsys) -> None:
    workspace = WorkspaceTools(tmp_path)
    config = AgentConfig(
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="test-chat",
        hf_api=None,
    )
    agent = Agent(
        config,
        FakeChatClient(config),
        FakeKnowledge(tmp_path, FakeEmbedder(config.embedding_model, None)),
        ToolManager(tmp_path),
    )
    agent._history.append({"role": "user", "content": "old turn"})

    should_continue = cli._handle_command("/clear", agent, workspace, config)

    assert should_continue
    assert agent.turn_count == 0
    assert agent._history == []
    assert "Stored knowledge and generated tools were kept" in capsys.readouterr().out


def test_approval_requires_explicit_yes(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert cli._approve_action("Run a risky command")

    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    assert not cli._approve_action("Run a risky command")
