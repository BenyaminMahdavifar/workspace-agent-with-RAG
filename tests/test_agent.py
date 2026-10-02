from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_cli.agent import Agent, AgentError
from agent_cli.config import AgentConfig
from agent_cli.knowledge import ErrorKnowledgeBase
from agent_cli.llm import ChatCompletionResult, ChatToolCall, ToolCallingUnsupported
from agent_cli.tools import ToolManager
from agent_cli.workspace import WorkspaceTools


class FakeEmbedder:
    model_name = "fake-model"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 17 + 1), 1.0] for text in texts]


class FakeCompletionClient:
    def __init__(
        self,
        responses: list[str | ChatCompletionResult | Exception],
    ) -> None:
        self.responses = iter(responses)
        self.requests: list[tuple[list[dict[str, object]], list[dict[str, object]] | None]] = []

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> str | ChatCompletionResult:
        self.requests.append((messages.copy(), tools))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def action(kind: str, **values: object) -> str:
    return json.dumps({"type": kind, **values})


def build_agent(root: Path, responses: list[str]) -> Agent:
    config = AgentConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-api-key",
        model="chat-model",
        hf_api="secret-hf-token",
        embedding_model="fake-model",
    )
    return Agent(
        config=config,
        completion_client=FakeCompletionClient(responses),
        knowledge=ErrorKnowledgeBase(root, FakeEmbedder()),
        tools=ToolManager(root),
    )


def test_agent_records_invalid_tool_then_repairs_and_resolves_it(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        [
            action(
                "create_tool",
                name="calculator",
                description="Calculator",
                source="def run(input_data):\n    return eval('2 + 2')\n",
            ),
            action(
                "repair_tool",
                name="calculator",
                description="Adds two values.",
                source=(
                    "def run(input_data):\n"
                    '    return input_data["left"] + input_data["right"]\n'
                ),
            ),
            "The calculator tool is ready.",
        ],
    )

    response = agent.respond("Create an addition tool.")
    entries = list((tmp_path / "knowledge").glob("*.md"))

    assert response == "The calculator tool is ready."
    assert len(entries) == 1
    assert "- Status: resolved" in entries[0].read_text(encoding="utf-8")
    assert agent.tools.run("calculator", {"left": 4, "right": 5}) == 9


def test_agent_retests_repaired_tool_and_resolves_execution_error(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        [
            action(
                "create_tool",
                name="converter",
                description="Broken converter.",
                source="def run(input_data):\n    return 1 / 0\n",
            ),
            action("run_tool", name="converter", input={"value": 6}),
            action(
                "repair_tool",
                name="converter",
                description="Returns the input value.",
                source="def run(input_data):\n    return input_data['value']\n",
            ),
            "The converter now works.",
        ],
    )

    response = agent.respond("Create and run a converter.")
    entries = list((tmp_path / "knowledge").glob("*.md"))

    assert response == "The converter now works."
    assert len(entries) == 1
    assert "- Status: resolved" in entries[0].read_text(encoding="utf-8")


def test_agent_resolves_each_error_after_multiple_repair_attempts(
    tmp_path: Path,
) -> None:
    agent = build_agent(
        tmp_path,
        [
            action(
                "create_tool",
                name="converter",
                description="Broken converter.",
                source="def run(input_data):\n    return 1 / 0\n",
            ),
            action("run_tool", name="converter", input={"value": 6}),
            action(
                "repair_tool",
                name="converter",
                description="Still broken.",
                source="def run(input_data):\n    return 1 / 0\n",
            ),
            action(
                "repair_tool",
                name="converter",
                description="Returns the input value.",
                source="def run(input_data):\n    return input_data['value']\n",
            ),
            "The converter is fixed.",
        ],
    )

    assert agent.respond("Repair the converter.") == "The converter is fixed."
    entries = list((tmp_path / "knowledge").glob("*.md"))

    assert len(entries) == 2
    assert all("- Status: resolved" in entry.read_text(encoding="utf-8") for entry in entries)


def test_api_secrets_are_redacted_from_persisted_errors(tmp_path: Path) -> None:
    agent = build_agent(tmp_path, ["unused"])
    error_id = agent._record_error(
        "Failure with secret-api-key",
        "request included secret-hf-token",
    )
    content = (tmp_path / "knowledge" / f"{error_id}.md").read_text(encoding="utf-8")

    assert "secret-api-key" not in content
    assert "secret-hf-token" not in content
    assert "[REDACTED]" in content


def test_api_secrets_are_redacted_from_resolutions(tmp_path: Path) -> None:
    agent = build_agent(tmp_path, ["unused"])
    error_id = agent._record_error("Tool failed.", "No secret here.")
    agent._resolve_error(error_id, "Tool returned secret-api-key and secret-hf-token.")
    content = (tmp_path / "knowledge" / f"{error_id}.md").read_text(encoding="utf-8")

    assert "secret-api-key" not in content
    assert "secret-hf-token" not in content


def test_agent_uses_internal_workspace_file_actions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = AgentConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-api-key",
        model="chat-model",
        hf_api=None,
        embedding_model="fake-model",
    )
    agent = Agent(
        config=config,
        completion_client=FakeCompletionClient(
            [
                action(
                    "write_file",
                    path="hello.py",
                    content="print('hello')\n",
                ),
                "Created hello.py.",
            ]
        ),
        knowledge=ErrorKnowledgeBase(tmp_path / "agent-data", FakeEmbedder()),
        tools=ToolManager(tmp_path / "agent-data"),
        workspace=WorkspaceTools(project),
    )

    assert agent.respond("Create hello.py.") == "Created hello.py."
    assert (project / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_declined_workspace_command_is_not_run_or_stored_as_error(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "agent-data"
    prompts: list[str] = []
    config = AgentConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-api-key",
        model="chat-model",
        hf_api=None,
        embedding_model="fake-model",
    )
    agent = Agent(
        config=config,
        completion_client=FakeCompletionClient(
            [
                action("run_command", command=["echo", "hello"]),
                "The command was not run because approval was denied.",
            ]
        ),
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(
            project,
            approve=lambda prompt: prompts.append(prompt) or False,
        ),
    )

    response = agent.respond("Run echo hello.")

    assert "denied" in response.lower()
    assert prompts and "echo hello" in prompts[0]
    assert list((data_root / "knowledge").glob("*.md")) == []


def test_agent_reports_progress_for_workspace_file_listing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    config = AgentConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-api-key",
        model="chat-model",
        hf_api=None,
        embedding_model="fake-model",
    )
    agent = Agent(
        config=config,
        completion_client=FakeCompletionClient(
            [action("list_files", path=".", recursive=True), "Found hello.py."]
        ),
        knowledge=ErrorKnowledgeBase(tmp_path / "agent-data", FakeEmbedder()),
        tools=ToolManager(tmp_path / "agent-data"),
        workspace=WorkspaceTools(project),
    )
    progress: list[str] = []
    agent.progress = progress.append

    assert agent.respond("List project files.") == "Found hello.py."
    assert "Listing workspace files" in progress


def test_native_tool_calls_complete_a_multi_step_project_task(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('hello')\n", encoding="utf-8")
    data_root = tmp_path / "agent-data"
    client = FakeCompletionClient(
        [
            ChatCompletionResult(
                content="I will inspect the project.",
                tool_calls=(
                    ChatToolCall(
                        "call-list",
                        "list_files",
                        json.dumps({"path": ".", "recursive": True}),
                    ),
                ),
            ),
            ChatCompletionResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        "call-read",
                        "read_file",
                        json.dumps({"path": "app.py"}),
                    ),
                ),
            ),
            ChatCompletionResult(content="The project prints hello."),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(project),
    )

    assert agent.respond("Explain this project.") == "The project prints hello."
    assert len(client.requests) == 3
    assert all(request[1] for request in client.requests)
    next_turn_messages = client.requests[1][0]
    assert any(message.get("role") == "tool" for message in next_turn_messages)
    assert any(
        message.get("tool_call_id") == "call-list"
        and "app.py" in str(message.get("content"))
        for message in next_turn_messages
    )


def test_prose_with_embedded_action_is_not_executed_and_uses_strict_fallback(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "agent-data"
    client = FakeCompletionClient(
        [
            'I will inspect files.\n{"type":"list_files","path":".","recursive":true}',
            json.dumps(
                {
                    "type": "action",
                    "action": {
                        "type": "list_files",
                        "path": ".",
                        "recursive": True,
                    },
                }
            ),
            json.dumps({"type": "final", "content": "The workspace is empty."}),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(project),
    )

    assert agent.respond("Check this project.") == "The workspace is empty."
    assert client.requests[0][1] is not None
    assert client.requests[1][1] is None
    assert client.requests[2][1] is None
    assert any("action-looking JSON" in str(message.get("content", ""))
               for message in client.requests[1][0] if message.get("role") == "system")


def test_provider_tool_call_rejection_switches_to_json_protocol(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "agent-data"
    client = FakeCompletionClient(
        [
            ToolCallingUnsupported("provider rejects native tools"),
            json.dumps(
                {
                    "type": "action",
                    "action": {
                        "type": "list_files",
                        "path": ".",
                        "recursive": True,
                    },
                }
            ),
            json.dumps({"type": "final", "content": "Workspace inspection complete."}),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(project),
    )

    assert agent.respond("Inspect this project.") == "Workspace inspection complete."
    assert client.requests[0][1] is not None
    assert all(request[1] is None for request in client.requests[1:])


def test_fallback_requires_strict_action_or_final_json(tmp_path: Path) -> None:
    data_root = tmp_path / "agent-data"
    client = FakeCompletionClient(
        [
            json.dumps(
                {
                    "type": "action",
                    "action": {
                        "type": "list_files",
                        "path": ".",
                        "recursive": True,
                    },
                }
            ),
            "I have now listed the files successfully.",
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(tmp_path),
    )
    agent._native_tool_calling_supported = False

    with pytest.raises(AgentError, match="strict JSON"):
        agent.respond("List files.")


def test_denied_native_action_is_returned_as_denial_and_stops_further_actions(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "agent-data"
    approval_prompts: list[str] = []
    client = FakeCompletionClient(
        [
            ChatCompletionResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        "call-risky",
                        "run_command",
                        json.dumps({"command": ["echo", "hello"]}),
                    ),
                ),
            ),
            ChatCompletionResult(content="The task is incomplete because approval was denied."),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(
            project,
            approve=lambda prompt: approval_prompts.append(prompt) or False,
        ),
    )

    assert "incomplete" in agent.respond("Run echo hello.").lower()
    assert len(approval_prompts) == 1
    denial_message = next(
        message
        for message in client.requests[1][0]
        if message.get("role") == "tool" and message.get("tool_call_id") == "call-risky"
    )
    assert '"ok": false' in str(denial_message["content"]).lower()
    assert client.requests[1][1] is None
    assert list((data_root / "knowledge").glob("*.md")) == []


def test_denied_native_action_skips_remaining_calls_in_same_batch(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "private.txt").write_text("must not be read", encoding="utf-8")
    data_root = tmp_path / "agent-data"
    approval_prompts: list[str] = []
    client = FakeCompletionClient(
        [
            ChatCompletionResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        "call-risky",
                        "run_command",
                        json.dumps({"command": ["echo", "hello"]}),
                    ),
                    ChatToolCall(
                        "call-read",
                        "read_file",
                        json.dumps({"path": "private.txt"}),
                    ),
                ),
            ),
            ChatCompletionResult(content="The task is incomplete because approval was denied."),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(
            project,
            approve=lambda prompt: approval_prompts.append(prompt) or False,
        ),
    )

    assert "incomplete" in agent.respond("Run the command and read the file.").lower()
    assert len(approval_prompts) == 1
    messages = client.requests[1][0]
    skipped_read = next(
        message
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "call-read"
    )
    assert "Not run because an earlier action" in str(skipped_read["content"])
    assert "must not be read" not in str(skipped_read["content"])


def test_action_budget_exhaustion_is_explicitly_incomplete(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "agent-data"
    client = FakeCompletionClient(
        [
            ChatCompletionResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        "call-list",
                        "list_files",
                        json.dumps({"path": ".", "recursive": True}),
                    ),
                ),
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="chat-model",
            hf_api=None,
            embedding_model="fake-model",
        ),
        completion_client=client,
        knowledge=ErrorKnowledgeBase(data_root, FakeEmbedder()),
        tools=ToolManager(data_root),
        workspace=WorkspaceTools(project),
        max_actions_per_turn=1,
    )

    with pytest.raises(AgentError, match="before the task was completed"):
        agent.respond("Inspect this project.")
