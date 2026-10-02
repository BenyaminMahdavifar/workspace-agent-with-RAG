from __future__ import annotations

import json
from pathlib import Path

from agent_cli.agent import Agent
from agent_cli.config import AgentConfig
from agent_cli.knowledge import ErrorKnowledgeBase
from agent_cli.tools import ToolManager
from agent_cli.workspace import WorkspaceTools


class FakeEmbedder:
    model_name = "fake-model"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 17 + 1), 1.0] for text in texts]


class FakeCompletionClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return next(self.responses)


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
