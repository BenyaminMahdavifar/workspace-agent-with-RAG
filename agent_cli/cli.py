from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .agent import Agent, AgentError
from .config import AgentConfig, DEFAULT_EMBEDDING_MODEL
from .embeddings import HuggingFaceEmbedder
from .knowledge import ErrorKnowledgeBase
from .llm import ChatCompletionClient
from .tools import ToolManager
from .workspace import WorkspaceError, WorkspaceTools


HELP_TEXT = """Commands:
  /help       Show this help
  /status     Show workspace, models, and session status
  /clear      Clear conversation history (knowledge and tools are kept)
  /exit       Exit the agent

Type any other text to send it to the coding agent. The active workspace is shown in the prompt.
"""


def _read_value(environment_name: str, prompt: str, secret: bool = False) -> str:
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    while True:
        value = getpass.getpass(prompt).strip() if secret else input(prompt).strip()
        if value:
            return value
        print(f"{environment_name} is required.", file=sys.stderr)


def _read_config() -> AgentConfig:
    base_url = _read_value("BASE_URL", "OpenAI-compatible API base URL: ")
    api_key = _read_value("API_KEY", "Chat API key: ", secret=True)
    model = _read_value("MODEL", "Chat model: ")
    hf_api = os.environ.get("HF_API", "").strip() or None
    if hf_api is None:
        entered_token = getpass.getpass(
            "Hugging Face token (leave blank for public models): "
        ).strip()
        hf_api = entered_token or None
    embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        embedding_model = input(
            f"Embedding model [{DEFAULT_EMBEDDING_MODEL}]: "
        ).strip() or DEFAULT_EMBEDDING_MODEL
    return AgentConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        hf_api=hf_api,
        embedding_model=embedding_model,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="full-agent",
        description="Interactive coding agent for a selected project workspace.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project directory (default: current terminal directory).",
    )
    return parser.parse_args(argv)


def _approve_action(message: str) -> bool:
    print(f"\nApproval required:\n{message}")
    try:
        answer = input("Approve this action? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAction denied.")
        return False
    return answer in {"y", "yes"}


def _show_status(agent: Agent, workspace: WorkspaceTools, config: AgentConfig) -> None:
    print(f"Workspace: {workspace.root}")
    print(f"Chat model: {config.model}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Generated tools: {len(agent.tools.list_tools())}")
    print(f"Conversation turns: {agent.turn_count}")


def _handle_command(
    command: str,
    agent: Agent,
    workspace: WorkspaceTools,
    config: AgentConfig,
) -> bool:
    normalized = command.strip().lower()
    if normalized in {"/exit", "/quit"}:
        return False
    if normalized == "/help":
        print(HELP_TEXT, end="")
    elif normalized == "/status":
        try:
            _show_status(agent, workspace, config)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Could not display status: {exc}", file=sys.stderr)
    elif normalized == "/clear":
        agent.clear_history()
        print("Conversation history cleared. Stored knowledge and generated tools were kept.")
    elif normalized.startswith("/"):
        print(f"Unknown command: {command.strip()}. Type /help to see available commands.")
    else:
        raise ValueError("Only slash-prefixed commands can be handled here.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    progress = lambda message: print(f"[status] {message}", flush=True)
    try:
        workspace = WorkspaceTools(
            args.workspace,
            approve=_approve_action,
            progress=progress,
        )
        print(f"Workspace: {workspace.root}")
        config = _read_config()
        root = Path(os.environ.get("AGENT_HOME", Path.home() / ".full-agent")).expanduser()
        print(f"Loading embedding model: {config.embedding_model}")
        embedder = HuggingFaceEmbedder(config.embedding_model, config.hf_api)
        knowledge = ErrorKnowledgeBase(root, embedder)
        print("Preparing agent services...")
        agent = Agent(
            config=config,
            completion_client=ChatCompletionClient(config),
            knowledge=knowledge,
            tools=ToolManager(root),
            workspace=workspace,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError, WorkspaceError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    print("Agent is ready. Type /help to see available commands.")
    while True:
        try:
            user_text = input(f"\nAgent[{workspace.root.name}]> ")
        except EOFError:
            print("\nGoodbye.")
            return 0
        except KeyboardInterrupt:
            print("\nCancelled. Type /exit to leave.")
            continue

        stripped = user_text.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            if not _handle_command(stripped, agent, workspace, config):
                print("Goodbye.")
                return 0
            continue
        try:
            response = agent.respond(user_text)
            print(f"\n{response}")
        except KeyboardInterrupt:
            agent.clear_history()
            print("\nOperation cancelled. Current conversation history was cleared.")
        except AgentError as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
        except (OSError, ValueError) as exc:
            print(f"Request error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
