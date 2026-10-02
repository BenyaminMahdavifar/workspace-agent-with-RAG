from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AgentConfig
from .knowledge import ErrorKnowledgeBase
from .tools import ToolError, ToolManager
from .workspace import WorkspaceApprovalDenied, WorkspaceError, WorkspaceTools


class CompletionClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingToolFailure:
    knowledge_ids: tuple[str, ...]
    input_data: object | None


SYSTEM_PROMPT = """You are a terminal coding agent. Use only the agent-managed tools listed below.
You are allowed to create or repair tools. A tool itself cannot create or register tools.
For project tasks, use these internal workspace actions; they do not create generated tools:
{{"type":"list_files","path":".","recursive":true}}
{{"type":"read_file","path":"relative/path"}}
{{"type":"write_file","path":"relative/path","content":"complete file content"}}
{{"type":"run_command","command":["python","-m","pytest"],"cwd":".","timeout_seconds":120}}
Use paths relative to the active workspace when possible. Commands are argument arrays, not shell
commands; do not use pipes, command chaining, or shell syntax. File operations outside the active
workspace and commands outside the development-check allowlist will trigger a user approval prompt.
Never claim that a denied operation was performed.
When a generated tool is needed, answer with exactly one JSON object:
{{"type":"create_tool","name":"snake_case_name","description":"...","source":"Python source"}}
{{"type":"run_tool","name":"existing_tool","input":{{}}}}
{{"type":"repair_tool","name":"existing_tool","description":"...","source":"complete replacement Python source"}}
For a normal response, answer with plain text. Generated source must define run(input_data),
return a JSON-serializable value, use only allowlisted standard-library imports, and avoid
filesystem, network, process, and shell access. If tool validation or execution fails, diagnose
the supplied error and repair the existing tool where appropriate. Do not claim success unless
validation or execution confirms it.

Available tools:
{tools}

Active workspace:
{workspace}

Relevant stored knowledge:
{knowledge}
"""

KNOWN_ACTIONS = {
    "create_tool",
    "repair_tool",
    "run_tool",
    "list_files",
    "read_file",
    "write_file",
    "run_command",
}
SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|token|authorization)(\s*[:=]\s*)(\S+)")
ProgressCallback = Callable[[str], None]


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        completion_client: CompletionClient,
        knowledge: ErrorKnowledgeBase,
        tools: ToolManager,
        workspace: WorkspaceTools | None = None,
        progress: ProgressCallback | None = None,
        max_actions_per_turn: int = 12,
    ) -> None:
        if max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be positive.")
        self.config = config
        self.completion_client = completion_client
        self.knowledge = knowledge
        self.tools = tools
        self.workspace = workspace
        self.progress = progress or (lambda _message: None)
        self.max_actions_per_turn = max_actions_per_turn
        self._history: list[dict[str, str]] = []
        self._pending_failures: dict[str, PendingToolFailure] = {}
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def clear_history(self) -> None:
        self._history.clear()
        self._pending_failures.clear()
        self._turn_count = 0

    def respond(self, user_text: str) -> str:
        if not user_text.strip():
            raise ValueError("User input must not be empty.")
        self._turn_count += 1
        self.progress("Searching stored knowledge")
        try:
            matches = self.knowledge.search(user_text)
        except Exception as exc:
            self._record_error(
                error=f"Knowledge retrieval failed: {type(exc).__name__}: {exc}",
                context=user_text,
            )
            raise AgentError(f"Could not retrieve stored knowledge: {exc}") from exc

        knowledge_text = "\n\n".join(match.content for match in matches)
        try:
            tool_list = self.tools.list_tools()
        except (OSError, ToolError, ValueError) as exc:
            self._record_error(
                error=f"Tool discovery failed: {type(exc).__name__}: {exc}",
                context=user_text,
            )
            raise AgentError(f"Could not list available tools: {exc}") from exc
        tools_text = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tool_list
        ) or "(none)"
        system_message = SYSTEM_PROMPT.format(
            tools=tools_text,
            workspace=str(self.workspace.root) if self.workspace else "(not configured)",
            knowledge=knowledge_text or "(none)",
        )
        self._history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system_message}, *self._history]

        for _ in range(self.max_actions_per_turn):
            try:
                self.progress("Waiting for model response")
                response = self.completion_client.complete(messages)
            except Exception as exc:
                self._record_error(
                    error=f"Chat completion failed: {type(exc).__name__}: {exc}",
                    context=user_text,
                )
                raise AgentError(f"Chat completion failed: {exc}") from exc

            messages.append({"role": "assistant", "content": response})
            self._history.append({"role": "assistant", "content": response})
            action = self._parse_action(response)
            if action is None:
                return response

            try:
                if action.get("type") == "run_command":
                    command = action.get("command")
                    if isinstance(command, list) and command:
                        self.progress("Running project command")
                elif action.get("type") == "list_files":
                    self.progress("Listing workspace files")
                elif action.get("type") == "read_file":
                    self.progress("Reading workspace file")
                elif action.get("type") == "write_file":
                    self.progress("Writing workspace file")
                result = self._run_action(action)
            except WorkspaceApprovalDenied as exc:
                result = (
                    f"Action denied by user: {exc}. Do not repeat this action unless "
                    "the user explicitly requests and approves it."
                )
            except (ToolError, WorkspaceError, OSError, ValueError, KeyError, TypeError) as exc:
                tool_name = self._optional_string(action.get("name")) or "unknown"
                context = (
                    f"User request: {user_text}\n"
                    f"Action: {json.dumps(action, ensure_ascii=False)}\n"
                    f"Failure: {type(exc).__name__}: {exc}"
                )
                knowledge_id = self._record_error(str(exc), context)
                previous = self._pending_failures.get(tool_name)
                failed_input = (
                    action.get("input")
                    if action.get("type") == "run_tool" and "input" in action
                    else previous.input_data if previous is not None else None
                )
                existing_ids = (
                    previous.knowledge_ids if previous is not None else ()
                )
                self._pending_failures[tool_name] = PendingToolFailure(
                    (*existing_ids, knowledge_id),
                    failed_input,
                )
                result = (
                    f"Tool action failed: {type(exc).__name__}: {exc}. "
                    "The failure has been recorded. Diagnose it and, if applicable, "
                    "return a repair_tool action."
                )
            feedback = {"role": "user", "content": result}
            messages.append(feedback)
            self._history.append(feedback)

        message = (
            f"The agent reached the limit of {self.max_actions_per_turn} actions "
            "without producing a final response."
        )
        self._record_error(message, f"User request: {user_text}")
        raise AgentError(message)

    def _run_action(self, action: dict[str, object]) -> str:
        action_type = action.get("type")
        if action_type in {"list_files", "read_file", "write_file", "run_command"}:
            if self.workspace is None:
                raise WorkspaceError("No project workspace is configured.")
            if action_type == "list_files":
                path = action.get("path", ".")
                recursive = action.get("recursive", True)
                if not isinstance(path, str) or not isinstance(recursive, bool):
                    raise ValueError("list_files expects a string path and boolean recursive.")
                files = self.workspace.list_files(path, recursive)
                return json.dumps(files, ensure_ascii=False)
            if action_type == "read_file":
                path = self._required_string(action, "path")
                return self.workspace.read_file(path)
            if action_type == "write_file":
                path = self._required_string(action, "path")
                content = action.get("content")
                if not isinstance(content, str):
                    raise ValueError("write_file requires string content.")
                return self.workspace.write_file(path, content)

            command = action.get("command")
            if not isinstance(command, list) or any(
                not isinstance(part, str) for part in command
            ):
                raise ValueError("run_command requires command as a list of strings.")
            cwd = action.get("cwd", ".")
            timeout = action.get("timeout_seconds")
            if not isinstance(cwd, str):
                raise ValueError("run_command cwd must be a string.")
            if timeout is not None and (
                not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            ):
                raise ValueError("run_command timeout_seconds must be numeric.")
            result = self.workspace.run_command(command, cwd, timeout)
            return json.dumps(
                {
                    "command": list(result.command),
                    "cwd": result.cwd,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                ensure_ascii=False,
            )

        name = self._required_string(action, "name")
        if action_type in {"create_tool", "repair_tool"}:
            description = self._required_string(action, "description")
            source = self._required_string(action, "source")
            if action_type == "create_tool":
                info = self.tools.create(name, description, source)
            elif self.tools.exists(name):
                info = self.tools.repair(name, description, source)
            else:
                pending_creation = self._pending_failures.get(name)
                if pending_creation is None:
                    raise ToolError(f"Cannot repair unknown tool {name!r}.")
                info = self.tools.create(name, description, source)
            pending = self._pending_failures.get(name)
            if pending is not None and pending.input_data is not None:
                try:
                    result = self.tools.run(name, pending.input_data)
                except ToolError as exc:
                    knowledge_id = self._record_error(
                        str(exc), f"Retest after repair failed for tool {name!r}."
                    )
                    self._pending_failures[name] = PendingToolFailure(
                        (*pending.knowledge_ids, knowledge_id),
                        pending.input_data,
                    )
                    return f"Tool {name!r} was updated but still fails its previous input: {exc}"
                for knowledge_id in pending.knowledge_ids:
                    self._resolve_error(
                        knowledge_id,
                        f"Repaired source validated and retested successfully. Result: {result!r}",
                    )
            elif pending is not None:
                for knowledge_id in pending.knowledge_ids:
                    self._resolve_error(
                        knowledge_id,
                        "The agent supplied corrected source and tool validation succeeded.",
                    )
            self._pending_failures.pop(name, None)
            action_verb = "created" if action_type == "create_tool" else "repaired"
            return f"Tool {info.name!r} was {action_verb} and validated."

        if action_type == "run_tool":
            if "input" not in action:
                raise ValueError("run_tool action must include an input field.")
            input_data = action["input"]
            result = self.tools.run(name, input_data)
            pending = self._pending_failures.pop(name, None)
            if pending is not None:
                for knowledge_id in pending.knowledge_ids:
                    self._resolve_error(
                        knowledge_id,
                        f"Tool execution succeeded after repair. Result: {result!r}",
                    )
            return f"Tool result from {name!r}: {json.dumps(result, ensure_ascii=False)}"

        raise ValueError(f"Unsupported tool action: {action_type!r}.")

    def _record_error(self, error: str, context: str) -> str:
        for secret in (self.config.api_key, self.config.hf_api or ""):
            if secret:
                error = error.replace(secret, "[REDACTED]")
                context = context.replace(secret, "[REDACTED]")
        error = SECRET_PATTERN.sub(r"\1\2[REDACTED]", error)
        context = SECRET_PATTERN.sub(r"\1\2[REDACTED]", context)
        return self.knowledge.record_error(error, context)

    def _resolve_error(self, knowledge_id: str, solution: str) -> None:
        for secret in (self.config.api_key, self.config.hf_api or ""):
            if secret:
                solution = solution.replace(secret, "[REDACTED]")
        solution = SECRET_PATTERN.sub(r"\1\2[REDACTED]", solution)
        self.knowledge.resolve_error(knowledge_id, solution)

    @staticmethod
    def _parse_action(response: str) -> dict[str, object] | None:
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or value.get("type") not in KNOWN_ACTIONS:
            return None
        return value

    @staticmethod
    def _required_string(action: dict[str, object], key: str) -> str:
        value = action.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Action field {key!r} must be a non-empty string.")
        return value

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None
