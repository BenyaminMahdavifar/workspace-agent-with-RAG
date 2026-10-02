from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AgentConfig
from .knowledge import ErrorKnowledgeBase
from .llm import ChatCompletionResult, ChatToolCall, ToolCallingUnsupported
from .tools import ToolError, ToolManager
from .workspace import WorkspaceApprovalDenied, WorkspaceError, WorkspaceTools


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatCompletionResult | str: ...


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingToolFailure:
    knowledge_ids: tuple[str, ...]
    input_data: object | None


SYSTEM_PROMPT = """You are a terminal coding agent. Use only the agent-managed tools listed below.
You are allowed to create or repair tools. A tool itself cannot create or register tools.
For project tasks, call the provided workspace functions to list/read/write files or run commands.
Use native function tool calls; never print a tool request as prose, a JSON example, or a code block.
Use paths relative to the active workspace when possible. Commands are argument arrays, not shell
commands; do not use pipes, command chaining, or shell syntax. File operations outside the active
workspace and commands outside the development-check allowlist will trigger a user approval prompt.
Never claim that a denied operation was performed.
When a generated tool is needed, call the matching generated-tool function. For a normal completed
task, answer the user with plain text. Generated source must define run(input_data),
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

ACTION_TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the active workspace or a subdirectory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a development command as an argument array, without a shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_tool",
            "description": "Create a generated Python tool. Tools cannot create tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["name", "description", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_tool",
            "description": "Replace the implementation of an existing generated Python tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["name", "description", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tool",
            "description": "Run a previously created generated Python tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "input": {},
                },
                "required": ["name", "input"],
                "additionalProperties": False,
            },
        },
    },
]
FALLBACK_PROTOCOL = """Native tool calls are unavailable for this response. Use this strict JSON-only
protocol, with no prose or Markdown fences:
For an action: {"type":"action","action":{"type":"list_files|read_file|write_file|run_command|create_tool|repair_tool|run_tool",...}}
For a final answer: {"type":"final","content":"Your answer to the user"}
Never include an action JSON object inside explanatory prose. The action result will be returned
to you, and you must choose the next action or provide a final answer based on that actual result.
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
        self._history: list[dict[str, object]] = []
        self._pending_failures: dict[str, PendingToolFailure] = {}
        self._turn_count = 0
        self._native_tool_calling_supported: bool | None = None

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
        use_native_tools = self._native_tool_calling_supported is not False
        fallback_prompt_added = not use_native_tools
        approval_denied = False
        action_count = 0
        if fallback_prompt_added:
            messages.append({"role": "system", "content": FALLBACK_PROTOCOL})

        for _ in range(self.max_actions_per_turn):
            try:
                self.progress("Waiting for model response")
                completion = self.completion_client.complete(
                    messages,
                    tools=ACTION_TOOL_SCHEMAS if use_native_tools else None,
                )
            except ToolCallingUnsupported:
                self._native_tool_calling_supported = False
                use_native_tools = False
                if not fallback_prompt_added:
                    messages.append({"role": "system", "content": FALLBACK_PROTOCOL})
                    fallback_prompt_added = True
                self.progress(
                    "Native tool calls are unavailable; switching to the strict JSON action protocol"
                )
                continue
            except Exception as exc:
                self._record_error(
                    error=f"Chat completion failed: {type(exc).__name__}: {exc}",
                    context=user_text,
                )
                raise AgentError(f"Chat completion failed: {exc}") from exc

            if isinstance(completion, str):
                completion = ChatCompletionResult(content=completion)
            if completion.tool_calls:
                if approval_denied:
                    raise AgentError(
                        "The task was paused because an operation was denied. "
                        "The model attempted another action after the denial; no further action was run."
                    )
                if action_count + len(completion.tool_calls) > self.max_actions_per_turn:
                    message = (
                        f"The task requested more than {self.max_actions_per_turn} actions "
                        "before completion."
                    )
                    self._record_error(message, f"User request: {user_text}")
                    raise AgentError(message)
                action_count += len(completion.tool_calls)
                assistant_message = completion.as_assistant_message()
                messages.append(assistant_message)
                self._history.append(assistant_message)
                for tool_call in completion.tool_calls:
                    if approval_denied:
                        result = json.dumps(
                            {
                                "ok": False,
                                "error": "Not run because an earlier action in this task was denied.",
                            },
                            ensure_ascii=False,
                        )
                        denied = False
                    else:
                        try:
                            action = self._action_from_tool_call(tool_call)
                            result, denied = self._execute_action(action, user_text)
                        except (ValueError, TypeError, KeyError) as exc:
                            self._record_error(
                                f"Invalid structured tool call: {type(exc).__name__}: {exc}",
                                f"User request: {user_text}\nTool call: {tool_call!r}",
                            )
                            result = self._format_action_error(exc)
                            denied = False
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                    self._history.append(messages[-1])
                    approval_denied = approval_denied or denied
                if approval_denied:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "A requested action was denied by the user. Do not request or "
                                "perform any more actions in this task. Explain that the task "
                                "is incomplete and identify the denied operation."
                            ),
                        }
                    )
                    use_native_tools = False
                continue

            response = completion.content or ""
            messages.append({"role": "assistant", "content": response})
            self._history.append({"role": "assistant", "content": response})
            parsed = self._parse_structured_response(response)
            if parsed is not None:
                kind, payload = parsed
                if kind == "final":
                    return payload
                if approval_denied:
                    raise AgentError(
                        "The task was paused because an operation was denied. "
                        "The model attempted another action after the denial; no further action was run."
                    )
                action_count += 1
                if action_count > self.max_actions_per_turn:
                    message = (
                        f"The task requested more than {self.max_actions_per_turn} actions "
                        "before completion."
                    )
                    self._record_error(message, f"User request: {user_text}")
                    raise AgentError(message)
                try:
                    action = payload
                    result, denied = self._execute_action(action, user_text)
                except (ValueError, TypeError, KeyError) as exc:
                    result = self._format_action_error(exc)
                    denied = False
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "type": "action_result",
                                "ok": self._result_is_success(result),
                                "result": result,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                self._history.append(messages[-1])
                approval_denied = approval_denied or denied
                if approval_denied:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The user denied an action. Do not request or perform another "
                                "action. Return a final response explaining that the task is incomplete."
                            ),
                        }
                    )
                continue

            if approval_denied:
                return response
            if self._contains_embedded_action(response):
                if use_native_tools:
                    self._native_tool_calling_supported = False
                    use_native_tools = False
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous reply contained prose and an action-looking JSON "
                                "fragment. No action was executed. Return actions only as a "
                                "strict JSON protocol response.\n" + FALLBACK_PROTOCOL
                            ),
                        }
                    )
                    fallback_prompt_added = True
                    self.progress(
                        "The model returned an unstructured action; requesting a strict JSON action"
                    )
                    continue
                message = (
                    "The model included an action in plain text instead of returning a structured "
                    "tool call. The task is incomplete; choose a provider/model that supports "
                    "tool calling or returns the strict JSON action protocol."
                )
                self._record_error(message, f"User request: {user_text}\nResponse: {response}")
                raise AgentError(message)
            if use_native_tools:
                return response
            message = (
                "The provider did not return the required strict JSON action/final response. "
                "The task is incomplete; no unstructured action text was executed."
            )
            self._record_error(message, f"User request: {user_text}\nResponse: {response}")
            raise AgentError(message)

        message = (
            f"The agent reached the limit of {self.max_actions_per_turn} actions "
            "before the task was completed."
        )
        self._record_error(message, f"User request: {user_text}")
        raise AgentError(message)

    def _execute_action(
        self,
        action: dict[str, object],
        user_text: str,
    ) -> tuple[str, bool]:
        try:
            action_type = action.get("type")
            if action_type == "run_command":
                self.progress("Running project command")
            elif action_type == "list_files":
                self.progress("Listing workspace files")
            elif action_type == "read_file":
                self.progress("Reading workspace file")
            elif action_type == "write_file":
                self.progress("Writing workspace file")
            value = self._run_action(action)
            return json.dumps({"ok": True, "result": value}, ensure_ascii=False), False
        except WorkspaceApprovalDenied as exc:
            return (
                json.dumps(
                    {"ok": False, "error": f"Action denied by user: {exc}"},
                    ensure_ascii=False,
                ),
                True,
            )
        except (ToolError, WorkspaceError, OSError, ValueError, KeyError, TypeError) as exc:
            self._record_action_error(action, user_text, exc)
            return self._format_action_error(exc), False

    def _record_action_error(
        self,
        action: dict[str, object],
        user_text: str,
        error: Exception,
    ) -> None:
        context = (
            f"User request: {user_text}\n"
            f"Action: {json.dumps(action, ensure_ascii=False)}\n"
            f"Failure: {type(error).__name__}: {error}"
        )
        knowledge_id = self._record_error(str(error), context)
        if action.get("type") not in {"create_tool", "repair_tool", "run_tool"}:
            return
        tool_name = self._optional_string(action.get("name")) or "unknown"
        previous = self._pending_failures.get(tool_name)
        failed_input = (
            action.get("input")
            if action.get("type") == "run_tool" and "input" in action
            else previous.input_data if previous is not None else None
        )
        existing_ids = previous.knowledge_ids if previous is not None else ()
        self._pending_failures[tool_name] = PendingToolFailure(
            (*existing_ids, knowledge_id),
            failed_input,
        )

    @staticmethod
    def _format_action_error(error: Exception) -> str:
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "recorded": True,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _result_is_success(result: str) -> bool:
        try:
            decoded = json.loads(result)
        except json.JSONDecodeError:
            return False
        return isinstance(decoded, dict) and decoded.get("ok") is True

    @classmethod
    def _action_from_tool_call(cls, call: ChatToolCall) -> dict[str, object]:
        if call.name not in KNOWN_ACTIONS:
            raise ValueError(f"Unsupported tool name: {call.name!r}.")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return {"type": call.name, **arguments}

    @classmethod
    def _parse_structured_response(
        cls,
        response: str,
    ) -> tuple[str, dict[str, object] | str] | None:
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict):
            return None
        if decoded.get("type") == "final":
            content = decoded.get("content")
            if not isinstance(content, str):
                raise AgentError("The structured final response must contain string content.")
            return "final", content
        if decoded.get("type") == "action":
            action = decoded.get("action")
            if not isinstance(action, dict) or action.get("type") not in KNOWN_ACTIONS:
                raise AgentError("The structured action response contains an invalid action.")
            return "action", action
        if decoded.get("type") in KNOWN_ACTIONS:
            return "action", decoded
        return None

    @staticmethod
    def _contains_embedded_action(response: str) -> bool:
        action_names = "|".join(sorted(KNOWN_ACTIONS))
        pattern = re.compile(
            rf'"type"\s*:\s*"(?:{action_names})"',
            flags=re.IGNORECASE,
        )
        if not pattern.search(response):
            return False
        try:
            json.loads(response)
        except json.JSONDecodeError:
            return True
        return False

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
