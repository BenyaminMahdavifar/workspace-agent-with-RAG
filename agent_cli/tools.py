from __future__ import annotations

import ast
import json
import keyword
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_TOOL_SOURCE_BYTES = 32_000
MAX_TOOL_INPUT_BYTES = 100_000
MAX_TOOL_OUTPUT_BYTES = 1_000_000
ALLOWED_IMPORTS = {"datetime", "json", "math", "re", "statistics"}
FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
}


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str


class ToolManager:
    def __init__(self, root: Path, timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.tools_dir = root / "tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.tools_dir / "tools.json"
        self.timeout_seconds = timeout_seconds

    def create(self, name: str, description: str, source: str) -> ToolInfo:
        path = self._path_for(name)
        if path.exists():
            raise ToolError(f"Tool {name!r} already exists; use repair_tool to update it.")
        self._validate(source)
        self._save_description(name, description)
        self._write_source(path, source)
        return ToolInfo(name=name, description=description.strip())

    def repair(self, name: str, description: str, source: str) -> ToolInfo:
        path = self._path_for(name)
        if not path.is_file():
            raise ToolError(f"Cannot repair unknown tool {name!r}.")
        self._validate(source)
        self._save_description(name, description)
        self._write_source(path, source)
        return ToolInfo(name=name, description=description.strip())

    def exists(self, name: str) -> bool:
        return self._path_for(name).is_file()

    def list_tools(self) -> list[ToolInfo]:
        descriptions = self._read_descriptions()
        infos: list[ToolInfo] = []
        for path in sorted(self.tools_dir.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                self._validate(source)
            except ToolError as exc:
                infos.append(ToolInfo(path.stem, f"INVALID TOOL: {exc}"))
                continue
            infos.append(
                ToolInfo(
                    path.stem,
                    descriptions.get(path.stem, self._description(source)),
                )
            )
        return infos

    def run(self, name: str, input_data: object) -> object:
        path = self._path_for(name)
        if not path.is_file():
            raise ToolError(f"Tool {name!r} does not exist.")
        source = path.read_text(encoding="utf-8")
        self._validate(source)
        serialized_input = json.dumps(input_data, ensure_ascii=False)
        if len(serialized_input.encode("utf-8")) > MAX_TOOL_INPUT_BYTES:
            raise ToolError("Tool input exceeds the configured size limit.")

        worker = Path(__file__).with_name("tool_worker.py")
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(worker), str(path)],
                input=serialized_input,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=self.tools_dir,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Tool {name!r} exceeded its {self.timeout_seconds:g}-second time limit."
            ) from exc
        except OSError as exc:
            raise ToolError(f"Could not start tool {name!r}: {exc}") from exc

        if len(result.stdout.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
            raise ToolError("Tool output exceeds the configured size limit.")
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or "worker exited without an error message"
            raise ToolError(f"Tool {name!r} failed: {diagnostic}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Tool {name!r} returned invalid JSON.") from exc

    def _path_for(self, name: str) -> Path:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ToolError("Tool names must be identifiers containing letters, digits, or underscores.")
        return self.tools_dir / f"{name}.py"

    def _read_descriptions(self) -> dict[str, str]:
        if not self.metadata_path.exists():
            return {}
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Could not read tool metadata: {exc}") from exc
        if not isinstance(value, dict) or any(
            not isinstance(name, str) or not isinstance(description, str)
            for name, description in value.items()
        ):
            raise ToolError("Tool metadata must be a JSON object of string descriptions.")
        return value

    def _save_description(self, name: str, description: str) -> None:
        descriptions = self._read_descriptions()
        descriptions[name] = description.strip()
        temporary_path = self.metadata_path.with_suffix(".json.tmp")
        try:
            temporary_path.write_text(
                json.dumps(descriptions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self.metadata_path)
        except OSError as exc:
            raise ToolError(f"Could not save tool metadata: {exc}") from exc

    @staticmethod
    def _write_source(path: Path, source: str) -> None:
        temporary_path = path.with_suffix(".py.tmp")
        try:
            temporary_path.write_text(source, encoding="utf-8")
            temporary_path.replace(path)
        except OSError as exc:
            raise ToolError(f"Could not save tool source: {exc}") from exc

    @staticmethod
    def _validate(source: str) -> None:
        if len(source.encode("utf-8")) > MAX_TOOL_SOURCE_BYTES:
            raise ToolError("Tool source exceeds the configured size limit.")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ToolError(f"Tool source has a syntax error: {exc}") from exc

        run_functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ]
        if len(run_functions) != 1 or isinstance(run_functions[0], ast.AsyncFunctionDef):
            raise ToolError("Tool source must define exactly one synchronous run(input_data) function.")
        run_args = run_functions[0].args
        if len(run_args.args) != 1 or run_args.vararg or run_args.kwarg or run_args.kwonlyargs:
            raise ToolError("The run function must accept exactly one positional argument.")
        if run_args.args[0].arg != "input_data":
            raise ToolError("The run function parameter must be named input_data.")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not run_functions[0]:
                raise ToolError("Generated tools may define only the run function.")
            if isinstance(node, ast.ClassDef):
                raise ToolError("Generated tools may not define classes.")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name.split(".")[0] for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module.split(".")[0] if node.module else ""]
                )
                if any(module not in ALLOWED_IMPORTS for module in modules):
                    raise ToolError(f"Only these imports are allowed: {', '.join(sorted(ALLOWED_IMPORTS))}.")
                if isinstance(node, ast.ImportFrom) and (
                    node.level or any(alias.name.startswith("_") for alias in node.names)
                ):
                    raise ToolError("Relative imports and private imports are not allowed.")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ToolError("Private and dunder attribute access is not allowed.")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ToolError("Dunder names are not allowed.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise ToolError(f"Calling {node.func.id} is not allowed.")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ToolError("Global and nonlocal declarations are not allowed.")

    @staticmethod
    def _description(source: str) -> str:
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if docstring:
            return docstring.strip()
        return "Agent-generated Python tool"
