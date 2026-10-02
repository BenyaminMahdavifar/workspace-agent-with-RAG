from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 1_000_000
MAX_LIST_ENTRIES = 500
IGNORED_DIRECTORIES = {".git", ".hg", ".svn", ".venv", "__pycache__", "node_modules"}
SHELL_OPERATORS = {"|", "||", "&", "&&", ";", ">", ">>", "<"}
DANGEROUS_COMMANDS = {
    "del", "erase", "format", "kill", "rd", "remove-item", "rmdir", "rm",
    "shutdown", "stop-process", "taskkill",
}
SAFE_COMMAND_PREFIXES = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python", "-m", "compileall"),
    ("uv", "run", "pytest"),
    ("uv", "run", "python", "-m", "pytest"),
    ("uv", "run", "python", "-m", "compileall"),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("npm", "run", "build"),
    ("npm", "run", "lint"),
    ("dotnet", "test"),
    ("dotnet", "build"),
    ("cargo", "test"),
    ("cargo", "build"),
    ("go", "test"),
    ("go", "build"),
)


class WorkspaceError(RuntimeError):
    pass


class WorkspaceApprovalDenied(WorkspaceError):
    pass


ApprovalCallback = Callable[[str], bool]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str


class WorkspaceTools:
    def __init__(
        self,
        root: Path,
        approve: ApprovalCallback | None = None,
        progress: ProgressCallback | None = None,
        command_timeout_seconds: float = 120.0,
    ) -> None:
        try:
            self.root = root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise WorkspaceError(f"Workspace does not exist: {root}") from exc
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace is not a directory: {self.root}")
        self.approve = approve or (lambda _message: False)
        self.progress = progress or (lambda _message: None)
        self.command_timeout_seconds = command_timeout_seconds

    def list_files(self, path: str = ".", recursive: bool = True) -> list[str]:
        directory = self._resolve(path)
        if not directory.is_dir():
            raise WorkspaceError(f"Not a directory: {path}")
        if not self._approve_external(
            directory,
            f"List files outside workspace: {directory}",
        ):
            raise WorkspaceApprovalDenied(
                f"Listing files outside the workspace was not approved: {directory}"
            )
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        results: list[str] = []
        for item in iterator:
            if any(part in IGNORED_DIRECTORIES for part in item.relative_to(directory).parts):
                continue
            if item.is_file():
                try:
                    display_path = item.relative_to(self.root).as_posix()
                except ValueError:
                    display_path = str(item)
                results.append(display_path)
                if len(results) >= MAX_LIST_ENTRIES:
                    break
        return sorted(results)

    def read_file(self, path: str) -> str:
        file_path = self._resolve(path)
        if not file_path.is_file():
            raise WorkspaceError(f"Not a file: {path}")
        if not self._approve_external(file_path, f"Read file outside workspace: {file_path}"):
            raise WorkspaceApprovalDenied(
                f"Reading outside the workspace was not approved: {file_path}"
            )
        try:
            with file_path.open("rb") as stream:
                content = stream.read(MAX_READ_BYTES + 1)
        except OSError as exc:
            raise WorkspaceError(f"Could not read {file_path}: {exc}") from exc
        if len(content) > MAX_READ_BYTES:
            raise WorkspaceError(f"File exceeds the {MAX_READ_BYTES}-byte read limit: {file_path}")
        return content.decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        file_path = self._resolve(path, allow_missing=True)
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise WorkspaceError(f"File content exceeds the {MAX_WRITE_BYTES}-byte write limit.")
        if not self._approve_external(file_path, f"Write file outside workspace: {file_path}"):
            raise WorkspaceApprovalDenied(
                f"Writing outside the workspace was not approved: {file_path}"
            )
        if file_path.exists() and not file_path.is_file():
            raise WorkspaceError(f"Cannot write to a non-file path: {file_path}")
        if not file_path.parent.is_dir():
            raise WorkspaceError(f"Parent directory does not exist: {file_path.parent}")
        try:
            file_path.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            raise WorkspaceError(f"Could not write {file_path}: {exc}") from exc
        target = file_path.relative_to(self.root) if self._is_inside(file_path) else file_path
        return f"Wrote {len(content.encode('utf-8'))} bytes to {target}."

    def run_command(
        self,
        command: Sequence[str],
        cwd: str = ".",
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise WorkspaceError("Command must be a non-empty list of non-empty strings.")
        if len(command) > 64 or sum(len(part) for part in command) > 16_000:
            raise WorkspaceError("Command exceeds configured argument limits.")
        if any(part in SHELL_OPERATORS for part in command):
            raise WorkspaceError("Shell operators are not supported; pass arguments as a JSON list.")
        workdir = self._resolve(cwd)
        if not workdir.is_dir():
            raise WorkspaceError(f"Command working directory is not a directory: {cwd}")
        risky_reason = self._command_risk(command, workdir)
        if not self._is_inside(workdir):
            outside_reason = f"Working directory is outside the workspace: {workdir}"
            risky_reason = (
                f"{outside_reason}\n{risky_reason}" if risky_reason else outside_reason
            )
        if risky_reason and not self.approve(
            f"Potentially risky command: {shlex.join(command)}\n"
            f"Working directory: {workdir}\nReason: {risky_reason}"
        ):
            raise WorkspaceApprovalDenied(
                f"Command was not approved: {shlex.join(command)}"
            )
        timeout = (
            self.command_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0 or timeout > 600:
            raise WorkspaceError("Command timeout must be between 0 and 600 seconds.")
        self.progress(f"Running: {shlex.join(command)}")
        try:
            completed = subprocess.run(
                list(command),
                cwd=workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceError(
                f"Command exceeded the {timeout:g}-second time limit: {shlex.join(command)}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(f"Could not execute {shlex.join(command)}: {exc}") from exc
        return CommandResult(
            command=tuple(command),
            cwd=str(workdir),
            returncode=completed.returncode,
            stdout=completed.stdout[-MAX_READ_BYTES:],
            stderr=completed.stderr[-MAX_READ_BYTES:],
        )

    def _resolve(self, value: str, allow_missing: bool = False) -> Path:
        if not value.strip():
            raise WorkspaceError("Workspace path must not be empty.")
        supplied = Path(value).expanduser()
        candidate = supplied if supplied.is_absolute() else self.root / supplied
        try:
            resolved = candidate.resolve(strict=not allow_missing)
        except OSError as exc:
            raise WorkspaceError(f"Could not resolve path {value!r}: {exc}") from exc
        if not allow_missing and not resolved.exists():
            raise WorkspaceError(f"Path does not exist: {value}")
        return resolved

    def _approve_external(self, path: Path, message: str) -> bool:
        return self._is_inside(path) or self.approve(message)

    def _is_inside(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False

    def _command_risk(self, command: Sequence[str], workdir: Path) -> str | None:
        normalized = [part.lower().replace("\\", "/") for part in command]
        executable = Path(command[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        normalized[0] = executable
        for argument in command[1:]:
            path_argument = argument.partition("=")[2] or argument
            argument_path = Path(path_argument).expanduser()
            if argument_path.is_absolute():
                try:
                    resolved_argument = argument_path.resolve(strict=False)
                except OSError:
                    return "A command argument refers to a path that cannot be resolved."
                if not self._is_inside(resolved_argument):
                    return f"The command references a path outside the workspace: {resolved_argument}"
            elif re.search(r"(^|/)\.\.(/|$)", path_argument.replace("\\", "/")):
                resolved_argument = (workdir / argument_path).resolve(strict=False)
                if not self._is_inside(resolved_argument):
                    return f"The command references a path outside the workspace: {resolved_argument}"
        if executable in DANGEROUS_COMMANDS:
            return "This command can delete data, affect system state, or terminate processes."
        if any(token in {"--force", "-force", "--hard", "--delete-branch"} for token in normalized[1:]):
            return "The command includes a force or destructive option."
        if any(phrase in " ".join(normalized) for phrase in ("git reset", "git clean", "git checkout --", "git restore")):
            return "The command can discard working-tree changes."
        if any(token in {"install", "add", "publish", "push", "deploy"} for token in normalized[1:]):
            return "The command can change dependencies or affect a remote service."
        if any(
            tuple(normalized[: len(prefix)]) == prefix
            for prefix in SAFE_COMMAND_PREFIXES
        ):
            return None
        return "This command is not in the development-check allowlist."
