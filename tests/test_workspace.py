from pathlib import Path
import sys

import pytest

from agent_cli.workspace import (
    WorkspaceApprovalDenied,
    WorkspaceError,
    WorkspaceTools,
)


def test_file_operations_are_confined_to_workspace(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "project"
    workspace_dir.mkdir()
    workspace = WorkspaceTools(workspace_dir)
    (workspace_dir / "src").mkdir()

    workspace.write_file("src/app.py", "print('hello')\n")
    assert workspace.read_file("src/app.py") == "print('hello')\n"
    assert "src/app.py" in workspace.list_files()
    with pytest.raises(WorkspaceError, match="Parent directory does not exist"):
        workspace.write_file("missing/new.py", "pass\n")


def test_external_file_access_requires_explicit_approval(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    prompts: list[str] = []
    workspace = WorkspaceTools(root, approve=lambda prompt: prompts.append(prompt) or False)

    with pytest.raises(WorkspaceApprovalDenied, match="not approved"):
        workspace.read_file(str(outside))
    assert prompts and "outside.txt" in prompts[0]


def test_external_file_access_is_allowed_only_after_approval(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    workspace = WorkspaceTools(root, approve=lambda _prompt: True)

    assert workspace.read_file(str(outside)) == "external"
    workspace.write_file(str(outside), "updated")
    assert outside.read_text(encoding="utf-8") == "updated"


def test_path_traversal_and_symlink_escape_are_detected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    workspace = WorkspaceTools(root)
    with pytest.raises(WorkspaceApprovalDenied):
        workspace.read_file("../outside.txt")

    link = root / "external-link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available.")
    with pytest.raises(WorkspaceApprovalDenied):
        workspace.read_file("external-link.txt")


def test_shell_operators_are_never_run(tmp_path: Path) -> None:
    workspace = WorkspaceTools(tmp_path, approve=lambda _prompt: True)

    with pytest.raises(WorkspaceError, match="Shell operators"):
        workspace.run_command(["python", "-m", "pytest", "&&", "echo", "unsafe"])


def test_safe_test_command_runs_without_prompt(tmp_path: Path) -> None:
    prompts: list[str] = []
    workspace = WorkspaceTools(
        tmp_path,
        approve=lambda prompt: prompts.append(prompt) or False,
    )

    result = workspace.run_command(
        [sys.executable, "-m", "pytest", "--version"],
        timeout_seconds=20,
    )

    assert result.returncode == 0
    assert "pytest" in result.stdout
    assert prompts == []


def test_unknown_and_outside_path_commands_require_approval(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    prompts: list[str] = []
    workspace = WorkspaceTools(root, approve=lambda prompt: prompts.append(prompt) or False)

    with pytest.raises(WorkspaceApprovalDenied):
        workspace.run_command(["echo", "hello"])
    assert "not in the development-check allowlist" in prompts[-1]

    prompts.clear()
    with pytest.raises(WorkspaceApprovalDenied):
        workspace.run_command(["pytest", str(tmp_path / "outside-test.py")])
    assert "outside the workspace" in prompts[-1]


def test_approved_unknown_command_executes_without_shell(tmp_path: Path) -> None:
    workspace = WorkspaceTools(tmp_path, approve=lambda _prompt: True)

    result = workspace.run_command(
        [sys.executable, "-c", "print('hello')"],
        timeout_seconds=20,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_force_command_is_gated(tmp_path: Path) -> None:
    workspace = WorkspaceTools(tmp_path, approve=lambda _prompt: False)

    with pytest.raises(WorkspaceApprovalDenied):
        workspace.run_command(["git", "push", "--force"])
