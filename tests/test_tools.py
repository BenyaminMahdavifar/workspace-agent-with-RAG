from pathlib import Path

import pytest

from agent_cli.tools import ToolError, ToolManager


def test_created_tool_runs_with_json_input_output(tmp_path: Path) -> None:
    manager = ToolManager(tmp_path)
    manager.create(
        "double_value",
        "Doubles a numeric input.",
        '"""Doubles a numeric input."""\n'
        "def run(input_data):\n"
        '    return {"result": input_data["value"] * 2}\n',
    )

    assert manager.run("double_value", {"value": 7}) == {"result": 14}


@pytest.mark.parametrize(
    "source, message",
    [
        ("def run(input_data):\n    open('x')\n", "open"),
        (
            "import os\ndef run(input_data):\n    return os.getcwd()\n",
            "Only these imports",
        ),
        (
            "def run(input_data):\n    return input_data.__class__\n",
            "Private and dunder",
        ),
        ("def run(input_data):\n    return eval('1 + 1')\n", "eval"),
        ("def run(value):\n    return value\n", "parameter must be named input_data"),
    ],
)
def test_validator_rejects_unsafe_or_invalid_source(
    tmp_path: Path, source: str, message: str
) -> None:
    manager = ToolManager(tmp_path)

    with pytest.raises(ToolError, match=message):
        manager.create("unsafe_tool", "Not allowed.", source)


def test_tool_repair_replaces_broken_implementation(tmp_path: Path) -> None:
    manager = ToolManager(tmp_path)
    manager.create(
        "broken",
        "Initially broken.",
        "def run(input_data):\n    return 1 / 0\n",
    )
    with pytest.raises(ToolError, match="failed"):
        manager.run("broken", {})

    manager.repair(
        "broken",
        "Returns the input value.",
        "def run(input_data):\n    return input_data\n",
    )

    assert manager.run("broken", {"ok": True}) == {"ok": True}


def test_tool_timeout_is_reported(tmp_path: Path) -> None:
    manager = ToolManager(tmp_path, timeout_seconds=0.1)
    manager.create(
        "loop_forever",
        "Used to verify timeout handling.",
        "def run(input_data):\n    while True:\n        pass\n",
    )

    with pytest.raises(ToolError, match="time limit"):
        manager.run("loop_forever", {})
