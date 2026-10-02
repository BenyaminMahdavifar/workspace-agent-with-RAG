from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ALLOWED_IMPORTS = {"datetime", "json", "math", "re", "statistics"}
SAFE_BUILTINS: dict[str, Any] = {
    "Exception": Exception,
    "False": False,
    "KeyError": KeyError,
    "None": None,
    "True": True,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _safe_import(
    name: str,
    globals_: dict[str, object] | None = None,
    locals_: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> ModuleType:
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"Import {name!r} is not allowed.")
    return __import__(name, globals_, locals_, fromlist, level)


def main() -> int:
    if len(sys.argv) != 2:
        print("Expected one tool source path.", file=sys.stderr)
        return 2
    source_path = Path(sys.argv[1])
    source = source_path.read_text(encoding="utf-8")
    input_data = json.load(sys.stdin)
    namespace: dict[str, object] = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
        "__name__": "generated_tool",
    }
    exec(compile(source, str(source_path), "exec"), namespace, namespace)
    result = namespace["run"](input_data)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
