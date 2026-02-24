#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Prefer local source tree over site-packages when run from examples/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "jarvis_operas").is_dir():
    sys.path.insert(0, str(_REPO_ROOT))

import sympy as sp
from sympy.utilities.lambdify import lambdify


def _parse_set_args(items: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"invalid assignment '{raw}', expected key=value")
        key, value_raw = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid assignment '{raw}', key cannot be empty")
        try:
            value = json.loads(value_raw)
        except json.JSONDecodeError:
            value = float(value_raw)
        values[key] = value
    return values


def main() -> int:
    # Configure test inputs here directly (no CLI args needed).
    expr_text = "dmdd.LZSI2024(mchi)"
    # Use key=value string format, same as previous CLI style.
    set_items = ["mchi=95.0"]

    from jarvis_operas import func_locals, numeric_funcs

    if not numeric_funcs:
        raise SystemExit("No numeric functions loaded from jarvis_operas.")

    expr = sp.sympify(expr_text, locals=func_locals)
    var_names = sorted(str(symbol) for symbol in expr.free_symbols)
    values = _parse_set_args(set_items)
    if set(var_names) - set(values.keys()):
        missing = sorted(set(var_names) - set(values.keys()))
        raise SystemExit(f"Missing values for symbols: {missing}")

    num_expr = lambdify(var_names, expr, modules=[numeric_funcs, "numpy"])
    result = num_expr(**{name: values[name] for name in var_names})

    print(f"Expression: {expr_text}")
    print(f"Parsed:      {expr}")
    print(f"Variables:   {var_names}")
    print(f"Inputs:      {values}")
    print(f"Result:      {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
