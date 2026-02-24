from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from typing import Any, Callable

import sympy as sp
from .api import get_global_registry


def build_register_dicts(registry=None) -> dict[str, Any]:
    """Return a plain name->callable table for all registered operators.

    Keys are exactly the full names returned by `registry.list()`, e.g.:
    - `math.add`
    - `dmdd.LZSI2024`
    """

    target_registry = registry or get_global_registry()
    table: dict[str, Any] = {}
    for full_name in target_registry.list():
        try:
            fn = target_registry.get(full_name)
        except Exception:
            continue
        if callable(fn):
            table[full_name] = fn
    return table


def _split_full_name(full_name: str) -> tuple[str, str] | None:
    normalized = full_name.strip()
    if not normalized:
        return None
    if ":" in normalized:
        return None
    if "." not in normalized:
        return None
    return normalized.split(".", 1)


def _to_symbolic_name(full_name: str) -> str:
    split = _split_full_name(full_name)
    if split is None:
        raise ValueError(f"invalid full operator name: {full_name!r}")
    namespace, short_name = split
    safe_ns = re.sub(r"[^0-9A-Za-z_]+", "_", namespace).strip("_") or "ns"
    safe_short = re.sub(r"[^0-9A-Za-z_]+", "_", short_name).strip("_") or "fn"
    digest = hashlib.blake2b(full_name.encode("utf-8"), digest_size=4).hexdigest()
    symbolic = f"{safe_ns}__{safe_short}__{digest}"
    if symbolic[0].isdigit():
        symbolic = f"f_{symbolic}"
    return symbolic


def build_sympy_dicts(
    mapping: dict[str, Any] | None = None,
    *,
    namespaces: list[str] | None = None,
    include_all: bool = False,
) -> tuple[dict[str, Any], dict[str, Callable[..., Any]]]:
    """Build `sympify` locals and `lambdify` numeric maps from register dicts.

    `func_locals` shape:
    - {"dmdd": SimpleNamespace(LZSI2024=Function("dmdd__LZSI2024__..."))}

    `numeric_funcs` shape:
    - {"dmdd__LZSI2024__...": <callable>}
    """

    source = mapping if mapping is not None else build_register_dicts()
    parse_locals: dict[str, Any] = {}
    numeric_funcs_map: dict[str, Callable[..., Any]] = {}
    namespace_attrs: dict[str, dict[str, Any]] = {}
    allowed = set(namespaces or [])

    for full_name, fn in source.items():
        if not callable(fn):
            continue
        split = _split_full_name(full_name)
        if split is None:
            continue
        namespace, short_name = split
        if allowed and not include_all and namespace not in allowed:
            continue
        symbolic_name = _to_symbolic_name(full_name)
        symbolic_fn = sp.Function(symbolic_name)
        namespace_attrs.setdefault(namespace, {})[short_name] = symbolic_fn
        numeric_funcs_map[symbolic_name] = fn

    for namespace, attrs in namespace_attrs.items():
        parse_locals[namespace] = SimpleNamespace(**attrs)

    return parse_locals, numeric_funcs_map


def _refresh_sympy_dicts() -> None:
    refreshed_func_locals, refreshed_numeric_funcs = build_sympy_dicts(
        build_register_dicts(),
        include_all=True,
    )
    func_locals.clear()
    func_locals.update(refreshed_func_locals)
    numeric_funcs.clear()
    numeric_funcs.update(refreshed_numeric_funcs)


def refresh_sympy_dicts_if_global_registry(registry: Any) -> bool:
    """Refresh public SymPy dict snapshots only when mutating global registry."""

    # Import lazily to avoid hard dependency on private module globals at import time.
    from . import api as _api

    global_registry = getattr(_api, "_global_registry", None)
    if global_registry is None or registry is not global_registry:
        return False
    _refresh_sympy_dicts()
    return True


func_locals, numeric_funcs = build_sympy_dicts(include_all=True)
