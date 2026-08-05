from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from difflib import get_close_matches
from types import SimpleNamespace
from typing import Any, Callable

import sympy as sp

from .api import get_global_operas_registry, is_global_operas_registry
from .core.registry import OperasRegistry
from .name_utils import try_split_full_name


class ConstantValue(sp.Float):
    """Parse-locals constant that rejects call syntax with a readable error."""

    def __new__(cls, value: float, full_name: str = ""):
        obj = sp.Float.__new__(cls, float(value))
        obj._jo_full_name = full_name
        return obj

    def __reduce__(self):  # type: ignore[override]
        return (type(self), (float(self), getattr(self, "_jo_full_name", "")))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name = getattr(self, "_jo_full_name", None) or str(self)
        raise TypeError(
            f"'{name}' is a Jarvis-Operas constant, not a function; "
            f"write '{name}' without parentheses."
        )


class ConstantNamespace(SimpleNamespace):
    """Namespace object that suggests nearby constant/function names on miss."""

    def __init__(self, _jo_namespace: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._jo_namespace = _jo_namespace

    def __reduce__(self):  # type: ignore[override]
        # D23.10: SimpleNamespace default reduce cannot rebuild a class whose
        # __init__ requires a positional namespace argument.
        public = {
            key: value
            for key, value in self.__dict__.items()
            if key != "_jo_namespace"
        }
        return (ConstantNamespace, (self._jo_namespace,), public)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        known = sorted(key for key in self.__dict__ if not str(key).startswith("_"))
        full = f"{self._jo_namespace}.{name}"
        # D23.11: case-only typos (pdg.mz vs mZ) lose under default difflib cutoff.
        folded_index = {str(key).casefold(): str(key) for key in known}
        if name.casefold() in folded_index:
            suggestions = [folded_index[name.casefold()]]
        else:
            suggestions = get_close_matches(name, known, n=3, cutoff=0.5)
            if not suggestions:
                cf_hits = get_close_matches(
                    name.casefold(), list(folded_index.keys()), n=3, cutoff=0.5
                )
                suggestions = [folded_index[hit] for hit in cf_hits]
        message = f"'{full}' is not registered in Jarvis-Operas."
        if suggestions:
            message += f" Did you mean {', '.join(suggestions)}?"
        if known:
            message += f" Known: {', '.join(known)}"
        raise AttributeError(message)


def build_register_dicts(registry: OperasRegistry | None = None) -> dict[str, Callable[..., Any]]:
    """Return full-name -> numeric callable table for all registered functions."""

    target_registry = registry or get_global_operas_registry()
    table: dict[str, Callable[..., Any]] = {}
    for full_name in target_registry.list():
        try:
            declaration = target_registry.get(full_name)
        except Exception:
            continue

        fn = declaration.numpy_impl
        if callable(fn):
            table[full_name] = fn
    return table


def build_constant_dicts(registry: OperasRegistry | None = None) -> dict[str, float]:
    """Return full-name -> value for declarations flagged ``constant``."""

    target_registry = registry or get_global_operas_registry()
    table: dict[str, float] = {}
    for full_name in target_registry.list():
        try:
            declaration = target_registry.get(full_name)
        except Exception:
            continue
        if "constant" not in declaration.flags:
            continue
        value = declaration.metadata.get("value")
        if value is None and callable(declaration.numpy_impl):
            value = declaration.numpy_impl()
        table[full_name] = float(value)
    return table


def _to_symbolic_name(full_name: str) -> str:
    split = try_split_full_name(full_name)
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
    constants: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Callable[..., Any]]]:
    """Build `sympify` locals and `lambdify` numeric maps from register dicts.

    Compatibility contract (do not break):
    - Return value is always a 2-tuple ``(parse_locals, numeric_funcs)``.
    - ``mapping`` remains the first positional parameter.
    - ``constants`` is keyword-only with default ``None``.

    When ``constants is None``:
    - if ``mapping is not None`` (V1/PLOT path): fold no constants;
    - if ``mapping is None`` (full-view / module snapshot): load via
      ``build_constant_dicts()``.
    """

    source = mapping if mapping is not None else build_register_dicts()
    if constants is None:
        resolved_constants: Mapping[str, float] = (
            build_constant_dicts() if mapping is None else {}
        )
    else:
        resolved_constants = constants

    parse_locals: dict[str, Any] = {}
    numeric_funcs_map: dict[str, Callable[..., Any]] = {}
    namespace_attrs: dict[str, dict[str, Any]] = {}
    allowed = set(namespaces or [])
    constant_names = set(resolved_constants)

    for full_name, fn in source.items():
        if full_name in constant_names:
            # Constant wins: skip its arity-0 impl so parse_locals gets Float.
            continue
        if not callable(fn):
            continue
        split = try_split_full_name(full_name)
        if split is None:
            continue
        namespace, short_name = split
        if allowed and not include_all and namespace not in allowed:
            continue
        symbolic_name = _to_symbolic_name(full_name)
        symbolic_fn = sp.Function(symbolic_name)
        namespace_attrs.setdefault(namespace, {})[short_name] = symbolic_fn
        numeric_funcs_map[symbolic_name] = fn

    namespaces_with_constants: set[str] = set()
    for full_name, value in resolved_constants.items():
        split = try_split_full_name(full_name)
        if split is None:
            continue
        namespace, short_name = split
        if allowed and not include_all and namespace not in allowed:
            continue
        namespace_attrs.setdefault(namespace, {})[short_name] = ConstantValue(
            value,
            full_name=full_name,
        )
        namespaces_with_constants.add(namespace)

    for namespace, attrs in namespace_attrs.items():
        if namespace in namespaces_with_constants:
            parse_locals[namespace] = ConstantNamespace(
                _jo_namespace=namespace,
                **attrs,
            )
        else:
            parse_locals[namespace] = SimpleNamespace(**attrs)

    return parse_locals, numeric_funcs_map


def _refresh_sympy_dicts() -> None:
    # Always fold constants into the module-level snapshot (full-view semantics).
    # Pass constants= explicitly because mapping is non-None here, which would
    # otherwise take the V1/PLOT "no fold" path.
    refreshed_func_locals, refreshed_numeric_funcs = build_sympy_dicts(
        build_register_dicts(),
        include_all=True,
        constants=build_constant_dicts(),
    )
    func_locals.clear()
    func_locals.update(refreshed_func_locals)
    numeric_funcs.clear()
    numeric_funcs.update(refreshed_numeric_funcs)


def refresh_sympy_dicts_if_global_registry(registry: Any) -> bool:
    """Refresh SymPy dict snapshots only when mutating the global registry."""

    if not is_global_operas_registry(registry):
        return False
    _refresh_sympy_dicts()
    return True


func_locals, numeric_funcs = build_sympy_dicts(include_all=True)
