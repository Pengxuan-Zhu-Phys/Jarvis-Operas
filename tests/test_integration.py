#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sympy as sp
from sympy.utilities.lambdify import lambdify

import jarvis_operas.api as api_mod
import jarvis_operas.integration as integration
from jarvis_operas import OperaFunction, OperasRegistry
from jarvis_operas.integration import (
    build_register_dicts,
    build_sympy_dicts,
    refresh_sympy_dicts_if_global_registry,
)


def _register(registry: OperasRegistry, namespace: str, name: str, fn) -> None:
    registry.register(
        OperaFunction(
            namespace=namespace,
            name=name,
            arity=None,
            return_dtype=None,
            numpy_impl=fn,
        )
    )


def test_build_register_dicts_returns_full_name_to_callable_only() -> None:
    registry = OperasRegistry()
    _register(registry, "dmdd", "LZSI2024", lambda x: x * 2.0)

    mapping = build_register_dicts(registry)
    assert callable(mapping["dmdd.LZSI2024"])
    assert set(mapping.keys()) == {"dmdd.LZSI2024"}


def test_build_sympy_dicts_for_namespace_function_parsing_and_eval() -> None:
    registry = OperasRegistry()
    _register(registry, "dmdd", "LZSI2024", lambda x: x * 2.0)
    _register(registry, "math", "add", lambda x, y: x + y)

    mapping = build_register_dicts(registry)
    func_locals, numeric_funcs = build_sympy_dicts(mapping, namespaces=["dmdd"])

    expr = sp.sympify("dmdd.LZSI2024(mchi)", locals=func_locals)
    num_expr = lambdify(["mchi"], expr, modules=[numeric_funcs, "numpy"])

    assert float(num_expr(95.0)) == 190.0
    assert "math" not in func_locals


def test_refresh_sympy_dicts_if_global_registry_updates_module_level_dicts(monkeypatch) -> None:
    sentinel_registry = object()
    monkeypatch.setattr(api_mod, "_global_operas_registry", sentinel_registry)
    monkeypatch.setattr(
        integration,
        "build_register_dicts",
        lambda registry=None: {"dmdd.LZSI2024": lambda x: x + 1.0},
    )

    old_func_locals = integration.func_locals
    old_numeric_funcs = integration.numeric_funcs
    refreshed = refresh_sympy_dicts_if_global_registry(sentinel_registry)

    expr = sp.sympify("dmdd.LZSI2024(mchi)", locals=integration.func_locals)
    num_expr = lambdify(["mchi"], expr, modules=[integration.numeric_funcs, "numpy"])
    assert refreshed is True
    assert old_func_locals is integration.func_locals
    assert old_numeric_funcs is integration.numeric_funcs
    assert float(num_expr(2.0)) == 3.0


def test_refresh_sympy_dicts_if_global_registry_skips_non_global_registry(monkeypatch) -> None:
    sentinel_registry = object()
    monkeypatch.setattr(api_mod, "_global_operas_registry", sentinel_registry)
    assert refresh_sympy_dicts_if_global_registry(object()) is False


def test_integration_does_not_access_api_private_global_symbol() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    body = (repo_root / "jarvis_operas" / "integration.py").read_text(encoding="utf-8")
    assert "._global_operas_registry" not in body
    assert 'getattr(_api, "_global_operas_registry"' not in body
