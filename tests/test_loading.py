from __future__ import annotations

import textwrap

import pytest

from jarvis_operas import (
    OperatorLoadError,
    OperasRegistry,
    func_locals,
    load_user_ops,
    numeric_funcs,
)
from jarvis_operas.api import get_global_operas_registry
from jarvis_operas.integration import refresh_sympy_dicts_if_global_registry


def test_load_user_ops_with_decorator(tmp_path) -> None:
    op_file = tmp_path / "decor_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            from jarvis_operas import oper

            @oper("double", tag="demo")
            def double(x):
                return x * 2
            """
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["decor_ops.double"]
    assert registry.call("decor_ops.double", x=6) == 12
    assert registry.info("decor_ops.double")["metadata"]["tag"] == "demo"


def test_load_user_ops_with_export_whitelist(tmp_path) -> None:
    op_file = tmp_path / "whitelist_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def triple(x):
                return x * 3

            __JARVIS_OPERAS__ = {
                "triple": triple,
            }
            """
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["whitelist_ops.triple"]
    assert registry.call("whitelist_ops.triple", x=5) == 15


def test_load_user_ops_wraps_import_errors(tmp_path) -> None:
    op_file = tmp_path / "broken_ops.py"
    op_file.write_text("def broken(:\n", encoding="utf-8")

    registry = OperasRegistry()

    with pytest.raises(OperatorLoadError) as exc:
        load_user_ops(str(op_file), registry)

    assert "Failed to load operators" in str(exc.value)
    assert exc.value.__cause__ is not None


def test_load_user_ops_refreshes_sympy_dicts_for_global_registry(tmp_path) -> None:
    op_file = tmp_path / "refresh_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def plus_seven(x):
                return x + 7

            __JARVIS_OPERAS__ = {
                "plus_seven": plus_seven,
            }
            """
        ),
        encoding="utf-8",
    )

    namespace = "autorefresh_ops"
    full_name = f"{namespace}.plus_seven"
    registry = get_global_operas_registry()
    registry.delete_namespace(namespace)
    refresh_sympy_dicts_if_global_registry(registry)
    assert namespace not in func_locals

    try:
        loaded = load_user_ops(str(op_file), registry, namespace=namespace)
        assert full_name in loaded
        assert namespace in func_locals
        symbol_name = str(getattr(func_locals[namespace], "plus_seven"))
        assert symbol_name in numeric_funcs
    finally:
        registry.delete_namespace(namespace)
        refresh_sympy_dicts_if_global_registry(registry)


def test_load_user_ops_rejects_protected_namespace(tmp_path) -> None:
    op_file = tmp_path / "ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def foo(x):
                return x

            __JARVIS_OPERAS__ = {
                "foo": foo,
            }
            """
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    with pytest.raises(OperatorLoadError) as exc:
        load_user_ops(str(op_file), registry, namespace="math")

    assert "protected" in str(exc.value).lower()
