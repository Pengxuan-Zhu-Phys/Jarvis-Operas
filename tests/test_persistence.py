from __future__ import annotations

import textwrap

import pytest

from jarvis_operas.errors import OperatorNotFound
from jarvis_operas.persistence import (
    delete_persisted_function,
    delete_persisted_namespace,
    list_persisted_user_ops,
    persist_user_ops,
    update_persisted_function,
    update_persisted_namespace,
)


def test_global_registry_autoloads_persisted_user_ops(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "auto_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def square(x):
                return x * x

            __JARVIS_OPERAS__ = {
                "square": square,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    persist_user_ops(str(op_file))
    entries = list_persisted_user_ops()
    assert entries == [{"path": str(op_file.resolve())}]

    # Reset the singleton to emulate a fresh process bootstrap.
    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_registry", None)
    registry = api.get_global_registry()

    assert registry.call("auto_ops.square", x=4) == 16


def test_persisted_function_delete_and_update_overrides(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "func_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def inc(x):
                return x + 1

            __JARVIS_OPERAS__ = {
                "inc": inc,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    persist_user_ops(str(op_file))
    update_persisted_function("func_ops.inc", "math.inc_user")
    delete_persisted_function("math.add")

    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_registry", None)
    registry = api.get_global_registry()

    assert registry.call("math.inc_user", x=2) == 3
    with pytest.raises(OperatorNotFound):
        registry.get("func_ops.inc")
    with pytest.raises(OperatorNotFound):
        registry.get("math.add")


def test_persisted_namespace_delete_and_update_overrides(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "ns_ops.py"
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
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    persist_user_ops(str(op_file))
    update_persisted_namespace("ns_ops", "my_ns")

    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_registry", None)
    registry = api.get_global_registry()
    assert registry.call("my_ns.foo", x=7) == 7

    delete_persisted_namespace("my_ns")
    monkeypatch.setattr(api, "_global_registry", None)
    registry = api.get_global_registry()

    with pytest.raises(OperatorNotFound):
        registry.get("my_ns.foo")
