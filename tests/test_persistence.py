from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from jarvis_operas.errors import OperatorNotFound
from jarvis_operas.persistence import (
    delete_persisted_function,
    delete_persisted_namespace,
    get_overrides_store_path,
    get_sources_store_path,
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
    assert len(entries) == 1
    assert entries[0]["path"] == str(op_file.resolve())
    assert entries[0]["namespace"] == "auto_ops"
    assert Path(entries[0]["load_path"]).exists()

    op_file.unlink()

    # Reset the singleton to emulate a fresh process bootstrap.
    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    registry = api.get_global_operas_registry()

    assert registry.call("auto_ops.square", x=4) == 16


def test_persisted_function_delete_and_update_overrides(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "func_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def inc(x):
                return x + 1

            def dec(x):
                return x - 1

            __JARVIS_OPERAS__ = {
                "inc": inc,
                "dec": dec,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    persist_user_ops(str(op_file))
    update_persisted_function("func_ops.inc", "renamed_ops.inc_user")
    delete_persisted_function("func_ops.dec")

    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    registry = api.get_global_operas_registry()

    assert registry.call("renamed_ops.inc_user", x=2) == 3
    with pytest.raises(OperatorNotFound):
        registry.get("func_ops.inc")
    with pytest.raises(OperatorNotFound):
        registry.get("func_ops.dec")


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

    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    registry = api.get_global_operas_registry()
    assert registry.call("my_ns.foo", x=7) == 7

    delete_persisted_namespace("my_ns")
    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    registry = api.get_global_operas_registry()

    with pytest.raises(OperatorNotFound):
        registry.get("my_ns.foo")


def test_persistence_rejects_protected_namespace_overrides(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    with pytest.raises(ValueError):
        delete_persisted_namespace("math")
    with pytest.raises(ValueError):
        update_persisted_namespace("user_ns", "helper")
    with pytest.raises(ValueError):
        delete_persisted_function("stat.chi2_cov")
    with pytest.raises(ValueError):
        update_persisted_function("user_ns.f", "math.f")


def test_persistence_splits_sources_and_overrides_files(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "split_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def f(x):
                return x + 1

            __JARVIS_OPERAS__ = {
                "f": f,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(store_path))

    persist_user_ops(str(op_file))
    update_persisted_function("split_ops.f", "split_ns.f2")

    sources_path = get_sources_store_path()
    overrides_path = get_overrides_store_path()
    assert sources_path.exists()
    assert overrides_path.exists()
    assert sources_path.name == "sources.json"
    assert overrides_path.name == "overrides.json"

    sources_payload = json.loads(sources_path.read_text(encoding="utf-8"))
    overrides_payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert isinstance(sources_payload.get("entries"), list)
    assert sources_payload["entries"][0]["path"] == str(op_file.resolve())
    assert "overrides" not in sources_payload
    assert overrides_payload["overrides"]["renamed_functions"]["split_ops.f"] == "split_ns.f2"
    assert "entries" not in overrides_payload


def test_persistence_can_read_legacy_combined_store(tmp_path, monkeypatch) -> None:
    op_file = tmp_path / "legacy_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def plus(x):
                return x + 5

            __JARVIS_OPERAS__ = {
                "plus": plus,
            }
            """
        ),
        encoding="utf-8",
    )
    legacy_store = tmp_path / "persist_store.json"
    legacy_store.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [{"path": str(op_file.resolve())}],
                "overrides": {
                    "deleted_functions": [],
                    "renamed_functions": {"legacy_ops.plus": "legacy_ns.plus2"},
                    "deleted_namespaces": [],
                    "renamed_namespaces": {},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(legacy_store))

    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    registry = api.get_global_operas_registry()

    assert registry.call("legacy_ns.plus2", x=10) == 15
    with pytest.raises(OperatorNotFound):
        registry.get("legacy_ops.plus")
