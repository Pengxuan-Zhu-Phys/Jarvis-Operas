from __future__ import annotations

import sys
import threading
from types import ModuleType

import pytest

from jarvis_operas import OperaFunction, OperatorLoadError, OperasRegistry, load_user_ops


OLD_SOURCE = '''from jarvis_operas import oper
@oper("first")
def first(): return 1
@oper("second")
def second(): return 2
'''
NEW_SOURCE = '''from jarvis_operas import oper
@oper("first")
def first(): return 11
@oper("second")
def second(): return 22
@oper("added")
def added(): return 33
'''


@pytest.fixture
def loaded(tmp_path):
    path = tmp_path / "atomic_ops.py"
    path.write_text(OLD_SOURCE)
    registry = OperasRegistry()
    load_user_ops(str(path), registry, refresh_sympy=False)
    return path, registry


def snapshot(registry):
    return {name: (registry.get(name), registry.get_id(name)) for name in registry.list()}


def assert_unchanged(registry, before):
    assert registry.list() == sorted(before)
    for name, (declaration, identifier) in before.items():
        assert registry.get(name) is declaration
        assert registry.get_id(name) == identifier
        assert registry.resolve_name(identifier) == name
    assert registry.call("atomic_ops.first") == 1
    assert registry.call("atomic_ops.second") == 2


@pytest.mark.parametrize("failure", [
    'raise RuntimeError("broken import")',
    '__JARVIS_OPERAS__ = {"extra": lambda: 44, "invalid": 0}',
    '__JARVIS_OPERAS__ = []',
    '@oper("bad", namespace="math")\ndef bad(): return 0',
])
def test_failed_load_preserves_all_declarations_ids_and_module_cache(loaded, failure):
    path, registry = loaded
    before = snapshot(registry)
    modules_before = {name for name in sys.modules if name.startswith("_jarvis_operas_user_")}
    path.write_text(NEW_SOURCE + failure + "\n")
    with pytest.raises(OperatorLoadError, match="Failed to load operators"):
        load_user_ops(str(path), registry, refresh_sympy=False)
    assert_unchanged(registry, before)
    assert {name for name in sys.modules if name.startswith("_jarvis_operas_user_")} == modules_before


def test_successful_load_updates_all_functions_and_preserves_ids(loaded):
    path, registry = loaded
    before = snapshot(registry)
    path.write_text(NEW_SOURCE + '__JARVIS_OPERAS__ = {"extra": lambda: 44}\n')
    assert load_user_ops(str(path), registry, refresh_sympy=False) == [
        "atomic_ops.added", "atomic_ops.extra", "atomic_ops.first", "atomic_ops.second",
    ]
    for name, value in [("first", 11), ("second", 22), ("added", 33), ("extra", 44)]:
        assert registry.call(f"atomic_ops.{name}") == value
        assert registry.resolve_name(registry.get_id(f"atomic_ops.{name}")) == f"atomic_ops.{name}"
    for name, (_, identifier) in before.items():
        assert registry.get_id(name) == identifier


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_initial_load_leaves_no_functions(loaded, interrupted):
    path, _ = loaded
    registry = OperasRegistry()
    failure = "KeyboardInterrupt()" if interrupted else 'RuntimeError("broken")'
    path.write_text(NEW_SOURCE + f"raise {failure}\n")
    with pytest.raises(KeyboardInterrupt if interrupted else OperatorLoadError):
        load_user_ops(str(path), registry, refresh_sympy=False)
    assert registry.list() == []


@pytest.mark.parametrize("concurrent_update", [False, True])
def test_incomplete_load_is_invisible_and_conflicting_commit_is_rejected(loaded, monkeypatch, concurrent_update):
    path, registry = loaded
    before = snapshot(registry)
    signals = ModuleType("_test_operas_load_signals")
    signals.ready = threading.Event()
    signals.finish = threading.Event()
    monkeypatch.setitem(sys.modules, signals.__name__, signals)
    path.write_text(NEW_SOURCE + '''from _test_operas_load_signals import ready, finish
ready.set()
if not finish.wait(5): raise RuntimeError("test synchronization timed out")
''')
    errors = []

    def load():
        try:
            load_user_ops(str(path), registry, refresh_sympy=False)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=load)
    thread.start()
    try:
        assert signals.ready.wait(3)
        assert_unchanged(registry, before)
        if concurrent_update:
            registry.register(OperaFunction(namespace="other", name="value", arity=0,
                                            return_dtype=None, numpy_impl=lambda: 100))
    finally:
        signals.finish.set()
        thread.join(6)
    assert not thread.is_alive()
    if concurrent_update:
        assert len(errors) == 1
        assert isinstance(errors[0], OperatorLoadError)
        assert "changed during" in str(errors[0])
        assert registry.call("other.value") == 100
        registry.delete("other.value")
        assert_unchanged(registry, before)
    else:
        assert errors == []
        assert registry.call("atomic_ops.first") == 11
        assert registry.call("atomic_ops.second") == 22


def test_sympy_refresh_failure_rolls_back_registry_and_views(loaded, monkeypatch):
    from jarvis_operas import api, integration

    path, registry = loaded
    before = snapshot(registry)
    monkeypatch.setattr(api, "_global_operas_registry", registry)
    old_locals = {"old": object()}
    old_numeric = {"old": lambda: 1}
    monkeypatch.setattr(integration, "func_locals", dict(old_locals))
    monkeypatch.setattr(integration, "numeric_funcs", dict(old_numeric))

    def broken_refresh(target):
        assert target is registry
        integration.func_locals.clear()
        integration.numeric_funcs.clear()
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(integration, "refresh_sympy_dicts_if_global_registry", broken_refresh)
    path.write_text(NEW_SOURCE)
    with pytest.raises(OperatorLoadError, match="refresh failed"):
        load_user_ops(str(path), registry)
    assert_unchanged(registry, before)
    assert integration.func_locals == old_locals
    assert integration.numeric_funcs == old_numeric


def test_invalid_persisted_source_is_skipped_without_partial_registration(loaded, monkeypatch):
    from jarvis_operas import persistence

    path, registry = loaded
    before = snapshot(registry)
    path.write_text(NEW_SOURCE + 'raise RuntimeError("broken snapshot")\n')
    monkeypatch.setattr(persistence, "list_persisted_user_ops", lambda **kwargs: [{"path": str(path)}])
    assert persistence.load_persisted_user_ops(registry, refresh_sympy=False) == []
    assert_unchanged(registry, before)
