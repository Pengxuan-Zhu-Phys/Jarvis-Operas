from __future__ import annotations

import pytest

from jarvis_operas import (
    OperatorCallError,
    OperatorConflict,
    get_global_registry,
)
from jarvis_operas.registry import OperatorRegistry


def test_conflict_rules_for_core_and_user() -> None:
    registry = OperatorRegistry()

    registry.register("chi2", lambda x: x, namespace="core")

    with pytest.raises(OperatorConflict):
        registry.register("chi2", lambda x: x, namespace="core")

    with pytest.raises(OperatorConflict):
        registry.register("chi2", lambda x: x, namespace="user")


def test_call_and_call_error_wrapping() -> None:
    registry = OperatorRegistry()
    registry.register("add", lambda a, b: a + b, namespace="core")
    assert registry.call("core:add", a=2, b=3) == 5

    def boom() -> None:
        raise ValueError("boom")

    registry.register("boom", boom, namespace="user")
    with pytest.raises(OperatorCallError) as exc:
        registry.call("user:boom")

    assert exc.value.operator == "user:boom"
    assert isinstance(exc.value.__cause__, ValueError)


def test_list_and_info() -> None:
    registry = OperatorRegistry()

    def scale(x: float, factor: float = 2.0) -> float:
        """Scale one value by a factor."""

        return x * factor

    registry.register(
        "scale",
        scale,
        namespace="user",
        metadata={"kind": "transform"},
    )

    assert registry.list() == ["user:scale"]
    assert registry.list(namespace="user") == ["user:scale"]

    info = registry.info("user:scale")
    assert info["name"] == "user:scale"
    assert info["signature"] == "(x: 'float', factor: 'float' = 2.0) -> 'float'"
    assert info["docstring"] == "Scale one value by a factor."
    assert info["metadata"] == {"kind": "transform"}


def test_global_registry_contains_builtin_ops() -> None:
    registry = get_global_registry()

    assert "core:add" in registry.list(namespace="core")
    assert registry.call("core:add", a=1, b=4) == 5
