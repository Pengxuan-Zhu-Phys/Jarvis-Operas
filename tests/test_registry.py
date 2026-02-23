from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import re

from jarvis_operas import (
    OperatorCallError,
    OperatorConflict,
    OperatorNotFound,
    get_global_registry,
)
from jarvis_operas.registry import OperatorRegistry


def test_conflict_rules_for_same_namespace_only() -> None:
    registry = OperatorRegistry()

    registry.register("chi2", lambda x: x, namespace="stat")

    with pytest.raises(OperatorConflict):
        registry.register("chi2", lambda x: x, namespace="stat")

    # Different namespaces can reuse short names.
    registry.register("chi2", lambda x: x, namespace="helper")


def test_call_and_call_error_wrapping() -> None:
    registry = OperatorRegistry()
    registry.register("add", lambda a, b: a + b, namespace="math")
    assert registry.call("math:add", a=2, b=3) == 5

    def boom() -> None:
        raise ValueError("boom")

    registry.register("boom", boom, namespace="bench")
    with pytest.raises(OperatorCallError) as exc:
        registry.call("bench:boom")

    assert exc.value.operator == "bench:boom"
    assert isinstance(exc.value.__cause__, ValueError)


def test_list_and_info() -> None:
    registry = OperatorRegistry()

    def scale(x: float, factor: float = 2.0) -> float:
        """Scale one value by a factor."""

        return x * factor

    registry.register(
        "scale",
        scale,
        namespace="script_ops",
        metadata={"kind": "transform"},
    )

    assert registry.list() == ["script_ops:scale"]
    assert registry.list(namespace="script_ops") == ["script_ops:scale"]

    info = registry.info("script_ops:scale")
    assert info["name"] == "script_ops:scale"
    assert info["signature"] == "(x: 'float', factor: 'float' = 2.0) -> 'float'"
    assert info["docstring"] == "Scale one value by a factor."
    assert info["metadata"] == {"kind": "transform"}
    assert info["is_async"] is False
    assert info["supports_async"] is True


def test_info_prefers_metadata_summary_then_note() -> None:
    registry = OperatorRegistry()

    def f(x):
        """Docstring summary line."""

        return x

    registry.register(
        "f_summary",
        f,
        namespace="meta",
        metadata={"summary": "Summary from metadata", "note": "Long note"},
    )
    info_summary = registry.info("meta:f_summary")
    assert info_summary["docstring"] == "Summary from metadata"

    registry.register(
        "f_note",
        f,
        namespace="meta",
        metadata={"note": "Summary from note"},
    )
    info_note = registry.info("meta:f_note")
    assert info_note["docstring"] == "Summary from note"

    registry.register("f_doc", f, namespace="meta")
    info_doc = registry.info("meta:f_doc")
    assert info_doc["docstring"] == "Docstring summary line."


def test_global_registry_contains_builtin_ops() -> None:
    registry = get_global_registry()

    assert "math:add" in registry.list(namespace="math")
    assert registry.call("math:add", a=1, b=4) == 5


def test_eggbox_builtin_supports_scalar_and_array_observables() -> None:
    registry = get_global_registry()

    assert "helper:eggbox" in registry.list(namespace="helper")
    assert "helper:eggbox2d" in registry.list(namespace="helper")

    scalar = registry.call("helper:eggbox", observables={"x": 0.5, "y": 0.0})
    assert scalar == pytest.approx(243.0)

    values = registry.call(
        "helper:eggbox",
        observables={"x": np.array([0.0, 0.5]), "y": np.array([0.0, 0.0])},
    )
    assert np.allclose(values, np.array([32.0, 243.0]))

    info = registry.info("helper:eggbox")
    assert info["metadata"]["category"] == "hep_scanner_benchmark"


def test_eggbox2d_builtin_returns_mapping_payload() -> None:
    registry = get_global_registry()

    scalar_result = registry.call("helper:eggbox2d", observables={"x": 0.5, "y": 0.0})
    assert scalar_result["z"] == pytest.approx(243.0)

    array_result = registry.call(
        "helper:eggbox2d",
        observables={"x": np.array([0.0, 0.5]), "y": np.array([0.0, 0.0])},
        sample_info={"uuid": "s-01"},
        cfg={"name": "EggBox"},
    )
    assert np.allclose(array_result["z"], np.array([32.0, 243.0]))


def test_eggbox_builtin_validates_observables_mapping() -> None:
    registry = get_global_registry()

    with pytest.raises(OperatorCallError) as exc:
        registry.call("helper:eggbox", observables={"x": 0.5})

    assert isinstance(exc.value.__cause__, ValueError)


def test_helper_namespace_sets_concurrent_metadata() -> None:
    registry = OperatorRegistry()
    registry.register(
        "demo",
        lambda x: x,
        namespace="helper",
        metadata={"note": "test"},
    )

    info = registry.info("helper:demo")
    assert info["metadata"]["concurrent"] is True
    assert info["metadata"]["note"] == "test"


def test_delete_and_rename_function() -> None:
    registry = OperatorRegistry()
    registry.register("foo", lambda x: x + 1, namespace="tmp")

    info_before = registry.info("tmp:foo")
    operator_id = info_before["id"]
    assert re.match(r"^[a-z][0-9]{3,5}$", operator_id)

    renamed = registry.rename("tmp:foo", new_full_name="user:bar")
    assert renamed == "user:bar"
    assert registry.call("user:bar", x=1) == 2
    # ID should stay usable after rename.
    assert registry.info("user:bar")["id"] == operator_id

    registry.delete(operator_id)
    with pytest.raises(OperatorNotFound):
        registry.get("user:bar")


def test_delete_and_rename_namespace() -> None:
    registry = OperatorRegistry()
    registry.register("a", lambda x: x, namespace="ns")
    registry.register("b", lambda x: x, namespace="ns")

    updated = registry.rename_namespace("ns", "newns")
    assert sorted(updated) == ["newns:a", "newns:b"]
    assert "newns:a" in registry.list(namespace="newns")
    assert "newns:b" in registry.list(namespace="newns")

    deleted = registry.delete_namespace("newns")
    assert sorted(deleted) == ["newns:a", "newns:b"]
    assert registry.list(namespace="newns") == []


def test_rename_namespace_conflict() -> None:
    registry = OperatorRegistry()
    registry.register("same", lambda x: x, namespace="old")
    registry.register("same", lambda x: x, namespace="target")

    with pytest.raises(OperatorConflict):
        registry.rename_namespace("old", "target")


def test_operator_ids_unique_across_namespaces() -> None:
    registry = OperatorRegistry()
    registry.register("x", lambda v: v, namespace="math")
    registry.register("x", lambda v: v, namespace="stat")
    registry.register("x", lambda v: v, namespace="helper")

    ids = {
        registry.info("math:x")["id"],
        registry.info("stat:x")["id"],
        registry.info("helper:x")["id"],
    }
    for operator_id in ids:
        assert re.match(r"^[a-z][0-9]{3,5}$", operator_id)
    assert len(ids) == 3


def test_builtin_ops_support_numpy_and_pandas_sync() -> None:
    registry = get_global_registry()

    np_add = registry.call("math:add", a=np.array([1.0, 2.0]), b=np.array([3.0, 4.0]))
    assert np.allclose(np_add, np.array([4.0, 6.0]))

    pd_add = registry.call(
        "math:add",
        a=pd.Series([1.0, 2.0], index=["a", "b"]),
        b=pd.Series([3.0, 4.0], index=["a", "b"]),
    )
    assert isinstance(pd_add, pd.Series)
    assert pd_add.to_dict() == {"a": 4.0, "b": 6.0}

    pd_identity_input = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    pd_identity = registry.call("math:identity", x=pd_identity_input)
    assert pd_identity is pd_identity_input

    pd_eggbox = registry.call(
        "helper:eggbox",
        observables={"x": pd.Series([0.0, 0.5]), "y": pd.Series([0.0, 0.0])},
    )
    assert isinstance(pd_eggbox, pd.Series)
    assert np.allclose(pd_eggbox.to_numpy(), np.array([32.0, 243.0]))

    pd_residual = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [0.0, 1.0],
        },
        index=["s1", "s2"],
    )
    pd_cov = pd.DataFrame(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        index=["x", "y"],
        columns=["x", "y"],
    )
    pd_chi2 = registry.call("stat:chi2_cov", residual=pd_residual, cov=pd_cov)
    assert isinstance(pd_chi2, pd.Series)
    assert list(pd_chi2.index) == ["s1", "s2"]
    assert np.allclose(pd_chi2.to_numpy(), np.array([0.5, 3.0]))


def test_builtin_ops_support_observables_dict_sync() -> None:
    registry = get_global_registry()

    assert registry.call("math:add", observables={"a": 2.0, "b": 3.0}) == 5.0
    assert registry.call("math:identity", observables={"x": 4.0}) == 4.0

    chi2 = registry.call(
        "stat:chi2_cov",
        observables={
            "residual": [1.0, 0.0],
            "cov": [[2.0, 0.0], [0.0, 1.0]],
        },
    )
    assert chi2 == pytest.approx(0.5)

    egg = registry.call(
        "helper:eggbox",
        observables={"x": 0.5, "y": 0.0},
    )
    assert egg == pytest.approx(243.0)
