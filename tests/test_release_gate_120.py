from __future__ import annotations

import asyncio
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from jarvis_operas.catalog import get_catalog_declarations
from jarvis_operas.core.registry import OperasRegistry


def _build_release_registry() -> OperasRegistry:
    registry = OperasRegistry()
    registry.register_many(
        tuple(
            get_catalog_declarations(
                include_interpolations=True,
            )
        )
    )
    return registry


def _assert_numpy_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(actual.keys()) == set(expected.keys())
        for key, expected_value in expected.items():
            _assert_numpy_equal(actual[key], expected_value)
        return
    if np.isscalar(expected):
        assert np.isclose(float(actual), float(expected))
        return
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))


def _eval_expr(expr: Any, frame: Any) -> list[Any]:
    return frame.select(expr.alias("__out")).get_column("__out").to_list()


def _assert_polars_values(actual: list[Any], expected: list[Any]) -> None:
    assert len(actual) == len(expected)
    if not expected:
        return
    if isinstance(expected[0], Mapping):
        for left, right in zip(actual, expected):
            assert isinstance(left, Mapping)
            assert set(left.keys()) == set(right.keys())
            for key in right.keys():
                assert np.isclose(float(left[key]), float(right[key]))
        return
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))


def _assert_numeric_close(actual: Any, expected: Any) -> None:
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    assert np.allclose(actual_arr, expected_arr, equal_nan=True)


def test_release_120_all_functions_have_numpy_and_polars_support() -> None:
    registry = _build_release_registry()

    missing: list[str] = []
    for full_name in registry.list():
        info = registry.info(full_name)
        capabilities = info.get("capabilities", {})
        declared_polars = capabilities.get("polars")
        has_polars = bool(info["supports_polars_native"] or info["supports_polars_fallback"])
        if not bool(info["supports_numpy"]):
            missing.append(full_name)
            continue
        if declared_polars is False:
            continue
        if not has_polars:
            missing.append(full_name)

    assert missing == []


def test_release_120_fixed_functions_support_scalar_array_sync_async_and_dual_backends() -> None:
    pl = pytest.importorskip("polars")
    registry = _build_release_registry()

    fixed_cases: dict[str, dict[str, Any]] = {
        "math.add": {
            "numpy_scalar_kwargs": {"a": 1.0, "b": 2.0},
            "numpy_scalar_expected": 3.0,
            "numpy_array_kwargs": {
                "a": np.array([1.0, 2.0]),
                "b": np.array([3.0, 4.0]),
            },
            "numpy_array_expected": np.array([4.0, 6.0]),
            "polars_args": (),
            "polars_kwargs": {"a": pl.col("a"), "b": pl.col("b")},
            "polars_frame_single": pl.DataFrame({"a": [1.0], "b": [2.0]}),
            "polars_expected_single": [3.0],
            "polars_frame_many": pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            "polars_expected_many": [4.0, 6.0],
        },
        "math.identity": {
            "numpy_scalar_kwargs": {"x": 2.5},
            "numpy_scalar_expected": 2.5,
            "numpy_array_kwargs": {"x": np.array([1.0, 2.0])},
            "numpy_array_expected": np.array([1.0, 2.0]),
            "polars_args": (),
            "polars_kwargs": {"x": pl.col("x")},
            "polars_frame_single": pl.DataFrame({"x": [2.5]}),
            "polars_expected_single": [2.5],
            "polars_frame_many": pl.DataFrame({"x": [1.0, 2.0]}),
            "polars_expected_many": [1.0, 2.0],
        },
        "helper.eggbox": {
            "numpy_scalar_kwargs": {"x": 0.5, "y": 0.0},
            "numpy_scalar_expected": 243.0,
            "numpy_array_kwargs": {
                "x": np.array([0.0, 0.5]),
                "y": np.array([0.0, 0.0]),
            },
            "numpy_array_expected": np.array([32.0, 243.0]),
            "polars_args": (),
            "polars_kwargs": {"x": pl.col("x"), "y": pl.col("y")},
            "polars_frame_single": pl.DataFrame({"x": [0.5], "y": [0.0]}),
            "polars_expected_single": [243.0],
            "polars_frame_many": pl.DataFrame({"x": [0.0, 0.5], "y": [0.0, 0.0]}),
            "polars_expected_many": [32.0, 243.0],
        },
        "helper.eggbox2d": {
            "numpy_scalar_kwargs": {"x": 0.5, "y": 0.0},
            "numpy_scalar_expected": {"z": 243.0},
            "numpy_array_kwargs": {
                "x": np.array([0.0, 0.5]),
                "y": np.array([0.0, 0.0]),
            },
            "numpy_array_expected": {"z": np.array([32.0, 243.0])},
            "polars_args": (),
            "polars_kwargs": {"x": pl.col("x"), "y": pl.col("y")},
            "polars_frame_single": pl.DataFrame({"x": [0.5], "y": [0.0]}),
            "polars_expected_single": [{"z": 243.0}],
            "polars_frame_many": pl.DataFrame({"x": [0.0, 0.5], "y": [0.0, 0.0]}),
            "polars_expected_many": [{"z": 32.0}, {"z": 243.0}],
        },
        "stat.chi2_cov": {
            "numpy_scalar_kwargs": {
                "residual": np.array([1.0, 0.0]),
                "cov": np.array([[2.0, 0.0], [0.0, 1.0]]),
            },
            "numpy_scalar_expected": 0.5,
            "numpy_array_kwargs": {
                "residual": np.array([[1.0, 0.0], [2.0, 1.0]]),
                "cov": np.array([[2.0, 0.0], [0.0, 1.0]]),
            },
            "numpy_array_expected": np.array([0.5, 3.0]),
            "polars_args": (),
            "polars_kwargs": {
                "residual": pl.struct(["r1", "r2"]),
                "cov": [[2.0, 0.0], [0.0, 1.0]],
            },
            "polars_frame_single": pl.DataFrame({"r1": [1.0], "r2": [0.0]}),
            "polars_expected_single": [0.5],
            "polars_frame_many": pl.DataFrame({"r1": [1.0, 2.0], "r2": [0.0, 1.0]}),
            "polars_expected_many": [0.5, 3.0],
        },
    }

    for full_name, case in fixed_cases.items():
        sync_scalar = registry.call(full_name, backend="numpy", **case["numpy_scalar_kwargs"])
        sync_array = registry.call(full_name, backend="numpy", **case["numpy_array_kwargs"])
        _assert_numpy_equal(sync_scalar, case["numpy_scalar_expected"])
        _assert_numpy_equal(sync_array, case["numpy_array_expected"])

        async_scalar = asyncio.run(
            registry.acall(full_name, backend="numpy", **case["numpy_scalar_kwargs"])
        )
        async_array = asyncio.run(
            registry.acall(full_name, backend="numpy", **case["numpy_array_kwargs"])
        )
        _assert_numpy_equal(async_scalar, case["numpy_scalar_expected"])
        _assert_numpy_equal(async_array, case["numpy_array_expected"])

        sync_expr = registry.call(
            full_name,
            *case["polars_args"],
            backend="polars",
            **case["polars_kwargs"],
        )
        sync_single = _eval_expr(sync_expr, case["polars_frame_single"])
        sync_many = _eval_expr(sync_expr, case["polars_frame_many"])
        _assert_polars_values(sync_single, case["polars_expected_single"])
        _assert_polars_values(sync_many, case["polars_expected_many"])

        async_expr = asyncio.run(
            registry.acall(
                full_name,
                *case["polars_args"],
                backend="polars",
                **case["polars_kwargs"],
            )
        )
        async_single = _eval_expr(async_expr, case["polars_frame_single"])
        async_many = _eval_expr(async_expr, case["polars_frame_many"])
        _assert_polars_values(async_single, case["polars_expected_single"])
        _assert_polars_values(async_many, case["polars_expected_many"])


def test_release_120_interpolation_functions_support_scalar_array_sync_async_and_dual_backends() -> None:
    pl = pytest.importorskip("polars")
    registry = _build_release_registry()

    interpolation_names = [
        name
        for name in registry.list()
        if name.startswith("interp1.") or name.startswith("dmdd.")
    ]
    assert interpolation_names

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in exp",
            category=RuntimeWarning,
        )

        for full_name in interpolation_names:
            sync_scalar = registry.call(full_name, x=2.0, backend="numpy")
            sync_array = registry.call(full_name, x=np.array([1.5, 2.5]), backend="numpy")
            assert np.asarray(sync_scalar, dtype=float).shape == ()
            assert np.asarray(sync_array, dtype=float).shape == (2,)

            async_scalar = asyncio.run(registry.acall(full_name, x=2.0, backend="numpy"))
            async_array = asyncio.run(
                registry.acall(full_name, x=np.array([1.5, 2.5]), backend="numpy")
            )
            _assert_numeric_close(async_scalar, sync_scalar)
            _assert_numeric_close(async_array, sync_array)

            sync_expr = registry.call(full_name, pl.col("x"), backend="polars")
            sync_single = _eval_expr(sync_expr, pl.DataFrame({"x": [2.0]}))
            sync_many = _eval_expr(sync_expr, pl.DataFrame({"x": [1.5, 2.5]}))
            assert len(sync_single) == 1
            assert len(sync_many) == 2

            async_expr = asyncio.run(registry.acall(full_name, pl.col("x"), backend="polars"))
            async_single = _eval_expr(async_expr, pl.DataFrame({"x": [2.0]}))
            async_many = _eval_expr(async_expr, pl.DataFrame({"x": [1.5, 2.5]}))
            assert len(async_single) == 1
            assert len(async_many) == 2
            _assert_numeric_close(async_single, sync_single)
            _assert_numeric_close(async_many, sync_many)
