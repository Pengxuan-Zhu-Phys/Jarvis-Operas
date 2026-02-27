from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from jarvis_operas import OperaFunction, Operas, OperasRegistry
from jarvis_operas.errors import OperatorCallError


def test_operas_numpy_backend_dispatch() -> None:
    registry = OperasRegistry()
    registry.register(
        OperaFunction(
            namespace="math",
            name="add_vec",
            arity=2,
            return_dtype=None,
            numpy_impl=lambda a, b: a + b,
        )
    )
    operas = Operas(registry, numpy_concurrency=2)

    left = np.array([1.0, 2.0, 3.0])
    right = np.array([4.0, 5.0, 6.0])
    value = operas.call("math.add_vec", left, right, backend="numpy")
    assert np.allclose(value, np.array([5.0, 7.0, 9.0]))


def test_operas_acall_numpy_backend_is_non_blocking() -> None:
    registry = OperasRegistry()

    def slow_add(a, b):
        time.sleep(0.2)
        return a + b

    registry.register(
        OperaFunction(
            namespace="math",
            name="slow_add",
            arity=2,
            return_dtype=None,
            numpy_impl=slow_add,
        )
    )
    operas = Operas(registry, numpy_concurrency=1)

    async def _runner() -> tuple[float, np.ndarray]:
        start = time.perf_counter()
        tick_task = asyncio.create_task(asyncio.sleep(0.02, result=True))
        call_task = asyncio.create_task(
            operas.acall(
                "math.slow_add",
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                backend="numpy",
            )
        )
        await tick_task
        elapsed = time.perf_counter() - start
        value = await call_task
        return elapsed, value

    elapsed, value = asyncio.run(_runner())
    assert elapsed < 0.15
    assert np.allclose(value, np.array([4.0, 6.0]))


def test_operas_polars_native_expr_dispatch() -> None:
    pl = pytest.importorskip("polars")
    registry = OperasRegistry()

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("numpy_impl should not be used for native polars dispatch")

    registry.register(
        OperaFunction(
            namespace="math",
            name="sum_native",
            arity=2,
            return_dtype=pl.Float64,
            numpy_impl=_should_not_run,
            polars_expr_impl=lambda x, y: x + y,
        )
    )

    operas = Operas(registry)
    expr = operas.call("math.sum_native", pl.col("a"), pl.col("b"), backend="polars")
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    values = frame.select(expr.alias("z")).get_column("z").to_list()
    assert values == [4.0, 6.0]


def test_operas_polars_fallback_map_batches_dispatch() -> None:
    pl = pytest.importorskip("polars")
    registry = OperasRegistry()

    registry.register(
        OperaFunction(
            namespace="math",
            name="sum_fallback",
            arity=2,
            return_dtype=pl.Float64,
            numpy_impl=lambda x, y: x + y,
            polars_expr_impl=None,
        )
    )

    operas = Operas(registry)
    expr = operas.call("math.sum_fallback", pl.col("a"), pl.col("b"), backend="polars")
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    values = frame.select(expr.alias("z")).get_column("z").to_list()
    assert values == [4.0, 6.0]


def test_operas_polars_fallback_requires_return_dtype() -> None:
    pl = pytest.importorskip("polars")
    registry = OperasRegistry()

    registry.register(
        OperaFunction(
            namespace="math",
            name="missing_dtype",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x,
            polars_expr_impl=None,
        )
    )

    operas = Operas(registry)
    with pytest.raises(OperatorCallError):
        operas.call("math.missing_dtype", pl.col("a"), backend="polars")
