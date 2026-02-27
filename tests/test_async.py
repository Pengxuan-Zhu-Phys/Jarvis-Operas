from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from jarvis_operas import OperaFunction, OperasRegistry
from jarvis_operas.api import get_global_operas_registry


def _register(
    registry: OperasRegistry,
    *,
    namespace: str,
    name: str,
    fn,
    arity: int | None = None,
) -> None:
    registry.register(
        OperaFunction(
            namespace=namespace,
            name=name,
            arity=arity,
            return_dtype=None,
            numpy_impl=fn,
        )
    )


def test_acall_supports_async_operator() -> None:
    registry = OperasRegistry()

    async def async_add(a, b):
        await asyncio.sleep(0)
        return a + b

    _register(registry, namespace="core", name="async_add", fn=async_add, arity=2)

    result = asyncio.run(registry.acall("core.async_add", a=2, b=5))
    assert result == 7


def test_acall_offloads_sync_operator() -> None:
    registry = OperasRegistry()

    def slow_add(a, b):
        time.sleep(0.2)
        return a + b

    _register(registry, namespace="core", name="slow_add", fn=slow_add, arity=2)

    async def runner() -> tuple[float, int]:
        start = time.perf_counter()
        tick_task = asyncio.create_task(asyncio.sleep(0.02, result=1))
        op_task = asyncio.create_task(registry.acall("core.slow_add", a=3, b=4))

        await tick_task
        tick_elapsed = time.perf_counter() - start
        value = await op_task
        return tick_elapsed, value

    tick_elapsed, value = asyncio.run(runner())

    assert value == 7
    assert tick_elapsed < 0.15


def test_acall_helper_many_runs_calls_concurrently() -> None:
    registry = OperasRegistry()

    def slow_square(x):
        time.sleep(0.2)
        return x * x

    _register(registry, namespace="helper", name="slow_square", fn=slow_square, arity=1)

    async def runner() -> tuple[float, list[int]]:
        start = time.perf_counter()
        values = await registry.acall_helper_many(
            "slow_square",
            [{"x": 2}, {"x": 3}, {"x": 4}],
        )
        elapsed = time.perf_counter() - start
        return elapsed, values

    elapsed, values = asyncio.run(runner())

    assert values == [4, 9, 16]
    assert elapsed < 0.55


def test_acall_many_accepts_full_operator_name() -> None:
    registry = OperasRegistry()
    _register(registry, namespace="helper", name="inc", fn=lambda x: x + 1, arity=1)

    values = asyncio.run(
        registry.acall_many(
            "helper.inc",
            [{"x": 1}, {"x": 3}],
        )
    )

    assert values == [2, 4]


def test_acall_many_default_concurrency_is_bounded_by_registry_limit() -> None:
    registry = OperasRegistry(numpy_concurrency=2)

    def slow_inc(x):
        time.sleep(0.1)
        return x + 1

    _register(registry, namespace="helper", name="slow_inc", fn=slow_inc, arity=1)

    async def runner() -> tuple[float, list[int]]:
        start = time.perf_counter()
        values = await registry.acall_many(
            "helper.slow_inc",
            [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}],
        )
        elapsed = time.perf_counter() - start
        return elapsed, values

    elapsed, values = asyncio.run(runner())
    assert values == [2, 3, 4, 5]
    assert elapsed >= 0.16
    assert elapsed < 0.60


def test_acall_uses_registry_numpy_concurrency_gate() -> None:
    registry = OperasRegistry(numpy_concurrency=2)

    def slow_square(x):
        time.sleep(0.1)
        return x * x

    _register(registry, namespace="core", name="slow_square", fn=slow_square, arity=1)

    async def runner() -> tuple[float, list[int]]:
        start = time.perf_counter()
        tasks = [
            asyncio.create_task(registry.acall("core.slow_square", x=2)),
            asyncio.create_task(registry.acall("core.slow_square", x=3)),
            asyncio.create_task(registry.acall("core.slow_square", x=4)),
            asyncio.create_task(registry.acall("core.slow_square", x=5)),
        ]
        values = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        return elapsed, values

    elapsed, values = asyncio.run(runner())
    assert values == [4, 9, 16, 25]
    assert elapsed >= 0.16
    assert elapsed < 0.60


def test_builtin_ops_support_numpy_and_pandas_async() -> None:
    registry = get_global_operas_registry()

    async def runner():
        add_np = await registry.acall(
            "math.add",
            a=np.array([1.0, 2.0]),
            b=np.array([3.0, 4.0]),
        )
        add_pd = await registry.acall(
            "math.add",
            a=pd.Series([1.0, 2.0]),
            b=pd.Series([3.0, 4.0]),
        )
        egg_pd = await registry.acall(
            "helper.eggbox",
            x=pd.Series([0.0, 0.5]),
            y=pd.Series([0.0, 0.0]),
        )
        chi2_pd = await registry.acall(
            "stat.chi2_cov",
            residual=pd.DataFrame({"x": [1.0], "y": [0.0]}),
            cov=pd.DataFrame([[2.0, 0.0], [0.0, 1.0]], columns=["x", "y"], index=["x", "y"]),
        )
        return add_np, add_pd, egg_pd, chi2_pd

    add_np, add_pd, egg_pd, chi2_pd = asyncio.run(runner())

    assert np.allclose(add_np, np.array([4.0, 6.0]))
    assert isinstance(add_pd, pd.Series)
    assert np.allclose(add_pd.to_numpy(), np.array([4.0, 6.0]))
    assert isinstance(egg_pd, pd.Series)
    assert np.allclose(egg_pd.to_numpy(), np.array([32.0, 243.0]))
    assert isinstance(chi2_pd, pd.Series)
    assert np.allclose(chi2_pd.to_numpy(), np.array([0.5]))


def test_builtin_ops_support_scalar_kwargs_async() -> None:
    registry = get_global_operas_registry()

    async def runner():
        add_value = await registry.acall("math.add", a=1.0, b=2.0)
        identity_value = await registry.acall("math.identity", x=3.0)
        chi2_value = await registry.acall(
            "stat.chi2_cov",
            residual=[1.0, 0.0],
            cov=[[2.0, 0.0], [0.0, 1.0]],
        )
        egg_value = await registry.acall(
            "helper.eggbox",
            x=0.5,
            y=0.0,
        )
        return add_value, identity_value, chi2_value, egg_value

    add_value, identity_value, chi2_value, egg_value = asyncio.run(runner())

    assert add_value == pytest.approx(3.0)
    assert identity_value == pytest.approx(3.0)
    assert chi2_value == pytest.approx(0.5)
    assert egg_value == pytest.approx(243.0)
