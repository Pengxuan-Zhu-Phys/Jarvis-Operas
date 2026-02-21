from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from jarvis_operas import get_global_registry
from jarvis_operas.registry import OperatorRegistry


def test_acall_supports_async_operator() -> None:
    registry = OperatorRegistry()

    async def async_add(a, b):
        await asyncio.sleep(0)
        return a + b

    registry.register("async_add", async_add, namespace="core")

    result = asyncio.run(registry.acall("core:async_add", a=2, b=5))
    assert result == 7


def test_acall_offloads_sync_operator() -> None:
    registry = OperatorRegistry()

    def slow_add(a, b):
        time.sleep(0.2)
        return a + b

    registry.register("slow_add", slow_add, namespace="core")

    async def runner() -> tuple[float, int]:
        start = time.perf_counter()
        tick_task = asyncio.create_task(asyncio.sleep(0.02, result=1))
        op_task = asyncio.create_task(registry.acall("core:slow_add", a=3, b=4))

        await tick_task
        tick_elapsed = time.perf_counter() - start
        value = await op_task
        return tick_elapsed, value

    tick_elapsed, value = asyncio.run(runner())

    assert value == 7
    assert tick_elapsed < 0.15


def test_acall_helper_many_runs_calls_concurrently() -> None:
    registry = OperatorRegistry()

    def slow_square(x):
        time.sleep(0.2)
        return x * x

    registry.register("slow_square", slow_square, namespace="helper")

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
    registry = OperatorRegistry()
    registry.register("inc", lambda x: x + 1, namespace="helper")

    values = asyncio.run(
        registry.acall_many(
            "helper:inc",
            [{"x": 1}, {"x": 3}],
        )
    )

    assert values == [2, 4]


def test_builtin_ops_support_numpy_and_pandas_async() -> None:
    registry = get_global_registry()

    async def runner():
        add_np = await registry.acall(
            "math:add",
            a=np.array([1.0, 2.0]),
            b=np.array([3.0, 4.0]),
        )
        add_pd = await registry.acall(
            "math:add",
            a=pd.Series([1.0, 2.0]),
            b=pd.Series([3.0, 4.0]),
        )
        egg_pd = await registry.acall(
            "helper:eggbox",
            inputs={"x": pd.Series([0.0, 0.5]), "y": pd.Series([0.0, 0.0])},
        )
        chi2_pd = await registry.acall(
            "stat:chi2_cov",
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


def test_builtin_ops_support_observables_dict_async() -> None:
    registry = get_global_registry()

    async def runner():
        add_value = await registry.acall("math:add", observables={"a": 1.0, "b": 2.0})
        identity_value = await registry.acall("math:identity", observables={"x": 3.0})
        chi2_value = await registry.acall(
            "stat:chi2_cov",
            observables={
                "residual": [1.0, 0.0],
                "cov": [[2.0, 0.0], [0.0, 1.0]],
            },
        )
        egg_value = await registry.acall(
            "helper:eggbox",
            observables={"x": 0.5, "y": 0.0},
        )
        return add_value, identity_value, chi2_value, egg_value

    add_value, identity_value, chi2_value, egg_value = asyncio.run(runner())

    assert add_value == pytest.approx(3.0)
    assert identity_value == pytest.approx(3.0)
    assert chi2_value == pytest.approx(0.5)
    assert egg_value == pytest.approx(243.0)
