from __future__ import annotations

import asyncio
import time

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
