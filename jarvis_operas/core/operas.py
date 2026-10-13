from __future__ import annotations

from concurrent.futures import Executor
from typing import Any

from .registry import OperasRegistry
from ..units import UnitCallPlan, UnitCallResult


class Operas:
    """Public function interface with sync + asyncio-compatible execution."""

    def __init__(
        self,
        registry: OperasRegistry | None = None,
        *,
        numpy_concurrency: int | None = None,
        executor: Executor | None = None,
    ) -> None:
        if numpy_concurrency is not None and numpy_concurrency < 1:
            raise ValueError("numpy_concurrency must be >= 1")
        if registry is None:
            if numpy_concurrency is None:
                self.registry = OperasRegistry(executor=executor)
            else:
                self.registry = OperasRegistry(
                    executor=executor,
                    numpy_concurrency=numpy_concurrency,
                )
            return

        self.registry = registry
        if numpy_concurrency is not None:
            self.registry.set_numpy_concurrency(numpy_concurrency)
        if executor is not None:
            self.registry.set_executor(executor)

    def call(
        self,
        name: str,
        *args: Any,
        backend: str = "numpy",
        **kwargs: Any,
    ) -> Any:
        return self.registry.call(name, *args, backend=backend, **kwargs)

    async def acall(
        self,
        name: str,
        *args: Any,
        backend: str = "numpy",
        **kwargs: Any,
    ) -> Any:
        return await self.registry.acall(name, *args, backend=backend, **kwargs)

    def prepare(self, name: str, **unit_options: Any) -> UnitCallPlan:
        return self.registry.prepare(name, **unit_options)

    def call_units(self, name: str, *args: Any, **kwargs: Any) -> UnitCallResult:
        return self.registry.call_units(name, *args, **kwargs)

    async def acall_units(self, name: str, *args: Any, **kwargs: Any) -> UnitCallResult:
        return await self.registry.acall_units(name, *args, **kwargs)

    async def acall_many_units(self, name: str, calls, **kwargs: Any) -> list[Any]:
        return await self.registry.acall_many_units(name, calls, **kwargs)


def get_global_operas_registry() -> OperasRegistry:
    # Single global state is managed in jarvis_operas.api.
    from ..api import get_global_operas_registry as _get_global_operas_registry

    return _get_global_operas_registry()


def get_global_operas(*, numpy_concurrency: int | None = None) -> Operas:
    # Single global state is managed in jarvis_operas.api.
    from ..api import get_global_operas as _get_global_operas

    return _get_global_operas(numpy_concurrency=numpy_concurrency)
