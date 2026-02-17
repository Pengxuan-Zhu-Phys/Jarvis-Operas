from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any

from .builtins import register_builtins
from .registry import OperatorRegistry


_registry_override: ContextVar[OperatorRegistry | None] = ContextVar(
    "jarvis_operas_registry_override",
    default=None,
)
_global_registry: OperatorRegistry | None = None
_global_registry_lock = Lock()


def get_global_registry() -> OperatorRegistry:
    global _global_registry

    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                registry = OperatorRegistry()
                register_builtins(registry)
                _global_registry = registry

    return _global_registry


def _resolve_registry(registry: OperatorRegistry | None = None) -> OperatorRegistry:
    if registry is not None:
        return registry

    overridden = _registry_override.get()
    if overridden is not None:
        return overridden

    return get_global_registry()


@contextmanager
def _use_registry(registry: OperatorRegistry):
    token = _registry_override.set(registry)
    try:
        yield
    finally:
        _registry_override.reset(token)


def oper(
    name: str,
    namespace: str = "core",
    registry: OperatorRegistry | None = None,
    **metadata: Any,
):
    """Decorator to register an operator into global or provided registry."""

    def decorator(fn):
        target_registry = _resolve_registry(registry)
        target_registry.register(
            name=name,
            fn=fn,
            namespace=namespace,
            metadata=metadata or None,
        )
        return fn

    return decorator
