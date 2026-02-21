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
_namespace_override: ContextVar[str | None] = ContextVar(
    "jarvis_operas_namespace_override",
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
                # Auto-load persisted user operator sources for cross-process reuse.
                from .persistence import apply_persisted_overrides, load_persisted_user_ops

                load_persisted_user_ops(registry)
                apply_persisted_overrides(registry)
                _global_registry = registry

    return _global_registry


def _resolve_registry(registry: OperatorRegistry | None = None) -> OperatorRegistry:
    if registry is not None:
        return registry

    overridden = _registry_override.get()
    if overridden is not None:
        return overridden

    return get_global_registry()


def _resolve_namespace(namespace: str | None) -> str:
    if namespace is not None:
        return namespace

    overridden = _namespace_override.get()
    if overridden is not None:
        return overridden

    return "core"


@contextmanager
def _use_registry(
    registry: OperatorRegistry,
    default_namespace: str | None = None,
):
    registry_token = _registry_override.set(registry)
    namespace_token = _namespace_override.set(default_namespace)
    try:
        yield
    finally:
        _namespace_override.reset(namespace_token)
        _registry_override.reset(registry_token)


def oper(
    name: str,
    namespace: str | None = None,
    registry: OperatorRegistry | None = None,
    **metadata: Any,
):
    """Decorator to register an operator into global or provided registry."""

    def decorator(fn):
        target_registry = _resolve_registry(registry)
        target_namespace = _resolve_namespace(namespace)
        target_registry.register(
            name=name,
            fn=fn,
            namespace=target_namespace,
            metadata=metadata or None,
        )
        return fn

    return decorator
