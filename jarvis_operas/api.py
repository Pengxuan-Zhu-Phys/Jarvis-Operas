from __future__ import annotations

import os
from threading import Lock
from typing import Any

from .bootstrap_state import (
    bootstrapping_global_registry,
    is_bootstrapping_global_registry as _is_bootstrapping_global_registry,
)
from .catalog import get_catalog_declarations
from .core.operas import Operas
from .core.registry import OperasRegistry
from .curves import init_builtin_curve_cache, register_hot_curves_in_registry
from .logging import get_logger
from .namespace_policy import ensure_user_namespace_allowed
from .registration import (
    current_namespace_override,
    current_registry_override,
    make_operafunction,
)
_global_operas_registry: OperasRegistry | None = None
_global_operas: Operas | None = None
_global_lock = Lock()
_CURVE_INDEX_ENV = "JARVIS_OPERAS_CURVE_INDEX"


def _bootstrap_global_registry(registry: OperasRegistry) -> None:
    local_logger = get_logger(action="bootstrap_global_registry")
    registry.register_many(tuple(get_catalog_declarations()))

    # Lazy imports avoid bootstrap cycles during package import.
    from .persistence import apply_persisted_overrides, load_persisted_user_ops

    load_persisted_user_ops(registry, refresh_sympy=False)
    curve_index_path: str | None = None
    if not os.getenv(_CURVE_INDEX_ENV):
        try:
            summary = init_builtin_curve_cache(logger=local_logger)
            curve_index_path = summary.get("index_path")
        except Exception as exc:
            local_logger.warning("failed to initialize built-in interpolation cache: {}", exc)

    register_hot_curves_in_registry(registry, index_path=curve_index_path, logger=local_logger)
    apply_persisted_overrides(registry)


def get_global_operas_registry() -> OperasRegistry:
    global _global_operas_registry

    if _global_operas_registry is None:
        with _global_lock:
            if _global_operas_registry is None:
                registry = OperasRegistry()
                with bootstrapping_global_registry():
                    _bootstrap_global_registry(registry)
                _global_operas_registry = registry
    return _global_operas_registry


def is_global_operas_registry(registry: Any) -> bool:
    """Return True only when `registry` is the initialized global registry."""

    return _global_operas_registry is not None and registry is _global_operas_registry


def is_bootstrapping_global_registry() -> bool:
    return _is_bootstrapping_global_registry()


def get_global_operas(*, numpy_concurrency: int | None = None) -> Operas:
    global _global_operas

    if numpy_concurrency is not None and numpy_concurrency < 1:
        raise ValueError("numpy_concurrency must be >= 1")

    if _global_operas is None:
        registry = get_global_operas_registry()
        with _global_lock:
            if _global_operas is None:
                _global_operas = Operas(
                    registry,
                    numpy_concurrency=numpy_concurrency,
                )
    elif numpy_concurrency is not None:
        current = _global_operas.registry.get_numpy_concurrency()
        if current != numpy_concurrency:
            get_logger(action="get_global_operas").warning(
                "global Operas already initialized with numpy_concurrency={}; "
                "ignoring new value {}",
                current,
                numpy_concurrency,
            )
    return _global_operas


def oper(
    name: str,
    namespace: str | None = None,
    registry: OperasRegistry | None = None,
    **metadata: Any,
):
    """Decorator to register an operator into global or provided registry."""

    def decorator(fn):
        target_registry = registry or current_registry_override() or get_global_operas_registry()
        target_namespace = namespace or current_namespace_override() or "core"
        target_namespace = ensure_user_namespace_allowed(
            target_namespace,
            action="@oper registration",
        )
        target_registry.register(
            make_operafunction(
                namespace=target_namespace,
                name=name,
                fn=fn,
                metadata=metadata or None,
            )
        )
        return fn

    return decorator
