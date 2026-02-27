from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .core.registry import OperasRegistry
from .core.spec import OperaFunction

_registry_override: ContextVar[OperasRegistry | None] = ContextVar(
    "jarvis_operas_registry_override",
    default=None,
)
_namespace_override: ContextVar[str | None] = ContextVar(
    "jarvis_operas_namespace_override",
    default=None,
)


def current_registry_override() -> OperasRegistry | None:
    return _registry_override.get()


def current_namespace_override() -> str | None:
    return _namespace_override.get()


@contextmanager
def use_registry(
    registry: OperasRegistry,
    *,
    default_namespace: str | None = None,
):
    registry_token = _registry_override.set(registry)
    namespace_token = _namespace_override.set(default_namespace)
    try:
        yield
    finally:
        _namespace_override.reset(namespace_token)
        _registry_override.reset(registry_token)


def make_operafunction(
    *,
    namespace: str,
    name: str,
    fn: Any,
    metadata: Mapping[str, Any] | None = None,
) -> OperaFunction:
    resolved = dict(metadata or {})
    arity = resolved.pop("arity", None)
    return_dtype = resolved.pop("return_dtype", None)
    polars_expr_impl = resolved.pop("polars_expr_impl", None)
    flags = resolved.pop("flags", ())
    numpy_impl = resolved.pop("numpy_impl", fn)
    return OperaFunction(
        namespace=namespace,
        name=name,
        arity=arity,
        return_dtype=return_dtype,
        numpy_impl=numpy_impl,
        polars_expr_impl=polars_expr_impl,
        flags=frozenset(flags or ()),
        metadata=resolved,
    )
