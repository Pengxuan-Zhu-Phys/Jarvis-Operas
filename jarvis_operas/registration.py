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
_replace_existing_names_override: ContextVar[frozenset[str] | None] = ContextVar(
    "jarvis_operas_replace_existing_names_override",
    default=None,
)
_metadata_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "jarvis_operas_metadata_override",
    default=None,
)
_loaded_names_override: ContextVar[list[str] | None] = ContextVar(
    "jarvis_operas_loaded_names_override",
    default=None,
)


def current_registry_override() -> OperasRegistry | None:
    return _registry_override.get()


def current_namespace_override() -> str | None:
    return _namespace_override.get()


def current_replace_existing_names() -> frozenset[str] | None:
    return _replace_existing_names_override.get()


def current_metadata_override() -> dict[str, Any] | None:
    metadata = _metadata_override.get()
    if metadata is None:
        return None
    return dict(metadata)


def note_registered_name(full_name: str) -> None:
    loaded_names = _loaded_names_override.get()
    if loaded_names is not None:
        loaded_names.append(full_name)


@contextmanager
def use_registry(
    registry: OperasRegistry,
    *,
    default_namespace: str | None = None,
    replace_existing_names: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None,
    metadata_override: Mapping[str, Any] | None = None,
    loaded_names: list[str] | None = None,
):
    registry_token = _registry_override.set(registry)
    namespace_token = _namespace_override.set(default_namespace)
    replace_token = _replace_existing_names_override.set(
        None
        if replace_existing_names is None
        else frozenset(str(item) for item in replace_existing_names)
    )
    metadata_token = _metadata_override.set(
        None if metadata_override is None else dict(metadata_override)
    )
    loaded_names_token = _loaded_names_override.set(loaded_names)
    try:
        yield
    finally:
        _loaded_names_override.reset(loaded_names_token)
        _metadata_override.reset(metadata_token)
        _replace_existing_names_override.reset(replace_token)
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
