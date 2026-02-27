from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .api import get_global_operas, get_global_operas_registry, oper
from .curves import (
    add_interpolation_namespace,
    build_interpolation_declarations,
    get_curve_index_path,
    init_curve_cache,
    interpolation_manifest_resource,
    list_interpolation_namespaces,
    list_interpolation_namespace_entries,
    load_hot_curve_function_table,
    load_hot_curve_namespace_function_table,
    load_interpolation_manifest_library,
    remove_interpolation_namespace,
    register_hot_curves,
    register_hot_curve_namespace_in_registry,
    register_hot_curves_in_registry,
    validate_interpolation_namespace_index,
)
from .core import (
    OperaFunction,
    Operas,
    OperasRegistry,
)
from .errors import (
    OperatorCallError,
    OperatorConflict,
    OperatorLoadError,
    OperatorNotFound,
)
from .loading import discover_entrypoints, load_user_ops
from .logging import configure_cli_logger, get_log_mode, get_logger, set_log_mode
from .persistence import (
    apply_persisted_overrides,
    delete_persisted_function,
    delete_persisted_namespace,
    get_overrides_store_path,
    get_persist_store_path,
    get_sources_store_path,
    list_persisted_user_ops,
    persist_user_ops,
    update_persisted_function,
    update_persisted_namespace,
)

if TYPE_CHECKING:
    from .integration import build_register_dicts, build_sympy_dicts, func_locals, numeric_funcs

    operas_registry: OperasRegistry
    operas: Operas

_LAZY_INTEGRATION_EXPORTS = frozenset(
    {
        "build_register_dicts",
        "build_sympy_dicts",
        "func_locals",
        "numeric_funcs",
    }
)

__all__ = [
    "OperaFunction",
    "OperasRegistry",
    "Operas",
    "OperatorNotFound",
    "OperatorConflict",
    "OperatorLoadError",
    "OperatorCallError",
    "build_register_dicts",
    "build_sympy_dicts",
    "get_logger",
    "set_log_mode",
    "get_log_mode",
    "configure_cli_logger",
    "oper",
    "operas_registry",
    "get_global_operas_registry",
    "operas",
    "get_global_operas",
    "load_user_ops",
    "discover_entrypoints",
    "persist_user_ops",
    "list_persisted_user_ops",
    "get_persist_store_path",
    "get_sources_store_path",
    "get_overrides_store_path",
    "apply_persisted_overrides",
    "delete_persisted_function",
    "update_persisted_function",
    "delete_persisted_namespace",
    "update_persisted_namespace",
    "get_curve_index_path",
    "build_interpolation_declarations",
    "add_interpolation_namespace",
    "init_curve_cache",
    "interpolation_manifest_resource",
    "list_interpolation_namespaces",
    "list_interpolation_namespace_entries",
    "load_hot_curve_function_table",
    "load_hot_curve_namespace_function_table",
    "load_interpolation_manifest_library",
    "remove_interpolation_namespace",
    "func_locals",
    "numeric_funcs",
    "register_hot_curves",
    "register_hot_curve_namespace_in_registry",
    "register_hot_curves_in_registry",
    "validate_interpolation_namespace_index",
]


def __getattr__(name: str) -> Any:
    if name == "operas_registry":
        return get_global_operas_registry()
    if name == "operas":
        return get_global_operas()
    if name in _LAZY_INTEGRATION_EXPORTS:
        from . import integration as _integration

        return getattr(_integration, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_INTEGRATION_EXPORTS | {"operas_registry", "operas"})
