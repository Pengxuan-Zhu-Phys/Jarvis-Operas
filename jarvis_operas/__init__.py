from __future__ import annotations

from .api import get_global_registry, oper
from .curves import (
    get_curve_index_path,
    init_curve_cache,
    interpolation_manifest_resource,
    load_hot_curve_function_table,
    load_interpolation_manifest_library,
    register_hot_curves,
    register_hot_curves_in_registry,
)
from .errors import (
    OperatorCallError,
    OperatorConflict,
    OperatorLoadError,
    OperatorNotFound,
)
from .integration import (
    build_register_dicts,
    build_sympy_dicts,
    func_locals,
    numeric_funcs,
)
from .loading import discover_entrypoints, load_user_ops
from .logging import configure_cli_logger, get_log_mode, get_logger, set_log_mode
from .persistence import (
    apply_persisted_overrides,
    delete_persisted_function,
    delete_persisted_namespace,
    get_persist_store_path,
    list_persisted_user_ops,
    persist_user_ops,
    update_persisted_function,
    update_persisted_namespace,
)
from .registry import OperatorRegistry

registry = get_global_registry()

__all__ = [
    "OperatorRegistry",
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
    "registry",
    "get_global_registry",
    "load_user_ops",
    "discover_entrypoints",
    "persist_user_ops",
    "list_persisted_user_ops",
    "get_persist_store_path",
    "apply_persisted_overrides",
    "delete_persisted_function",
    "update_persisted_function",
    "delete_persisted_namespace",
    "update_persisted_namespace",
    "get_curve_index_path",
    "init_curve_cache",
    "interpolation_manifest_resource",
    "load_hot_curve_function_table",
    "load_interpolation_manifest_library",
    "func_locals",
    "numeric_funcs",
    "register_hot_curves",
    "register_hot_curves_in_registry",
]
