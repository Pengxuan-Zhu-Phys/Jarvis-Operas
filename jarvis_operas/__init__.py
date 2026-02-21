from __future__ import annotations

from .api import get_global_registry, oper
from .errors import (
    OperatorCallError,
    OperatorConflict,
    OperatorLoadError,
    OperatorNotFound,
)
from .loading import discover_entrypoints, load_user_ops
from .logging import get_log_mode, get_logger, set_log_mode
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
    "get_logger",
    "set_log_mode",
    "get_log_mode",
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
]
