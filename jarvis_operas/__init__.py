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
]
