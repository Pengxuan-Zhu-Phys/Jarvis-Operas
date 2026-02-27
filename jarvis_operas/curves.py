from __future__ import annotations

from .curve_cache import (
    init_builtin_curve_cache,
    init_curve_cache,
    load_hot_curve_function_table,
    load_hot_curve_namespace_function_table,
    register_hot_curves,
)
from .curve_interpolator import Curve1DInterpolator
from .curve_manifest import (
    add_interpolation_namespace,
    build_interpolation_declarations,
    interpolation_manifest_resource,
    list_interpolation_namespace_entries,
    list_interpolation_namespaces,
    load_interpolation_manifest_library,
    remove_interpolation_namespace,
    validate_interpolation_namespace_index,
)
from .curve_registration import (
    register_hot_curve_namespace_in_registry,
    register_hot_curves_in_registry,
)
from .curve_storage import get_curve_cache_root, get_curve_index_path

__all__ = [
    "Curve1DInterpolator",
    "get_curve_cache_root",
    "get_curve_index_path",
    "interpolation_manifest_resource",
    "init_builtin_curve_cache",
    "init_curve_cache",
    "load_interpolation_manifest_library",
    "list_interpolation_namespaces",
    "list_interpolation_namespace_entries",
    "add_interpolation_namespace",
    "remove_interpolation_namespace",
    "validate_interpolation_namespace_index",
    "build_interpolation_declarations",
    "load_hot_curve_function_table",
    "register_hot_curves",
    "load_hot_curve_namespace_function_table",
    "register_hot_curves_in_registry",
    "register_hot_curve_namespace_in_registry",
]
