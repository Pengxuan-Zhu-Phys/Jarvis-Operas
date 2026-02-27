from __future__ import annotations

from typing import Any, Mapping

from .bootstrap_state import is_bootstrapping_global_registry
from .core.spec import OperaFunction
from .curve_cache import load_hot_curve_function_table
from .curve_manifest import (
    _build_curve_metadata,
    _normalize_selected_namespaces,
    _resolve_curve_bounds,
    _resolve_curve_namespace,
    _resolve_curve_return_dtype,
)
from .curve_storage import (
    _read_json_object,
    get_curve_index_path,
)
from .errors import OperatorConflict, OperatorNotFound
from .logging import get_logger
from .name_utils import try_split_full_name


def _build_existing_interpolation_name_index(registry: Any) -> dict[str, list[str]]:
    """Build short-name index for already-registered interpolation declarations."""

    indexed: dict[str, list[str]] = {}
    for existing_name in registry.list():
        split = try_split_full_name(existing_name)
        if split is None:
            continue
        _, short_name = split
        try:
            declaration = registry.get(existing_name)
        except OperatorNotFound:
            continue

        metadata = declaration.metadata
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("category") != "interpolation":
            continue
        indexed.setdefault(short_name, []).append(existing_name)
    return indexed


def register_hot_curves_in_registry(
    registry: Any,
    *,
    namespace: str = "interp",
    cache_root: str | None = None,
    index_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    include_cold: bool = False,
    overwrite: bool = True,
    logger: Any | None = None,
) -> list[str]:
    """Register hot curves into a function namespace."""

    local_logger = get_logger(logger, action="register_hot_curves_in_registry")
    resolved_index = get_curve_index_path(cache_root=cache_root, index_path=index_path)
    if not resolved_index.exists():
        return []

    raw_index = _read_json_object(resolved_index)
    raw_curves = raw_index.get("curves", {})
    if not isinstance(raw_curves, Mapping):
        raise ValueError(f"invalid curve index format: {resolved_index}")

    selected_namespaces = _normalize_selected_namespaces(namespaces)
    loaded = load_hot_curve_function_table(
        cache_root=cache_root,
        index_path=index_path,
        namespaces=namespaces,
        include_cold=include_cold,
        logger=logger,
    )
    existing_interpolation_names = (
        _build_existing_interpolation_name_index(registry) if overwrite else {}
    )
    registered: list[str] = []
    for curve_id, curve_fn in loaded.items():
        curve_entry = raw_curves.get(curve_id, {})
        target_namespace = _resolve_curve_namespace(
            curve_entry=curve_entry if isinstance(curve_entry, Mapping) else {},
            default_namespace=namespace,
        )
        resolved_x_min, resolved_x_max = _resolve_curve_bounds(
            curve_entry=curve_entry if isinstance(curve_entry, Mapping) else {},
            curve_fn=curve_fn,
        )
        metadata_override = (
            curve_entry.get("metadata")
            if isinstance(curve_entry, Mapping) and isinstance(curve_entry.get("metadata"), Mapping)
            else None
        )
        metadata = _build_curve_metadata(
            curve_id=curve_id,
            target_namespace=target_namespace,
            source=(
                str(curve_entry.get("source"))
                if isinstance(curve_entry, Mapping) and curve_entry.get("source") is not None
                else None
            ),
            kind=(
                str(curve_entry.get("kind"))
                if isinstance(curve_entry, Mapping) and curve_entry.get("kind") is not None
                else None
            ),
            log_x=bool(curve_entry.get("logX", False)) if isinstance(curve_entry, Mapping) else False,
            log_y=bool(curve_entry.get("logY", False)) if isinstance(curve_entry, Mapping) else False,
            hot=bool(curve_entry.get("hot", True)) if isinstance(curve_entry, Mapping) else True,
            points=curve_entry.get("points") if isinstance(curve_entry, Mapping) else None,
            x_min=resolved_x_min,
            x_max=resolved_x_max,
            metadata_override=metadata_override,
        )

        if selected_namespaces is not None and target_namespace not in selected_namespaces:
            continue

        full_name = f"{target_namespace}.{curve_id}"
        if overwrite:
            for existing_name in existing_interpolation_names.get(curve_id, []):
                try:
                    registry.delete(existing_name)
                except OperatorNotFound:
                    continue
            existing_interpolation_names[curve_id] = []

        declaration = OperaFunction(
            namespace=target_namespace,
            name=curve_id,
            arity=1,
            return_dtype=_resolve_curve_return_dtype(curve_entry),
            numpy_impl=curve_fn,
            metadata=metadata,
        )
        try:
            registry.register(declaration)
            registered.append(full_name)
            if overwrite:
                existing_interpolation_names[curve_id] = [full_name]
        except OperatorConflict:
            if overwrite:
                raise
            continue

    local_logger.info(
        "registered {} interpolation operators",
        len(registered),
    )
    if not is_bootstrapping_global_registry():
        from .integration import refresh_sympy_dicts_if_global_registry

        refresh_sympy_dicts_if_global_registry(registry)
    return sorted(registered)


def register_hot_curve_namespace_in_registry(
    registry: Any,
    namespace: str,
    *,
    cache_root: str | None = None,
    index_path: str | None = None,
    include_cold: bool = False,
    overwrite: bool = True,
    logger: Any | None = None,
) -> list[str]:
    """Register hot interpolation functions for one namespace."""

    return register_hot_curves_in_registry(
        registry,
        namespace=namespace,
        cache_root=cache_root,
        index_path=index_path,
        namespaces=[namespace],
        include_cold=include_cold,
        overwrite=overwrite,
        logger=logger,
    )
