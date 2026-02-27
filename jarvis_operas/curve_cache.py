from __future__ import annotations

import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .curve_interpolator import Curve1DInterpolator
from .curve_manifest import (
    _MANIFEST_LIBRARY_DIR,
    _MANIFEST_LIBRARY_NAME,
    _extract_curve_points,
    _normalize_selected_namespaces,
    _parse_manifest,
    _resolve_curve_namespace,
    _resolve_source_path,
)
from .curve_storage import (
    _INDEX_VERSION,
    _PICKLE_PROTOCOL,
    _read_index,
    _read_json_object,
    _relative_or_absolute,
    _resolve_index_ref,
    _write_json,
    _write_pickle,
    get_curve_index_path,
)
from .logging import get_logger


def init_builtin_curve_cache(
    *,
    source_root: str | None = None,
    cache_root: str | None = None,
    index_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    force: bool = False,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Initialize cache from packaged interpolation manifests."""

    manifest_dir_resource = resources.files("jarvis_operas").joinpath(
        _MANIFEST_LIBRARY_DIR
    )
    with resources.as_file(manifest_dir_resource) as manifest_dir:
        return init_curve_cache(
            str((manifest_dir / _MANIFEST_LIBRARY_NAME).resolve()),
            source_root=source_root,
            cache_root=cache_root,
            index_path=index_path,
            namespaces=namespaces,
            force=force,
            logger=logger,
        )


def init_curve_cache(
    manifest_path: str,
    *,
    source_root: str | None = None,
    cache_root: str | None = None,
    index_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    force: bool = False,
    logger: Any | None = None,
) -> dict[str, Any]:
    local_logger = get_logger(logger, action="init_curve_cache")
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_manifest.exists() or not resolved_manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {resolved_manifest}")

    manifest_obj = _read_json_object(resolved_manifest)
    curve_specs = _parse_manifest(
        manifest_obj,
        resolved_manifest,
        selected_namespaces=_normalize_selected_namespaces(namespaces),
    )
    resolved_source_root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else resolved_manifest.parent
    )

    resolved_index = get_curve_index_path(cache_root=cache_root, index_path=index_path)
    resolved_index.parent.mkdir(parents=True, exist_ok=True)
    (resolved_index.parent / "curves").mkdir(parents=True, exist_ok=True)

    old_index = _read_index(resolved_index)
    old_entries = old_index.get("curves", {}) if isinstance(old_index, Mapping) else {}

    entries: dict[str, dict[str, Any]] = {}
    compiled: list[str] = []
    cached: list[str] = []

    for spec in curve_specs:
        curve_id = spec["curve_id"]
        source_path = _resolve_source_path(
            source_ref=spec["source"],
            source_root=resolved_source_root,
            manifest_dir=Path(spec["manifest_path"]).parent,
        )
        payload = _read_json_object(source_path)
        x_values, y_values = _extract_curve_points(payload, source_path, curve_id)

        content_hash = _compute_curve_hash(spec, x_values, y_values)
        old_entry = old_entries.get(curve_id) if isinstance(old_entries, Mapping) else None
        reuse_existing = _can_reuse_pickle(
            old_entry=old_entry,
            expected_hash=content_hash,
            index_parent=resolved_index.parent,
            force=force,
        )

        if reuse_existing and isinstance(old_entry, Mapping):
            pickle_ref = str(old_entry["pickle"])
            cached.append(curve_id)
        else:
            safe_name = _safe_curve_filename(curve_id)
            pickle_ref = f"curves/{safe_name}-{content_hash[:12]}.pkl"
            pickle_path = resolved_index.parent / pickle_ref
            curve_callable = Curve1DInterpolator(
                curve_id=curve_id,
                x_values=x_values,
                y_values=y_values,
                kind=spec["kind"],
                log_x=spec["log_x"],
                log_y=spec["log_y"],
                extrapolation=spec["extrapolation"],
                metadata=spec["metadata"],
            )
            _write_pickle(pickle_path, curve_callable)
            compiled.append(curve_id)

        entries[curve_id] = {
            "curve_id": curve_id,
            "source": _relative_or_absolute(source_path, resolved_index.parent),
            "pickle": pickle_ref,
            "hash": content_hash,
            "namespace": spec["namespace"],
            "hot": spec["hot"],
            "kind": spec["kind"],
            "logX": spec["log_x"],
            "logY": spec["log_y"],
            "extrapolation": spec["extrapolation"],
            "points": int(x_values.size),
            "x_min": float(np.min(x_values)),
            "x_max": float(np.max(x_values)),
            "metadata": dict(spec["metadata"]),
        }

    index_payload = {
        "version": _INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(resolved_manifest),
        "source_root": str(resolved_source_root),
        "pickle_protocol": _PICKLE_PROTOCOL,
        "curves": entries,
    }
    _write_json(resolved_index, index_payload)

    local_logger.info(
        "initialized curve cache from {} (compiled={}, cached={}, total={})",
        str(resolved_manifest),
        len(compiled),
        len(cached),
        len(curve_specs),
    )
    return {
        "manifest_path": str(resolved_manifest),
        "index_path": str(resolved_index),
        "compiled": sorted(compiled),
        "cached": sorted(cached),
        "updated": sorted(compiled),
        "total": len(curve_specs),
    }


def load_hot_curve_function_table(
    *,
    cache_root: str | None = None,
    index_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    include_cold: bool = False,
    logger: Any | None = None,
) -> dict[str, Callable[..., Any]]:
    local_logger = get_logger(logger, action="load_hot_curve_function_table")
    resolved_index = get_curve_index_path(cache_root=cache_root, index_path=index_path)
    if not resolved_index.exists():
        return {}

    raw_index = _read_json_object(resolved_index)
    raw_curves = raw_index.get("curves", {})
    if not isinstance(raw_curves, Mapping):
        raise ValueError(f"invalid curve index format: {resolved_index}")

    selected_namespaces = _normalize_selected_namespaces(namespaces)
    function_table: dict[str, Callable[..., Any]] = {}
    for curve_id, entry in raw_curves.items():
        if not isinstance(entry, Mapping):
            continue
        entry_namespace = _resolve_curve_namespace(
            curve_entry=entry,
            default_namespace="interp",
        )
        if selected_namespaces is not None and entry_namespace not in selected_namespaces:
            continue
        is_hot = bool(entry.get("hot", True))
        if not include_cold and not is_hot:
            continue

        pickle_ref = entry.get("pickle")
        if not isinstance(pickle_ref, str) or not pickle_ref.strip():
            continue
        pickle_path = _resolve_index_ref(str(pickle_ref), resolved_index.parent)
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"missing curve pickle for '{curve_id}': {pickle_path}"
            )

        with pickle_path.open("rb") as fp:
            curve_fn = pickle.load(fp)
        if not callable(curve_fn):
            raise TypeError(f"curve '{curve_id}' is not callable after load")
        function_table[str(curve_id)] = curve_fn

    local_logger.info(
        "loaded {} hot curve functions from {}",
        len(function_table),
        str(resolved_index),
    )
    return function_table


def register_hot_curves(
    function_table: dict[str, Callable[..., Any]],
    *,
    cache_root: str | None = None,
    index_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    include_cold: bool = False,
    overwrite: bool = True,
    logger: Any | None = None,
) -> list[str]:
    loaded = load_hot_curve_function_table(
        cache_root=cache_root,
        index_path=index_path,
        namespaces=namespaces,
        include_cold=include_cold,
        logger=logger,
    )
    updated: list[str] = []
    for curve_id, curve_fn in loaded.items():
        if not overwrite and curve_id in function_table:
            continue
        function_table[curve_id] = curve_fn
        updated.append(curve_id)
    return sorted(updated)


def load_hot_curve_namespace_function_table(
    namespace: str,
    *,
    cache_root: str | None = None,
    index_path: str | None = None,
    include_cold: bool = False,
    logger: Any | None = None,
) -> dict[str, Callable[..., Any]]:
    """Load hot interpolation callables for one namespace."""

    return load_hot_curve_function_table(
        cache_root=cache_root,
        index_path=index_path,
        namespaces=[namespace],
        include_cold=include_cold,
        logger=logger,
    )


def _compute_curve_hash(
    spec: Mapping[str, Any],
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> str:
    payload = {
        "curve_id": spec["curve_id"],
        "kind": spec["kind"],
        "log_x": bool(spec["log_x"]),
        "log_y": bool(spec["log_y"]),
        "extrapolation": spec["extrapolation"],
        "x": [float(v) for v in np.asarray(x_values, dtype=float)],
        "y": [float(v) for v in np.asarray(y_values, dtype=float)],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=20).hexdigest()


def _safe_curve_filename(curve_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", curve_id).strip("._")
    if not safe:
        return "curve"
    return safe


def _can_reuse_pickle(
    *,
    old_entry: Any,
    expected_hash: str,
    index_parent: Path,
    force: bool,
) -> bool:
    if force:
        return False
    if not isinstance(old_entry, Mapping):
        return False
    if str(old_entry.get("hash", "")) != expected_hash:
        return False
    old_pickle = old_entry.get("pickle")
    if not isinstance(old_pickle, str) or not old_pickle.strip():
        return False
    path = _resolve_index_ref(old_pickle, index_parent)
    return path.exists() and path.is_file()
