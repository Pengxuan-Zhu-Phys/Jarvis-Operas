from __future__ import annotations

import hashlib
import io
import json
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
    _ARTIFACT_FORMAT,
    _INDEX_VERSION,
    _read_index,
    _read_json_object,
    _relative_or_absolute,
    _resolve_index_ref,
    _write_json,
    get_curve_index_path,
)
from .io_utils import atomic_write_bytes
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
    selected_namespaces = _normalize_selected_namespaces(namespaces)
    curve_specs = _parse_manifest(
        manifest_obj,
        resolved_manifest,
        selected_namespaces=selected_namespaces,
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
    if not isinstance(old_entries, Mapping):
        old_entries = {}
    owned_namespaces = _owned_namespaces(manifest_obj, curve_specs, selected_namespaces)

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
            artifact_ref = str(old_entry["artifact"])
            cached.append(curve_id)
        else:
            safe_name = _safe_curve_filename(curve_id)
            artifact_ref = f"curves/{safe_name}-{content_hash[:12]}.npz"
            artifact_path = resolved_index.parent / artifact_ref
            _write_curve_artifact(artifact_path, x_values, y_values)
            compiled.append(curve_id)

        entries[curve_id] = {
            "curve_id": curve_id,
            "source": _relative_or_absolute(source_path, resolved_index.parent),
            "format": _ARTIFACT_FORMAT,
            "artifact": artifact_ref,
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

    # This init owns the namespaces it compiled. Curves in every other
    # namespace stay, including a custom manifest sharing the default cache.
    preserved: dict[str, dict[str, Any]] = {}
    for curve_id, old_entry in old_entries.items():
        if not isinstance(curve_id, str) or not isinstance(old_entry, Mapping):
            continue
        if curve_id in entries:
            continue
        namespace = _resolve_curve_namespace(curve_entry=old_entry, default_namespace="interp")
        if namespace in owned_namespaces:
            continue
        preserved[curve_id] = dict(old_entry)
    merged_entries = {**preserved, **entries}

    index_payload = {
        "version": _INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(resolved_manifest),
        "source_root": str(resolved_source_root),
        "artifact_format": _ARTIFACT_FORMAT,
        "curves": merged_entries,
    }
    _write_json(resolved_index, index_payload)
    _delete_replaced_artifacts(resolved_index.parent / "curves", old_entries, merged_entries)

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

        artifact_ref = entry.get("artifact")
        if entry.get("format") != _ARTIFACT_FORMAT or not isinstance(artifact_ref, str) or not artifact_ref.strip():
            raise ValueError(
                f"curve cache entry '{curve_id}' uses a retired pickle format; "
                "run `jopera init` to rebuild it"
            )
        artifact_path = _resolve_index_ref(artifact_ref, resolved_index.parent)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"missing curve artifact for '{curve_id}': {artifact_path}"
            )
        x_values, y_values = _read_curve_artifact(artifact_path)
        metadata = entry.get("metadata")
        curve_fn = Curve1DInterpolator(
            curve_id=str(entry.get("curve_id") or curve_id),
            x_values=x_values,
            y_values=y_values,
            kind=str(entry.get("kind", "cubic")),
            log_x=bool(entry.get("logX", False)),
            log_y=bool(entry.get("logY", False)),
            extrapolation=str(entry.get("extrapolation", "extrapolate")),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
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
    if old_entry.get("format") != _ARTIFACT_FORMAT:
        return False
    artifact = old_entry.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        return False
    path = _resolve_index_ref(artifact, index_parent)
    return path.exists() and path.is_file() and path.suffix == ".npz"


def _write_curve_artifact(path: Path, x_values: np.ndarray, y_values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        x=np.asarray(x_values, dtype=float),
        y=np.asarray(y_values, dtype=float),
    )
    atomic_write_bytes(path, buffer.getvalue(), fsync=True, temp_prefix=f"{path.name}.")


def _read_curve_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "x" not in payload.files or "y" not in payload.files:
            raise ValueError(f"curve artifact '{path}' must contain x and y arrays")
        return np.asarray(payload["x"], dtype=float), np.asarray(payload["y"], dtype=float)


def _owned_namespaces(
    manifest_obj: Mapping[str, Any],
    curve_specs: list[dict[str, Any]],
    selected_namespaces: set[str] | None,
) -> set[str]:
    """Namespaces this init may add, replace, or remove."""

    if selected_namespaces is not None:
        return set(selected_namespaces)
    raw_namespaces = manifest_obj.get("namespaces")
    if isinstance(raw_namespaces, list):
        declared: set[str] = set()
        for item in raw_namespaces:
            if not isinstance(item, Mapping):
                continue
            name = item.get("namespace")
            if isinstance(name, str) and name.strip():
                declared.add(name.strip())
        if declared:
            return declared
    owned = {str(spec["namespace"]) for spec in curve_specs if spec.get("namespace")}
    manifest_namespace = manifest_obj.get("namespace")
    if isinstance(manifest_namespace, str) and manifest_namespace.strip():
        owned.add(manifest_namespace.strip())
    return owned


def _artifact_basename(entry: Mapping[str, Any]) -> str | None:
    for key in ("artifact", "pickle"):
        ref = entry.get(key)
        if isinstance(ref, str) and ref.strip():
            return Path(ref).name
    return None


def _delete_replaced_artifacts(
    curves_dir: Path,
    old_entries: Mapping[str, Any],
    merged_entries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Delete artifacts left behind by curves this init replaced or removed."""

    referenced: set[str] = set()
    for entry in merged_entries.values():
        if isinstance(entry, Mapping):
            name = _artifact_basename(entry)
            if name:
                referenced.add(name)
    if not curves_dir.is_dir():
        return
    for curve_id, old_entry in old_entries.items():
        if not isinstance(old_entry, Mapping):
            continue
        kept = merged_entries.get(curve_id)
        if isinstance(kept, Mapping) and _artifact_basename(kept) == _artifact_basename(old_entry):
            continue
        name = _artifact_basename(old_entry)
        if not name or name in referenced:
            continue
        path = curves_dir / name
        if path.is_file() and path.suffix in {".npz", ".pkl"}:
            path.unlink()
