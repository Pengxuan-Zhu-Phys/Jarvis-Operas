from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.interpolate import interp1d

from .errors import OperatorConflict, OperatorNotFound
from .logging import get_logger

_CACHE_ROOT_ENV = "JARVIS_OPERAS_CURVE_CACHE_ROOT"
_INDEX_PATH_ENV = "JARVIS_OPERAS_CURVE_INDEX"
_INDEX_FILENAME = "index.json"
_INDEX_VERSION = 1
_PICKLE_PROTOCOL = 5
_MANIFEST_LIBRARY_DIR = "manifests"
_MANIFEST_LIBRARY_NAME = "interpolations.manifest.json"


@dataclass
class Curve1DInterpolator:
    """Pickle-safe 1D interpolation callable."""

    curve_id: str
    x_values: np.ndarray
    y_values: np.ndarray
    kind: str = "cubic"
    log_x: bool = False
    log_y: bool = False
    extrapolation: str = "extrapolate"
    metadata: dict[str, Any] | None = None
    _interp: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x_values = np.asarray(self.x_values, dtype=float)
        y_values = np.asarray(self.y_values, dtype=float)

        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError(f"curve '{self.curve_id}' requires 1D x/y arrays")
        if x_values.size < 2:
            raise ValueError(f"curve '{self.curve_id}' requires at least two points")
        if x_values.size != y_values.size:
            raise ValueError(f"curve '{self.curve_id}' x/y lengths do not match")

        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        if np.any(np.diff(x_values) <= 0):
            raise ValueError(f"curve '{self.curve_id}' x values must be strictly increasing")
        if self.log_x and np.any(x_values <= 0):
            raise ValueError(f"curve '{self.curve_id}' has non-positive x values under logX")
        if self.log_y and np.any(y_values <= 0):
            raise ValueError(f"curve '{self.curve_id}' has non-positive y values under logY")

        object.__setattr__(self, "x_values", x_values)
        object.__setattr__(self, "y_values", y_values)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        self._interp = self._build_interp()

    def _build_interp(self):
        src_x = np.log(self.x_values) if self.log_x else self.x_values
        src_y = np.log(self.y_values) if self.log_y else self.y_values

        extrapolation = self.extrapolation.strip().lower()
        if extrapolation == "extrapolate":
            kwargs = {"bounds_error": False, "fill_value": "extrapolate"}
        elif extrapolation == "clip":
            kwargs = {"bounds_error": False, "fill_value": (src_y[0], src_y[-1])}
        elif extrapolation == "error":
            kwargs = {"bounds_error": True}
        else:
            raise ValueError(
                f"curve '{self.curve_id}' has invalid extrapolation policy: {self.extrapolation}"
            )

        return interp1d(
            src_x,
            src_y,
            kind=self.kind,
            assume_sorted=True,
            **kwargs,
        )

    def __call__(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        is_scalar_input = np.isscalar(x)
        values = np.asarray(x, dtype=float)

        eval_x = values
        if self.extrapolation.strip().lower() == "clip":
            eval_x = np.clip(eval_x, self.x_values[0], self.x_values[-1])

        if self.log_x:
            if np.any(eval_x <= 0):
                raise ValueError(f"curve '{self.curve_id}' received non-positive x under logX")
            eval_x = np.log(eval_x)

        result = self._interp(eval_x)
        if self.log_y:
            result = np.exp(result)

        result = np.asarray(result)
        if is_scalar_input:
            return float(result.item())
        return result

    def __getstate__(self) -> dict[str, Any]:
        return {
            "curve_id": self.curve_id,
            "x_values": np.asarray(self.x_values),
            "y_values": np.asarray(self.y_values),
            "kind": self.kind,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "extrapolation": self.extrapolation,
            "metadata": dict(self.metadata or {}),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.curve_id = str(state["curve_id"])
        self.x_values = np.asarray(state["x_values"], dtype=float)
        self.y_values = np.asarray(state["y_values"], dtype=float)
        self.kind = str(state.get("kind", "cubic"))
        self.log_x = bool(state.get("log_x", False))
        self.log_y = bool(state.get("log_y", False))
        self.extrapolation = str(state.get("extrapolation", "extrapolate"))
        self.metadata = dict(state.get("metadata", {}))
        self._interp = self._build_interp()


def get_curve_cache_root(cache_root: str | None = None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser().resolve()
    env_value = os.getenv(_CACHE_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / ".jarvis-operas" / "curve-cache").resolve()


def interpolation_manifest_resource() -> str:
    """Return package resource path for built-in interpolation manifest library."""

    return f"jarvis_operas/{_MANIFEST_LIBRARY_DIR}/{_MANIFEST_LIBRARY_NAME}"


def load_interpolation_manifest_library() -> dict[str, Any]:
    """Load bundled interpolation manifest library JSON."""

    resource = resources.files("jarvis_operas").joinpath(
        _MANIFEST_LIBRARY_DIR,
        _MANIFEST_LIBRARY_NAME,
    )
    raw = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("interpolation manifest library must be a JSON object")
    curves = raw.get("curves")
    if not isinstance(curves, list):
        raise ValueError("interpolation manifest library must contain a 'curves' list")
    return dict(raw)


def get_curve_index_path(
    *,
    cache_root: str | None = None,
    index_path: str | None = None,
) -> Path:
    if index_path:
        return Path(index_path).expanduser().resolve()
    env_value = os.getenv(_INDEX_PATH_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return get_curve_cache_root(cache_root) / _INDEX_FILENAME


def init_curve_cache(
    manifest_path: str,
    *,
    source_root: str | None = None,
    cache_root: str | None = None,
    index_path: str | None = None,
    force: bool = False,
    logger: Any | None = None,
) -> dict[str, Any]:
    local_logger = get_logger(logger, action="init_curve_cache")
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_manifest.exists() or not resolved_manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {resolved_manifest}")

    manifest_obj = _read_json_object(resolved_manifest)
    curve_specs = _parse_manifest(manifest_obj, resolved_manifest)
    resolved_source_root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else resolved_manifest.parent
    )

    resolved_index = get_curve_index_path(cache_root=cache_root, index_path=index_path)
    resolved_index.parent.mkdir(parents=True, exist_ok=True)
    curve_dir = resolved_index.parent / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

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
            manifest_dir=resolved_manifest.parent,
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

    function_table: dict[str, Callable[..., Any]] = {}
    for curve_id, entry in raw_curves.items():
        if not isinstance(entry, Mapping):
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
    include_cold: bool = False,
    overwrite: bool = True,
    logger: Any | None = None,
) -> list[str]:
    loaded = load_hot_curve_function_table(
        cache_root=cache_root,
        index_path=index_path,
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


def register_hot_curves_in_registry(
    registry: Any,
    *,
    namespace: str = "interp",
    cache_root: str | None = None,
    index_path: str | None = None,
    include_cold: bool = False,
    overwrite: bool = True,
    logger: Any | None = None,
) -> list[str]:
    """Register hot curves into OperatorRegistry namespace."""

    local_logger = get_logger(logger, action="register_hot_curves_in_registry")
    resolved_index = get_curve_index_path(cache_root=cache_root, index_path=index_path)
    if not resolved_index.exists():
        return []

    raw_index = _read_json_object(resolved_index)
    raw_curves = raw_index.get("curves", {})
    if not isinstance(raw_curves, Mapping):
        raise ValueError(f"invalid curve index format: {resolved_index}")

    loaded = load_hot_curve_function_table(
        cache_root=cache_root,
        index_path=index_path,
        include_cold=include_cold,
        logger=logger,
    )
    registered: list[str] = []
    for curve_id, curve_fn in loaded.items():
        curve_entry = raw_curves.get(curve_id, {})
        target_namespace = _resolve_curve_namespace(
            curve_entry=curve_entry if isinstance(curve_entry, Mapping) else {},
            default_namespace=namespace,
        )
        metadata = {
            "category": "interpolation",
            "curve_id": curve_id,
            "source": curve_entry.get("source") if isinstance(curve_entry, Mapping) else None,
            "kind": curve_entry.get("kind") if isinstance(curve_entry, Mapping) else None,
            "logX": bool(curve_entry.get("logX", False)) if isinstance(curve_entry, Mapping) else False,
            "logY": bool(curve_entry.get("logY", False)) if isinstance(curve_entry, Mapping) else False,
            "hot": bool(curve_entry.get("hot", True)) if isinstance(curve_entry, Mapping) else True,
            "points": curve_entry.get("points") if isinstance(curve_entry, Mapping) else None,
        }
        if isinstance(curve_entry, Mapping) and isinstance(curve_entry.get("metadata"), Mapping):
            metadata.update(dict(curve_entry.get("metadata", {})))
        metadata["group"] = target_namespace

        full_name = f"{target_namespace}:{curve_id}"
        if overwrite:
            for existing_name in registry.list():
                if ":" not in existing_name:
                    continue
                if existing_name.split(":", 1)[1] != curve_id:
                    continue
                try:
                    registry.delete(existing_name)
                except OperatorNotFound:
                    continue

        try:
            registry.register(
                name=curve_id,
                fn=curve_fn,
                namespace=target_namespace,
                metadata=metadata,
            )
            registered.append(full_name)
        except OperatorConflict:
            if overwrite:
                raise
            continue

    local_logger.info(
        "registered {} interpolation operators",
        len(registered),
    )
    return sorted(registered)


def _parse_manifest(raw: Mapping[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    raw_curves = raw.get("curves")
    if not isinstance(raw_curves, list):
        raise ValueError(
            f"manifest '{manifest_path}' must contain a 'curves' list"
        )

    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_curves):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest curve item #{idx} must be an object")

        curve_id = _normalize_required_string(item.get("curve_id"), "curve_id", idx)
        source_ref = item.get("source")
        if source_ref is None:
            source_ref = item.get("file", item.get("path", item.get("json")))
        source = _normalize_required_string(source_ref, "source", idx)

        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError(f"manifest curve '{curve_id}' metadata must be an object")
        metadata_dict = dict(metadata)

        item_group = _normalize_optional_namespace(
            item.get("group"),
            field_name=f"curves[{idx}].group",
        )
        metadata_group = _normalize_optional_namespace(
            metadata_dict.get("group"),
            field_name=f"curves[{idx}].metadata.group",
        )
        resolved_group = item_group or metadata_group
        if resolved_group is not None:
            metadata_dict["group"] = resolved_group

        item_namespace = _normalize_optional_namespace(
            item.get("namespace"),
            field_name=f"curves[{idx}].namespace",
        )
        resolved_namespace = item_namespace or resolved_group

        parsed.append(
            {
                "curve_id": curve_id,
                "source": source,
                "namespace": resolved_namespace,
                "kind": str(item.get("kind", "cubic")).strip() or "cubic",
                "log_x": bool(item.get("logX", item.get("log_x", False))),
                "log_y": bool(item.get("logY", item.get("log_y", False))),
                "extrapolation": str(item.get("extrapolation", "extrapolate")).strip()
                or "extrapolate",
                "hot": bool(item.get("hot", True)),
                "metadata": metadata_dict,
            }
        )

    curve_ids = [item["curve_id"] for item in parsed]
    if len(curve_ids) != len(set(curve_ids)):
        raise ValueError(f"manifest '{manifest_path}' contains duplicated curve_id values")
    return parsed


def _normalize_required_string(value: Any, field_name: str, index: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"manifest curve item #{index} missing string field '{field_name}'")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"manifest curve item #{index} has empty field '{field_name}'")
    return normalized


def _normalize_optional_namespace(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided")
    normalized = value.strip()
    if not normalized:
        return None
    if ":" in normalized:
        raise ValueError(f"{field_name} cannot contain ':'")
    return normalized


def _coerce_optional_namespace(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or ":" in normalized:
        return None
    return normalized


def _resolve_curve_namespace(
    *,
    curve_entry: Mapping[str, Any],
    default_namespace: str,
) -> str:
    namespace = _coerce_optional_namespace(curve_entry.get("namespace"))
    if namespace is not None:
        return namespace

    metadata = curve_entry.get("metadata")
    if isinstance(metadata, Mapping):
        group = _coerce_optional_namespace(metadata.get("group"))
        if group is not None:
            return group

    return default_namespace


def _resolve_source_path(source_ref: str, source_root: Path, manifest_dir: Path) -> Path:
    candidate = Path(source_ref).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (source_root / candidate).resolve()
        if not resolved.exists():
            resolved_alt = (manifest_dir / candidate).resolve()
            if resolved_alt.exists():
                resolved = resolved_alt
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"curve source file not found: {resolved}")
    return resolved


def _extract_curve_points(
    payload: Mapping[str, Any],
    source_path: Path,
    curve_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw = payload.get("x", payload.get("x_values"))
    y_raw = payload.get("y", payload.get("y_values"))
    if x_raw is None or y_raw is None:
        raise ValueError(
            f"curve source '{source_path}' for '{curve_id}' must contain x/y (or x_values/y_values)"
        )
    x_values = np.asarray(x_raw, dtype=float)
    y_values = np.asarray(y_raw, dtype=float)
    return x_values, y_values


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


def _resolve_index_ref(path_ref: str, index_parent: Path) -> Path:
    path = Path(path_ref).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (index_parent / path).resolve()


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as fp:
        pickle.dump(value, fp, protocol=_PICKLE_PROTOCOL)
    temp.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid json file '{path}': {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"json file '{path}' must contain an object")
    return dict(raw)


def _read_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": _INDEX_VERSION, "curves": {}}
    return _read_json_object(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)
