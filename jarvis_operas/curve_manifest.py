from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .core.spec import OperaFunction
from .curve_interpolator import Curve1DInterpolator
from .curve_storage import (
    _read_json_object,
    _resolve_index_ref,
    _write_json,
)

_MANIFEST_LIBRARY_DIR = "manifests"
_MANIFEST_LIBRARY_NAME = "interpolations.manifest.json"


def _sorted_namespace_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: item["namespace"])


def _with_resolved_manifest_path(
    manifest_path: str | None,
    handler: Callable[[Path], Any],
) -> Any:
    if manifest_path is not None:
        return handler(Path(manifest_path).expanduser().resolve())

    manifest_dir_resource = resources.files("jarvis_operas").joinpath(_MANIFEST_LIBRARY_DIR)
    with resources.as_file(manifest_dir_resource) as manifest_dir:
        resolved_manifest = (manifest_dir / _MANIFEST_LIBRARY_NAME).resolve()
        return handler(resolved_manifest)


def interpolation_manifest_resource() -> str:
    """Return package resource path for built-in interpolation manifest library."""

    return f"jarvis_operas/{_MANIFEST_LIBRARY_DIR}/{_MANIFEST_LIBRARY_NAME}"


def load_interpolation_manifest_library() -> dict[str, Any]:
    """Load bundled interpolation manifest library JSON.

    Returns the raw manifest payload plus a flattened `curves` list for
    compatibility with existing callers.
    """

    def _load_payload(manifest_path: Path) -> dict[str, Any]:
        raw = _read_json_object(manifest_path)
        parsed_specs = _parse_manifest(raw, manifest_path)
        payload = dict(raw)
        payload["curves"] = [
            {
                "curve_id": spec["curve_id"],
                "namespace": spec["namespace"],
                "source": spec["source"],
                "kind": spec["kind"],
                "logX": spec["log_x"],
                "logY": spec["log_y"],
                "hot": spec["hot"],
                "extrapolation": spec["extrapolation"],
                "metadata": dict(spec["metadata"]),
            }
            for spec in parsed_specs
        ]
        return payload

    return _with_resolved_manifest_path(None, _load_payload)


def list_interpolation_namespaces(
    *,
    manifest_path: str | None = None,
) -> list[str]:
    """List interpolation namespaces from manifest index."""

    def _list_namespaces(resolved_manifest: Path) -> list[str]:
        raw = _read_json_object(resolved_manifest)
        return sorted({item["namespace"] for item in _parse_manifest(raw, resolved_manifest)})

    return _with_resolved_manifest_path(manifest_path, _list_namespaces)


def list_interpolation_namespace_entries(
    *,
    manifest_path: str | None = None,
) -> list[dict[str, Any]]:
    """List namespace index entries from interpolation root manifest."""

    return _with_resolved_manifest_path(
        manifest_path,
        _list_interpolation_namespace_entries_from_path,
    )


def add_interpolation_namespace(
    *,
    manifest_path: str,
    namespace: str,
    namespace_manifest: str,
    description: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Add one namespace entry into interpolation root manifest."""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if resolved_manifest.exists():
        root_payload = _read_json_object(resolved_manifest)
    else:
        root_payload = {
            "version": 2,
            "kind": "jarvis_operas_interpolation_namespace_index",
            "description": "Interpolation namespace index.",
            "namespaces": [],
        }

    normalized_namespace = _normalize_optional_namespace(
        namespace,
        field_name="namespace",
    )
    if normalized_namespace is None:
        raise ValueError("namespace cannot be empty")
    normalized_manifest_ref = str(namespace_manifest).strip()
    if not normalized_manifest_ref:
        raise ValueError("namespace_manifest cannot be empty")

    target_manifest_path = _resolve_index_ref(
        normalized_manifest_ref,
        resolved_manifest.parent,
    )
    if not target_manifest_path.exists() or not target_manifest_path.is_file():
        raise FileNotFoundError(f"namespace manifest not found: {target_manifest_path}")

    # Validate namespace manifest shape before saving.
    target_manifest_raw = _read_json_object(target_manifest_path)
    _parse_manifest_function_entries(
        raw=target_manifest_raw,
        manifest_path=target_manifest_path,
        namespace_hint=normalized_namespace,
        selected_namespaces=None,
    )

    normalized_description: str | None = None
    if description is not None:
        normalized_description = str(description).strip() or None

    raw_namespaces = root_payload.get("namespaces")
    if raw_namespaces is None:
        raw_namespaces = []
    if not isinstance(raw_namespaces, list):
        raise ValueError(
            f"manifest '{resolved_manifest}' must contain a 'namespaces' list"
        )

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_namespaces):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest namespaces item #{idx} must be an object")
        item_namespace = _normalize_optional_namespace(
            item.get("namespace"),
            field_name=f"namespaces[{idx}].namespace",
        )
        if item_namespace is None:
            raise ValueError(f"namespaces[{idx}].namespace cannot be empty")
        if item_namespace in seen:
            raise ValueError(
                f"manifest '{resolved_manifest}' contains duplicated namespace '{item_namespace}'"
            )
        seen.add(item_namespace)
        item_manifest = _normalize_required_string(item.get("manifest"), "manifest", idx)
        item_entry: dict[str, Any] = {
            "namespace": item_namespace,
            "manifest": item_manifest,
        }
        item_description = item.get("description")
        if item_description is not None:
            if not isinstance(item_description, str):
                raise ValueError(
                    f"namespaces[{idx}].description must be a string when provided"
                )
            description_text = item_description.strip()
            if description_text:
                item_entry["description"] = description_text
        entries.append(item_entry)

    existing_index = next(
        (idx for idx, item in enumerate(entries) if item["namespace"] == normalized_namespace),
        None,
    )
    next_entry: dict[str, Any] = {
        "namespace": normalized_namespace,
        "manifest": normalized_manifest_ref,
    }
    if normalized_description is not None:
        next_entry["description"] = normalized_description

    added = False
    updated = False
    if existing_index is None:
        entries.append(next_entry)
        added = True
    else:
        current = entries[existing_index]
        if current == next_entry:
            return {
                "manifest_path": str(resolved_manifest),
                "namespace": normalized_namespace,
                "added": False,
                "updated": False,
                "unchanged": True,
                "entry": current,
            }
        if not overwrite:
            raise ValueError(
                f"namespace '{normalized_namespace}' already exists; use overwrite=True to replace"
            )
        entries[existing_index] = next_entry
        updated = True

    entries_sorted = _sorted_namespace_entries(entries)
    candidate_payload = dict(root_payload)
    candidate_payload.setdefault("version", 2)
    candidate_payload.setdefault("kind", "jarvis_operas_interpolation_namespace_index")
    candidate_payload["namespaces"] = entries_sorted

    # Parse-check combined root/namespace consistency before write.
    _parse_manifest(candidate_payload, resolved_manifest)
    _write_json(resolved_manifest, candidate_payload)
    return {
        "manifest_path": str(resolved_manifest),
        "namespace": normalized_namespace,
        "added": added,
        "updated": updated,
        "unchanged": False,
        "entry": next_entry,
    }


def remove_interpolation_namespace(
    *,
    manifest_path: str,
    namespace: str,
) -> dict[str, Any]:
    """Remove one namespace entry from interpolation root manifest."""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_manifest.exists() or not resolved_manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {resolved_manifest}")

    normalized_namespace = _normalize_optional_namespace(
        namespace,
        field_name="namespace",
    )
    if normalized_namespace is None:
        raise ValueError("namespace cannot be empty")

    root_payload = _read_json_object(resolved_manifest)
    entries = _list_interpolation_namespace_entries_from_path(resolved_manifest)
    filtered = [item for item in entries if item["namespace"] != normalized_namespace]
    removed = len(filtered) != len(entries)

    if not removed:
        return {
            "manifest_path": str(resolved_manifest),
            "namespace": normalized_namespace,
            "removed": False,
            "remaining": len(entries),
        }

    next_payload = dict(root_payload)
    next_payload["namespaces"] = _sorted_namespace_entries(filtered)
    _write_json(resolved_manifest, next_payload)
    return {
        "manifest_path": str(resolved_manifest),
        "namespace": normalized_namespace,
        "removed": True,
        "remaining": len(filtered),
    }


def validate_interpolation_namespace_index(
    *,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Validate interpolation root namespace index and all namespace manifests."""

    return _with_resolved_manifest_path(
        manifest_path,
        _validate_interpolation_namespace_index_from_path,
    )


def build_interpolation_declarations(
    *,
    manifest_path: str | None = None,
    namespaces: list[str] | tuple[str, ...] | None = None,
    source_root: str | None = None,
    include_cold: bool = True,
) -> list[OperaFunction]:
    """Build interpolation OperaFunction declarations directly from manifest data."""

    def _build(resolved_manifest: Path) -> list[OperaFunction]:
        return _build_interpolation_declarations_from_manifest_path(
            resolved_manifest,
            namespaces=namespaces,
            source_root=source_root,
            include_cold=include_cold,
        )

    return _with_resolved_manifest_path(manifest_path, _build)


def _build_interpolation_declarations_from_manifest_path(
    manifest_path: Path,
    *,
    namespaces: list[str] | tuple[str, ...] | None,
    source_root: str | None,
    include_cold: bool,
) -> list[OperaFunction]:
    raw = _read_json_object(manifest_path)
    selected_namespaces = _normalize_selected_namespaces(namespaces)
    specs = _parse_manifest(
        raw,
        manifest_path,
        selected_namespaces=selected_namespaces,
    )
    resolved_source_root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else manifest_path.parent
    )

    declarations: list[OperaFunction] = []
    for spec in specs:
        if not include_cold and not spec["hot"]:
            continue
        curve_id = spec["curve_id"]
        target_namespace = spec["namespace"] or "interp"
        source_path = _resolve_source_path(
            source_ref=spec["source"],
            source_root=resolved_source_root,
            manifest_dir=Path(spec["manifest_path"]).parent,
        )
        payload = _read_json_object(source_path)
        x_values, y_values = _extract_curve_points(payload, source_path, curve_id)
        curve_fn = Curve1DInterpolator(
            curve_id=curve_id,
            x_values=x_values,
            y_values=y_values,
            kind=spec["kind"],
            log_x=spec["log_x"],
            log_y=spec["log_y"],
            extrapolation=spec["extrapolation"],
            metadata=dict(spec["metadata"]),
        )
        metadata = _build_curve_metadata(
            curve_id=curve_id,
            target_namespace=target_namespace,
            source=str(source_path),
            kind=spec["kind"],
            log_x=bool(spec["log_x"]),
            log_y=bool(spec["log_y"]),
            hot=bool(spec["hot"]),
            points=int(x_values.size),
            x_min=float(np.min(x_values)),
            x_max=float(np.max(x_values)),
            metadata_override=spec["metadata"],
        )
        declaration = OperaFunction(
            namespace=target_namespace,
            name=curve_id,
            arity=1,
            return_dtype=_resolve_curve_return_dtype({"metadata": metadata}),
            numpy_impl=curve_fn,
            metadata=metadata,
        )
        declarations.append(declaration)

    dedup: dict[str, OperaFunction] = {}
    for declaration in declarations:
        dedup[declaration.full_name] = declaration
    return sorted(dedup.values(), key=lambda item: item.full_name)


def _list_interpolation_namespace_entries_from_path(
    manifest_path: Path,
) -> list[dict[str, Any]]:
    root_payload = _read_json_object(manifest_path)
    raw_namespaces = root_payload.get("namespaces")
    if not isinstance(raw_namespaces, list):
        raise ValueError(
            f"manifest '{manifest_path}' must contain a 'namespaces' list"
        )

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_namespaces):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest namespaces item #{idx} must be an object")
        namespace = _normalize_optional_namespace(
            item.get("namespace"),
            field_name=f"namespaces[{idx}].namespace",
        )
        if namespace is None:
            raise ValueError(f"namespaces[{idx}].namespace cannot be empty")
        if namespace in seen:
            raise ValueError(
                f"manifest '{manifest_path}' contains duplicated namespace '{namespace}'"
            )
        seen.add(namespace)

        manifest_ref = _normalize_required_string(item.get("manifest"), "manifest", idx)
        entry: dict[str, Any] = {
            "namespace": namespace,
            "manifest": manifest_ref,
        }

        description = item.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(
                    f"namespaces[{idx}].description must be a string when provided"
                )
            text = description.strip()
            if text:
                entry["description"] = text
        entries.append(entry)
    return _sorted_namespace_entries(entries)


def _validate_interpolation_namespace_index_from_path(
    manifest_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    namespaces: list[str] = []
    function_count = 0

    if not manifest_path.exists() or not manifest_path.is_file():
        errors.append(f"manifest file not found: {manifest_path}")
        return {
            "ok": False,
            "manifest_path": str(manifest_path),
            "namespace_count": 0,
            "function_count": 0,
            "namespaces": [],
            "errors": errors,
            "warnings": warnings,
        }

    root_payload: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    try:
        root_payload = _read_json_object(manifest_path)
    except Exception as exc:
        errors.append(str(exc))

    if root_payload is not None:
        try:
            entries = _list_interpolation_namespace_entries_from_path(manifest_path)
            namespaces = [item["namespace"] for item in entries]
        except Exception as exc:
            errors.append(str(exc))

    if root_payload is not None and not errors:
        try:
            parsed_specs = _parse_manifest(root_payload, manifest_path)
            function_count = len(parsed_specs)
        except Exception as exc:
            errors.append(str(exc))

    return {
        "ok": not errors,
        "manifest_path": str(manifest_path),
        "namespace_count": len(namespaces),
        "function_count": function_count,
        "namespaces": namespaces,
        "errors": errors,
        "warnings": warnings,
    }


def _parse_manifest(
    raw: Mapping[str, Any],
    manifest_path: Path,
    *,
    selected_namespaces: set[str] | None = None,
) -> list[dict[str, Any]]:
    raw_namespaces = raw.get("namespaces")
    if isinstance(raw_namespaces, list):
        return _parse_manifest_namespace_index(
            raw=raw,
            manifest_path=manifest_path,
            selected_namespaces=selected_namespaces,
        )
    return _parse_manifest_function_entries(
        raw=raw,
        manifest_path=manifest_path,
        namespace_hint=None,
        selected_namespaces=selected_namespaces,
    )


def _parse_manifest_namespace_index(
    *,
    raw: Mapping[str, Any],
    manifest_path: Path,
    selected_namespaces: set[str] | None,
) -> list[dict[str, Any]]:
    raw_namespaces = raw.get("namespaces")
    if not isinstance(raw_namespaces, list):
        raise ValueError(
            f"manifest '{manifest_path}' must contain a 'namespaces' list"
        )

    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(raw_namespaces):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest namespaces item #{index} must be an object")
        namespace = _normalize_optional_namespace(
            item.get("namespace"),
            field_name=f"namespaces[{index}].namespace",
        )
        if namespace is None:
            raise ValueError(f"namespaces[{index}].namespace cannot be empty")
        if selected_namespaces is not None and namespace not in selected_namespaces:
            continue

        manifest_ref = _normalize_required_string(item.get("manifest"), "manifest", index)
        namespace_manifest_path = _resolve_index_ref(manifest_ref, manifest_path.parent)
        if not namespace_manifest_path.exists():
            raise FileNotFoundError(
                f"namespace manifest not found: {namespace_manifest_path}"
            )
        namespace_raw = _read_json_object(namespace_manifest_path)
        parsed.extend(
            _parse_manifest_function_entries(
                raw=namespace_raw,
                manifest_path=namespace_manifest_path,
                namespace_hint=namespace,
                selected_namespaces=selected_namespaces,
            )
        )

    curve_ids = [item["curve_id"] for item in parsed]
    if len(curve_ids) != len(set(curve_ids)):
        raise ValueError(
            f"manifest '{manifest_path}' contains duplicated curve/function names"
        )
    return parsed


def _parse_manifest_function_entries(
    *,
    raw: Mapping[str, Any],
    manifest_path: Path,
    namespace_hint: str | None,
    selected_namespaces: set[str] | None,
) -> list[dict[str, Any]]:
    raw_functions = raw.get("functions")
    raw_curves = raw.get("curves")
    if isinstance(raw_functions, list):
        entries = raw_functions
        entry_label = "functions"
    elif isinstance(raw_curves, list):
        entries = raw_curves
        entry_label = "curves"
    else:
        raise ValueError(
            f"manifest '{manifest_path}' must contain a 'functions' or 'curves' list"
        )

    manifest_namespace = _normalize_optional_namespace(
        raw.get("namespace"),
        field_name="namespace",
    )
    if manifest_namespace is None:
        manifest_namespace = namespace_hint

    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest {entry_label} item #{idx} must be an object")

        raw_name = item.get("name", item.get("curve_id"))
        curve_id = _normalize_required_string(raw_name, "name", idx)
        source_ref = item.get("source")
        if source_ref is None:
            source_ref = item.get("file", item.get("path", item.get("json")))
        source = _normalize_required_string(source_ref, "source", idx)

        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError(f"manifest function '{curve_id}' metadata must be an object")
        metadata_dict = dict(metadata)

        item_group = _normalize_optional_namespace(
            item.get("group"),
            field_name=f"{entry_label}[{idx}].group",
        )
        metadata_group = _normalize_optional_namespace(
            metadata_dict.get("group"),
            field_name=f"{entry_label}[{idx}].metadata.group",
        )
        item_namespace = _normalize_optional_namespace(
            item.get("namespace"),
            field_name=f"{entry_label}[{idx}].namespace",
        )
        resolved_namespace = (
            item_namespace
            or item_group
            or metadata_group
            or manifest_namespace
            or "interp"
        )
        if resolved_namespace is not None:
            metadata_dict.setdefault("group", resolved_namespace)
        if selected_namespaces is not None and resolved_namespace not in selected_namespaces:
            continue

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
                "manifest_path": str(manifest_path),
            }
        )

    curve_ids = [item["curve_id"] for item in parsed]
    if len(curve_ids) != len(set(curve_ids)):
        raise ValueError(f"manifest '{manifest_path}' contains duplicated curve/function names")
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
    if "." in normalized:
        raise ValueError(f"{field_name} cannot contain '.'")
    return normalized


def _coerce_optional_namespace(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or "." in normalized:
        return None
    return normalized


def _normalize_selected_namespaces(
    value: list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if value is None:
        return None
    normalized: set[str] = set()
    for item in value:
        candidate = _normalize_optional_namespace(item, field_name="namespaces[]")
        if candidate is None:
            continue
        normalized.add(candidate)
    return normalized or None


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


def _resolve_curve_return_dtype(curve_entry: Mapping[str, Any]) -> Any | None:
    dtype_hint = curve_entry.get("return_dtype")
    metadata = curve_entry.get("metadata")
    if dtype_hint is None and isinstance(metadata, Mapping):
        dtype_hint = metadata.get("return_dtype")

    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError:
        return None

    if dtype_hint is None:
        return pl.Float64

    if isinstance(dtype_hint, str):
        candidate = getattr(pl, dtype_hint, None)
        if candidate is None:
            raise ValueError(
                f"invalid return_dtype '{dtype_hint}' for interpolation function"
            )
        return candidate

    return dtype_hint


def _build_curve_metadata(
    *,
    curve_id: str,
    target_namespace: str,
    source: str | None,
    kind: str | None,
    log_x: bool,
    log_y: bool,
    hot: bool,
    points: Any,
    x_min: Any,
    x_max: Any,
    metadata_override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "category": "interpolation",
        "curve_id": curve_id,
        "source": source,
        "kind": kind,
        "logX": bool(log_x),
        "logY": bool(log_y),
        "hot": bool(hot),
        "points": points,
        "x_min": x_min,
        "x_max": x_max,
    }
    if isinstance(metadata_override, Mapping):
        metadata.update(dict(metadata_override))

    metadata["group"] = target_namespace
    metadata["backend_support"] = _build_curve_backend_support_metadata(
        metadata.get("backend_support")
    )
    note = _compose_curve_note_with_input_range(
        existing_note=metadata.get("note"),
        x_min=metadata.get("x_min"),
        x_max=metadata.get("x_max"),
    )
    if note is not None:
        metadata["note"] = note
    return metadata


def _compose_curve_note_with_input_range(
    *,
    existing_note: Any,
    x_min: Any,
    x_max: Any,
) -> str | None:
    range_text = _format_curve_input_range(x_min=x_min, x_max=x_max)
    if range_text is None:
        if isinstance(existing_note, str):
            stripped = existing_note.strip()
            return stripped or None
        return None

    range_note = f"Input range: {range_text}."

    if isinstance(existing_note, str):
        stripped = existing_note.strip()
        if stripped:
            lowered = stripped.lower()
            if "input range" in lowered:
                return stripped
            return f"{stripped}\n\t{range_note}"
    return range_note


def _format_curve_input_range(*, x_min: Any, x_max: Any) -> str | None:
    min_value = _coerce_finite_float(x_min)
    max_value = _coerce_finite_float(x_max)
    if min_value is None or max_value is None:
        return None
    return f"x in [{min_value:.12g}, {max_value:.12g}]"


def _coerce_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _resolve_curve_bounds(
    *,
    curve_entry: Mapping[str, Any],
    curve_fn: Callable[..., Any],
) -> tuple[float | None, float | None]:
    min_value = _coerce_finite_float(curve_entry.get("x_min"))
    max_value = _coerce_finite_float(curve_entry.get("x_max"))
    if min_value is not None and max_value is not None:
        return min_value, max_value

    x_values = getattr(curve_fn, "x_values", None)
    if x_values is None:
        return min_value, max_value

    try:
        arr = np.asarray(x_values, dtype=float)
    except Exception:
        return min_value, max_value
    if arr.size == 0:
        return min_value, max_value

    inferred_min = float(np.min(arr))
    inferred_max = float(np.max(arr))
    if min_value is None:
        min_value = inferred_min
    if max_value is None:
        max_value = inferred_max
    return min_value, max_value


def _build_curve_backend_support_metadata(raw_value: Any) -> dict[str, str]:
    defaults = {
        "numpy": "native",
        "polars": "fallback_map_batches",
        "sync": "call",
        "async": "acall",
    }
    if not isinstance(raw_value, Mapping):
        return defaults
    merged = dict(defaults)
    for key, value in raw_value.items():
        if isinstance(key, str) and isinstance(value, str):
            merged[key] = value
    return merged


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
