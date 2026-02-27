from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import OperatorNotFound
from .io_utils import atomic_write_bytes
from .logging import get_logger
from .name_utils import (
    FULL_NAME_SEPARATOR as _FULL_NAME_SEPARATOR,
    namespace_from_source_path,
    normalize_full_name as normalize_common_full_name,
    normalize_namespace as normalize_common_namespace,
    try_split_full_name as try_split_common_full_name,
)
from .namespace_policy import (
    ensure_user_namespace_allowed,
    is_protected_namespace,
    split_full_name as split_policy_full_name,
)
from .core.registry import OperasRegistry

_STORE_LOCK = RLock()
_ENV_STORE_PATH = "JARVIS_OPERAS_PERSIST_FILE"
_ENV_SOURCES_PATH = "JARVIS_OPERAS_SOURCES_FILE"
_ENV_OVERRIDES_PATH = "JARVIS_OPERAS_OVERRIDES_FILE"
_ENV_USER_SOURCE_DIR = "JARVIS_OPERAS_USER_SOURCE_DIR"


def _legacy_env_base_path() -> Path | None:
    raw = os.getenv(_ENV_STORE_PATH)
    if not raw:
        return None
    return Path(raw).expanduser()


def _resolve_store_file_path(
    *,
    explicit_env: str,
    legacy_filename: str,
    default_filename: str,
) -> Path:
    raw = os.getenv(explicit_env)
    if raw:
        return Path(raw).expanduser()

    legacy_path = _legacy_env_base_path()
    if legacy_path is not None:
        if legacy_path.suffix:
            return legacy_path.with_name(legacy_filename)
        return legacy_path / legacy_filename

    return Path.home() / ".jarvis-operas" / default_filename


def get_persist_store_path() -> Path:
    """Return legacy combined persistence file path for compatibility."""

    legacy_path = _legacy_env_base_path()
    if legacy_path is not None:
        return legacy_path
    return Path.home() / ".jarvis-operas" / "user_ops.json"


def get_sources_store_path() -> Path:
    """Return source entries persistence file path."""

    return _resolve_store_file_path(
        explicit_env=_ENV_SOURCES_PATH,
        legacy_filename="sources.json",
        default_filename="sources.json",
    )


def get_overrides_store_path() -> Path:
    """Return override rules persistence file path."""

    return _resolve_store_file_path(
        explicit_env=_ENV_OVERRIDES_PATH,
        legacy_filename="overrides.json",
        default_filename="overrides.json",
    )


def get_user_source_store_dir(*, store_path: Path | None = None) -> Path:
    """Return managed snapshot directory for user operator source files."""

    raw = os.getenv(_ENV_USER_SOURCE_DIR)
    if raw:
        return Path(raw).expanduser()
    base_store = (store_path or get_sources_store_path()).expanduser()
    return base_store.parent / "user_sources"


def persist_user_ops(
    path: str,
    *,
    namespace: str | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a user operator script path so future processes auto-load it."""

    local_logger = get_logger(logger, action="persist_user_ops")
    store_path = get_sources_store_path()
    resolved_source_path = Path(path).expanduser().resolve()
    requested_namespace = namespace.strip() if isinstance(namespace, str) and namespace.strip() else None
    effective_namespace = ensure_user_namespace_allowed(
        requested_namespace or namespace_from_source_path(resolved_source_path),
        action="persisting user operators",
    )

    snapshot = _snapshot_user_source(
        resolved_source_path,
        store_path=store_path,
    )
    entry: dict[str, Any] = {
        "path": str(resolved_source_path),
        "load_path": snapshot["load_path"],
        "source_hash": snapshot["source_hash"],
        "namespace": effective_namespace,
    }

    with _STORE_LOCK:
        store = _read_sources_store(store_path, local_logger)
        entries = store["entries"]
        matched_index = next(
            (
                idx
                for idx, item in enumerate(entries)
                if item.get("path") == entry["path"]
                and item.get("namespace") == entry.get("namespace")
            ),
            None,
        )
        created = matched_index is None
        updated = False

        if created:
            entries.append(entry)
            _write_sources_store(store_path, store)
        else:
            existing_entry = entries[matched_index]
            changed = any(existing_entry.get(k) != v for k, v in entry.items())
            if changed:
                entries[matched_index] = {**existing_entry, **entry}
                _write_sources_store(store_path, store)
                updated = True

    local_logger.info(
        "persisted user operator source: {} (snapshot: {})",
        str(resolved_source_path),
        entry["load_path"],
    )
    return {
        "entry": dict(entry),
        "store_path": str(store_path),
        "created": created,
        "updated": updated,
    }


def list_persisted_user_ops(*, logger: Any | None = None) -> list[dict[str, Any]]:
    """Return persisted user operator script entries."""

    local_logger = get_logger(logger, action="list_persisted_user_ops")
    store = _read_sources_store(get_sources_store_path(), local_logger)
    return [dict(entry) for entry in store["entries"]]


def delete_persisted_function(
    full_name: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a function deletion override by full operator name."""

    normalized = _normalize_full_name(full_name)
    namespace, _ = split_policy_full_name(normalized)
    ensure_user_namespace_allowed(
        namespace,
        action="function deletion override",
    )
    local_logger = get_logger(logger, action="delete_persisted_function")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]
        deleted: list[str] = overrides["deleted_functions"]
        created = normalized not in deleted
        if created:
            deleted.append(normalized)
        overrides["renamed_functions"].pop(normalized, None)
        _write_overrides_store(store_path, store)

    local_logger.info("persisted function deletion override: {}", normalized)
    return {"full_name": normalized, "created": created, "store_path": str(store_path)}


def clear_persisted_function_overrides(
    full_name: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Clear function-level override rules for a specific operator."""

    normalized = _normalize_full_name(full_name)
    local_logger = get_logger(logger, action="clear_persisted_function_overrides")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]
        deleted_functions: list[str] = overrides["deleted_functions"]
        renamed_functions: dict[str, str] = overrides["renamed_functions"]

        before_deleted = len(deleted_functions)
        before_renamed = len(renamed_functions)

        overrides["deleted_functions"] = [item for item in deleted_functions if item != normalized]
        overrides["renamed_functions"] = {
            source: target
            for source, target in renamed_functions.items()
            if source != normalized and target != normalized
        }
        _write_overrides_store(store_path, store)

    local_logger.info("cleared function overrides for {}", normalized)
    return {
        "full_name": normalized,
        "removed_deleted": before_deleted - len(store["overrides"]["deleted_functions"]),
        "removed_renamed": before_renamed - len(store["overrides"]["renamed_functions"]),
        "store_path": str(store_path),
    }


def update_persisted_function(
    full_name: str,
    new_full_name: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a function rename/update override."""

    source = _normalize_full_name(full_name)
    target = _normalize_full_name(new_full_name)
    source_namespace, _ = split_policy_full_name(source)
    target_namespace, _ = split_policy_full_name(target)
    ensure_user_namespace_allowed(
        source_namespace,
        action="function rename override source",
    )
    ensure_user_namespace_allowed(
        target_namespace,
        action="function rename override target",
    )
    if source == target:
        raise ValueError("full_name and new_full_name must be different")

    local_logger = get_logger(logger, action="update_persisted_function")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]
        overrides["renamed_functions"][source] = target
        overrides["deleted_functions"] = [
            item
            for item in overrides["deleted_functions"]
            if item not in (source, target)
        ]
        _write_overrides_store(store_path, store)

    local_logger.info("persisted function rename override: {} -> {}", source, target)
    return {
        "from": source,
        "to": target,
        "store_path": str(store_path),
    }


def delete_persisted_namespace(
    namespace: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a namespace deletion override."""

    normalized = normalize_common_namespace(namespace)
    ensure_user_namespace_allowed(
        normalized,
        action="namespace deletion override",
    )
    local_logger = get_logger(logger, action="delete_persisted_namespace")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]
        deleted: list[str] = overrides["deleted_namespaces"]
        created = normalized not in deleted
        if created:
            deleted.append(normalized)
        overrides["renamed_namespaces"].pop(normalized, None)
        _write_overrides_store(store_path, store)

    local_logger.info("persisted namespace deletion override: {}", normalized)
    return {"namespace": normalized, "created": created, "store_path": str(store_path)}


def clear_persisted_namespace_overrides(
    namespace: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Clear namespace-level (and related function-level) override rules."""

    normalized = normalize_common_namespace(namespace)
    local_logger = get_logger(logger, action="clear_persisted_namespace_overrides")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]

        deleted_namespaces: list[str] = overrides["deleted_namespaces"]
        renamed_namespaces: dict[str, str] = overrides["renamed_namespaces"]
        deleted_functions: list[str] = overrides["deleted_functions"]
        renamed_functions: dict[str, str] = overrides["renamed_functions"]

        before_deleted_ns = len(deleted_namespaces)
        before_renamed_ns = len(renamed_namespaces)
        before_deleted_fn = len(deleted_functions)
        before_renamed_fn = len(renamed_functions)

        overrides["deleted_namespaces"] = [
            item for item in deleted_namespaces if item != normalized
        ]
        overrides["renamed_namespaces"] = {
            source: target
            for source, target in renamed_namespaces.items()
            if source != normalized and target != normalized
        }
        overrides["deleted_functions"] = [
            item
            for item in deleted_functions
            if _namespace_from_maybe_full_name(item) != normalized
        ]
        overrides["renamed_functions"] = {
            source: target
            for source, target in renamed_functions.items()
            if _namespace_from_maybe_full_name(source) != normalized
            and _namespace_from_maybe_full_name(target) != normalized
        }
        _write_overrides_store(store_path, store)

    local_logger.info("cleared namespace overrides for {}", normalized)
    return {
        "namespace": normalized,
        "removed_deleted_namespaces": before_deleted_ns - len(store["overrides"]["deleted_namespaces"]),
        "removed_renamed_namespaces": before_renamed_ns - len(store["overrides"]["renamed_namespaces"]),
        "removed_deleted_functions": before_deleted_fn - len(store["overrides"]["deleted_functions"]),
        "removed_renamed_functions": before_renamed_fn - len(store["overrides"]["renamed_functions"]),
        "store_path": str(store_path),
    }


def update_persisted_namespace(
    namespace: str,
    new_namespace: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a namespace rename/update override."""

    source = normalize_common_namespace(namespace)
    target = normalize_common_namespace(new_namespace)
    ensure_user_namespace_allowed(
        source,
        action="namespace rename override source",
    )
    ensure_user_namespace_allowed(
        target,
        action="namespace rename override target",
    )
    if source == target:
        raise ValueError("namespace and new_namespace must be different")

    local_logger = get_logger(logger, action="update_persisted_namespace")
    store_path = get_overrides_store_path()

    with _STORE_LOCK:
        store = _read_overrides_store(store_path, local_logger)
        overrides = store["overrides"]
        overrides["renamed_namespaces"][source] = target
        overrides["deleted_namespaces"] = [
            item for item in overrides["deleted_namespaces"] if item not in (source, target)
        ]
        _write_overrides_store(store_path, store)

    local_logger.info("persisted namespace rename override: {} -> {}", source, target)
    return {"from": source, "to": target, "store_path": str(store_path)}


def apply_persisted_overrides(
    registry: OperasRegistry,
    *,
    logger: Any | None = None,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Apply persisted delete/update overrides for functions and namespaces."""

    local_logger = get_logger(logger, action="apply_persisted_overrides")
    store = _read_overrides_store(get_overrides_store_path(), local_logger)
    overrides = store["overrides"]

    renamed_namespaces = dict(overrides["renamed_namespaces"])
    renamed_functions = dict(overrides["renamed_functions"])
    deleted_namespaces = list(overrides["deleted_namespaces"])
    deleted_functions = list(overrides["deleted_functions"])

    summary = {
        "updated_namespaces": [],
        "updated_functions": [],
        "deleted_namespaces": [],
        "deleted_functions": [],
    }

    for source_ns, target_ns in renamed_namespaces.items():
        if source_ns == target_ns:
            continue
        if is_protected_namespace(source_ns) or is_protected_namespace(target_ns):
            local_logger.debug(
                "skip protected namespace rename override {} -> {}",
                source_ns,
                target_ns,
            )
            continue
        try:
            updated = registry.rename_namespace(source_ns, target_ns)
            if updated:
                summary["updated_namespaces"].extend(updated)
        except Exception as exc:
            if strict:
                raise
            local_logger.warning(
                "skip namespace rename override {} -> {}: {}",
                source_ns,
                target_ns,
                exc,
            )

    for source_full, target_full in renamed_functions.items():
        source_effective = _remap_full_name_namespace(source_full, renamed_namespaces)
        target_effective = _remap_full_name_namespace(target_full, renamed_namespaces)
        if _is_protected_full_name(source_effective) or _is_protected_full_name(target_effective):
            local_logger.debug(
                "skip protected function rename override {} -> {}",
                source_full,
                target_full,
            )
            continue
        try:
            updated_name = registry.rename(source_effective, new_full_name=target_effective)
            summary["updated_functions"].append(updated_name)
        except OperatorNotFound:
            continue
        except Exception as exc:
            if strict:
                raise
            local_logger.warning(
                "skip function rename override {} -> {}: {}",
                source_full,
                target_full,
                exc,
            )

    for namespace in deleted_namespaces:
        namespace_effective = renamed_namespaces.get(namespace, namespace)
        if is_protected_namespace(namespace_effective):
            local_logger.debug(
                "skip protected namespace deletion override {}",
                namespace_effective,
            )
            continue
        try:
            deleted = registry.delete_namespace(namespace_effective)
            if deleted:
                summary["deleted_namespaces"].extend(deleted)
        except Exception as exc:
            if strict:
                raise
            local_logger.warning(
                "skip namespace deletion override {}: {}",
                namespace,
                exc,
            )

    for full_name in deleted_functions:
        candidates = [
            full_name,
            _remap_full_name_namespace(full_name, renamed_namespaces),
            renamed_functions.get(full_name),
        ]
        candidates = [c for c in candidates if c]
        deleted_any = False
        for candidate in dict.fromkeys(candidates):
            if _is_protected_full_name(candidate):
                local_logger.debug(
                    "skip protected function deletion override {}",
                    candidate,
                )
                continue
            try:
                registry.delete(candidate)
                summary["deleted_functions"].append(candidate)
                deleted_any = True
                break
            except OperatorNotFound:
                continue
            except Exception as exc:
                if strict:
                    raise
                local_logger.warning(
                    "skip function deletion override {} (candidate {}): {}",
                    full_name,
                    candidate,
                    exc,
                )
                break
        if not deleted_any:
            continue

    return {
        key: list(dict.fromkeys(values))
        for key, values in summary.items()
    }


def load_persisted_user_ops(
    registry: OperasRegistry,
    *,
    logger: Any | None = None,
    strict: bool = False,
    refresh_sympy: bool = True,
) -> list[str]:
    """Load all persisted user operator scripts into the provided registry."""

    local_logger = get_logger(logger, action="load_persisted_user_ops")
    entries = list_persisted_user_ops(logger=local_logger)
    if not entries:
        return []

    # Import lazily to avoid module cycle during API bootstrap.
    from .loading import load_user_ops

    loaded: list[str] = []
    for entry in entries:
        source_path = entry["path"]
        load_path = entry.get("load_path")
        primary_path = str(load_path).strip() if isinstance(load_path, str) and load_path.strip() else source_path
        namespace = entry.get("namespace")
        candidate_paths = [primary_path]
        if source_path not in candidate_paths:
            candidate_paths.append(source_path)

        selected_path: str | None = None
        for candidate in candidate_paths:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                selected_path = str(candidate_path)
                break

        if selected_path is None:
            local_logger.warning(
                "skip missing persisted operator file: source={} snapshot={}",
                source_path,
                primary_path,
            )
            continue
        try:
            loaded.extend(
                load_user_ops(
                    selected_path,
                    registry,
                    namespace=namespace,
                    refresh_sympy=refresh_sympy,
                    logger=local_logger,
                )
            )
        except Exception as exc:
            if strict:
                raise
            local_logger.warning("skip invalid persisted operator file '{}': {}", selected_path, exc)

    return list(dict.fromkeys(loaded))


def _empty_sources_store() -> dict[str, Any]:
    return {
        "version": 2,
        "entries": [],
    }


def _empty_overrides_store() -> dict[str, Any]:
    return {
        "version": 2,
        "overrides": {
            "deleted_functions": [],
            "renamed_functions": {},
            "deleted_namespaces": [],
            "renamed_namespaces": {},
        },
    }


def _normalize_full_name(full_name: str) -> str:
    return normalize_common_full_name(full_name)


def _is_protected_full_name(full_name: str) -> bool:
    try:
        namespace, _ = split_policy_full_name(full_name)
    except Exception:
        return False
    return is_protected_namespace(namespace)


def _namespace_from_maybe_full_name(value: str) -> str:
    split = try_split_common_full_name(value)
    if split is None:
        return ""
    namespace, _ = split
    return namespace


def _remap_full_name_namespace(
    full_name: str,
    renamed_namespaces: Mapping[str, str],
) -> str:
    normalized = _normalize_full_name(full_name)
    namespace, short_name = normalized.split(_FULL_NAME_SEPARATOR, 1)
    mapped_namespace = renamed_namespaces.get(namespace, namespace)
    return f"{mapped_namespace}{_FULL_NAME_SEPARATOR}{short_name}"


def _snapshot_user_source(
    source_path: Path,
    *,
    store_path: Path,
) -> dict[str, str]:
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"user operator source does not exist: {source_path}")

    payload = source_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    snapshot_dir = get_user_source_store_dir(store_path=store_path)
    safe_stem = _sanitize_snapshot_stem(source_path.stem)
    suffix = source_path.suffix or ".py"
    snapshot_path = snapshot_dir / f"{safe_stem}-{source_hash[:16]}{suffix}"

    if not snapshot_path.exists():
        atomic_write_bytes(snapshot_path, payload)

    return {
        "load_path": str(snapshot_path.resolve()),
        "source_hash": source_hash,
    }


def _sanitize_snapshot_stem(stem: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_")
    if not normalized:
        return "user_ops"
    return normalized


def _legacy_combined_store_path() -> Path:
    return get_persist_store_path().expanduser()


def _read_store_mapping_with_legacy_fallback(
    path: Path,
    logger: Any,
    *,
    label: str,
) -> Mapping[str, Any] | None:
    raw = _read_json_mapping(path, logger, label=label)
    if raw is not None:
        return raw

    legacy_path = _legacy_combined_store_path()
    if legacy_path == path:
        return None
    return _read_json_mapping(
        legacy_path,
        logger,
        label="legacy persist store",
    )


def _read_sources_store(path: Path, logger: Any) -> dict[str, Any]:
    raw = _read_store_mapping_with_legacy_fallback(
        path,
        logger,
        label="sources persist store",
    )
    if raw is None:
        return _empty_sources_store()
    entries = _normalize_persist_entries(raw.get("entries", []))
    return {
        "version": 2,
        "entries": entries,
    }


def _read_overrides_store(path: Path, logger: Any) -> dict[str, Any]:
    raw = _read_store_mapping_with_legacy_fallback(
        path,
        logger,
        label="overrides persist store",
    )
    if raw is None:
        return _empty_overrides_store()

    raw_overrides = raw.get("overrides")
    if raw_overrides is None:
        raw_overrides = raw
    if not isinstance(raw_overrides, Mapping):
        raw_overrides = {}

    return {
        "version": 2,
        "overrides": _normalize_override_rules(raw_overrides),
    }


def _normalize_persist_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, Mapping):
            continue
        item_path = item.get("path")
        if not isinstance(item_path, str) or not item_path.strip():
            continue
        normalized: dict[str, Any] = {"path": item_path.strip()}

        item_namespace = item.get("namespace")
        if isinstance(item_namespace, str) and item_namespace.strip():
            normalized["namespace"] = item_namespace.strip()

        item_load_path = item.get("load_path")
        if isinstance(item_load_path, str) and item_load_path.strip():
            normalized["load_path"] = item_load_path.strip()

        item_source_hash = item.get("source_hash")
        if isinstance(item_source_hash, str) and item_source_hash.strip():
            normalized["source_hash"] = item_source_hash.strip()
        entries.append(normalized)

    return entries


def _normalize_override_rules(raw_overrides: Mapping[str, Any]) -> dict[str, Any]:
    raw_deleted_functions = raw_overrides.get("deleted_functions", [])
    if not isinstance(raw_deleted_functions, list):
        raw_deleted_functions = []

    raw_renamed_functions = raw_overrides.get("renamed_functions", {})
    if not isinstance(raw_renamed_functions, Mapping):
        raw_renamed_functions = {}

    raw_deleted_namespaces = raw_overrides.get("deleted_namespaces", [])
    if not isinstance(raw_deleted_namespaces, list):
        raw_deleted_namespaces = []

    raw_renamed_namespaces = raw_overrides.get("renamed_namespaces", {})
    if not isinstance(raw_renamed_namespaces, Mapping):
        raw_renamed_namespaces = {}

    deleted_functions: list[str] = []
    for item in raw_deleted_functions:
        if not isinstance(item, str):
            continue
        try:
            deleted_functions.append(_normalize_full_name(item))
        except ValueError:
            continue

    renamed_functions: dict[str, str] = {}
    for source, target in raw_renamed_functions.items():
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        try:
            source_normalized = _normalize_full_name(source)
            target_normalized = _normalize_full_name(target)
        except ValueError:
            continue
        renamed_functions[source_normalized] = target_normalized

    deleted_namespaces: list[str] = []
    for item in raw_deleted_namespaces:
        if not isinstance(item, str):
            continue
        try:
            deleted_namespaces.append(normalize_common_namespace(item))
        except ValueError:
            continue

    renamed_namespaces: dict[str, str] = {}
    for source, target in raw_renamed_namespaces.items():
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        try:
            source_normalized = normalize_common_namespace(source)
            target_normalized = normalize_common_namespace(target)
        except ValueError:
            continue
        renamed_namespaces[source_normalized] = target_normalized

    return {
        "deleted_functions": list(dict.fromkeys(deleted_functions)),
        "renamed_functions": renamed_functions,
        "deleted_namespaces": list(dict.fromkeys(deleted_namespaces)),
        "renamed_namespaces": renamed_namespaces,
    }


def _read_json_mapping(path: Path, logger: Any, *, label: str) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("invalid {} '{}': {}. ignoring.", label, str(path), exc)
        return None
    if not isinstance(raw, Mapping):
        logger.warning("invalid {} '{}': expected object. ignoring.", label, str(path))
        return None
    return raw


def _write_store_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _write_sources_store(path: Path, store: Mapping[str, Any]) -> None:
    entries = _normalize_persist_entries(store.get("entries", []))
    payload = {
        "version": 2,
        "entries": [
            {
                "path": entry["path"],
                **({"namespace": entry["namespace"]} if "namespace" in entry else {}),
                **({"load_path": entry["load_path"]} if "load_path" in entry else {}),
                **({"source_hash": entry["source_hash"]} if "source_hash" in entry else {}),
            }
            for entry in entries
        ],
    }
    _write_store_json(path, payload)


def _write_overrides_store(path: Path, store: Mapping[str, Any]) -> None:
    raw_overrides = store.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raw_overrides = {}
    payload = {
        "version": 2,
        "overrides": _normalize_override_rules(raw_overrides),
    }
    _write_store_json(path, payload)
