from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import OperatorNotFound
from .logging import get_logger
from .registry import OperatorRegistry

_STORE_LOCK = RLock()
_ENV_STORE_PATH = "JARVIS_OPERAS_PERSIST_FILE"
_FULL_NAME_SEPARATOR = "."
_LEGACY_FULL_NAME_SEPARATOR = ":"


def get_persist_store_path() -> Path:
    """Return the persistence file path for user-registered operator scripts."""

    raw = os.getenv(_ENV_STORE_PATH)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".jarvis-operas" / "user_ops.json"


def persist_user_ops(
    path: str,
    *,
    namespace: str | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a user operator script path so future processes auto-load it."""

    local_logger = get_logger(logger, action="persist_user_ops")
    store_path = get_persist_store_path()
    resolved_path = str(Path(path).expanduser().resolve())
    resolved_namespace = namespace.strip() if isinstance(namespace, str) and namespace.strip() else None

    entry: dict[str, Any] = {"path": resolved_path}
    if resolved_namespace is not None:
        entry["namespace"] = resolved_namespace

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
        entries = store["entries"]
        exists = any(
            item.get("path") == entry["path"]
            and item.get("namespace") == entry.get("namespace")
            for item in entries
        )
        if not exists:
            entries.append(entry)
            _write_store(store_path, store)

    local_logger.info("persisted user operator source: {}", resolved_path)
    return {
        "entry": dict(entry),
        "store_path": str(store_path),
        "created": not exists,
    }


def list_persisted_user_ops(*, logger: Any | None = None) -> list[dict[str, Any]]:
    """Return persisted user operator script entries."""

    local_logger = get_logger(logger, action="list_persisted_user_ops")
    store = _read_store(get_persist_store_path(), local_logger)
    return [dict(entry) for entry in store["entries"]]


def delete_persisted_function(
    full_name: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Persist a function deletion override by full operator name."""

    normalized = _normalize_full_name(full_name)
    local_logger = get_logger(logger, action="delete_persisted_function")
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
        overrides = store["overrides"]
        deleted: list[str] = overrides["deleted_functions"]
        created = normalized not in deleted
        if created:
            deleted.append(normalized)
        overrides["renamed_functions"].pop(normalized, None)
        _write_store(store_path, store)

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
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
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
        _write_store(store_path, store)

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
    if source == target:
        raise ValueError("full_name and new_full_name must be different")

    local_logger = get_logger(logger, action="update_persisted_function")
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
        overrides = store["overrides"]
        overrides["renamed_functions"][source] = target
        overrides["deleted_functions"] = [
            item
            for item in overrides["deleted_functions"]
            if item not in (source, target)
        ]
        _write_store(store_path, store)

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

    normalized = _normalize_namespace(namespace)
    local_logger = get_logger(logger, action="delete_persisted_namespace")
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
        overrides = store["overrides"]
        deleted: list[str] = overrides["deleted_namespaces"]
        created = normalized not in deleted
        if created:
            deleted.append(normalized)
        overrides["renamed_namespaces"].pop(normalized, None)
        _write_store(store_path, store)

    local_logger.info("persisted namespace deletion override: {}", normalized)
    return {"namespace": normalized, "created": created, "store_path": str(store_path)}


def clear_persisted_namespace_overrides(
    namespace: str,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Clear namespace-level (and related function-level) override rules."""

    normalized = _normalize_namespace(namespace)
    local_logger = get_logger(logger, action="clear_persisted_namespace_overrides")
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
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
            if (_split_full_name(item) or ("", ""))[0] != normalized
        ]
        overrides["renamed_functions"] = {
            source: target
            for source, target in renamed_functions.items()
            if (_split_full_name(source) or ("", ""))[0] != normalized
            and (_split_full_name(target) or ("", ""))[0] != normalized
        }
        _write_store(store_path, store)

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

    source = _normalize_namespace(namespace)
    target = _normalize_namespace(new_namespace)
    if source == target:
        raise ValueError("namespace and new_namespace must be different")

    local_logger = get_logger(logger, action="update_persisted_namespace")
    store_path = get_persist_store_path()

    with _STORE_LOCK:
        store = _read_store(store_path, local_logger)
        overrides = store["overrides"]
        overrides["renamed_namespaces"][source] = target
        overrides["deleted_namespaces"] = [
            item for item in overrides["deleted_namespaces"] if item not in (source, target)
        ]
        _write_store(store_path, store)

    local_logger.info("persisted namespace rename override: {} -> {}", source, target)
    return {"from": source, "to": target, "store_path": str(store_path)}


def apply_persisted_overrides(
    registry: OperatorRegistry,
    *,
    logger: Any | None = None,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Apply persisted delete/update overrides for functions and namespaces."""

    local_logger = get_logger(logger, action="apply_persisted_overrides")
    store = _read_store(get_persist_store_path(), local_logger)
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
    registry: OperatorRegistry,
    *,
    logger: Any | None = None,
    strict: bool = False,
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
        path = entry["path"]
        namespace = entry.get("namespace")
        if not Path(path).expanduser().exists():
            local_logger.warning("skip missing persisted operator file: {}", path)
            continue
        try:
            loaded.extend(
                load_user_ops(
                    path,
                    registry,
                    namespace=namespace,
                    logger=local_logger,
                )
            )
        except Exception as exc:
            if strict:
                raise
            local_logger.warning("skip invalid persisted operator file '{}': {}", path, exc)

    return list(dict.fromkeys(loaded))


def _empty_store() -> dict[str, Any]:
    return {
        "version": 2,
        "entries": [],
        "overrides": {
            "deleted_functions": [],
            "renamed_functions": {},
            "deleted_namespaces": [],
            "renamed_namespaces": {},
        },
    }


def _normalize_full_name(full_name: str) -> str:
    normalized = full_name.strip()
    namespace, name = _split_full_name(normalized)
    namespace = _normalize_namespace(namespace)
    name = name.strip()
    if not name:
        raise ValueError("operator name cannot be empty")
    if _FULL_NAME_SEPARATOR in name:
        raise ValueError(f"operator name cannot contain '{_FULL_NAME_SEPARATOR}'")
    if _LEGACY_FULL_NAME_SEPARATOR in name:
        raise ValueError(f"operator name cannot contain '{_LEGACY_FULL_NAME_SEPARATOR}'")
    return f"{namespace}{_FULL_NAME_SEPARATOR}{name}"


def _normalize_namespace(namespace: str) -> str:
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace cannot be empty")
    if _FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"namespace cannot contain '{_FULL_NAME_SEPARATOR}'")
    if _LEGACY_FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"namespace cannot contain '{_LEGACY_FULL_NAME_SEPARATOR}'")
    return normalized


def _remap_full_name_namespace(
    full_name: str,
    renamed_namespaces: Mapping[str, str],
) -> str:
    normalized = _normalize_full_name(full_name)
    namespace, short_name = normalized.split(_FULL_NAME_SEPARATOR, 1)
    mapped_namespace = renamed_namespaces.get(namespace, namespace)
    return f"{mapped_namespace}{_FULL_NAME_SEPARATOR}{short_name}"


def _split_full_name(value: str) -> tuple[str, str]:
    has_new = _FULL_NAME_SEPARATOR in value
    has_legacy = _LEGACY_FULL_NAME_SEPARATOR in value
    if has_new and has_legacy:
        raise ValueError(
            "full_name cannot mix '.' and ':' separators; use '<namespace>.<name>'"
        )
    if has_new:
        return value.split(_FULL_NAME_SEPARATOR, 1)
    if has_legacy:
        return value.split(_LEGACY_FULL_NAME_SEPARATOR, 1)
    raise ValueError("full_name must be in '<namespace>.<name>' format")


def _read_store(path: Path, logger: Any) -> dict[str, Any]:
    if not path.exists():
        return _empty_store()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("invalid persist store '{}': {}. ignoring.", str(path), exc)
        return _empty_store()

    if not isinstance(raw, Mapping):
        logger.warning("invalid persist store '{}': expected object. ignoring.", str(path))
        return _empty_store()

    raw_entries = raw.get("entries", [])
    if not isinstance(raw_entries, list):
        logger.warning("invalid persist store '{}': entries must be list. ignoring.", str(path))
        return _empty_store()

    raw_overrides = raw.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raw_overrides = {}

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

    entries: list[dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, Mapping):
            continue
        item_path = item.get("path")
        if not isinstance(item_path, str) or not item_path.strip():
            continue
        item_namespace = item.get("namespace")
        if item_namespace is not None and not isinstance(item_namespace, str):
            item_namespace = None
        normalized: dict[str, Any] = {"path": item_path}
        if item_namespace and item_namespace.strip():
            normalized["namespace"] = item_namespace.strip()
        entries.append(normalized)

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
            deleted_namespaces.append(_normalize_namespace(item))
        except ValueError:
            continue

    renamed_namespaces: dict[str, str] = {}
    for source, target in raw_renamed_namespaces.items():
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        try:
            source_normalized = _normalize_namespace(source)
            target_normalized = _normalize_namespace(target)
        except ValueError:
            continue
        renamed_namespaces[source_normalized] = target_normalized

    return {
        "version": 2,
        "entries": entries,
        "overrides": {
            "deleted_functions": list(dict.fromkeys(deleted_functions)),
            "renamed_functions": renamed_functions,
            "deleted_namespaces": list(dict.fromkeys(deleted_namespaces)),
            "renamed_namespaces": renamed_namespaces,
        },
    }


def _write_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "entries": [
            {
                "path": entry["path"],
                **({"namespace": entry["namespace"]} if "namespace" in entry else {}),
            }
            for entry in store["entries"]
        ],
        "overrides": {
            "deleted_functions": list(store["overrides"]["deleted_functions"]),
            "renamed_functions": dict(store["overrides"]["renamed_functions"]),
            "deleted_namespaces": list(store["overrides"]["deleted_namespaces"]),
            "renamed_namespaces": dict(store["overrides"]["renamed_namespaces"]),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
