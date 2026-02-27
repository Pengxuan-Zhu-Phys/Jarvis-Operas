from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from .io_utils import atomic_write_bytes

_CACHE_ROOT_ENV = "JARVIS_OPERAS_CURVE_CACHE_ROOT"
_INDEX_PATH_ENV = "JARVIS_OPERAS_CURVE_INDEX"
_INDEX_FILENAME = "index.json"
_INDEX_VERSION = 1
_PICKLE_PROTOCOL = 5


def get_curve_cache_root(cache_root: str | None = None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser().resolve()
    env_value = os.getenv(_CACHE_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / ".jarvis-operas" / "curve-cache").resolve()


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
    payload = pickle.dumps(value, protocol=_PICKLE_PROTOCOL)
    atomic_write_bytes(path, payload, fsync=True, temp_prefix=f"{path.name}.")


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
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, raw, fsync=True, temp_prefix=f"{path.name}.")
