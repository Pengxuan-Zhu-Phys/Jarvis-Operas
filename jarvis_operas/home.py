"""Process state for Jarvis-Operas under the shared Jarvis home directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from threading import Lock

_HOME_ENV = "JARVIS_HOME"
_OPERAS_HOME_ENV = "JARVIS_OPERAS_HOME"
_LEGACY_DIR_NAME = ".jarvis-operas"
_STATE_NAMES = (
    "curve-cache",
    "sources.json",
    "overrides.json",
    "user_sources",
    "user_ops.json",
)
_migrate_lock = Lock()


def user_home() -> Path:
    return Path.home()


def jarvis_home() -> Path:
    """Return the shared Jarvis directory, honoring ``JARVIS_HOME``."""

    raw = os.getenv(_HOME_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    return (user_home() / ".jarvis").resolve()


def operas_home() -> Path:
    """Return ``~/.jarvis/operas``, or ``JARVIS_OPERAS_HOME`` when set."""

    raw = os.getenv(_OPERAS_HOME_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    return jarvis_home() / "operas"


def legacy_operas_home() -> Path:
    return (user_home() / _LEGACY_DIR_NAME).resolve()


def migrate_legacy_operas_state() -> Path:
    """Move leftover ``~/.jarvis-operas`` entries into the current operas home.

    Existing files in the new directory win. An empty legacy directory is
    removed. Explicit environment paths never come through here.
    """

    target_root = operas_home()
    with _migrate_lock:
        legacy_root = legacy_operas_home()
        if legacy_root.exists() and legacy_root != target_root:
            for name in _STATE_NAMES:
                _move_legacy_entry(legacy_root / name, target_root / name)
            if legacy_root.is_dir() and not any(legacy_root.iterdir()):
                legacy_root.rmdir()
    return target_root


def _move_legacy_entry(source: Path, target: Path) -> None:
    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
