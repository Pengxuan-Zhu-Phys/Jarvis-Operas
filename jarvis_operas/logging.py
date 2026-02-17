from __future__ import annotations

import os
import sys
from threading import RLock
from typing import Any

from loguru import logger as _default_logger

_LOG_MODE_TO_LEVEL = {
    "warning": "WARNING",
    "info": "INFO",
    "debug": "DEBUG",
}
_DEFAULT_MODE = "warning"
_CURRENT_MODE = _DEFAULT_MODE
_SINK_ID: int | None = None
_INITIALIZED = False
_LOCK = RLock()


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _LOG_MODE_TO_LEVEL:
        valid = ", ".join(sorted(_LOG_MODE_TO_LEVEL.keys()))
        raise ValueError(f"Invalid log mode '{mode}'. Expected one of: {valid}")
    return normalized


def _initial_mode_from_env() -> str:
    raw = os.getenv("JARVIS_OPERAS_LOG_MODE", _DEFAULT_MODE)
    try:
        return _normalize_mode(raw)
    except ValueError:
        return _DEFAULT_MODE


def _custom_format(record: dict[str, Any]) -> str:
    module = record["extra"].get("module", "Jarvis-Operas")
    if "raw" in record["extra"]:
        return "{message}\n"
    return (
        f"\n <cyan>{module}</cyan> "
        f"\n\t-> <green>{record['time']:MM-DD HH:mm:ss.SSS}</green> - "
        f"[<level>{record['level']}</level>] >>> \n"
        f"<level>{record['message']}</level> "
    )


def _configure_default_logger(mode: str) -> None:
    normalized_mode = _normalize_mode(mode)
    level = _LOG_MODE_TO_LEVEL[normalized_mode]

    global _SINK_ID, _CURRENT_MODE, _INITIALIZED
    with _LOCK:
        if _INITIALIZED and _SINK_ID is not None and _CURRENT_MODE == normalized_mode:
            return

        if not _INITIALIZED:
            # Remove loguru's implicit default sink so Jarvis-Operas controls
            # both formatting and default verbosity.
            try:
                _default_logger.remove(0)
            except ValueError:
                pass
            _INITIALIZED = True

        if _SINK_ID is not None:
            try:
                _default_logger.remove(_SINK_ID)
            except ValueError:
                pass
            _SINK_ID = None

        _SINK_ID = _default_logger.add(
            sys.stdout,
            format=_custom_format,
            colorize=True,
            enqueue=True,
            level=level,
        )
        _CURRENT_MODE = normalized_mode


def set_log_mode(mode: str) -> None:
    """Set default log mode for Jarvis-Operas internal logger."""

    _configure_default_logger(mode)


def get_log_mode() -> str:
    """Return current default log mode."""

    return _CURRENT_MODE


def get_logger(
    logger: Any | None = None,
    *,
    mode: str | None = None,
    **bind_kwargs: Any,
):
    """Return a usable loguru logger without creating duplicate handlers."""

    global _CURRENT_MODE

    if logger is None:
        if mode is None:
            if not _INITIALIZED:
                _CURRENT_MODE = _initial_mode_from_env()
            target_mode = _CURRENT_MODE
        else:
            target_mode = mode
        _configure_default_logger(target_mode)
        active_logger = _default_logger.bind(
            module="Jarvis-Operas",
            to_console=True,
            Jarvis=True,
        )
    else:
        active_logger = logger

    if bind_kwargs:
        return active_logger.bind(**bind_kwargs)
    return active_logger
