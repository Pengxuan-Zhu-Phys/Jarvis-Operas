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
_OPERAS_LOG_DOMAIN = "jarvis_operas"


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
    level_prefix = (
        f"[<level>{record['level']}</level>] >>> "
        if _CURRENT_MODE == "debug"
        else ">>> "
    )
    return (
        f"\n <cyan>{module}</cyan> "
        f"\n\t-> <green>{record['time']:MM-DD HH:mm:ss.SSS}</green> - "
        f"{level_prefix}\n"
        f"<level>{{message}}</level> "
    )


def _stream_filter(record: dict[str, Any]) -> bool:
    extra = record.get("extra", {})
    domain = extra.get("_log_domain")
    if domain is not None:
        return domain == _OPERAS_LOG_DOMAIN
    if extra.get("_jarvis_operas", False):
        return True
    module = extra.get("module", "")
    return isinstance(module, str) and module.startswith("Jarvis-Operas")


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
            filter=_stream_filter,
        )
        _CURRENT_MODE = normalized_mode


def set_log_mode(mode: str) -> None:
    """Configure Jarvis-Operas CLI sink and set its verbosity."""

    _configure_default_logger(mode)


def configure_cli_logger(mode: str | None = None) -> None:
    """Configure console sink for jopera CLI only."""

    target_mode = _initial_mode_from_env() if mode is None else mode
    _configure_default_logger(target_mode)


def get_log_mode() -> str:
    """Return current default log mode."""

    return _CURRENT_MODE


def get_logger(
    logger: Any | None = None,
    *,
    mode: str | None = None,
    **bind_kwargs: Any,
):
    """Return a bound logger.

    In library mode (logger=None), initialize Jarvis-Operas sink lazily so
    default output follows warning-level Operas formatting instead of loguru's
    implicit sink.
    """

    if logger is None:
        target_mode = _initial_mode_from_env() if mode is None else mode
        _configure_default_logger(target_mode)
        active_logger = _default_logger
    else:
        active_logger = logger

    bind_kwargs.setdefault("module", "Jarvis-Operas")
    bind_kwargs.setdefault("to_console", False)
    bind_kwargs.setdefault("Jarvis", False)
    bind_kwargs.setdefault("_jarvis_operas", True)
    bind_kwargs.setdefault("_log_domain", _OPERAS_LOG_DOMAIN)
    return active_logger.bind(**bind_kwargs)
