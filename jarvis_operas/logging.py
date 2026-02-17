from __future__ import annotations

from typing import Any

from loguru import logger as _default_logger


def get_logger(logger: Any | None = None, **bind_kwargs: Any):
    """Return a usable loguru logger without creating duplicate handlers."""

    active_logger = logger if logger is not None else _default_logger.bind(module="Jarvis-Operas")
    if bind_kwargs:
        return active_logger.bind(**bind_kwargs)
    return active_logger
