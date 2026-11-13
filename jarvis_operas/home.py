"""Process state for Jarvis-Operas under the shared Jarvis home directory."""

from __future__ import annotations

import os
from pathlib import Path

_HOME_ENV = "JARVIS_HOME"
_OPERAS_HOME_ENV = "JARVIS_OPERAS_HOME"


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
