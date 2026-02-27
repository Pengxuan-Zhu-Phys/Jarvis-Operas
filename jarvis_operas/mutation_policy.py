from __future__ import annotations

import os

_DEV_WRITE_ENV = "JARVIS_OPERAS_DEV_WRITE"
_SUPPORTED_USER_WRITE_COMMANDS = frozenset({"load"})


def dev_write_enabled() -> bool:
    raw = os.getenv(_DEV_WRITE_ENV, "")
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def ensure_cli_write_allowed(command: str) -> None:
    normalized = str(command).strip().lower()
    if normalized in _SUPPORTED_USER_WRITE_COMMANDS:
        return
    if dev_write_enabled():
        return
    raise ValueError(
        f"CLI write command '{command}' is disabled by JO policy. "
        "Supported user write path is 'jopera load'. "
        f"For developer maintenance, set {_DEV_WRITE_ENV}=1."
    )

