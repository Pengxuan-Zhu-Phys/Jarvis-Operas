from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    fsync: bool = False,
    temp_prefix: str | None = None,
) -> None:
    """Write bytes atomically by replacing destination with a temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = temp_prefix if temp_prefix is not None else f".{path.name}."
    fd, temp_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(payload)
            fp.flush()
            if fsync:
                os.fsync(fp.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
