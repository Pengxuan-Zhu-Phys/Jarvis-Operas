from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FULL_NAME_SEPARATOR = "."
_NAMESPACE_FROM_STEM_RE = re.compile(r"[^0-9A-Za-z_]+")


def normalize_namespace(namespace: Any) -> str:
    normalized = str(namespace).strip()
    if not normalized:
        raise ValueError("namespace cannot be empty")
    if FULL_NAME_SEPARATOR in normalized:
        raise ValueError("namespace cannot contain '.'")
    return normalized


def normalize_short_name(name: Any, *, field_label: str = "name") -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError(f"{field_label} cannot be empty")
    if FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"{field_label} cannot contain '.'")
    return normalized


def split_full_name(value: Any) -> tuple[str, str]:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("full_name cannot be empty")
    if FULL_NAME_SEPARATOR not in normalized:
        raise ValueError("full_name must be in '<namespace>.<name>' format")
    namespace, short_name = normalized.split(FULL_NAME_SEPARATOR, 1)
    return normalize_namespace(namespace), short_name.strip()


def try_split_full_name(value: Any) -> tuple[str, str] | None:
    try:
        return split_full_name(value)
    except Exception:
        return None


def normalize_full_name(value: Any) -> str:
    namespace, short_name = split_full_name(value)
    normalized_short = normalize_short_name(short_name, field_label="operator name")
    return f"{namespace}{FULL_NAME_SEPARATOR}{normalized_short}"


def namespace_from_source_path(path: str | Path, *, default: str = "user_ops") -> str:
    stem = Path(path).stem.strip()
    normalized = _NAMESPACE_FROM_STEM_RE.sub("_", stem).strip("_")
    return normalized or default
