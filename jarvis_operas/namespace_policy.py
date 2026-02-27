from __future__ import annotations

from functools import lru_cache

from .curves import list_interpolation_namespaces
from .name_utils import normalize_namespace, split_full_name

# Built-in, internal, and interpolation runtime namespaces that user ops
# cannot write to.
_STATIC_PROTECTED_NAMESPACES = frozenset(
    {
        "math",
        "stat",
        "helper",
        "interp",
        "core",
        "user",
    }
)


@lru_cache(maxsize=1)
def protected_namespaces() -> frozenset[str]:
    names = set(_STATIC_PROTECTED_NAMESPACES)
    try:
        names.update(list_interpolation_namespaces())
    except Exception:
        # Never break namespace guard when packaged manifests are unavailable.
        pass
    return frozenset(names)


def is_protected_namespace(namespace: str) -> bool:
    normalized = normalize_namespace(namespace)
    return normalized in protected_namespaces()


def ensure_user_namespace_allowed(namespace: str, *, action: str | None = None) -> str:
    normalized = normalize_namespace(namespace)
    if normalized in protected_namespaces():
        if action:
            raise ValueError(
                f"namespace '{normalized}' is protected and cannot be used for {action}"
            )
        raise ValueError(
            f"namespace '{normalized}' is protected and cannot be used for user functions"
        )
    return normalized
