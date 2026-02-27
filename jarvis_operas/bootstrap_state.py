from __future__ import annotations

from contextlib import contextmanager

_BOOTSTRAPPING_GLOBAL_REGISTRY = False


def is_bootstrapping_global_registry() -> bool:
    return _BOOTSTRAPPING_GLOBAL_REGISTRY


@contextmanager
def bootstrapping_global_registry():
    global _BOOTSTRAPPING_GLOBAL_REGISTRY
    previous = _BOOTSTRAPPING_GLOBAL_REGISTRY
    _BOOTSTRAPPING_GLOBAL_REGISTRY = True
    try:
        yield
    finally:
        _BOOTSTRAPPING_GLOBAL_REGISTRY = previous
