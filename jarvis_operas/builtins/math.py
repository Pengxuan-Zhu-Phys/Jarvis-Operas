from __future__ import annotations

from collections.abc import Mapping


def identity(x=None, observables=None, key=None, logger=None):
    """Return input unchanged, with optional value lookup from observables dict."""

    source = _as_mapping(observables, name="observables")
    if x is None and source is not None:
        if key is not None:
            if key not in source:
                raise ValueError(f"observables does not contain key '{key}'")
            x = source[key]
        elif "x" in source:
            x = source["x"]
        else:
            x = source

    if x is None:
        raise ValueError("identity requires 'x' or 'observables'.")

    if logger is not None:
        logger.debug("identity called")
    return x


def add(a=None, b=None, observables=None, logger=None):
    """Return a + b, with optional lookup from observables dict."""

    source = _as_mapping(observables, name="observables")
    if b is None and isinstance(a, Mapping):
        source = dict(a) if source is None else {**source, **a}
        a = None

    if a is None and source is not None and "a" in source:
        a = source["a"]
    if b is None and source is not None and "b" in source:
        b = source["b"]

    if a is None or b is None:
        raise ValueError("add requires 'a' and 'b', or observables containing both keys.")

    if logger is not None:
        logger.debug("add called")
    return a + b


def _as_mapping(value, *, name: str) -> Mapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
