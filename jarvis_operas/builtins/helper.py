from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def eggbox(inputs=None, observables=None, logger=None):
    """Evaluate EggBox benchmark with x/y from inputs or observables mapping."""

    source = None
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping with keys 'x' and 'y'")
        source = dict(inputs)
    if observables is not None:
        if not isinstance(observables, Mapping):
            raise ValueError("observables must be a mapping when provided")
        if source is None:
            source = dict(observables)
        else:
            source = {**observables, **source}

    if source is None:
        raise ValueError("eggbox requires 'inputs' or 'observables' mapping.")
    if "x" not in source or "y" not in source:
        raise ValueError("mapping must contain both 'x' and 'y'")

    x = _to_numeric_array_like(source["x"])
    y = _to_numeric_array_like(source["y"])
    z = (np.sin(np.pi * x) * np.cos(np.pi * y) + 2.0) ** 5

    if logger is not None:
        logger.debug("eggbox called")

    if _is_numpy_scalar(z):
        return float(z)
    return z


def eggbox2d(inputs=None, observables=None, logger=None):
    """Backward-compatible alias for eggbox."""
    return eggbox(inputs=inputs, observables=observables, logger=logger)


def _to_numeric_array_like(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.astype(float)
    return np.asarray(value, dtype=float)


def _is_numpy_scalar(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 0
