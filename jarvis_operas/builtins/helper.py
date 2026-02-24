from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def eggbox(observables=None, logger=None):
    """Evaluate EggBox benchmark with x/y from observables mapping."""

    if observables is None:
        if logger is not None:
            logger.error("eggbox requires 'observables' mapping.")
        raise ValueError("eggbox requires 'observables' mapping.")
    if not isinstance(observables, Mapping):
        if logger is not None:
            logger.error("observables must be a mapping when provided")
        raise ValueError("observables must be a mapping when provided")
    if "x" not in observables or "y" not in observables:
        if logger is not None:
            logger.error("mapping must contain both 'x' and 'y'")
        raise ValueError("mapping must contain both 'x' and 'y'")

    x = _to_numeric_array_like(observables["x"])
    y = _to_numeric_array_like(observables["y"])
    if logger is not None: 
        logger.info("Eggbox 2D function input loaded: \n\t x -> {}, y -> {}".format(x, y))
    z = (np.sin(np.pi * x) * np.cos(np.pi * y) + 2.0) ** 5

    if logger is not None:
        logger.debug("Jarvis-Operas -> EggBox 2D function called: \n\t z -> {}".format(z))

    if _is_numpy_scalar(z):
        return float(z)
    return z


def eggbox2d(observables=None, logger=None, **_):
    """EggBox operator with dict-style output for mapping workflows."""
    return {
        "z": eggbox(
            observables=observables,
            logger=logger,
        )
    }


def _to_numeric_array_like(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.astype(float)
    return np.asarray(value, dtype=float)


def _is_numpy_scalar(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 0
