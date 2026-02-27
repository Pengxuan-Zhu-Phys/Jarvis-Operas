from __future__ import annotations

import numpy as np
import pandas as pd


def eggbox_numpy(x, y, logger=None, **_params):
    x_values = _to_numeric_array_like(x)
    y_values = _to_numeric_array_like(y)

    if logger is not None:
        logger.info("EggBox input loaded: x -> {}, y -> {}", x_values, y_values)

    z = (np.sin(np.pi * x_values) * np.cos(np.pi * y_values) + 2.0) ** 5

    if logger is not None:
        logger.debug("EggBox called: z -> {}", z)

    if _is_numpy_scalar(z):
        return float(z)
    return z


def eggbox2d_numpy(x, y, logger=None, **_params):
    return {"z": eggbox_numpy(x, y, logger=logger)}


def eggbox_polars_expr(x, y, **_params):
    return ((x * np.pi).sin() * (y * np.pi).cos() + 2.0) ** 5


def eggbox2d_polars_expr(x, y, **_params):
    import polars as pl  # type: ignore[import-not-found]

    return pl.struct(eggbox_polars_expr(x, y).alias("z"))


def _to_numeric_array_like(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.astype(float)
    return np.asarray(value, dtype=float)


def _is_numpy_scalar(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 0
