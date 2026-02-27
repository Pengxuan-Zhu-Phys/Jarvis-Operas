from __future__ import annotations


def add_numpy(a, b, logger=None, **_params):
    if logger is not None:
        logger.debug("math.add called")
    return a + b


def add_polars_expr(a, b, **_params):
    return a + b


def identity_numpy(x, logger=None, **_params):
    if logger is not None:
        logger.debug("math.identity called")
    return x


def identity_polars_expr(x, **_params):
    return x
