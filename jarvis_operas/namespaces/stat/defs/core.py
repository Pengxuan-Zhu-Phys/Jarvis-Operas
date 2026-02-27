from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg


def chi2_cov_numpy(residual, cov, logger=None, **_params):
    """Compute chi2 = r^T * C^{-1} * r for residual and covariance."""

    if isinstance(residual, pd.DataFrame):
        result = _chi2_cov_dataframe(residual, cov)
    else:
        result = _chi2_cov_array_like(residual, cov)

    if logger is not None:
        logger.debug("stat.chi2_cov called")

    return result


def chi2_cov_polars_expr(residual, cov, **_params):
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("polars is required for stat.chi2_cov polars backend") from exc

    cov_mat = _cov_to_numpy(cov)
    if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
        raise ValueError("cov must be a square matrix")
    expected_dim = int(cov_mat.shape[0])

    def _udf(batch: Any):
        residual_mat = _batch_to_residual_matrix(batch, expected_dim=expected_dim)
        if residual_mat.shape[0] == 0:
            return pl.Series([], dtype=pl.Float64)
        solutions = linalg.solve(cov_mat, residual_mat.T, assume_a="sym")
        chi2_vals = np.einsum("nd,dn->n", residual_mat, solutions)
        return pl.Series(chi2_vals, dtype=pl.Float64)

    try:
        return residual.map_batches(_udf, return_dtype=pl.Float64)
    except TypeError:
        return residual.map_batches(_udf, pl.Float64)


def _chi2_cov_array_like(residual, cov):
    residual_arr = np.asarray(residual, dtype=float)
    cov_mat = _cov_to_numpy(cov)

    if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
        raise ValueError("cov must be a square matrix")

    if residual_arr.ndim == 1:
        if cov_mat.shape[0] != residual_arr.shape[0]:
            raise ValueError("residual length must match covariance dimension")
        solution = linalg.solve(cov_mat, residual_arr, assume_a="sym")
        return float(np.dot(residual_arr, solution))

    if residual_arr.ndim == 2:
        if cov_mat.shape[0] != residual_arr.shape[1]:
            raise ValueError("residual width must match covariance dimension")
        solutions = linalg.solve(cov_mat, residual_arr.T, assume_a="sym")
        return np.einsum("nd,dn->n", residual_arr, solutions)

    raise ValueError("residual must be a 1D or 2D array-like")


def _chi2_cov_dataframe(residual: pd.DataFrame, cov):
    residual_df = residual.astype(float)

    if residual_df.shape[1] == 0:
        raise ValueError("residual dataframe must have at least one column")

    if isinstance(cov, pd.DataFrame):
        cov_df = cov.astype(float)
        if cov_df.shape[0] != cov_df.shape[1]:
            raise ValueError("cov must be a square matrix")

        residual_cols = list(residual_df.columns)
        missing = [col for col in residual_cols if col not in cov_df.index or col not in cov_df.columns]
        if missing:
            raise ValueError(
                "cov dataframe must contain residual columns on both index and columns: "
                + ", ".join(map(str, missing))
            )
        cov_mat = cov_df.loc[residual_cols, residual_cols].to_numpy(dtype=float)
    else:
        cov_mat = _cov_to_numpy(cov)
        if cov_mat.shape[0] != residual_df.shape[1]:
            raise ValueError("residual width must match covariance dimension")

    residual_mat = residual_df.to_numpy(dtype=float)
    solutions = linalg.solve(cov_mat, residual_mat.T, assume_a="sym")
    chi2_vals = np.einsum("nd,dn->n", residual_mat, solutions)
    return pd.Series(chi2_vals, index=residual_df.index, name="chi2")


def _cov_to_numpy(cov) -> np.ndarray:
    if isinstance(cov, pd.DataFrame):
        return cov.to_numpy(dtype=float)
    return np.asarray(cov, dtype=float)


def _batch_to_residual_matrix(batch: Any, *, expected_dim: int) -> np.ndarray:
    values = batch.to_list()
    if not values:
        return np.empty((0, expected_dim), dtype=float)

    first = values[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        matrix = np.asarray(
            [[row[key] for key in keys] for row in values],
            dtype=float,
        )
    elif isinstance(first, (list, tuple, np.ndarray)):
        matrix = np.asarray(values, dtype=float)
    else:
        matrix = np.asarray(values, dtype=float).reshape(-1, 1)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("residual expression must yield scalar/list/struct rows")
    if matrix.shape[1] != expected_dim:
        raise ValueError("residual width must match covariance dimension")
    return matrix
