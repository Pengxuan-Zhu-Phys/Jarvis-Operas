from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import linalg


def chi2_cov(residual=None, cov=None, observables=None, logger=None):
    """Compute chi2 = r^T * C^{-1} * r for residual and covariance."""

    source = _as_mapping(observables, name="observables")
    if cov is None and isinstance(residual, Mapping):
        source = dict(residual) if source is None else {**source, **residual}
        residual = None

    if residual is None and source is not None and "residual" in source:
        residual = source["residual"]
    if cov is None and source is not None and "cov" in source:
        cov = source["cov"]

    if residual is None or cov is None:
        raise ValueError(
            "chi2_cov requires 'residual' and 'cov', or observables containing both keys."
        )

    if isinstance(residual, pd.DataFrame):
        result = _chi2_cov_dataframe(residual, cov)
    else:
        result = _chi2_cov_array_like(residual, cov)

    if logger is not None:
        logger.debug("chi2_cov called")

    return result


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


def _as_mapping(value, *, name: str) -> Mapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
