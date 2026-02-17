from __future__ import annotations

import numpy as np
from scipy import linalg


def identity(x, logger=None):
    """Return input unchanged."""

    if logger is not None:
        logger.debug("identity called")
    return x


def add(a, b, logger=None):
    """Return a + b."""

    if logger is not None:
        logger.debug("add called")
    return a + b


def chi2_cov(residual, cov, logger=None) -> float:
    """Compute chi2 = r^T * C^{-1} * r for residual vector r and covariance C."""

    residual_vec = np.asarray(residual, dtype=float)
    cov_mat = np.asarray(cov, dtype=float)

    if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
        raise ValueError("cov must be a square matrix")
    if residual_vec.ndim != 1:
        raise ValueError("residual must be a 1D vector")
    if cov_mat.shape[0] != residual_vec.shape[0]:
        raise ValueError("residual length must match covariance dimension")

    solution = linalg.solve(cov_mat, residual_vec, assume_a="sym")
    chi2_value = float(np.dot(residual_vec, solution))

    if logger is not None:
        logger.debug("chi2_cov called", chi2=chi2_value)

    return chi2_value
