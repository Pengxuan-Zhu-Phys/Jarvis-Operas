from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jarvis_operas import OperasRegistry
from jarvis_operas.errors import OperatorCallError
from jarvis_operas.namespaces.stat.decls.core import STAT_CORE_DECLARATIONS


@pytest.fixture
def registry():
    registry = OperasRegistry()
    registry.register(STAT_CORE_DECLARATIONS[0])
    return registry


def named_call(registry, backend, residual, covariance):
    if backend == "numpy":
        return registry.call("stat.chi2_cov", residual=residual, cov=covariance).to_numpy()
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({name: residual[name].to_numpy() for name in residual.columns})
    expression = registry.call(
        "stat.chi2_cov", backend="polars", residual=pl.struct(list(residual.columns)), cov=covariance,
    )
    return frame.select(expression.alias("chi2")).get_column("chi2").to_numpy()


@pytest.mark.parametrize("residual_order", [["x", "y"], ["y", "x"]])
@pytest.mark.parametrize("row_order", [["x", "y"], ["y", "x"]])
@pytest.mark.parametrize("column_order", [["x", "y"], ["y", "x"]])
def test_named_chi2_is_invariant_to_all_axis_orders(registry, residual_order, row_order, column_order):
    residual = pd.DataFrame({"x": [1., 2.], "y": [2., -1.]})[residual_order]
    covariance = pd.DataFrame([[2., .5], [.5, 1.]], index=["x", "y"], columns=["x", "y"])
    covariance = covariance.loc[row_order, column_order]
    # Analytic inverse: chi2 = (x*x - x*y + 2*y*y) / 1.75.
    for backend in ("numpy", "polars"):
        np.testing.assert_allclose(named_call(registry, backend, residual, covariance), [4., 32. / 7.])


@pytest.mark.parametrize("backend", ["numpy", "polars"])
def test_named_chi2_diagonal_reordered_covariance(registry, backend):
    residual = pd.DataFrame({"x": [1.], "y": [2.]})
    covariance = pd.DataFrame([[4., 0.], [0., 1.]], index=["y", "x"], columns=["y", "x"])
    np.testing.assert_allclose(named_call(registry, backend, residual, covariance), [2.])


@pytest.mark.parametrize("backend", ["numpy", "polars"])
@pytest.mark.parametrize("empty", [False, True])
def test_named_chi2_selects_covariance_subset_and_supports_empty_rows(registry, backend, empty):
    residual = pd.DataFrame({"x": [1.], "y": [2.]})
    if empty:
        residual = residual.iloc[:0]
    covariance = pd.DataFrame(np.diag([9., 4., 1.]), index=["z", "y", "x"], columns=["z", "y", "x"])
    np.testing.assert_allclose(named_call(registry, backend, residual, covariance), [] if empty else [2.])


@pytest.mark.parametrize("backend", ["numpy", "polars"])
@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize("empty", [False, True])
def test_named_chi2_rejects_missing_labels_even_for_empty_rows(registry, backend, axis, empty):
    residual = pd.DataFrame({"x": [1.], "y": [2.]})
    if empty:
        residual = residual.iloc[:0]
    covariance = pd.DataFrame(np.eye(2), index=["x", "y"], columns=["x", "y"])
    setattr(covariance, axis, ["x", "z"])
    with pytest.raises((ValueError, OperatorCallError), match="contain residual columns.*y"):
        named_call(registry, backend, residual, covariance)


@pytest.mark.parametrize("backend", ["numpy", "polars"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_named_chi2_rejects_duplicate_covariance_labels(registry, backend, axis):
    residual = pd.DataFrame({"x": [1.], "y": [2.]})
    covariance = pd.DataFrame(np.eye(2), index=["x", "y"], columns=["x", "y"])
    setattr(covariance, axis, ["x", "x"])
    with pytest.raises((ValueError, OperatorCallError), match="cov dataframe.*unique"):
        named_call(registry, backend, residual, covariance)


def test_named_chi2_rejects_duplicate_residual_labels(registry):
    residual = pd.DataFrame([[1., 2.]], columns=["x", "x"])
    covariance = pd.DataFrame(np.eye(2), index=["x", "y"], columns=["x", "y"])
    with pytest.raises(OperatorCallError, match="residual columns must be unique"):
        named_call(registry, "numpy", residual, covariance)


@pytest.mark.parametrize("labelled_covariance", [False, True])
def test_unnamed_residuals_keep_positional_semantics(registry, labelled_covariance):
    pl = pytest.importorskip("polars")
    covariance = [[4., 0.], [0., 1.]]
    if labelled_covariance:
        covariance = pd.DataFrame(covariance, index=["y", "x"], columns=["y", "x"])
    expected = [4.25, 2.]
    numpy_result = registry.call("stat.chi2_cov", residual=[[1., 2.], [2., 1.]], cov=covariance)
    expression = registry.call("stat.chi2_cov", backend="polars", residual=pl.col("r"), cov=covariance)
    polars_result = pl.DataFrame({"r": [[1., 2.], [2., 1.]]}).select(expression).to_series()
    np.testing.assert_allclose(numpy_result, expected)
    np.testing.assert_allclose(polars_result.to_numpy(), expected)


@pytest.mark.parametrize("backend", ["numpy", "polars"])
def test_named_residuals_with_unnamed_covariance_remain_positional(registry, backend):
    residual = pd.DataFrame({"y": [2.], "x": [1.]})
    np.testing.assert_allclose(named_call(registry, backend, residual, [[4., 0.], [0., 1.]]), [2.])
