from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.core import chi2_cov_numpy, chi2_cov_polars_expr


STAT_CORE_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="stat",
        name="chi2_cov",
        arity=2,
        return_dtype=None,
        numpy_impl=chi2_cov_numpy,
        polars_expr_impl=chi2_cov_polars_expr,
        metadata={
            "category": "statistics",
            "summary": "Compute chi-square using residual vector(s) and covariance matrix.",
            "params": {
                "residual": "Residual vector or matrix (samples x observables).",
                "cov": "Covariance matrix.",
            },
            "return": "Chi-square scalar (single sample) or vector (multiple samples).",
            "unit_contract": {
                "version": "1.0",
                "inputs": {},
                "outputs": {
                    "": {"unit": "1", "shape": "scalar_or_array", "nullable": False},
                },
            },
            "note": (
                "The chi-square result is dimensionless. Residual and covariance units depend "
                "on the observables; covariance must carry the products of their units. "
                "Version 1.0 has no fixed input contract or covariance unit inference."
            ),
            "backend_support": {
                "numpy": "native",
                "polars": "native_expr",
            },
            "examples": [
                "jopera call stat.chi2_cov --kwargs '{\"residual\":[1.0,0.0],\"cov\":[[2.0,0.0],[0.0,1.0]]}'",
            ],
            "since": "1.2.0",
            "tags": ["statistics", "chi2", "covariance"],
        },
    ),
)
