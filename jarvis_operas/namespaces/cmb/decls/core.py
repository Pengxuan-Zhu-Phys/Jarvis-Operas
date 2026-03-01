from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.core import (
    loglike_numpy,
    tt_cosmopower_numpy,
)


CMB_CORE_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="cmb",
        name="loglike",
        arity=None,
        return_dtype=None,
        numpy_impl=loglike_numpy,
        polars_expr_impl=None,
        metadata={
            "category": "cosmology_likelihood",
            "summary": "Evaluate CMB TT log-likelihood on fixed internal Planck 111-point grid.",
            "params": {
                "payload": "Optional input mapping; supports model_cl/noise_cl/apply_noise_* keys.",
                "observables": "Optional mapping for pipeline calls; used when payload is omitted.",
                "model_cl": "Required theoretical C_l spectrum on internal Planck 111-point l_center grid.",
                "noise_cl": "Optional noise C_l spectrum on the same 111-point grid.",
                "apply_noise_to_data": "Whether noise is added to data_cl (default true).",
                "apply_noise_to_model": "Whether noise is added to model_cl (default true).",
            },
            "return": "Dictionary with loglike and runtime info (n_ell is fixed to 111; used_bins is always false).",
            "backend_support": {
                "numpy": "native",
                "polars": "unsupported",
            },
            "examples": [
                "python -c \"from jarvis_operas.api import get_global_operas_registry as g; r=g(); m=r.call('cmb.tt_cosmopower',omegabh2=0.022,omegach2=0.12,tau=0.055,ns=0.965,As=3.0,h=0.67)['model_cl']; print(r.call('cmb.loglike',model_cl=m))\"",
            ],
            "since": "1.3.x",
            "tags": ["cmb", "likelihood", "cosmology"],
            "note": "Source: https://github.com/htjb/cmb-likelihood (cmblike/cmb.py + data/TT_power_spec.txt), adapted with JO internal Planck TT dataset on fixed 111-point l_center grid. Constraints: model_cl/noise_cl must match length 111; do not pass ell/l/data/data_cl/bins.",
        },
    ),
    OperaFunction(
        namespace="cmb",
        name="tt_cosmopower",
        arity=None,
        return_dtype=None,
        numpy_impl=tt_cosmopower_numpy,
        polars_expr_impl=None,
        metadata={
            "category": "cosmology_model",
            "summary": "Predict TT C_l from LambdaCDM on fixed internal Planck 111-point grid.",
            "params": {
                "omegabh2": "Physical baryon density parameter.",
                "omegach2": "Physical cold dark matter density parameter.",
                "thetaMC": "Accepted for compatibility; ignored by CosmoPower backend.",
                "tau": "Optical depth.",
                "ns": "Scalar spectral index.",
                "As": "ln(10^10 A_s) parameterization used by upstream.",
                "h": "Reduced Hubble parameter.",
            },
            "return": "Dictionary with model_cl only on internal Planck 111-point l_center grid.",
            "backend_support": {
                "numpy": "native",
                "polars": "unsupported",
            },
            "examples": [
                "jopera call cmb.tt_cosmopower --kwargs '{\"omegabh2\":0.022,\"omegach2\":0.12,\"tau\":0.055,\"ns\":0.965,\"As\":3.0,\"h\":0.67}'",
            ],
            "since": "1.3.x",
            "tags": ["cmb", "model", "cosmopower", "lcdm"],
            "note": "Source: https://github.com/htjb/cmb-likelihood (cmblike/cmb.py:get_cosmopower_model), adapted with JO-bundled model path, internal numpy forward pass, and Planck 111-point grid projection. Compatibility: ell_min/ell_max are accepted but ignored.",
        },
    ),
)
