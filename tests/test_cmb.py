from __future__ import annotations

import asyncio
import sys

import numpy as np
import pytest
from scipy.stats import chi2

from jarvis_operas import OperatorCallError, OperatorNotFound
from jarvis_operas.api import get_global_operas_registry
from jarvis_operas.namespaces.cmb.defs.core import (
    _load_internal_planck_tt_cl,
    _load_internal_planck_tt_ell,
)


def _manual_loglike(data_cl, model_cl, ell):
    data = np.asarray(data_cl, dtype=float)
    model = np.asarray(model_cl, dtype=float)
    l = np.asarray(ell, dtype=float)
    prefactor = 2.0 * l + 1.0
    x = prefactor * data / model
    return float(-0.5 * (-2.0 * chi2(prefactor).logpdf(x) - 2.0 * np.log(prefactor / model)).sum())


def test_cmb_only_keeps_loglike_and_tt_cosmopower() -> None:
    registry = get_global_operas_registry()
    names = set(registry.list(namespace="cmb"))
    assert names == {"cmb.loglike", "cmb.tt_cosmopower"}


def test_cmb_removed_helpers_are_not_registered() -> None:
    registry = get_global_operas_registry()
    for name in (
        "cmb.observe",
        "cmb.observe_lcdm",
        "cmb.rebin",
        "cmb.noise_planck",
        "cmb.noise_wmap",
    ):
        with pytest.raises(OperatorNotFound):
            registry.get(name)


def test_cmb_loglike_builtin_registered_with_metadata() -> None:
    registry = get_global_operas_registry()
    info = registry.info("cmb.loglike")
    assert info["metadata"]["category"] == "cosmology_likelihood"
    assert "cmb-likelihood" in str(info["metadata"].get("note", ""))
    assert info["metadata"]["supports"]["call"] is True
    assert info["metadata"]["supports"]["acall"] is True
    assert info["metadata"]["supports"]["numpy"] is True
    assert info["metadata"]["supports"]["polars"] is False


def test_cmb_loglike_call_with_direct_kwargs() -> None:
    registry = get_global_operas_registry()

    data_cl = _load_internal_planck_tt_cl()
    model_cl = data_cl * 1.01
    ell = _load_internal_planck_tt_ell()

    result = registry.call("cmb.loglike", model_cl=model_cl)

    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["n_ell"] == 111
    assert result["used_bins"] is False

    expected = _manual_loglike(data_cl, model_cl, ell)
    assert result["loglike"] == pytest.approx(expected)


def test_cmb_loglike_call_with_payload_dict_and_noise() -> None:
    registry = get_global_operas_registry()
    data_cl = _load_internal_planck_tt_cl()
    ell = _load_internal_planck_tt_ell()
    model = data_cl * 0.98
    noise = np.full_like(data_cl, 1.0e-3)

    result = registry.call(
        "cmb.loglike",
        payload={
            "model_cl": model.tolist(),
            "noise_cl": noise.tolist(),
        },
    )

    assert result["ok"] is True
    assert result["used_bins"] is False
    assert result["n_ell"] == 111

    expected = _manual_loglike(
        data_cl + noise,
        model + noise,
        ell,
    )
    assert result["loglike"] == pytest.approx(expected)


def test_cmb_loglike_acall_matches_call() -> None:
    registry = get_global_operas_registry()
    model = (_load_internal_planck_tt_cl() * 1.02).tolist()
    sync_result = registry.call("cmb.loglike", model_cl=model)
    async_result = asyncio.run(registry.acall("cmb.loglike", model_cl=model))
    assert async_result["loglike"] == pytest.approx(sync_result["loglike"])
    assert async_result["n_ell"] == sync_result["n_ell"]


def test_cmb_loglike_missing_required_key_raises_operator_error() -> None:
    registry = get_global_operas_registry()
    with pytest.raises(OperatorCallError) as exc:
        registry.call("cmb.loglike", noise_cl=[1.0, 2.0])
    assert "cmb.loglike" in str(exc.value)
    assert isinstance(exc.value.__cause__, KeyError)


def test_cmb_loglike_rejects_forbidden_keys() -> None:
    registry = get_global_operas_registry()
    for key, value in (
        ("data_cl", [1.0, 2.0]),
        ("ell", [2.0, 3.0]),
        ("bins", [[2, 30]]),
    ):
        with pytest.raises(OperatorCallError) as exc:
            registry.call("cmb.loglike", model_cl=(_load_internal_planck_tt_cl() * 1.01).tolist(), **{key: value})
        assert isinstance(exc.value.__cause__, ValueError)
        assert "must not be provided" in str(exc.value.__cause__)


def test_cmb_loglike_rejects_wrong_model_length() -> None:
    registry = get_global_operas_registry()
    with pytest.raises(OperatorCallError) as exc:
        registry.call("cmb.loglike", model_cl=[1.0, 2.0, 3.0])
    assert isinstance(exc.value.__cause__, ValueError)
    assert "111-point grid" in str(exc.value.__cause__)


def test_cmb_tt_cosmopower_predicts_fixed_111_grid_without_external_dependency(monkeypatch) -> None:
    registry = get_global_operas_registry()
    monkeypatch.delitem(sys.modules, "cosmopower", raising=False)

    first = registry.call(
        "cmb.tt_cosmopower",
        omegabh2=0.022,
        omegach2=0.12,
        thetaMC=1.04,
        tau=0.055,
        ns=0.965,
        As=3.0,
        h=0.67,
        ell_min=3,
        ell_max=50,
    )
    second = registry.call(
        "cmb.tt_cosmopower",
        omegabh2=0.022,
        omegach2=0.12,
        thetaMC=1.04,
        tau=0.055,
        ns=0.965,
        As=3.0,
        h=0.67,
    )

    model_first = np.asarray(first["model_cl"], dtype=float)
    model_second = np.asarray(second["model_cl"], dtype=float)
    assert model_first.shape == (111,)
    assert np.all(model_first > 0.0)
    assert np.allclose(model_first, model_second)
    assert "ell" not in first
    assert "lcdm" not in first

    info = registry.info("cmb.tt_cosmopower")
    assert "internal numpy forward pass" in str(info["metadata"].get("note", ""))


def test_cmb_tt_cosmopower_rejects_user_provided_model_path() -> None:
    registry = get_global_operas_registry()
    with pytest.raises(OperatorCallError) as exc:
        registry.call("cmb.tt_cosmopower", cp_model_path="/tmp/should-not-be-allowed")
    assert isinstance(exc.value.__cause__, ValueError)


def test_cmb_tt_cosmopower_output_can_feed_loglike() -> None:
    registry = get_global_operas_registry()
    model = registry.call(
        "cmb.tt_cosmopower",
        omegabh2=0.022,
        omegach2=0.12,
        tau=0.055,
        ns=0.965,
        As=3.0,
        h=0.67,
    )["model_cl"]
    result = registry.call("cmb.loglike", model_cl=model)
    assert result["n_ell"] == 111
    assert result["ok"] is True
