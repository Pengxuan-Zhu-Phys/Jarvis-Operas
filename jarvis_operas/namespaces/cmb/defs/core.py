from __future__ import annotations

import importlib.resources as ir
import pickle
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2


def loglike_numpy(
    payload: Mapping[str, Any] | None = None,
    observables: Mapping[str, Any] | None = None,
    logger=None,
    **params: Any,
) -> dict[str, Any]:
    data = _resolve_payload(payload=payload, observables=observables, params=params)

    for forbidden_key in ("data_cl", "data", "ell", "l", "bins"):
        if forbidden_key in data:
            raise ValueError(
                f"'{forbidden_key}' is managed internally by cmb.loglike and must not be provided"
            )

    ell = _load_internal_planck_tt_ell()
    data_cl = _load_internal_planck_tt_cl()
    model_cl = _as_1d_float_array("model_cl", _pick_required(data, "model_cl", "model", "cl_model"))
    if model_cl.shape[0] != data_cl.shape[0]:
        raise ValueError(
            f"model_cl must have length {int(data_cl.shape[0])} to match internal Planck 111-point grid"
        )

    noise_raw = _pick_optional(data, "noise_cl", "noise")
    apply_noise_to_data = bool(data.get("apply_noise_to_data", True))
    apply_noise_to_model = bool(data.get("apply_noise_to_model", True))

    _ensure_same_length("data_cl", data_cl, "model_cl", model_cl)
    _ensure_same_length("ell", ell, "model_cl", model_cl)
    data_eval = data_cl
    model_eval = model_cl
    ell_eval = ell

    noise_eval = _as_1d_float_array("noise_cl", noise_raw) if noise_raw is not None else None
    if noise_eval is not None:
        _ensure_same_length("noise_cl", noise_eval, "model_cl", model_eval)

    if noise_eval is not None:
        if apply_noise_to_data:
            data_eval = data_eval + noise_eval
        if apply_noise_to_model:
            model_eval = model_eval + noise_eval

    if np.any(model_eval <= 0.0):
        raise ValueError("model_cl must be strictly positive after optional noise addition")

    prefactor = 2.0 * ell_eval + 1.0
    if np.any(prefactor <= 0.0):
        raise ValueError("ell values must satisfy 2*ell+1 > 0")

    x = prefactor * data_eval / model_eval
    loglike = -0.5 * (-2.0 * chi2(prefactor).logpdf(x) - 2.0 * np.log(prefactor / model_eval)).sum()
    if not np.isfinite(loglike):
        raise ValueError("computed loglike is not finite; check inputs")

    if logger is not None:
        logger.debug("cmb.loglike called: n_ell={}", int(ell_eval.size))

    return {
        "loglike": float(loglike),
        "ok": True,
        "n_ell": int(ell_eval.size),
        "used_bins": False,
    }


def tt_cosmopower_numpy(
    payload: Mapping[str, Any] | None = None,
    observables: Mapping[str, Any] | None = None,
    logger=None,
    **params: Any,
) -> dict[str, Any]:
    data = _resolve_payload(payload=payload, observables=observables, params=params)

    for forbidden_key in ("cp_model_path", "path_to_cp", "cp_root"):
        if forbidden_key in data:
            raise ValueError(
                f"'{forbidden_key}' is no longer accepted by cmb.tt_cosmopower; "
                "JO uses an internal bundled model path."
            )

    omegabh2 = float(data.get("omegabh2", 0.022))
    omegach2 = float(data.get("omegach2", 0.12))
    _ = float(data.get("thetaMC", 1.04))  # Kept for compatibility; Cosmopower model ignores it.
    tau = float(data.get("tau", 0.055))
    ns = float(data.get("ns", 0.965))
    ln10_10_as = float(data.get("As", data.get("ln10_10_As", 3.0)))
    h = float(data.get("h", 0.67))

    if logger is not None and ("ell_min" in data or "ell_max" in data):
        logger.debug(
            "cmb.tt_cosmopower ignores ell_min/ell_max and always returns the fixed Planck 111-point grid"
        )

    model_cl, ell = _build_cosmopower_tt_model(
        omegabh2=omegabh2,
        omegach2=omegach2,
        tau=tau,
        ns=ns,
        ln10_10_as=ln10_10_as,
        h=h,
    )

    if logger is not None:
        logger.debug("cmb.tt_cosmopower called: n_ell={}", int(ell.size))

    return {
        "model_cl": model_cl,
    }


def _resolve_payload(
    *,
    payload: Mapping[str, Any] | None,
    observables: Mapping[str, Any] | None,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    if payload is not None:
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
        resolved.update({str(k): v for k, v in payload.items()})

    if observables is not None:
        if not isinstance(observables, Mapping):
            raise TypeError(f"observables must be a mapping, got {type(observables).__name__}")
        for key, value in observables.items():
            resolved.setdefault(str(key), value)

    for key, value in params.items():
        resolved.setdefault(str(key), value)

    return resolved


def _pick_required(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"missing required key, expected one of: {keys}")


def _pick_optional(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_1d_float_array(field_name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D array-like")
    if arr.size == 0:
        raise ValueError(f"{field_name} cannot be empty")
    return arr


def _build_cosmopower_tt_model(
    *,
    omegabh2: float,
    omegach2: float,
    tau: float,
    ns: float,
    ln10_10_as: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ = _resolve_internal_cosmopower_tt_model_path()

    params = {
        "omega_b": float(omegabh2),
        "omega_cdm": float(omegach2),
        "h": float(h),
        "tau_reio": float(tau),
        "n_s": float(ns),
        "ln10^{10}A_s": float(ln10_10_as),
    }
    tt = _cosmopower_tt_forward_np(params)
    if tt.size == 0:
        raise ValueError("internal CosmoPower TT prediction output is empty")

    tt = tt * 1.0e12 * 2.725**2
    ell_full = np.arange(2.0, tt.shape[0] + 2.0, dtype=float)
    ell = _load_internal_planck_tt_ell()
    if float(ell[0]) < float(ell_full[0]) or float(ell[-1]) > float(ell_full[-1]):
        raise ValueError("internal Planck ell grid is outside CosmoPower output domain")

    model = np.interp(ell, ell_full, tt)
    if np.any(model <= 0.0):
        raise ValueError("CosmoPower TT spectrum produced non-positive C_l in requested ell range")
    return model, ell


def _resolve_internal_planck_tt_data_path() -> Path:
    data_rel = "models/cmb/planck/data/TT_power_spec.txt"
    try:
        data_file = ir.files("jarvis_operas").joinpath(data_rel)
        data_path = Path(data_file)
    except Exception as exc:
        raise RuntimeError("failed to resolve internal Planck TT data resource path") from exc
    if not data_path.is_file():
        raise RuntimeError(
            "JO internal Planck TT data is missing at "
            f"'{data_path}'. Expected bundled file 'TT_power_spec.txt'."
        )
    return data_path


@lru_cache(maxsize=1)
def _load_internal_planck_tt_points() -> tuple[np.ndarray, np.ndarray]:
    data_path = _resolve_internal_planck_tt_data_path()
    ell_values: list[float] = []
    dl_values: list[float] = []

    for raw_line in data_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) < 5:
            continue
        if parts[0] not in {"Planck binned", "Planck unbinned"}:
            continue
        ell_values.append(float(parts[2]))
        dl_values.append(float(parts[4]))

    if not ell_values:
        raise ValueError("internal Planck TT dataset does not contain usable Planck rows")

    grouped: dict[float, list[float]] = defaultdict(list)
    for ell, dl in zip(ell_values, dl_values):
        grouped[float(ell)].append(float(dl))

    ell_sorted = np.array(sorted(grouped.keys()), dtype=float)
    dl_sorted = np.array([float(np.mean(grouped[ell])) for ell in ell_sorted], dtype=float)
    cl_sorted = dl_sorted * (2.0 * np.pi) / (ell_sorted * (ell_sorted + 1.0))
    if np.any(cl_sorted <= 0.0):
        raise ValueError("internal Planck TT data produced non-positive C_l values")
    return ell_sorted, cl_sorted


def _load_internal_planck_tt_ell() -> np.ndarray:
    ell, _ = _load_internal_planck_tt_points()
    return ell.copy()


def _load_internal_planck_tt_cl() -> np.ndarray:
    _, cl = _load_internal_planck_tt_points()
    return cl.copy()


def _resolve_internal_cosmopower_tt_model_path() -> str:
    model_rel = "models/cosmopower/CP_paper/CMB/cmb_TT_NN.pkl"
    try:
        model_file = ir.files("jarvis_operas").joinpath(model_rel)
        model_path = Path(model_file)
    except Exception as exc:
        raise RuntimeError("failed to resolve internal CosmoPower model resource path") from exc

    if not model_path.is_file():
        raise RuntimeError(
            "JO internal CosmoPower model is missing at "
            f"'{model_path}'. Expected bundled file 'cmb_TT_NN.pkl'."
        )
    return str(model_path.with_suffix(""))


class _ListWrapperCompatUnpickler(pickle.Unpickler):
    """Compat loader for cosmopower pickle without tensorflow runtime."""

    def find_class(self, module: str, name: str):
        if module == "tensorflow.python.training.tracking.data_structures" and name == "ListWrapper":
            return list
        return super().find_class(module, name)


@lru_cache(maxsize=1)
def _load_internal_cosmopower_tt_weights() -> dict[str, Any]:
    model_no_suffix = _resolve_internal_cosmopower_tt_model_path()
    model_path = Path(f"{model_no_suffix}.pkl")
    with model_path.open("rb") as handle:
        payload = _ListWrapperCompatUnpickler(handle).load()
    if not isinstance(payload, list) or len(payload) != 15:
        raise ValueError("invalid internal CosmoPower model payload format")

    (
        W_list,
        b_list,
        alpha_list,
        beta_list,
        parameters_mean,
        parameters_std,
        features_mean,
        features_std,
        _n_parameters,
        parameters,
        _n_modes,
        _modes,
        _n_hidden,
        n_layers,
        _architecture,
    ) = payload

    return {
        "W": [np.asarray(item, dtype=float) for item in W_list],
        "b": [np.asarray(item, dtype=float) for item in b_list],
        "alpha": [np.asarray(item, dtype=float) for item in alpha_list],
        "beta": [np.asarray(item, dtype=float) for item in beta_list],
        "parameters_mean": np.asarray(parameters_mean, dtype=float),
        "parameters_std": np.asarray(parameters_std, dtype=float),
        "features_mean": np.asarray(features_mean, dtype=float),
        "features_std": np.asarray(features_std, dtype=float),
        "parameters": [str(item) for item in parameters],
        "n_layers": int(n_layers),
    }


def _cosmopower_tt_forward_np(parameters: Mapping[str, float]) -> np.ndarray:
    model = _load_internal_cosmopower_tt_weights()
    ordered = np.array([parameters[name] for name in model["parameters"]], dtype=float)[None, :]
    layers = (ordered - model["parameters_mean"][None, :]) / model["parameters_std"][None, :]

    for index in range(model["n_layers"] - 1):
        linear = np.dot(layers, model["W"][index]) + model["b"][index]
        sigma = 1.0 / (1.0 + np.exp(-model["alpha"][index] * linear))
        layers = (model["beta"][index] + (1.0 - model["beta"][index]) * sigma) * linear

    linear_out = np.dot(layers, model["W"][-1]) + model["b"][-1]
    predictions = linear_out * model["features_std"][None, :] + model["features_mean"][None, :]
    return np.power(10.0, predictions[0])


def _ensure_same_length(name_a: str, array_a: np.ndarray, name_b: str, array_b: np.ndarray) -> None:
    if array_a.shape[0] != array_b.shape[0]:
        raise ValueError(f"{name_a} and {name_b} must have the same length")
