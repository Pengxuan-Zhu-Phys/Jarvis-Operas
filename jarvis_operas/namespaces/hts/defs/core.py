"""Model-independent optional HiggsTools execution support."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


class HiggsToolsUnavailableError(RuntimeError):
    """Raised internally when an optional runtime resource is unavailable."""


class PredictionContractError(ValueError):
    """Raised internally for a malformed native prediction payload."""


@dataclass(frozen=True)
class DatasetResource:
    path: str
    source: str
    version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "version": self.version, "source": self.source}


@dataclass(frozen=True)
class HiggsToolsBackend:
    bounds: Any
    signals: Any
    input_module: Any


_BACKEND_CACHE: dict[tuple[str, str], HiggsToolsBackend] = {}
_BACKEND_LOCK = RLock()
_DEFAULT_DATASET_PATHS: dict[str, str | None] = {"hb": None, "hs": None}
PREDICTION_ENVELOPE_SCHEMA = "higgstools.prediction-envelope/1.0"
EVALUATION_RESULT_SCHEMA = "higgstools.evaluation-result/1.0"


def configure_default_dataset_paths(
    *, hb_dataset_path: str | None = None, hs_dataset_path: str | None = None
) -> None:
    """Configure process-local fallback paths for the optional evaluator."""

    _DEFAULT_DATASET_PATHS["hb"] = hb_dataset_path
    _DEFAULT_DATASET_PATHS["hs"] = hs_dataset_path


def resolve_dataset_resources(
    *,
    hb_dataset_path: str | None = None,
    hs_dataset_path: str | None = None,
    payload: Mapping[str, Any] | None = None,
    observables: Mapping[str, Any] | None = None,
    dataset_defaults: Mapping[str, Any] | None = None,
) -> dict[str, DatasetResource]:
    """Resolve HB/HS directories by explicit, runtime, environment, then default priority."""

    return {
        "hb_dataset": _resolve_dataset(
            label="HBDataSet",
            explicit=hb_dataset_path,
            payload=payload,
            observables=observables,
            environment="HIGGSTOOLS_HBDATASET",
            default=_default_path("hb", dataset_defaults),
        ),
        "hs_dataset": _resolve_dataset(
            label="HSDataSet",
            explicit=hs_dataset_path,
            payload=payload,
            observables=observables,
            environment="HIGGSTOOLS_HSDATASET",
            default=_default_path("hs", dataset_defaults),
        ),
    }


def get_higgstools_backend(hb_dataset_path: str, hs_dataset_path: str) -> HiggsToolsBackend:
    """Return a process-local backend cached by normalized dataset directories."""

    key = (str(Path(hb_dataset_path).expanduser().resolve()), str(Path(hs_dataset_path).expanduser().resolve()))
    with _BACKEND_LOCK:
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            return cached

        higgs = _import_higgs()
        try:
            bounds = higgs.bounds.Bounds(key[0])
            signals = higgs.signals.Signals(key[1])
            input_module = higgs.tools.Input
        except Exception as exc:
            raise HiggsToolsUnavailableError(
                f"failed to initialize HiggsTools datasets: {exc}"
            ) from exc

        backend = HiggsToolsBackend(bounds=bounds, signals=signals, input_module=input_module)
        _BACKEND_CACHE[key] = backend
        return backend


def clear_higgstools_backend_cache() -> None:
    """Clear the current-process backend cache (principally useful for tests)."""

    with _BACKEND_LOCK:
        _BACKEND_CACHE.clear()


def evaluate_numpy(
    payload: Mapping[str, Any] | None = None,
    *,
    prediction: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    hb_dataset_path: str | None = None,
    hs_dataset_path: str | None = None,
    observables: Mapping[str, Any] | None = None,
    dataset_defaults: Mapping[str, Any] | None = None,
    strict_validation: bool = False,
    logger=None,
    **params: Any,
) -> dict[str, Any]:
    """Evaluate one native HiggsTools prediction without interpreting model-specific fields."""

    resources: dict[str, DatasetResource] | None = None
    try:
        envelope, native_prediction = _resolve_prediction_payload(
            payload=payload, prediction=prediction, metadata=metadata, params=params
        )
        validate_prediction_contract(native_prediction, strict=strict_validation)
    except PredictionContractError as exc:
        return _result(status="invalid_input", errors=[str(exc)])

    try:
        runtime_config = _runtime_config(envelope, observables)
        resources = resolve_dataset_resources(
            hb_dataset_path=hb_dataset_path,
            hs_dataset_path=hs_dataset_path,
            payload=runtime_config,
            observables=observables,
            dataset_defaults=dataset_defaults,
        )
        backend = get_higgstools_backend(
            resources["hb_dataset"].path, resources["hs_dataset"].path
        )
    except PredictionContractError as exc:
        return _result(status="invalid_input", errors=[str(exc)])
    except HiggsToolsUnavailableError as exc:
        return _result(status="unavailable", errors=[str(exc)], resources=resources)

    try:
        predictions = backend.input_module.predictionsFromDict(native_prediction)
        bounds_result = backend.bounds(predictions)
        signals_result = backend.signals(predictions)
        result = _result(
            status="ok",
            resources=resources,
            direct_exclusion=_normalize_bounds(bounds_result),
            signal_measurements=_normalize_signals(signals_result),
        )
        if logger is not None:
            logger.debug("HTs.evaluate completed")
        return result
    except Exception as exc:
        return _result(status="evaluation_error", errors=[f"HiggsTools evaluation failed: {exc}"], resources=resources)


def validate_prediction_contract(prediction: Mapping[str, Any], *, strict: bool = False) -> None:
    """Perform schema-forward validation without assigning model meaning to fields."""

    if not isinstance(prediction, Mapping):
        raise PredictionContractError(
            f"prediction must be a mapping, got {type(prediction).__name__}"
        )
    if not prediction:
        raise PredictionContractError("prediction must not be empty")
    _validate_value(prediction, path="prediction")
    if strict and not any(isinstance(value, (Mapping, Sequence)) and not isinstance(value, str) for value in prediction.values()):
        raise PredictionContractError(
            "prediction must contain at least one structured HiggsTools input field when strict_validation is enabled"
        )


def _resolve_prediction_payload(
    *, payload: Mapping[str, Any] | None, prediction: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None, params: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if payload is not None and not isinstance(payload, Mapping):
        raise PredictionContractError(f"payload must be a mapping, got {type(payload).__name__}")
    if prediction is not None and not isinstance(prediction, Mapping):
        raise PredictionContractError(f"prediction must be a mapping, got {type(prediction).__name__}")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise PredictionContractError(f"metadata must be a mapping, got {type(metadata).__name__}")

    envelope = dict(payload or {})
    if prediction is not None:
        envelope["prediction"] = prediction
    if metadata is not None:
        envelope["metadata"] = metadata
    if "prediction" in envelope:
        native = envelope["prediction"]
    else:
        native = {key: value for key, value in envelope.items() if key not in _RUNTIME_KEYS}
        native.update({key: value for key, value in params.items() if key not in _RUNTIME_KEYS})
    if not isinstance(native, Mapping):
        raise PredictionContractError(f"prediction must be a mapping, got {type(native).__name__}")
    return envelope, native


_RUNTIME_KEYS = frozenset({"metadata", "hb_dataset_path", "hs_dataset_path", "dataset_defaults", "observables"})


def _runtime_config(envelope: Mapping[str, Any], observables: Mapping[str, Any] | None) -> dict[str, Any]:
    config = {key: value for key, value in envelope.items() if key != "prediction"}
    if observables is not None:
        if not isinstance(observables, Mapping):
            raise PredictionContractError(f"observables must be a mapping, got {type(observables).__name__}")
        for key, value in observables.items():
            config.setdefault(str(key), value)
    return config


def _resolve_dataset(*, label: str, explicit: str | None, payload: Mapping[str, Any] | None,
                     observables: Mapping[str, Any] | None, environment: str, default: str | None) -> DatasetResource:
    key = "hb_dataset_path" if label == "HBDataSet" else "hs_dataset_path"
    candidates = (
        (explicit, "explicit"),
        (_mapping_value(payload, key), "payload"),
        (_mapping_value(observables, key), "observables"),
        (os.getenv(environment), "env"),
        (default, "default"),
    )
    for raw_path, source in candidates:
        if raw_path is None or str(raw_path).strip() == "":
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_dir():
            raise HiggsToolsUnavailableError(f"{label} directory from {source} does not exist: {path}")
        return DatasetResource(path=str(path), source=source, version=_dataset_version(path))
    raise HiggsToolsUnavailableError(
        f"{label} directory is not configured; set {key}, {environment}, or a project default"
    )


def _mapping_value(mapping: Mapping[str, Any] | None, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _default_path(kind: str, defaults: Mapping[str, Any] | None) -> str | None:
    if isinstance(defaults, Mapping):
        keys = (f"{kind}_dataset_path", f"{kind}_dataset", kind)
        for key in keys:
            if defaults.get(key) is not None:
                return str(defaults[key])
    return _DEFAULT_DATASET_PATHS[kind]


def _dataset_version(path: Path) -> str | None:
    for filename in ("VERSION", "version", ".version"):
        candidate = path / filename
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            return value or None
    return None


def _import_higgs() -> Any:
    try:
        return importlib.import_module("Higgs")
    except ModuleNotFoundError as exc:
        if exc.name == "Higgs":
            raise HiggsToolsUnavailableError(
                "HiggsTools Python package is unavailable; install the optional HiggsTools dependency"
            ) from exc
        raise HiggsToolsUnavailableError(f"HiggsTools import failed: {exc}") from exc
    except Exception as exc:
        raise HiggsToolsUnavailableError(f"HiggsTools import failed: {exc}") from exc


def _validate_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{path}.{key}"
            _validate_value(child, path=field)
            if "br" in str(key).lower():
                _validate_branching_ratio(child, field)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_value(child, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
        raise PredictionContractError(f"{path} must be finite")


def _validate_branching_ratio(value: Any, path: str) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise PredictionContractError(f"{path} must be within [0, 1]")


def _normalize_bounds(value: Any) -> dict[str, Any]:
    allowed = _value_from(value, "allowed", "isAllowed", default=None)
    selected = _value_from(value, "selectedLimits", "selected_limits", "limits", default=[])
    if selected is None:
        selected = []
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes, bytearray)):
        selected = [selected]
    return {"allowed": None if allowed is None else bool(allowed), "selected_limits": [_normalize_limit(item) for item in selected]}


def _normalize_limit(value: Any) -> dict[str, Any]:
    return {
        "analysis_id": _json_safe(_value_from(value, "analysis_id", "analysisId", "id", "name", default=None)),
        "reference": _json_safe(_value_from(value, "reference", "citation", default=None)),
        "observed_ratio": _json_safe(_value_from(value, "observed_ratio", "obsRatio", "obsratio", default=None)),
        "expected_ratio": _json_safe(_value_from(value, "expected_ratio", "expectedRatio", "expratio", default=None)),
    }


def _normalize_signals(value: Any) -> dict[str, Any]:
    return {
        "chisq": _json_safe(_value_from(value, "chisq", "chi2", "chiSquared", default=None)),
        "loglike": _json_safe(_value_from(value, "loglike", "logLike", "log_likelihood", default=None)),
    }


def _value_from(value: Any, *names: str, default: Any) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            candidate = getattr(value, name)
            if callable(candidate):
                try:
                    return candidate()
                except TypeError:
                    continue
            return candidate
    return default


def _result(*, status: str, errors: list[str] | None = None,
            resources: Mapping[str, DatasetResource] | None = None,
            direct_exclusion: Mapping[str, Any] | None = None,
            signal_measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _json_safe({
        "schema": EVALUATION_RESULT_SCHEMA,
        "status": status,
        "ready": status == "ok",
        "direct_exclusion": dict(direct_exclusion or {"allowed": None, "selected_limits": []}),
        "signal_measurements": dict(signal_measurements or {"chisq": None, "loglike": None}),
        "provenance": {
            "evaluator": "HiggsTools",
            "hb_dataset": resources["hb_dataset"].as_dict() if resources else None,
            "hs_dataset": resources["hs_dataset"].as_dict() if resources else None,
        },
        "warnings": [],
        "errors": errors or [],
    })


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


# Public helper API.  These functions deliberately do not encode any model
# assumptions; they are also registered as HTs.<name> declarations.
def dataset_environment_candidates(kind: str) -> tuple[str, ...]:
    """Return canonical then legacy environment-variable names for a dataset kind."""

    normalized = str(kind).strip().lower()
    if normalized == "hb":
        return ("HIGGSTOOLS_HBDATASET", "HIGGSTOOLS_HB_DATASET")
    if normalized == "hs":
        return ("HIGGSTOOLS_HSDATASET", "HIGGSTOOLS_HS_DATASET")
    raise ValueError("kind must be 'hb' or 'hs'")


def make_diagnostic(
    code: str,
    message: str,
    *,
    field: str | None = None,
    exception: BaseException | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    """Create a JSON-safe, model-independent diagnostic record."""

    result: dict[str, Any] = {
        "code": str(code),
        "message": str(message),
        "field": field,
        "exception_type": type(exception).__name__ if exception is not None else None,
        "retryable": bool(retryable),
    }
    return result


def to_json_safe(value: Any, *, nonfinite_policy: str = "null") -> Any:
    """Convert common evaluator values to JSON-safe data without model interpretation."""

    if nonfinite_policy != "null":
        raise ValueError("only nonfinite_policy='null' is currently supported")
    return _json_safe(value)


def parse_prediction_envelope(
    payload: Mapping[str, Any] | None = None,
    *,
    prediction: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    compatibility_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse direct or enveloped prediction input into the versioned public envelope."""

    envelope, native = _resolve_prediction_payload(
        payload=payload,
        prediction=prediction,
        metadata=metadata,
        params=compatibility_params or {},
    )
    supplied_runtime = runtime if runtime is not None else envelope.get("runtime", {})
    if not isinstance(supplied_runtime, Mapping):
        raise PredictionContractError(
            f"runtime must be a mapping, got {type(supplied_runtime).__name__}"
        )
    return {
        "schema": str(envelope.get("schema") or PREDICTION_ENVELOPE_SCHEMA),
        "prediction": native,
        "metadata": dict(envelope.get("metadata") or {}),
        "runtime": dict(supplied_runtime),
    }


def inspect_dataset_resource(
    kind: str,
    path: str | Path,
    *,
    source: str = "explicit",
    expected_version: str | None = None,
    strict_version: bool = False,
) -> dict[str, Any]:
    """Inspect one existing local dataset directory without network access."""

    label = "HBDataSet" if str(kind).lower() in {"hb", "hbdataset"} else "HSDataSet"
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise HiggsToolsUnavailableError(f"{label} directory from {source} does not exist: {resolved}")
    version = _dataset_version(resolved)
    if strict_version and expected_version is not None and version != expected_version:
        raise PredictionContractError(
            f"{label} version mismatch: expected {expected_version!r}, found {version!r}"
        )
    return {"path": str(resolved), "source": str(source), "version": version, "revision": None, "fingerprint": None}


def build_higgstools_predictions(prediction: Mapping[str, Any], backend: HiggsToolsBackend) -> Any:
    """Build native HiggsTools predictions without altering the caller mapping."""

    return backend.input_module.predictionsFromDict(prediction)


def normalize_selected_limit(value: Any, *, particle: Any | None = None) -> dict[str, Any]:
    """Normalize one selected HiggsBounds limit record."""

    result = _normalize_limit(value)
    result["particle"] = _json_safe(particle)
    result["process"] = _json_safe(_value_from(value, "process", "channel", default=None))
    return result


def normalize_bounds_result(value: Any) -> dict[str, Any]:
    """Normalize a HiggsBounds result from mapping, attribute, or accessor APIs."""

    normalized = _normalize_bounds(value)
    normalized["selected_limits"] = [normalize_selected_limit(item) for item in normalized["selected_limits"]]
    return normalized


def normalize_signals_result(value: Any) -> dict[str, Any]:
    """Normalize a HiggsSignals result from mapping, attribute, or accessor APIs."""

    return _normalize_signals(value)


def evaluate_bounds(predictions: Any, backend: HiggsToolsBackend) -> dict[str, Any]:
    """Run and normalize the optional HiggsBounds component."""

    if backend.bounds is None:
        raise HiggsToolsUnavailableError("HiggsBounds was not initialized in this backend")
    return normalize_bounds_result(backend.bounds(predictions))


def evaluate_signals(predictions: Any, backend: HiggsToolsBackend) -> dict[str, Any]:
    """Run and normalize the optional HiggsSignals component."""

    if backend.signals is None:
        raise HiggsToolsUnavailableError("HiggsSignals was not initialized in this backend")
    return normalize_signals_result(backend.signals(predictions))


def make_evaluation_result(**kwargs: Any) -> dict[str, Any]:
    """Build the stable public result shape used by HTs.evaluate."""

    return _result(
        status=str(kwargs.get("status", "internal_error")),
        errors=list(kwargs.get("errors") or []),
        resources=kwargs.get("resources"),
        direct_exclusion=kwargs.get("direct_exclusion"),
        signal_measurements=kwargs.get("signal_measurements"),
    )


def flatten_evaluation_result(
    result: Mapping[str, Any], *, prefix: str = "higgstools", include_selected_limits: bool = False
) -> dict[str, Any]:
    """Flatten canonical result fields for scalar-observable consumers."""

    direct = result.get("direct_exclusion") if isinstance(result.get("direct_exclusion"), Mapping) else {}
    signals = result.get("signal_measurements") if isinstance(result.get("signal_measurements"), Mapping) else {}
    limits = direct.get("selected_limits") if isinstance(direct, Mapping) else []
    limits = limits if isinstance(limits, Sequence) and not isinstance(limits, (str, bytes)) else []
    flat = {
        f"{prefix}_status": result.get("status"),
        f"{prefix}_ready": result.get("ready"),
        f"{prefix}_hb_allowed": direct.get("allowed") if isinstance(direct, Mapping) else None,
        f"{prefix}_hb_selected_count": len(limits),
        f"{prefix}_hs_chisq": signals.get("chisq") if isinstance(signals, Mapping) else None,
        f"{prefix}_hs_loglike": signals.get("loglike") if isinstance(signals, Mapping) else None,
    }
    if include_selected_limits:
        flat[f"{prefix}_hb_selected_limits"] = _json_safe(limits)
    return _json_safe(flat)


def import_higgstools_package() -> dict[str, Any]:
    """Import the optional package and expose only serializable package metadata."""

    module = _import_higgs()
    return {"version": getattr(module, "__version__", None), "source_path": getattr(module, "__file__", None)}
