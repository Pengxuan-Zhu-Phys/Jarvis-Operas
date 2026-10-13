"""Versioned, dependency-free-from-host unit contracts for numerical operators.

Only explicitly registered symbols are convertible. Angles and dimensionless
numbers are distinct; natural-unit physical conversions are never inferred.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import math
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import OperatorCallError

UNIT_CONTRACT_VERSION = "1.0"


class UnitContractError(ValueError):
    """A unit failure with a stable, machine-readable ``code``."""

    def __init__(
        self, code: str, message: str, *, field: str | None = None,
        source_unit: str | None = None, target_unit: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.source_unit = source_unit
        self.target_unit = target_unit

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    dimension: str
    scale: float


def _catalog() -> dict[str, UnitDefinition]:
    result: dict[str, UnitDefinition] = {}
    families = {
        "energy": {"eV": 1e-9, "keV": 1e-6, "MeV": 1e-3, "GeV": 1., "TeV": 1e3},
        "cross_section": {"b": 1e12, "mb": 1e9, "ub": 1e6, "nb": 1e3, "pb": 1., "fb": 1e-3,
                          "m^2": 1e40, "cm^2": 1e36, "mm^2": 1e34},
        "length": {"m": 1e3, "cm": 10., "mm": 1., "um": 1e-3, "fm": 1e-12},
        "angle": {"rad": 1., "deg": math.pi / 180.},
        "dimensionless": {"1": 1.},
        "energy_length": {"MeV·fm": 1.},
        "speed": {"m/s": 1.},
        "temperature_squared": {"K^2": 1., "mK^2": 1e-6, "uK^2": 1e-12},
    }
    for dimension, symbols in families.items():
        for symbol, scale in symbols.items():
            result[symbol] = UnitDefinition(symbol, dimension, scale)
    for symbol in ("eV", "keV", "MeV", "GeV", "TeV"):
        for power in (2, -2):
            name = f"{symbol}^{power}"
            result[name] = UnitDefinition(name, f"energy^{power}", result[symbol].scale ** power)
    return result


SUPPORTED_UNITS: Mapping[str, UnitDefinition] = MappingProxyType(_catalog())
_ALIASES = MappingProxyType({"MeV*fm": "MeV·fm", "µK^2": "uK^2", "μK^2": "uK^2"})
DIAGNOSTIC_CODES = (
    "unknown_unit", "unsupported_unit", "dimension_mismatch", "missing_unit",
    "source_conflict", "type_error", "shape_mismatch", "non_finite",
    "contract_missing", "invalid_contract", "output_path_missing", "unsupported_backend",
)


def unit_catalog() -> dict[str, dict[str, Any]]:
    """Return an independent JSON-compatible copy of the unit directory."""
    return {
        symbol: {"dimension": definition.dimension, "scale": definition.scale}
        for symbol, definition in SUPPORTED_UNITS.items()
    }


def normalize_unit(unit: Any) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise UnitContractError("unknown_unit", f"Unknown unit symbol {unit!r}.")
    symbol = _ALIASES.get(unit.strip(), unit.strip())
    if symbol not in SUPPORTED_UNITS:
        code = "unsupported_unit" if re.search(r"[*/·^]", symbol) else "unknown_unit"
        raise UnitContractError(code, f"Unit symbol {symbol!r} is not registered.")
    return symbol


def conversion_factor(source_unit: str, target_unit: str) -> float:
    """Compute a same-dimension scale; no physical conversion is implicit."""
    source, target = normalize_unit(source_unit), normalize_unit(target_unit)
    a, b = SUPPORTED_UNITS[source], SUPPORTED_UNITS[target]
    if a.dimension != b.dimension:
        raise UnitContractError(
            "dimension_mismatch", f"Cannot convert {source!r} to {target!r}.",
            source_unit=source, target_unit=target,
        )
    return a.scale / b.scale


def _shape(shape: Any, field: str) -> str | list[int]:
    if isinstance(shape, str) and shape in {"scalar", "array", "scalar_or_array"}:
        return shape
    if isinstance(shape, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape
    ):
        return list(shape)
    raise UnitContractError("invalid_contract", f"Invalid shape for {field!r}.", field=field)


def normalize_output_path(path: Any) -> str:
    """Normalize an output key to a JSON Pointer; the empty pointer is root."""
    if not isinstance(path, str):
        raise UnitContractError("invalid_contract", "Output paths must be strings.")
    if path == "":
        return ""
    if not path.startswith("/"):
        return "/" + path.replace("~", "~0").replace("/", "~1")
    if re.search(r"~(?![01])", path):
        raise UnitContractError("invalid_contract", f"Invalid JSON Pointer {path!r}.")
    return path


def validate_unit_contract(
    contract: Any, callable_impl: Any, *, documented_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and copy a v1 contract; never modify declaration metadata.

    Required: ``version``. Optional maps: ``inputs`` keyed by signature names
    and ``outputs`` keyed by JSON Pointer (plain keys are accepted). Each spec
    requires ``unit``, defaults ``shape`` to scalar_or_array and ``nullable``
    to False. Unknown fields are rejected. Exact shapes use integer extents.
    Logical keyword inputs are accepted for **kwargs implementations; when
    supplied, documented_parameters limits them to the documented names.
    """
    if contract is None:
        return None
    if not isinstance(contract, Mapping) or set(contract) - {"version", "inputs", "outputs"}:
        raise UnitContractError("invalid_contract", "Unit contract must be a v1 mapping.")
    if contract.get("version") != UNIT_CONTRACT_VERSION:
        raise UnitContractError("invalid_contract", "Unsupported unit contract version.")
    normalized: dict[str, Any] = {"version": UNIT_CONTRACT_VERSION, "inputs": {}, "outputs": {}}
    try:
        signature = inspect.signature(callable_impl)
    except (TypeError, ValueError):
        signature = None
    for section in ("inputs", "outputs"):
        specs = contract.get(section, {})
        if not isinstance(specs, Mapping):
            raise UnitContractError("invalid_contract", f"Contract {section} must be a mapping.")
        for raw_field, spec in specs.items():
            if not isinstance(raw_field, str) or (section == "inputs" and not raw_field):
                raise UnitContractError("invalid_contract", "Contract fields must be strings.")
            field = raw_field if section == "inputs" else normalize_output_path(raw_field)
            if field in normalized[section]:
                raise UnitContractError("invalid_contract", f"Duplicate output path {field!r}.")
            if section == "inputs":
                has_keywords = signature is not None and any(
                    p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
                )
                logical_keyword = has_keywords and (
                    documented_parameters is None or field in documented_parameters
                )
                if signature is None or (field not in signature.parameters and not logical_keyword):
                    raise UnitContractError("invalid_contract", f"Unknown parameter {field!r}.", field=field)
                parameter = signature.parameters.get(field)
                if parameter is not None and parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                    raise UnitContractError("invalid_contract", f"Variadic parameter {field!r} is unsupported.", field=field)
            if not isinstance(spec, Mapping) or set(spec) - {"unit", "shape", "nullable"} or "unit" not in spec:
                raise UnitContractError("invalid_contract", f"Invalid specification for {field!r}.", field=field)
            nullable = spec.get("nullable", False)
            if not isinstance(nullable, bool):
                raise UnitContractError("invalid_contract", f"nullable must be bool for {field!r}.", field=field)
            normalized[section][field] = {
                "unit": normalize_unit(spec["unit"]),
                "shape": _shape(spec.get("shape", "scalar_or_array"), field),
                "nullable": nullable,
            }
    return normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _has_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_bool(item) for item in value)
    return False


def _numeric(value: Any, shape: Any, nullable: bool, field: str) -> None:
    if value is None:
        if nullable:
            return
        raise UnitContractError("type_error", f"None is not allowed for {field!r}.", field=field)
    if _has_bool(value):
        raise UnitContractError("type_error", f"bool is not a numerical quantity for {field!r}.", field=field)
    scalar = isinstance(value, numbers.Real)
    if not scalar and not (
        isinstance(value, (np.ndarray, list, tuple))
        or type(value).__module__ == "pandas"
        or type(value).__module__.startswith("pandas.")
    ):
        raise UnitContractError("type_error", f"Expected a real scalar or numeric array for {field!r}.", field=field)
    try:
        array = np.asarray(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise UnitContractError("type_error", f"Invalid numeric array for {field!r}.", field=field) from exc
    if array.dtype.kind not in "iuf":
        raise UnitContractError("type_error", f"Expected finite real numbers for {field!r}.", field=field)
    if not np.all(np.isfinite(array)):
        raise UnitContractError("non_finite", f"Non-finite number in {field!r}.", field=field)
    actual_shape = array.shape
    if shape == "scalar" and actual_shape != ():
        raise UnitContractError("shape_mismatch", f"Expected scalar for {field!r}.", field=field)
    if shape == "array" and actual_shape == ():
        raise UnitContractError("shape_mismatch", f"Expected array for {field!r}.", field=field)
    if isinstance(shape, (list, tuple)) and tuple(shape) != actual_shape:
        raise UnitContractError("shape_mismatch", f"Expected shape {tuple(shape)} for {field!r}; got {actual_shape}.", field=field)


def _scale(value: Any, factor: float, *, copy_value: bool) -> Any:
    if value is None:
        return None
    if factor == 1.:
        return copy.deepcopy(value) if copy_value else value
    if isinstance(value, list):
        return [_scale(item, factor, copy_value=True) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale(item, factor, copy_value=True) for item in value)
    return value * factor


def convert_value(
    value: Any, source_unit: str, target_unit: str, *,
    shape: Any = "scalar_or_array", nullable: bool = False,
) -> Any:
    """Convert real values without mutation or broadcasting, preserving containers."""
    shape = _shape(shape, "value")
    _numeric(value, shape, nullable, "value")
    converted = _scale(value, conversion_factor(source_unit, target_unit), copy_value=True)
    _numeric(converted, shape, nullable, "value")
    return converted


def _unit_map(units: Any, *, outputs: bool = False) -> dict[str, str]:
    if units is None:
        return {}
    if not isinstance(units, Mapping):
        raise UnitContractError("invalid_contract", "Unit declarations must be mappings.")
    result: dict[str, str] = {}
    for key, unit in units.items():
        if not isinstance(key, str):
            raise UnitContractError("invalid_contract", "Unit declaration keys must be strings.")
        name = normalize_output_path(key) if outputs else key
        if isinstance(unit, Mapping):
            unit = unit.get("unit")
        if name in result:
            raise UnitContractError("invalid_contract", f"Duplicate unit declaration {name!r}.")
        result[name] = normalize_unit(unit)
    return result


def _audit(field: str, source: str, target: str, factor: float, origin: str, kind: str) -> dict[str, Any]:
    return {
        "field": field, "kind": kind, "source_unit": source, "target_unit": target,
        "factor": factor, "origin": origin, "contract_version": UNIT_CONTRACT_VERSION,
    }


def _get_path(result: Any, path: str) -> Any:
    value = result
    if path:
        for encoded in path[1:].split("/"):
            key = encoded.replace("~1", "/").replace("~0", "~")
            try:
                if isinstance(value, Mapping):
                    value = value[key]
                elif isinstance(value, (list, tuple, np.ndarray)) and key.isdigit() and (key == "0" or not key.startswith("0")):
                    value = value[int(key)]
                else:
                    raise KeyError(key)
            except (KeyError, IndexError, TypeError) as exc:
                raise UnitContractError("output_path_missing", f"Output path {path!r} is absent.", field=path) from exc
    return value


@dataclass(frozen=True)
class UnitProjection:
    """Independent observable values, units and projection audit."""
    values: dict[str, Any]
    units: dict[str, str | None]
    audit: list[dict[str, Any]]


@dataclass(frozen=True)
class UnitCallResult:
    """The complete canonical result and separately projected observables."""
    canonical_result: Any
    values: dict[str, Any]
    units: dict[str, str | None]
    audit: list[dict[str, Any]]
    normalized_inputs: dict[str, Any]


@dataclass(frozen=True)
class UnitCallPlan:
    """Immutable prepared binding; all per-call data stays local.

    ``normalize`` returns ``(args, kwargs, audit)``. ``bind_inputs`` builds an
    independent input snapshot. ``project`` is usable for a host-managed call;
    ``finalize`` joins its audit and input snapshot into ``UnitCallResult``.
    ``run`` and ``arun`` execute the declaration's NumPy implementation.
    """
    declaration: Any
    signature: inspect.Signature
    contract: Mapping[str, Any] | None
    input_units: Mapping[str, str]
    output_units: Mapping[str, str]
    source_units: Mapping[str, str]
    output_bindings: Mapping[str, str]
    strict: bool
    input_factors: Mapping[str, float]
    output_factors: Mapping[str, float]

    @property
    def __signature__(self) -> inspect.Signature:
        return self.signature

    def __call__(self, *args: Any, **kwargs: Any) -> UnitCallResult:
        return self.run(*args, **kwargs)

    def bind_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        for name, parameter in self.signature.parameters.items():
            if parameter.kind == parameter.VAR_KEYWORD:
                inputs.update(inputs.pop(name, {}))
        return copy.deepcopy(inputs)

    def normalize(
        self, *args: Any, source_units: Mapping[str, Any] | None = None, **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any], list[dict[str, Any]]]:
        return self.normalize_inputs(args, kwargs, source_units=source_units)

    def normalize_inputs(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any], *,
        source_units: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any], list[dict[str, Any]]]:
        """Normalize an explicit argument container without reserving kwargs.

        Hosts and legacy wrappers can use this path when the physical function
        has its own parameter named ``source_units``.
        """
        original = self.signature.bind(*args, **kwargs)
        bound = self.signature.bind(*args, **kwargs)
        keyword_name = next((name for name, parameter in self.signature.parameters.items()
                             if parameter.kind == parameter.VAR_KEYWORD), None)
        supplied = set(bound.arguments)
        if keyword_name:
            supplied.update(bound.arguments.get(keyword_name, {}))
        bound.apply_defaults()
        runtime_sources = _unit_map(source_units)
        specs = self.contract["inputs"] if self.contract else {}
        for field in runtime_sources:
            if field not in specs:
                raise UnitContractError("contract_missing", f"No input unit contract for {field!r}.", field=field)
            if field in self.source_units and runtime_sources[field] != self.source_units[field]:
                raise UnitContractError("source_conflict", f"Source units disagree for {field!r}.", field=field)
        sources = dict(self.source_units)
        sources.update(runtime_sources)
        audit: list[dict[str, Any]] = []
        for field, spec in specs.items():
            logical_keyword = field not in self.signature.parameters
            arguments = bound.arguments.get(keyword_name, {}) if logical_keyword else bound.arguments
            explicit = self.input_units.get(field)
            actual = sources.get(field)
            if explicit is not None and actual is not None and explicit != actual:
                raise UnitContractError("source_conflict", f"Declared and recorded source units disagree for {field!r}.", field=field)
            if logical_keyword and (explicit is not None or actual is not None or self.strict):
                for container_name, parameter in self.signature.parameters.items():
                    container = bound.arguments.get(container_name)
                    if (
                        parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
                        and container_name not in specs and isinstance(container, Mapping)
                    ):
                        raise UnitContractError(
                            "contract_missing", f"Opaque mapping input {container_name!r} may shadow contracted input {field!r}; pass direct keyword inputs without a mapping or use a typed numeric wrapper.",
                            field=field,
                        )
            if field not in arguments:
                if field in self.input_units or field in sources:
                    raise UnitContractError("missing_unit", f"Bound input {field!r} was not supplied directly.", field=field)
                continue
            if field not in supplied:
                source, origin = spec["unit"], "default_canonical"
            elif actual is not None:
                source, origin = actual, "source_record_asserted" if explicit else "source_record"
            elif explicit is not None:
                source, origin = explicit, "declared"
            else:
                if self.strict and arguments[field] is not None:
                    raise UnitContractError("missing_unit", f"A confirmed source unit is required for {field!r}.", field=field)
                source, origin = spec["unit"], "assumed_canonical"
            if origin in {"default_canonical", "assumed_canonical"}:
                factor = 1.
            elif field in self.input_factors:
                factor = self.input_factors[field]
            else:
                factor = conversion_factor(source, spec["unit"])
            value = arguments[field]
            _numeric(value, spec["shape"], spec["nullable"], field)
            converted = _scale(value, factor, copy_value=True)
            _numeric(converted, spec["shape"], spec["nullable"], field)
            if field in supplied:
                if logical_keyword:
                    original.arguments[keyword_name][field] = converted
                else:
                    original.arguments[field] = converted
            audit.append(_audit(field, source, spec["unit"], factor, origin, "input"))
        # BoundArguments.args promotes keyword calls to positional calls. Keep
        # the caller's actual layout so legacy arity checks remain unchanged.
        normalized_args = list(args)
        position = 0
        for name, parameter in self.signature.parameters.items():
            if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}:
                if position < len(args):
                    normalized_args[position] = original.arguments[name]
                    position += 1
            elif parameter.kind == parameter.VAR_POSITIONAL:
                position = len(args)
        normalized_kwargs = dict(kwargs)
        for name in kwargs:
            parameter = self.signature.parameters.get(name)
            if parameter is not None and parameter.kind in {
                parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY,
            }:
                normalized_kwargs[name] = original.arguments[name]
            elif keyword_name and name in original.arguments.get(keyword_name, {}):
                normalized_kwargs[name] = original.arguments[keyword_name][name]
        return tuple(normalized_args), normalized_kwargs, audit

    def project(self, result: Any) -> UnitProjection:
        specs = self.contract["outputs"] if self.contract else {}
        values: dict[str, Any] = {}
        units: dict[str, str | None] = {}
        audit: list[dict[str, Any]] = []
        for name, path in self.output_bindings.items():
            value = _get_path(result, path)
            spec = specs.get(path)
            if spec is None:
                values[name], units[name] = copy.deepcopy(value), None
                continue
            source = spec["unit"]
            if path:
                parent_path, leaf = path.rsplit("/", 1)
                parent = _get_path(result, parent_path)
                if leaf == "value" and isinstance(parent, Mapping) and "unit" in parent:
                    recorded = normalize_unit(parent["unit"])
                    if recorded != source:
                        raise UnitContractError("source_conflict", f"Output record unit disagrees with contract at {path!r}.", field=path)
            target = self.output_units.get(name, source)
            factor = self.output_factors.get(name, 1.)
            _numeric(value, spec["shape"], spec["nullable"], path)
            projected = _scale(value, factor, copy_value=True)
            _numeric(projected, spec["shape"], spec["nullable"], path)
            values[name], units[name] = projected, target
            audit.append(_audit(path, source, target, factor, "contract", "output"))
        return UnitProjection(values, units, audit)

    def finalize(
        self, result: Any, input_audit: list[dict[str, Any]], normalized_inputs: Mapping[str, Any],
    ) -> UnitCallResult:
        projection = self.project(result)
        return UnitCallResult(
            result, projection.values, projection.units,
            copy.deepcopy(input_audit) + projection.audit, copy.deepcopy(dict(normalized_inputs)),
        )

    def _dispatch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return self.declaration.dispatch(*args, backend="numpy", **kwargs)
        except (UnitContractError, OperatorCallError):
            raise
        except Exception as exc:
            name = self.declaration.full_name
            raise OperatorCallError(name, f"Operator '{name}' failed on backend='numpy': {exc}") from exc

    def run(self, *args: Any, **kwargs: Any) -> UnitCallResult:
        return self.run_inputs(args, kwargs)

    def run_inputs(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any], *,
        source_units: Mapping[str, Any] | None = None,
    ) -> UnitCallResult:
        """Execute argument containers with separate trusted unit records."""
        normalized_args, normalized_kwargs, audit = self.normalize_inputs(args, kwargs, source_units=source_units)
        inputs = self.bind_inputs(normalized_args, normalized_kwargs)
        result = self._dispatch(normalized_args, normalized_kwargs)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise OperatorCallError(self.declaration.full_name, "Sync call returned an awaitable; use arun().")
        return self.finalize(result, audit, inputs)

    async def arun(self, *args: Any, **kwargs: Any) -> UnitCallResult:
        return await self.arun_inputs(args, kwargs)

    async def arun_inputs(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any], *,
        source_units: Mapping[str, Any] | None = None,
    ) -> UnitCallResult:
        """Asynchronously execute containers with separate source records."""
        normalized_args, normalized_kwargs, audit = self.normalize_inputs(args, kwargs, source_units=source_units)
        inputs = self.bind_inputs(normalized_args, normalized_kwargs)
        if self.declaration.is_numpy_async():
            result = self._dispatch(normalized_args, normalized_kwargs)
        else:
            result = await asyncio.to_thread(self._dispatch, normalized_args, normalized_kwargs)
        if inspect.isawaitable(result):
            try:
                result = await result
            except (UnitContractError, OperatorCallError):
                raise
            except Exception as exc:
                name = self.declaration.full_name
                raise OperatorCallError(name, f"Operator '{name}' failed during async call: {exc}") from exc
        return self.finalize(result, audit, inputs)


def prepare_unit_call(
    declaration: Any, *, input_units: Mapping[str, Any] | None = None,
    output_units: Mapping[str, Any] | None = None,
    source_units: Mapping[str, Any] | None = None, strict: bool = False,
    output_bindings: Mapping[str, str] | None = None,
) -> UnitCallPlan:
    """Precheck metadata and fixed units for a reusable NumPy call plan.

    Input unit declarations assert the caller value's source. Trusted records
    use ``source_units``. Output unit keys are observable names when explicit
    bindings exist, otherwise JSON Pointers. No implementation is executed.
    """
    if not isinstance(strict, bool):
        raise UnitContractError("invalid_contract", "strict must be a bool.")
    if declaration.numpy_impl is None:
        raise UnitContractError("unsupported_backend", "Unit call plans require a NumPy implementation.")
    documented = declaration.metadata.get("params")
    contract = validate_unit_contract(
        declaration.metadata.get("unit_contract"), declaration.numpy_impl,
        documented_parameters=documented if isinstance(documented, Mapping) else None,
    )
    try:
        signature = inspect.signature(declaration.numpy_impl)
    except (TypeError, ValueError) as exc:
        raise UnitContractError("invalid_contract", "Unit call plans require an inspectable signature.") from exc
    inputs, sources = _unit_map(input_units), _unit_map(source_units)
    outputs = _unit_map(output_units, outputs=output_bindings is None)
    input_specs = contract["inputs"] if contract else {}
    output_specs = contract["outputs"] if contract else {}
    input_factors: dict[str, float] = {}
    for field in set(inputs) | set(sources):
        if field not in input_specs:
            raise UnitContractError("contract_missing", f"No input unit contract for {field!r}.", field=field)
        if field in inputs and field in sources and inputs[field] != sources[field]:
            raise UnitContractError("source_conflict", f"Declared and recorded source units disagree for {field!r}.", field=field)
        input_factors[field] = conversion_factor(sources.get(field, inputs.get(field)), input_specs[field]["unit"])
    if output_bindings is None:
        bindings = {path: path for path in output_specs}
        if not bindings:
            bindings = {"": ""}
    else:
        if not isinstance(output_bindings, Mapping) or not all(isinstance(key, str) and key for key in output_bindings):
            raise UnitContractError("invalid_contract", "Output bindings require nonempty observable names.")
        bindings = {name: normalize_output_path(path) for name, path in output_bindings.items()}
    output_factors: dict[str, float] = {}
    for name, target in outputs.items():
        if name not in bindings or bindings[name] not in output_specs:
            raise UnitContractError("contract_missing", f"No output unit contract for {name!r}.", field=name)
        output_factors[name] = conversion_factor(output_specs[bindings[name]]["unit"], target)
    return UnitCallPlan(
        declaration, signature, _freeze(contract), _freeze(inputs), _freeze(outputs),
        _freeze(sources), _freeze(bindings), strict, _freeze(input_factors), _freeze(output_factors),
    )
