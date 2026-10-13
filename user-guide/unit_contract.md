# Unit contract 1.0

Jarvis-Operas owns unit declarations, conversion, validation, per-call audit, and
output projection. Physical functions receive ordinary canonical numeric values.
This capability has no HiggsTools, GM, Jarvis-HEP, or sample-type dependency.

## Authoritative metadata

`OperaFunction.metadata.unit_contract` is the only authoritative declaration.
Existing `metadata.params` descriptions remain descriptions. Parameter names,
including suffixes such as `_GeV`, do not determine units. The same validation
applies to built-ins and `@oper` / `__JARVIS_OPERAS__` user operators.

```python
unit_contract = {
    "version": "1.0",
    "inputs": {
        "mass_GeV": {"unit": "GeV", "shape": "scalar", "nullable": False},
    },
    "outputs": {
        "/value": {"unit": "pb", "shape": "scalar", "nullable": True},
    },
}
```

The contract requires `version="1.0"`; `inputs` and `outputs` are optional maps
defaulting to `{}`. No other contract-level or specification fields are allowed.
Input keys name actual callable parameters or declared logical keyword inputs
for a callable accepting `**kwargs`. When `metadata.params` is a mapping, those
logical keywords must be listed there; otherwise the unit contract defines
their accepted names. A variadic container
itself cannot receive a numeric specification. Output keys are JSON Pointers:
`""` denotes the complete result; `/value` denotes a dictionary field; `/a/0`
selects the first item of field `a`. Escape a literal `~` as `~0` and a literal
`/` as `~1`. A bare field name such as `value` is accepted as shorthand and
normalized to `/value`. Use JSON Pointer notation for new declarations; it keeps
the root distinct from every legal dictionary key.

Each unit specification has `unit`, `shape`, and `nullable`. Supported shapes are
`scalar`, `array`, `scalar_or_array`, or an explicit list of nonnegative integer
dimensions such as `[3]` or `[2, 3]`. Defaults are `scalar_or_array` and
`nullable=False`. Unknown units, invalid versions, unknown input parameters,
invalid shapes, and malformed output paths fail at declaration time. A missing
runtime output path is a separate `output_path_missing` error.

`jopera info --json` retains the full normalized declaration in metadata; readable
`info` and `scripts/generate_function_docs.py` render input/output tables derived
from that declaration. There is no separate manually maintained function-unit
table.

## Supported units and dimensions

| Dimension | Symbols | Canonical scale |
| --- | --- | --- |
| Energy / mass / width | eV, keV, MeV, GeV, TeV | GeV |
| Cross section | b, mb, ub, nb, pb, fb | pb |
| Length | m, cm, mm, um, fm | mm |
| Area (same cross-section dimension) | m^2, cm^2, mm^2 | pb |
| Angle | rad, deg | rad |
| Dimensionless | 1 | 1 |
| Energy squared | eV^2, keV^2, MeV^2, GeV^2, TeV^2 | GeV^2 |
| Inverse energy squared | eV^-2, keV^-2, MeV^-2, GeV^-2, TeV^-2 | GeV^-2 |
| Energy times length | MeV·fm, MeV*fm | MeV·fm |
| Speed | m/s | m/s |
| CMB temperature squared | K^2, mK^2, uK^2 (aliases µK^2, μK^2) | K^2 |

The public `unit_catalog()` provides the authoritative supported-unit inventory
and scale definitions. Symbols preserve case; `MeV` is not `mev`. Unsupported
compound symbols fail explicitly. Angle and dimensionless are distinct
dimensions. No implicit natural-unit conversion connects GeV^-2 to pb or GeV
to mm; such a conversion requires a separate physical transformation.

Conversion multiplies by a fixed source-to-target factor. Inputs may be finite
real scalars, NumPy arrays, or numeric lists/tuples. Arrays preserve shape;
lists and tuples preserve their container type. The engine does not add
broadcasting between arguments; the function's own existing broadcasting and
return convention apply. Zero and negative values convert normally. Booleans,
complex/non-numeric values, and non-finite contracted inputs or projected
outputs fail. `None` is preserved only
when `nullable=True`. Neither inputs nor the original returned record are
mutated by normalization or projection.

Factors use Python floating-point arithmetic. The scalar pb ↔ fb round-trip
test uses a relative tolerance of `1e-13`; array precision follows its numeric
dtype. Conversions do not promise bit-for-bit equality for arbitrary scales.

## Source declarations, defaults, and errors

`input_units={"mass_GeV": "TeV"}` asserts the unit of the number being supplied.
`source_units` carries a trusted source-unit record obtained from the host. If
both are present, they must identify the same source unit after supported
symbol normalization. An assertion cannot overwrite a known `TeV` source with
`GeV`, even though both have the energy dimension.

With neither source nor assertion, compatibility mode interprets an input in
its declared canonical unit and emits `assumed_canonical` in that call's audit.
It does not verify upstream units. `strict=True` fails before execution when a
unit-contracted numeric parameter lacks a confirmed source unit. Metadata-only
non-numeric arguments, such as an analysis identifier or quantile string, need
no unit declaration.

Logical keyword quantities must be supplied as direct kwargs during explicit
or strict unit processing. For a logical `**kwargs` unit contract, an active
input unit/source binding or strict mode rejects any opaque mapping supplied
through an explicit signature parameter, including empty `payload={}` and
mapping fields using aliases. JO cannot establish whether such a mapping
overrides the normalized direct kwargs, so this combination fails before
execution with `contract_missing`. Omit opaque `payload` / `observables` maps
and flatten numeric values into direct kwargs, or expose the numerical
quantities through a wrapper with named numeric signature parameters. Named
numeric parameters can coexist with a separate configuration mapping. Raw
compatibility calls retain their existing mapping behavior. An explicitly
unit-bound logical keyword with no direct value fails with `missing_unit`.

An omitted output target uses the declared source output unit. Explicit unit
requests require a usable contract for the requested parameter or output.
Legacy functions without a contract retain ordinary `call` / `acall` behavior;
an explicit request cannot be silently ignored.

`UnitContractError.code` is a stable diagnostic identifier:

| Code | Meaning |
| --- | --- |
| unknown_unit | Unregistered unit symbol |
| unsupported_unit | Unsupported compound unit syntax |
| dimension_mismatch | Source and target dimensions differ |
| missing_unit | Strict input lacks a confirmed source unit |
| source_conflict | Assertion disagrees with trusted source record |
| type_error | Unsupported numeric type, including bool |
| shape_mismatch | Value shape violates its specification |
| non_finite | NaN or infinity in numeric data |
| contract_missing | Explicit unit handling has no usable contract |
| invalid_contract | Malformed metadata, version, parameter, or path |
| output_path_missing | Declared/bound path absent from returned result |
| unsupported_backend | Explicit unit execution requested for an unsupported backend |

Treat these as contract failures, not experimental exclusions or zero-valued
limits. CLI failures return exit status 1 and include the diagnostic code.

## Public preparation and execution API

```python
registry.prepare(
    name, *, input_units=None, source_units=None, output_units=None,
    strict=False, output_bindings=None,
)  # -> UnitCallPlan

registry.call_units(
    name, *args, input_units=None, source_units=None, output_units=None,
    strict=False, output_bindings=None, backend="numpy", **kwargs,
)  # -> UnitCallResult

await registry.acall_units(
    name, *args, input_units=None, source_units=None, output_units=None,
    strict=False, output_bindings=None, backend="numpy", **kwargs,
)  # -> UnitCallResult
```

`output_bindings` maps observable names to paths, for example
`{"h5pp_limit_fb": "/value", "h5pp_limit_status": "/status"}`. With explicit
bindings, `output_units` uses those observable names, for example
`{"h5pp_limit_fb": "fb"}`. Without explicit bindings, `values`, `units`, and
`output_units` use normalized output JSON Pointers. An uncontracted field can be
projected unchanged with `units[name] = None`; requesting conversion for it
fails with `contract_missing`.

`prepare` validates fixed declarations and precomputes conversion factors so a
host can prepare one plan during Worker preload. `plan.signature` exposes the
operator signature for inspection. `plan.run(*args, **kwargs)` and
`await plan.arun(*args, **kwargs)` normalize, execute, and project with that
fixed contract. `plan.arun` awaits native async implementations or offloads sync
ones to a thread; the hosting caller controls plan-level concurrency.
`registry.acall_units` additionally applies the registry concurrency gate and
configured executor. Each result has its own audit; there is no global last-result or
last-audit state.

`await registry.acall_many_units(name, calls, concurrency=None,
return_exceptions=False, **unit_options)` accepts a list of independent kwargs
payloads sharing fixed unit options. Its results preserve input order and each
carry an independent audit. `Operas` exposes the same preparation and unit-call
methods as its registry.

For point-specific source records, use
`plan.run_inputs(args_tuple, kwargs_mapping, source_units=...)` or
`await plan.arun_inputs(args_tuple, kwargs_mapping, source_units=...)`.
`plan.run` / `arun` treat every kwarg as a physical function argument, including
a parameter named `source_units`. The container APIs keep source metadata
separate from such physical arguments. Runtime records must agree with fixed
plan source records and assertions, even when a logical parameter is absent.

For hosts that already own execution dispatch,
`plan.normalize_inputs(args_tuple, kwargs_mapping, source_units=None)` returns a
canonical `(args_tuple, kwargs_dict, input_audit_list)`, and
`plan.project(canonical_result)` returns a `UnitProjection` with independent
`values`, `units`, and output `audit`. Prefer `run` / `arun` for ordinary host
execution so normalization cannot be skipped. The shorthand
`plan.normalize(*args, source_units=None, **kwargs)` reserves `source_units` as
runtime metadata; use `normalize_inputs` when that name is a physical parameter.

`UnitCallResult` fields:

| Field | Meaning |
| --- | --- |
| canonical_result | Complete original result, retaining its canonical record identity |
| normalized_inputs | Canonical parameter values used for execution |
| values | Separately projected values keyed by output path/binding |
| units | Units associated with those separately projected values |
| audit | Per-call input/output conversion and assumption records |

Audit entry fields are `field`, `kind` (`input` or `output`), `source_unit`,
`target_unit`, `factor`, `origin`, and `contract_version`. Origins include
`assumed_canonical`, `default_canonical` (an omitted Python parameter default),
`declared`, `source_record`, `source_record_asserted`, and `contract`. Operator
defaults are already canonical; strictness applies to supplied numeric values.
An HTs-style record keeps its original `value`, `unit`, `record_id`, provenance,
and dependency references intact; storing the projected value does not relabel
the canonical record.

Existing `registry.call(name, *args, backend=..., **kwargs)` and `acall` preserve
their signatures and raw return values. For contracted NumPy functions they
use the same canonical-input compatibility checks. Use `call_units` or a
prepared plan when source units, strictness, projections, or audit are needed.
Contracted input NaN/infinity now fails with `non_finite` before execution.
A finite but out-of-range interpolation input retains its existing raw NaN
return through plain `call` / `acall`; projecting that NaN as a unit-contracted
output fails with `non_finite`.
Explicit unit processing currently supports NumPy execution. Polars expression
construction retains its existing dispatch behavior and rejects explicit unit
processing with `unsupported_backend`.

## Runnable Python and CLI examples

From the repository root:

```bash
python examples/unit_contract_demo.py

python -m jarvis_operas.cli info unit_contract_ops.limit_prediction \
  --user-ops examples/unit_contract_ops.py --json

python -m jarvis_operas.cli call unit_contract_ops.limit_prediction \
  --user-ops examples/unit_contract_ops.py \
  --arg mass_GeV=0.2 \
  --source-units '{"mass_GeV":"TeV"}' \
  --input-units '{"mass_GeV":"TeV"}' \
  --output-units '{"/value":"fb"}' --strict-units --json
```

The fixture receives `200.0` GeV, retains an original result with `0.038332` pb,
and projects `/value` to `38.332` fb. Replace `call` with `acall` for the same
unit behavior through async execution. Add `--unit-audit` alone to inspect
compatibility assumptions. Any of the unit options selects the five-field
result envelope; a plain call retains the legacy raw result.
Without `--json`, unit calls print compact input/projection previews and a
canonical result type/record ID summary. Use `--json` explicitly for the full
canonical record and arrays.

## Integration and supported scope

Downstream applications prepare the public unit plan during preload, then call
`plan.run()` / `plan.arun()` and store `result.values` together with
`result.units`. Preserve `canonical_result` and its provenance when a complete
physical record is needed. Direct calls to an operator's implementation bypass
unit validation; use the public registry or prepared plan.

Jarvis-Operas does not parse host YAML, resolve sample fields, or infer the units
of arbitrary expressions. A host supplies source records and input assertions
separately. This release does not add Jarvis-HEP YAML/executor fields or change
HTs/GM physics implementations. Explicit unit processing supports NumPy;
Polars expressions and implicit natural-unit transformations are outside
contract 1.0.

The [JSON Schema](unit_contract_schema.json) describes the declaration's
structure. Runtime declaration validation also checks registered unit symbols,
callable input names, and output paths. Use `unit_catalog()` for the supported
unit inventory and `jopera info NAME --json` for a function's normalized
metadata. See [built-in units](builtin_units.md) for all 64 contracted functions.

Runnable fixtures are available in `examples/unit_contract_demo.py` and
`examples/unit_contract_ops.py` in the source distribution.
