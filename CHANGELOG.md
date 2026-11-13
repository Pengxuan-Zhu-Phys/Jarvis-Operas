# Changelog

Release history of `Jarvis-Operas` (CLI `jopera`). User guides and API reference
are available in [Jarvis-Books](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/README.md).

## Unreleased

### Fixed

- Accept `-V` as an alias of `-v` / `--version`.
- `jopera -h`, `jopera --help`, `jopera --version`, and `jopera` with no
  command no longer load the global registry. SymPy snapshots `func_locals`
  and `numeric_funcs` are built on first use, so asking for help does not load
  persisted operator scripts or print their warnings.
- Import `joperas.<namespace>` from a persisted operator script without
  deadlocking startup. Importing `joperas` publishes the registry before any
  namespace module loads, and `from joperas.<namespace> import <name>` only
  reads that published registry. It does not start or wait for bootstrap
  while the import lock is held. Built-in operators and interpolation curves
  are registered before persisted scripts run, so module-level imports such
  as `from joperas.math import add` and `from joperas.dmddxn import LZSI2024`
  resolve during load.
- Pass an operator argument named `backend` through `registry.call` when its
  value is not the dispatch name `numpy` or `polars`.
- Reuse the default unit plan for ordinary NumPy calls instead of validating
  the contract and inspecting the signature on every call.
- Keep curves from other namespaces when initializing a cache. Startup
  initialization of the bundled library, and `jopera init --namespaces`,
  update only the namespaces they compile. A custom manifest compiled into
  the default cache stays in `index.json`, and its `.npz` files stay on disk.
  Recompiling a namespace still removes curves deleted from that namespace
  and deletes the artifact replaced by the new file.

### Changed

- Render `jopera -h` with the Jarvis CLI help palette: bold usage, cyan long
  options, green short options, yellow metavars, cyan command names, and dim
  panel borders. The usage line sits above the panels, and a command with no
  positional arguments does not show an Arguments panel. `jopera list`,
  `jopera namespaces`, and `jopera info` use that same dim border and cyan
  text. `jopera --help-advanced` describes curve cache files as `.npz`.
- Read and write Jarvis-Operas state only under `~/.jarvis/operas/`
  (`JARVIS_HOME` selects the Jarvis directory, `JARVIS_OPERAS_HOME` selects
  the operas directory). `~/.jarvis-operas/` is not read and is not moved.
- Store compiled curves as numeric `.npz` artifacts referenced by `index.json`.
  Loading rebuilds `Curve1DInterpolator` and does not execute pickle. An older
  pickle entry is rebuilt by `jopera init` and then removed.
- Imported `joperas` names are the NumPy implementation, with canonical unit
  normalization applied when the operator declares a unit contract. A name
  already bound in the caller keeps that function object. Polars dispatch
  stays on `registry.call`. Namespace attributes are read-only.
- Leave the HiggsTools helpers out of the public registry. `HTs` is not
  registered, so `jopera list`, `registry.list()` and `joperas.HTs` do not
  expose it. The implementation remains in `jarvis_operas.namespaces.hts`
  for development, and user operators still cannot register into that
  namespace.

### Added

- Import any registered operator in a Python script. The namespace and function
  name are the same ones accepted by `registry.call` and `jopera call`:

  ```python
  from joperas.math import add
  from joperas.pdg import mZ
  from joperas.dmddxn import LZSI2024
  from joperas.dmddxe import CDEXLM2022

  add(1, 2)
  mZ()
  LZSI2024(30.0)
  ```

  This covers built-in operators, interpolation curves, and user operators
  already loaded into the global registry. A call uses the same dispatch as
  `registry.call`, including canonical unit handling. Constants are
  zero-argument calls. An unknown namespace names the closest registered
  namespaces, so `joperas.dmdd` points at `dmddxn` (DM-nucleon) and `dmddxe`
  (DM-electron).

## 1.5.2

### Fixed

- Report whether an operator implementation is asynchronous in the human info
  card, independently of its support for asynchronous invocation.
- Print each operator list once when stdout is redirected.
- Load or reload a user operator file as one complete update. If loading or
  validation fails, the previously registered operators remain available;
  partially loaded definitions are not exposed. Conflicting concurrent
  registry changes cause the load to fail without a partial update.
- Match labelled covariance matrices to residual names in both NumPy/pandas
  and Polars `stat.chi2_cov` calls. Missing or duplicate labels raise an error,
  including for empty batches. Unnamed inputs still use positional ordering.

### Changed

- Display `Jarvis-Operas` with `-v` / `--version`, including through
  `Jarvis operas`. Use `--json` for structured version output.

## 1.5.1 — 2026-10-18

### Documentation

- Find the user guide, unit contract JSON Schema, function reference and HTs
  notes in [Jarvis-Books](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/README.md).
- Correct outdated CLI examples and curve namespace references in the README.

## 1.5.0 — 2026-10-13

This release adds explicit unit handling and canonical unit declarations for
64 existing physical operators.

### Added

- Declare input and output units with `metadata.unit_contract`, version `1.0`,
  including `unit`, `shape` and `nullable`. Invalid declarations raise
  `UnitContractError` with a stable diagnostic code. Existing `metadata.params`
  descriptions remain supported.
- Use `prepare`, `call_units`, `acall_units` and `acall_many_units` on the registry
  or `Operas` interface. Prepared `UnitCallPlan` objects validate fixed bindings
  and expose sync, async, input-normalization and output-projection methods.
- Access the unit API through `UNIT_CONTRACT_VERSION`, `UnitCallPlan`,
  `UnitCallResult`, `UnitProjection`, `prepare_unit_call`,
  `validate_unit_contract`, `unit_catalog`, `conversion_factor` and `convert_value`.
- Convert between 37 registered unit symbols. The catalog covers energy, mass,
  width, cross section, area, length, angle, dimensionless values and CMB
  temperature squared. See the [unit guide and declaration schema](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract.md)
  for supported symbols, aliases and JSON Pointer output paths.
- Inspect the complete unit-aware result: `canonical_result`, `normalized_inputs`,
  projected `values`, `units` and a per-call `audit`. Output projections retain
  canonical values and provenance; prepared plans can be reused across calls.
- Supply `--input-units`, `--source-units`, `--output-units`, `--strict-units` and
  `--unit-audit` to `jopera call` and `jopera acall`. Unit-aware JSON output returns
  the complete result envelope. `jopera info` displays input/output contracts.
- Use canonical units for the physical operators in `helper`, `stat`, `cmb`,
  PDG constants and the `dmddxn` / `dmddxe` curves. The
  [per-function reference](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/builtin_units.md)
  lists all declarations and examples.

### Fixed

- Serialize dataclass results and zero-dimensional NumPy arrays in CLI JSON
  output while preserving existing array and container support.
- Keep edits to returned registry information from changing registered unit
  declarations.

### Compatibility and upgrade notes

- Ordinary `call` / `acall` keep their signatures and raw return values. Physical
  constants, interpolation data and numerical implementations are unchanged.
- Unit-aware execution supports the NumPy backend. Ordinary Polars calls remain
  supported; explicit unit-aware Polars execution raises an error.
- Without source-unit declarations, contracted NumPy inputs use canonical units
  and are marked `assumed_canonical` in the audit. Omitted Python defaults are
  marked `default_canonical`. Strict mode rejects unconfirmed numeric sources.
- Explicit units must agree with trusted source records. Incompatible dimensions,
  unknown units and implicit natural-unit conversions such as GeV^-2 to pb are
  rejected. Angles remain distinct from dimensionless values.
- Unit processing preserves scalar, array and pandas shapes without implicit
  broadcasting or input mutation. It rejects boolean, complex and non-finite
  values; `None` requires a nullable declaration.
- Contracted physical curve calls reject NaN/Inf inputs. Finite out-of-range
  ordinary calls still return NaN; unit-projected outputs must be finite.
- Unit-aware CMB and logical-keyword calls require direct named inputs; opaque
  payload mappings cannot bypass unit checks. Existing ordinary payload calls
  remain supported.
- Generic `math.add`, `math.identity` and `interp1.*` operators have no fixed
  physical units. Explicit conversions require a declared unit contract.
- This release provides Python and CLI unit APIs. It does not add unit processing
  to HEP YAML, infer units for arbitrary expressions, or add HiggsTools/GM physics
  functions or native dependencies.

## 1.4.0

### Breaking changes

- `helper.eggbox` uses the formula without the `pi` factor. Existing code that
  depends on the previous formula must account for this change.
