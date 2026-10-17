# Changelog

## Unreleased

- Move all documentation, including the user guide, the unit contract JSON
  Schema and the HTs notes, to [Jarvis-Books](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/README.md). The repository
  and the source distribution keep only `README.md` and `CHANGELOG.md` as
  documentation.
- Consolidate duplicate README sections and correct obsolete CLI, curve
  namespace, and release instructions.
- Archive redundant document copies and historical task/design reports, and
  regenerate the function reference from declaration metadata.

## 1.5.0 — 2026-10-13

Released on PyPI on 2026-10-13. This release adds the shared unit contract
and canonical units
for 64 existing physical operators. See the [complete per-function update list](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/builtin_units.md).

### Shared unit contract and metadata

- Add `metadata.unit_contract`, version `1.0`, as the authoritative input/output
  unit declaration. Specifications contain `unit`, `shape`, and `nullable`;
  existing `metadata.params` text descriptions remain compatible.
- Validate declarations at `OperaFunction` construction, including built-ins,
  decorator registrations, and loaded user operators. Reject unknown units,
  unsupported versions, nonexistent parameters, invalid shapes/specification
  fields, malformed output paths, and duplicate normalized output paths.
- Support named logical keyword inputs for `**kwargs` implementations. When
  `metadata.params` is a mapping, logical input names must also be documented there.
- Use JSON Pointer output paths: `""` identifies the root result, `/value`
  identifies a dictionary field, and escaped/nested paths support arbitrary
  dictionary keys and sequence items. Plain field names normalize to pointers.
- Add 37 registered unit symbols covering energy/mass/width, cross section,
  area, length, angle, dimensionless values, energy squared/inverse squared,
  `MeV·fm`, `m/s`, and CMB temperature squared. Support explicit `MeV*fm` and
  microkelvin spelling aliases while preserving case-sensitive unit symbols.
- Keep angle distinct from dimensionless values. Reject unregistered compound
  units and implicit natural-unit conversions such as GeV^-2 → pb or GeV → mm.
- Expose stable `UnitContractError.code` diagnostics for unknown/unsupported
  units, dimension mismatch, missing units/contracts, source conflicts, type,
  shape, non-finite values, missing output paths, invalid contracts, and
  unsupported backends.

### Public execution and result protection

- Add registry and `Operas` methods `prepare`, `call_units`, `acall_units`, and
  `acall_many_units`. Preparation validates fixed bindings and precomputes
  conversion factors without executing the physical function.
- Add public `UNIT_CONTRACT_VERSION`, `UnitCallPlan`, `UnitCallResult`, `UnitProjection`,
  `prepare_unit_call`, `validate_unit_contract`, `unit_catalog`,
  `conversion_factor`, and `convert_value` exports.
- Prepared plans expose the physical function signature and provide `run`,
  `arun`, `normalize`, `normalize_inputs`, `run_inputs`, `arun_inputs`,
  `bind_inputs`, `project`, and `finalize`. Argument-container APIs keep
  point-specific source records separate from physical kwargs, including a
  physical parameter named `source_units`.
- Preserve ordinary `call`/`acall` signatures and raw results. Contracted NumPy
  inputs without source declarations use canonical units; audited execution
  marks supplied values `assumed_canonical` and omitted Python defaults
  `default_canonical`. Optional strict mode rejects unconfirmed numeric sources.
- Treat explicit input units as source assertions. Trusted source records and
  assertions must agree; one same-dimension unit cannot overwrite another.
  Check runtime conflicts even when a logical input value is absent.
- Normalize finite real scalars, NumPy arrays, numeric lists/tuples, and pandas
  containers without engine broadcasting or caller-input mutation. Preserve
  container types, shapes, pandas indexes, zero, and negative values; allow
  `None` only for nullable specifications and reject bool/complex/non-finite data.
- Preserve positional/keyword layouts and legacy arity checks. Unit controls
  are never passed to physical functions as incidental kwargs.
- Return `canonical_result`, `normalized_inputs`, projected `values`, `units`,
  and a separate per-call `audit`. Keep canonical value/unit/record ID/provenance
  and dependency references intact; do not regenerate HTs-specific fingerprints.
- Retain canonical output units when no target is requested. Explicit output
  conversion requires an output contract, and return-record source units must
  agree with it. Non-unit fields such as status may be projected unchanged.
- Keep prepared bindings immutable and audits independent across sync, async,
  batch, threaded, and forked calls. Registry async unit calls retain semaphore
  and configured-executor behavior; hosts control concurrency for direct plans.
- Prevent explicit or strict logical-keyword unit processing from being bypassed
  or shadowed by opaque mapping parameters, including alias-only or empty CMB
  payloads. Reject ambiguous mappings before execution; explicitly bound logical
  inputs must be supplied directly. Canonical legacy payload calls remain supported.

### Internal function library

- `helper.eggbox` and `helper.eggbox2d`: declare `x`/`y` in rad and outputs in
  `1`; support degree inputs while preserving broadcasting and scalar/array/pandas
  return conventions. `eggbox2d` declares output `/z`.
- `stat.chi2_cov`: declare the root chi-square output in `1`. Residual/covariance
  inputs remain without fixed units; heterogeneous covariance dimensions are
  not inferred. Preserve float, array, and pandas Series results.
- `cmb.loglike`: declare direct `model_cl`/`noise_cl` inputs in `uK^2`, nullable
  noise, and `/loglike` in `1`. Preserve the function's 111-point grid validation.
- `cmb.tt_cosmopower`: declare `omegabh2`, `omegach2`, `thetaMC`, `tau`, `ns`,
  `As`, and `h` in `1`; declare `/model_cl` in `uK^2` with shape `[111]`.
  Unit-aware CMB calls require direct kwargs instead of opaque payload/observables.
- PDG mass constants `mZ`, `mW`, `mt`, `mh`, `me`, `mmu`, `mtau`: root outputs
  in GeV. Couplings `alphaEM`/`alphaSMZ`: `1`; `GF`: GeV^-2; `hbarc`: MeV·fm;
  `c`: m/s. Derive the legacy `metadata.unit` display alias from the normalized
  root output contract and reject conflicting declarations. Constant values are unchanged.
- All 27 `dmddxn.*` curves: manifest-declared `x` in GeV and root output in
  cm^2. All 20 `dmddxe.*` curves: `x` in MeV and root output in cm^2. Cold
  declarations and hot-cache registration use the same manifest contracts.
- Document `math.add`/`math.identity` as generic operators without fixed physical
  units; preserve arbitrary-object identity and existing addition behavior.
  Generic `interp1.*` examples remain uncontracted. Explicit conversions on
  uncontracted inputs/outputs fail instead of silently assuming dimensionless units.
- Intentional validation tightening: contracted physical curve calls now reject
  NaN/Inf inputs before interpolation. Finite out-of-range legacy calls still
  return NaN; unit-projected outputs must be finite. Underlying interpolation
  points, numerical implementations, and constant values are unchanged.

### CLI and introspection

- Add `--input-units`, `--source-units`, `--output-units`, `--strict-units`, and
  `--unit-audit` to both `jopera call` and `jopera acall`.
- Unit-aware JSON calls return the complete five-field result envelope; ordinary
  calls retain raw output. Human unit output shows bounded value previews and
  canonical result type/record ID instead of dumping full records or large arrays.
- Display input/output contracts in human `jopera info`; expose normalized
  contracts in JSON metadata and registry `info.unit_contract`.
- Copy the unit contract in registry info snapshots so edits to returned info do
  not alter the registered declaration. Surface stable unit diagnostic codes in CLI errors.
- Extend CLI JSON serialization to dataclass results and fix zero-dimensional
  NumPy array serialization; preserve recursive array/scalar/container handling.
- Generate readable unit tables and machine-readable `unit_contract` fields from
  declaration metadata in function documentation/catalog generation.

### Documentation, examples, and integration boundaries

- Publish [the implemented unit API and schema guide](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract.md),
  its [JSON Schema](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract_schema.json), and
  [all 64 built-in function declarations](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/builtin_units.md).
- Add runnable Python and user-operator CLI fixtures demonstrating 0.2 TeV →
  200 GeV, 0.038332 pb → 38.332 fb, fixed/dynamic sources, sync/async agreement,
  and protection of canonical records.
- Update README, architecture, and maintainer instructions with unit metadata,
  public APIs, default policy, preload preparation, and contract maintenance.
- Document public host integration using source assertions/records, prepared
  plans, and projected values with units. This release does not implement HEP
  YAML/executor changes or HTs/GM physics functions.
- Explicit unit execution supports the NumPy backend. Existing Polars calls
  remain compatible; explicit Polars unit execution is rejected. Arbitrary
  expression dimension inference is outside v1.0. No new HiggsTools/GM/native dependency.

### Release automation and maintainer workflow

- Replace the local release path with `scripts/full-release.sh` and the root
  canonical `jorel` wrapper, correcting the previous filename typo. Follow
  Jarvis-HEP verification and Jarvis-PLOT source/artifact consistency checks.
  The wrapper is executable and sourceable from Bash/Zsh as the `jorel` function.
- Per maintainer policy, `./jorel 1.5.0` builds, verifies, and publishes to PyPI
  by default; `--dry` prepares verified artifacts without publication. Upload
  only the exact wheel/sdist after all gates pass. Accept an optional `X.Y.Z`,
  `--output`, `--python`, `--wheelhouse`, `--no-index`, and `--no-isolation`;
  reject unsupported options and nonempty output directories. Leave local
  packages, caches, commits, tags, and pushes alone.
- Snapshot tracked files with current edits and explicitly include required new
  release helpers and public user documentation. Require the source version
  and root changelog entry to agree; update a requested package version only
  after changelog preflight. Preserve the `dev` extra and the independent
  unit-contract schema version.
- Build wheel/sdist and run `twine check`, then compare metadata, entry points,
  Python modules, curve manifests, and model resources against the candidate
  source with the read-only, standard-library `scripts/check_release.py` gate
  (including Python 3.10 support). Reject missing/stale content, unrelated
  packages, invalid entry points, and duplicate/unsafe archive paths. Install
  the candidate in separate test tooling before running the release gate,
  full regression suite, and
  function-doc generation.
- Install the ordinary wheel and every runtime dependency in a clean virtual
  environment, run `pip check`, and exercise public units and the installed
  `jopera` CLI from outside the source checkout with `scripts/smoke_install.py`.
  Local wheelhouses support dependency verification without accessing indexes;
  preserve wheelhouse paths containing spaces through file-URI build settings
  and resolve relative Python paths from the invoking directory.
- Add explicit `MANIFEST.in` entries for the public wrapper, release helpers,
  canonical release/unit documents, and runnable unit examples in the sdist.
- Fail if tracked runtime files are missing; record the source-snapshot SHA-256
  before tests/doc generation and isolate test cache/persistence state.
- Produce `release-verification.json` and artifact `SHA256SUMS` only after
  required checks pass, and copy only that verified candidate into the selected
  output directory. Default publication uses those exact wheel/sdist files.
- Maintain local release procedures and maintainer checklists, and document
  candidate preparation in README. Replace broad cleanup/upload
  wildcard and automatic Git/local-install instructions with reviewable
  default publication after verification or preparation-only dry runs. Git
  release actions remain separate.

### Public documentation and history cleanup

- Move the complete changelog to root `CHANGELOG.md`. Publish the implemented
  unit guide, a standalone declaration JSON Schema, the 64-function built-in
  unit reference, and legacy migration notes under `user-guide/`.
- Keep maintenance instructions, architecture/design reviews, downstream
  implementation proposals, and internal handoff/verification records local.
  Ignore the entire `docs/` directory and remove its paths from all local Git
  history, including earlier root/`agent_maintenance/` paths of the internal
  maintenance guides. Preserve a backup and the local documentation files.
- Update README links and release packaging to use only public documentation.
  Require the root changelog for release version preflight; remove dependence
  on the internal handoff's package version. Source snapshots and source
  distributions exclude `docs/`, even when local internal files are present.
- Keep the HTs README's dataset-installation and bounded logging behavior
  explicitly marked as planned prototype work; remove links to local design
  reviews. Existing numerical implementations are unchanged.

### Validation

- Complete isolated release workflow: 344 tests passed, with one Python 3.13
  multithreaded-fork warning from the isolation test; release gate — 3 passed.
- Focused source/artifact contract checks — 54 passed; public-documentation
  and shell workflow checks — 34 passed; unit/registry/built-in/CMB/CLI checks
  — 88 passed.
- Wheel and sdist passed `twine check` and all 121 runtime-file comparisons.
  The sdist includes the public changelog and unit guides and excludes internal
  documents. The workflow separately generated reference pages for 90 functions.
- A clean environment installed the wheel and all runtime dependencies;
  `pip check` and installed CLI/unit smoke passed for 64 physical unit contracts,
  47 curves, CMB resources, async/batch calls, diagnostics, and record protection.
- Both distributions were published to [PyPI](https://pypi.org/project/Jarvis-Operas/1.5.0/).
  Published SHA-256 values match the local verification report. Release tag
  `1.5.0` points to `c4af128`; the cleaned branch and historical tag were pushed.

## 1.4.0

- Breaking change: `helper.eggbox` now uses the no-`pi` formula.
- Updated the corresponding NumPy, Polars, async, and release-gate tests to match the new contract.
