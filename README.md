# Jarvis-Operas

Jarvis-Operas is a standalone operator layer for registering, loading, and calling Python callables.

- Scope: operators only (likelihood, chi2, prior mapping, data transforms, etc.)
- No dependency on Jarvis-HEP internals
- Native async entrypoint for Jarvis-HEP style execution (`await registry.acall(...)`)
- Stable `"<namespace>.<name>"` naming for references from Jarvis-HEP/Jarvis-PLOT YAML

Documentation lives in [Jarvis-Books](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/README.md):

- [User guide](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/README.md)
- [Unit contract v1.0](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract.md)
- [Built-in function units](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/builtin_units.md)
- [Unit contract JSON Schema](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract_schema.json)
- [Function reference](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/functions/index.md)
- [Legacy 1.2.x migration notes](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/archive/migration-1.2.md)
- [Changelog](CHANGELOG.md)

## Install

```bash
pip install .
```

Or for development (includes `pytest` and installs terminal command):

```bash
pip install -e ".[dev]"
```

## Terminal command (`jopera`)

After install, you can use:

```bash
jopera
jopera help examples
jopera namespaces
jopera list --namespace math
jopera list --namespace math --json
jopera info stat.chi2_cov --json
jopera call math.add --kwargs '{"a": 1, "b": 2}'
jopera acall stat.chi2_cov --arg residual=[1.0,-0.5] --arg cov=[[2.0,0.1],[0.1,1.0]]
jopera call helper.eggbox --kwargs '{"x":0.5,"y":0.0}'
jopera call helper.eggbox --arg x=90 --arg y=0 --input-units '{"x":"deg","y":"rad"}' --unit-audit --json
jopera call pdg.mZ --output-units '{"":"MeV"}' --json
jopera list --namespace math --log-mode info
jopera call math.add --kwargs '{"a": 1, "b": 2}' --log-mode debug
jopera init
jopera init --manifest ./manifest.json --cache-root ~/.jarvis-operas/curve-cache
jopera interp list
jopera interp validate
jopera --help-advanced
```

Load user operators directly in CLI:

```bash
jopera load /absolute/path/to/my_ops.py
jopera call my_ops.my_op --user-ops /absolute/path/to/my_ops.py --arg x=10
jopera info my_ops.my_op --json
```

CLI behavior:

- `jopera` (no args) prints a quick start card
- `jopera list` prints grouped human-readable output by default (`id + name`)
- `jopera list --json` prints machine-readable JSON objects with `id/name/namespace`
- `jopera info <name_or_id>` prints metadata (including a unique id) and a suggested next command
- `call` / `acall` accept `--input-units`, `--source-units`, `--output-units`, `--strict-units`, and `--unit-audit`; unit options return the canonical result, projected values, units, normalized inputs, and audit
- `jopera load <path>` persists a JO-managed source snapshot for future processes
- Re-running `jopera load <path>` replaces existing user operators with the same `<namespace>.<name>`
- Removing a function from a user source file does not auto-delete its prior registration
- `jopera load <path> --session-only` loads without persisting
- `jopera init` precompiles bundled interpolation manifest library
- `jopera init --manifest <manifest.json>` precompiles a custom curve JSON manifest
- `jopera interp list` lists root interpolation namespaces
- `jopera interp validate` validates root+namespace manifests
- User write policy: the supported user write path is `jopera load`

## Core API

Physical operators can declare a versioned `metadata.unit_contract`. Calls without
source units retain canonical-unit defaults; `call_units` / `acall_units` return
the original result, separately projected values and a per-call audit. Prepare
reusable plans with `registry.prepare(...)` during Worker preload. For example:

```python
from jarvis_operas import get_global_operas_registry

registry = get_global_operas_registry()
result = registry.call_units("pdg.mZ", output_units={"": "MeV"})
assert result.values[""] == 91187.6
```

Use `jopera call pdg.mZ --output-units '{"":"MeV"}' --json` from the CLI.
The [unit guide](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract.md) includes the metadata schema,
strict source validation, TeV → GeV input, and pb → fb output examples. The
[JSON Schema](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/unit_contract_schema.json) describes declaration
structure; registered unit symbols and callable parameters are validated by JO.
HEP YAML unit fields and its executor bridge remain downstream work.

```python
from jarvis_operas import get_global_operas_registry, get_global_operas

registry = get_global_operas_registry()
registry.call("math.add", a=1, b=2)
registry.call("math.identity", x={"k": 1})

operas = get_global_operas()
operas.call("math.add", a=1, b=2)
```

Supported namespace convention:

- Built-ins use feature namespaces, e.g.:
  - `math.<name>`
  - `stat.<name>`
  - `helper.<name>`
- User operator files default to `<script_name>.<func_name>`
- User operators cannot register into protected built-in namespaces (for example
  `math`, `stat`, `helper`, `interp`, built-in interpolation namespaces).

## Async call (Jarvis-HEP Factory/Module friendly)

```python
result = await registry.acall(
    "stat.chi2_cov",
    residual=[1.0, -0.5],
    cov=[[2.0, 0.1], [0.1, 1.0]],
)
```

Behavior:

- Async operator: awaited directly
- Sync operator: offloaded via `asyncio.to_thread` by default (or custom executor)
- Default numpy async concurrency is CPU-scaled (`os.cpu_count() * 3`); override with `get_global_operas(numpy_concurrency=...)`
- Batch helper concurrency: `await registry.acall_helper_many("eggbox", [...])`

Example batch helper execution for Factory-side concurrent scans:

```python
results = await registry.acall_helper_many(
    "eggbox",
    [
        {"x": 0.1, "y": 0.2},
        {"x": 0.3, "y": 0.4},
        {"x": 0.5, "y": 0.6},
    ],
)
```

## Logger injection

`registry` methods accept optional `logger` and operators can optionally define `logger` argument.

- If no logger is provided, Jarvis-Operas uses `loguru.logger` bound with `module="Jarvis-Operas"`
- If logger is provided, Jarvis-Operas reuses it (no duplicate handler creation)
- Console format follows Jarvis-HEP style (`module -> time - [level] >>> message`)
- Default mode is `warning` (only warning/error/critical are shown)
- Optional modes: `info`, `debug`

Utility:

```python
from jarvis_operas import get_logger, set_log_mode

set_log_mode("info")   # or "debug", default is "warning"
logger = get_logger()
```

## Load user operators from file

```python
from jarvis_operas import OperasRegistry, load_user_ops

registry = OperasRegistry()
loaded = load_user_ops("./my_ops.py", registry)
print(loaded)
```

Reload behavior:

- `load_user_ops(...)` and `jopera load` always execute the current source file contents
- If a user operator resolves to an existing `<namespace>.<name>`, the new definition replaces the old one
- If a function is removed from the file, JO does not auto-prune the old registration; delete/rename remains an explicit override action

Persistent registration for future Python processes:

```python
from jarvis_operas import get_global_operas_registry, persist_user_ops

persist_user_ops("/absolute/path/to/my_ops.py")
registry = get_global_operas_registry()  # auto-loads persisted sources
```

Persistence store location:

- Default sources registry: `~/.jarvis-operas/sources.json`
- Default override rules: `~/.jarvis-operas/overrides.json`
- Legacy base env (still supported): `JARVIS_OPERAS_PERSIST_FILE=/custom/path/persist_store.json`
  - maps to sibling files `sources.json` and `overrides.json`
- Optional explicit envs:
  - `JARVIS_OPERAS_SOURCES_FILE=/custom/path/sources.json`
  - `JARVIS_OPERAS_OVERRIDES_FILE=/custom/path/overrides.json`
- Managed user source snapshots: `~/.jarvis-operas/user_sources/`
- Override snapshot dir: `JARVIS_OPERAS_USER_SOURCE_DIR=/custom/path/user_sources`
- Legacy single-file `user_ops.json` is still readable for backward compatibility

By default, `load_user_ops("./my_ops.py", ...)` uses `my_ops` as namespace, so
`my_op` becomes `my_ops.my_op`.

`my_ops.py` can export operators with either style:

1. Decorator (recommended)

```python
from jarvis_operas import oper

@oper("my_chi2")
def my_chi2(residual, cov, logger=None):
    ...
```

When loaded by `load_user_ops("./my_ops.py", ...)`, this is registered as
`my_ops.my_chi2`.

2. Explicit whitelist

```python
def my_op(x):
    return x

__JARVIS_OPERAS__ = {
    "my_op": my_op,
}
```

## Integration Contract (Jarvis-HEP / Jarvis-PLOT)

Import integration entry points from `jarvis_operas`: `OperasRegistry`, `Operas`,
`get_global_operas_registry`, `get_global_operas`, `oper`, `load_user_ops`, and
`persist_user_ops`. Downstream modules should use the public registry methods:

- registry provider: `get_global_operas_registry()`
- runtime calls: `call/acall`
- lookup/introspection: `resolve_name/get/list/info`
- unit execution: `prepare/call_units/acall_units/acall_many_units`

Prepared plans expose `run/arun`, `run_inputs/arun_inputs`, and `project`.
Prepare registrations and fixed unit plans during host initialization, then
reuse them during execution. Direct implementation calls bypass unit validation;
downstream integrations must use these public entry points. JO provides
declarations, registration, and dispatch; hosts own sample objects, YAML parsing,
and observable storage. The [user guide](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/user-guide/README.md) links to current
unit execution examples and archived migration notes.

## Function documentation generation

Function docs are generated from `OperaFunction.metadata`:

```bash
python scripts/generate_function_docs.py --out docs/functions
```

Generated artifacts are local outputs under the ignored `docs/` directory. The
published copy is the [function reference](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-Operas/functions/index.md) in
Jarvis-Books.


- `docs/functions/index.md`
- `docs/functions/<namespace>/index.md`
- `docs/functions/functions.json`
- `docs/functions/<namespace>/<function>.md`

Optional filters and output directory:

```bash
python scripts/generate_function_docs.py --namespaces math,stat
python scripts/generate_function_docs.py --out docs/custom_functions
```

## Curve publish/runtime cache (namespace manifests + JSON sources)

Use this flow when you have many 1D interpolation curves and want runtime speed:

1. Source of truth:
   - root index manifest (`interpolations.manifest.json`) with namespace list
   - one manifest per namespace (function metadata list)
   - per-function JSON source (`x`/`y` arrays)
2. Precompile once: `jopera init` (bundled library) or `jopera init --manifest ./manifest.json` (custom)
   - optional: `jopera init --namespaces dmddxn,interp1` for partial namespace processing
   - inspect manifests with `jopera interp list` and `jopera interp validate`
3. Runtime auto-registration: `get_global_operas_registry()` initializes the bundled cache and registers hot curves in their resolved namespaces, such as `dmddxn.LZSI2024` and `dmddxe.CDEXLM2022`
4. After initialization, cached curve loading uses `index.json` + `*.pkl`; evaluation reuses the loaded callable

Namespace rule for registered interpolation operators:

- Use the first available declaration: the curve item's `namespace`, its `group`, its `metadata.group`, then the namespace manifest or root-index namespace.
- Register as `<resolved_namespace>.<curve_id>`; use `interp.<curve_id>` only when no namespace is declared.
- Bundled interpolation namespaces are `interp1`, `dmddxn` (DM-nucleon), and `dmddxe` (DM-electron).

Bundled interpolation manifest library resource:

- `jarvis_operas/manifests/interpolations.manifest.json`

Root index example:

```json
{
  "version": 2,
  "kind": "jarvis_operas_interpolation_namespace_index",
  "namespaces": [
    {
      "namespace": "demo",
      "manifest": "demo/demo.manifest.json"
    }
  ]
}
```

Namespace manifest example:

```json
{
  "namespace": "demo",
  "functions": [
    {
      "name": "demo_curve",
      "source": "demo/demo_curve.json",
      "kind": "linear",
      "hot": true,
      "metadata": {
        "group": "demo"
      }
    }
  ]
}
```

Curve source JSON example:

```json
{
  "x": [0.0, 1.0, 2.0],
  "y": [0.0, 1.0, 4.0]
}
```

Python integration API:

```python
from jarvis_operas import (
    init_curve_cache,
    interpolation_manifest_resource,
    load_hot_curve_function_table,
    load_interpolation_manifest_library,
    register_hot_curves,
)

library_manifest = load_interpolation_manifest_library()
library_path = interpolation_manifest_resource()

init_curve_cache("./manifest.json")
table = load_hot_curve_function_table()

funcs = {}
updated = register_hot_curves(funcs)
```

Interpolation functions support:
- `numpy` backend: native callable
- `polars` backend: `map_batches` fallback via `OperaFunction.return_dtype`
- sync: `registry.call(...)`
- async: `await registry.acall(...)`

## Built-in operators

| Namespace | Available operators |
| --- | --- |
| `math` | Generic `add` and `identity`. |
| `stat` | Covariance chi-square with `chi2_cov`. |
| `helper` | Benchmark functions `eggbox` and `eggbox2d`. |
| `pdg` | Particle masses, couplings, and physical constants, including `mZ`, `GF`, and `hbarc`. |
| `cmb` | `tt_cosmopower` prediction and `loglike` on the bundled 111-point Planck TT grid. |
| `interp1` | Four general interpolation examples. |
| `dmddxn` | 27 DM-nucleon exclusion curves; input in GeV and output in cm². |
| `dmddxe` | 20 DM-electron exclusion curves; input in MeV and output in cm². |
| `HTs` | Optional HiggsTools evaluator and helpers; the current prototype requires an external package and local datasets. See its [status and setup notes](jarvis_operas/namespaces/hts/README.md). |

Use `jopera namespaces`, `jopera list --namespace NAME`, and `jopera info NAME`
to discover functions and their signatures, units, and supported backends.
Sync/async dispatch is available through the registry; accepted containers and
backend support depend on each function. Examples:

```python
registry.call("math.add", a=1.0, b=2.0)
registry.call("stat.chi2_cov", residual=[1.0, 0.0], cov=[[2.0, 0.0], [0.0, 1.0]])
registry.call("helper.eggbox", x=0.5, y=0.0)
```

## Query registry for external UIs (JHEP/JPlot)

```python
registry.list()
registry.list(namespace="math")
registry.info("stat.chi2_cov")
```

`registry.info(...)` returns metadata, signature, docstring summary, backend support fields, and async flag.

## Releases

Version `1.5.0` is already published. For a new release, add its version heading
and changes to `CHANGELOG.md`, then build, verify, and publish to PyPI. These
commands use `1.5.1` only as an example of a new version:

```bash
source ./jorel
jorel 1.5.1
```

`jorel` bumps `pyproject.toml`, runs the release gate and the full test suite,
builds and checks the wheel/sdist, and uploads them to PyPI. It then
reinstalls that version from PyPI and checks the import. Configure Twine
credentials before running it. An already published version cannot be uploaded
again with replacement files.

To build and check without publishing, add `--dry`, the only option:

```bash
jorel 1.5.1 --dry
```

Release changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Workspace clean

Clean local cache/build artifacts:

```bash
./scripts/clean_workspace.sh
```
