# Jarvis-Operas

Jarvis-Operas is a standalone operator layer for registering, loading, and calling Python callables.

- Scope: operators only (likelihood, chi2, prior mapping, data transforms, etc.)
- No dependency on Jarvis-HEP internals
- Native async entrypoint for Jarvis-HEP style execution (`await registry.acall(...)`)
- Stable `"<namespace>.<name>"` naming for references from Jarvis-HEP/Jarvis-PLOT YAML

Architecture and migration references:

- `docs/architecture.md`
- `docs/migration.md`

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
jopera list --namespace math
jopera list --namespace math --json
jopera info stat.chi2_cov --json
jopera call math.add --kwargs '{"a": 1, "b": 2}'
jopera acall stat.chi2_cov --arg residual=[1.0,-0.5] --arg cov=[[2.0,0.1],[0.1,1.0]]
jopera call helper.eggbox --kwargs '{"x":0.5,"y":0.0}'
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
- `jopera load <path>` persists a JO-managed source snapshot for future processes
- `jopera load <path> --session-only` loads without persisting
- `jopera init` precompiles bundled interpolation manifest library
- `jopera init --manifest <manifest.json>` precompiles a custom curve JSON manifest
- `jopera interp list` lists root interpolation namespaces
- `jopera interp validate` validates root+namespace manifests
- User write policy: the supported user write path is `jopera load`
- Developer-only write commands (`update`, `delete-func`, `delete-namespace`, `interp add/remove`) require `JARVIS_OPERAS_DEV_WRITE=1`

## Core API

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

Downstream modules should depend on JO public API only:

- registry provider: `get_global_operas_registry()`
- runtime calls: `call/acall`
- lookup/introspection: `resolve_name/get/list/info`

Avoid importing JO private modules or private symbols.

## Function Docs Generation

Function docs are generated from `OperaFunction.metadata`:

```bash
python scripts/generate_function_docs.py --out docs/functions
```

Generated artifacts:

- `docs/functions/functions.json`
- `docs/functions/<namespace>/<function>.md`

## Curve publish/runtime cache (namespace manifests + JSON sources)

Use this flow when you have many 1D interpolation curves and want runtime speed:

1. Source of truth:
   - root index manifest (`interpolations.manifest.json`) with namespace list
   - one manifest per namespace (function metadata list)
   - per-function JSON source (`x`/`y` arrays)
2. Precompile once: `jopera init` (bundled library) or `jopera init --manifest ./manifest.json` (custom)
   - optional: `jopera init --namespaces dmdd,interp1` for partial namespace processing
   - optional maintenance: `jopera interp list|validate|add|remove`
3. Runtime auto-registration: `get_global_operas_registry()` registers hot curves as `interp.<curve_id>`
4. Runtime load path uses `index.json` + `*.pkl` only (no source JSON in hot path)

Namespace rule for registered interpolation operators:

- If `namespace` is set in curve item, register as `<namespace>.<curve_id>`
- Else if `metadata.group` (or `group`) is set, register as `<group>.<curve_id>`
- Else fallback to `interp.<curve_id>`

Bundled interpolation manifest library resource:

- `jarvis_operas/manifests/interpolations.manifest.json`

Root index example:

```json
{
  "version": 2,
  "kind": "jarvis_operas_interpolation_namespace_index",
  "namespaces": [
    {
      "namespace": "dmdd",
      "manifest": "dmdd/dmdd.manifest.json"
    }
  ]
}
```

Namespace manifest example:

```json
{
  "namespace": "dmdd",
  "functions": [
    {
      "name": "demo_curve",
      "source": "dmdd/demo_curve.json",
      "kind": "linear",
      "hot": true,
      "metadata": {
        "group": "dmdd"
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

- `math.add(a, b)`
- `stat.chi2_cov(residual, cov)`
- `helper.eggbox(x, y)`
- `math.identity(x)`

All built-ins can be called via sync/async registry APIs and accept scalar, NumPy, and Pandas inputs where applicable.
Example:

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

## Function docs generation (OperaFunction-driven)

Jarvis-Operas can auto-generate per-function docs from `OperaFunction` declarations.

```bash
python scripts/generate_function_docs.py
```

Output is generated under `docs/functions/`:

- `docs/functions/index.md`
- `docs/functions/<namespace>/index.md`
- `docs/functions/<namespace>/<function>.md`
- `docs/functions/functions.json`

Optional filters:

```bash
python scripts/generate_function_docs.py --namespaces math,stat
python scripts/generate_function_docs.py --out docs/custom_functions
```

## Workspace clean

Clean local cache/build artifacts:

```bash
./scripts/clean_workspace.sh
```
