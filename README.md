# Jarvis-Operas

Jarvis-Operas is a standalone operator layer for registering, loading, and calling Python callables.

- Scope: operators only (likelihood, chi2, prior mapping, data transforms, etc.)
- No dependency on Jarvis-HEP internals
- Native async entrypoint for Jarvis-HEP style execution (`await registry.acall(...)`)
- Stable `"<namespace>:<name>"` naming for references from Jarvis-HEP/Jarvis-PLOT YAML

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
jopera info stat:chi2_cov --json
jopera call math:add --kwargs '{"a": 1, "b": 2}'
jopera acall stat:chi2_cov --arg residual=[1.0,-0.5] --arg cov=[[2.0,0.1],[0.1,1.0]]
jopera call helper:eggbox --kwargs '{"inputs":{"x":0.5,"y":0.0}}'
jopera list --namespace math --log-mode info
jopera call math:add --kwargs '{"a": 1, "b": 2}' --log-mode debug
jopera --help-advanced
```

Load user operators directly in CLI:

```bash
jopera load /absolute/path/to/my_ops.py
jopera call my_ops:my_op --user-ops /absolute/path/to/my_ops.py --arg x=10
jopera info my_ops:my_op --json
jopera update /absolute/path/to/my_ops.py
jopera update /absolute/path/to/my_ops.py --function my_op
jopera delete-func my_ops:old_op
jopera delete-func h12345
jopera delete-namespace legacy_ops
```

CLI behavior:

- `jopera` (no args) prints a quick start card
- `jopera list` prints grouped human-readable output by default (`id + name`)
- `jopera list --json` prints machine-readable JSON objects with `id/name/namespace`
- `jopera info <name_or_id>` prints metadata (including a unique id) and a suggested next command
- `jopera load <path>` persists this source path by default for future processes
- `jopera load <path> --session-only` loads without persisting
- `jopera update <path>` updates all functions in script namespaces (default behavior)
- `jopera update <path> --function <name>` updates one function from script
- `jopera delete-func <name_or_id>` / `delete-namespace` remove persisted functions/namespaces

## Core API

```python
from jarvis_operas import get_global_registry

registry = get_global_registry()
registry.call("math:add", a=1, b=2)
registry.call("math:identity", x={"k": 1})
```

Supported namespace convention:

- Built-ins use feature namespaces, e.g.:
  - `math:<name>`
  - `stat:<name>`
  - `helper:<name>`
- User operator files default to `<script_name>:<func_name>`

## Async call (Jarvis-HEP Factory/Module friendly)

```python
result = await registry.acall(
    "stat:chi2_cov",
    residual=[1.0, -0.5],
    cov=[[2.0, 0.1], [0.1, 1.0]],
    observables={"obs": 1.0},
    sample_info={"id": 42},
    cfg={"mode": "demo"},
)
```

Behavior:

- Async operator: awaited directly
- Sync operator: offloaded via `asyncio.to_thread` by default (or custom executor)
- Batch helper concurrency: `await registry.acall_helper_many("eggbox", [...])`

Example batch helper execution for Factory-side concurrent scans:

```python
results = await registry.acall_helper_many(
    "eggbox",
    [
        {"inputs": {"x": 0.1, "y": 0.2}},
        {"inputs": {"x": 0.3, "y": 0.4}},
        {"inputs": {"x": 0.5, "y": 0.6}},
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
from jarvis_operas import OperatorRegistry, load_user_ops

registry = OperatorRegistry()
loaded = load_user_ops("./my_ops.py", registry)
print(loaded)
```

Persistent registration for future Python processes:

```python
from jarvis_operas import get_global_registry, persist_user_ops

persist_user_ops("/absolute/path/to/my_ops.py")
registry = get_global_registry()  # auto-loads persisted sources
```

Persistence store location:

- Default: `~/.jarvis-operas/user_ops.json`
- Override with env: `JARVIS_OPERAS_PERSIST_FILE=/custom/path/user_ops.json`
- The same store keeps persistent delete/update overrides for functions and namespaces

By default, `load_user_ops("./my_ops.py", ...)` uses `my_ops` as namespace, so
`my_op` becomes `my_ops:my_op`.

`my_ops.py` can export operators with either style:

1. Decorator (recommended)

```python
from jarvis_operas import oper

@oper("my_chi2")
def my_chi2(residual, cov, logger=None):
    ...
```

When loaded by `load_user_ops("./my_ops.py", ...)`, this is registered as
`my_ops:my_chi2`.

2. Explicit whitelist

```python
def my_op(x):
    return x

__JARVIS_OPERAS__ = {
    "my_op": my_op,
}
```

## Built-in operators

- `math:add(a, b)`
- `stat:chi2_cov(residual, cov)`
- `helper:eggbox(inputs)` where `inputs` must be `{"x": ..., "y": ...}` (scalar, NumPy, or Pandas)
- `math:identity(x)`

All built-ins can be called via sync/async registry APIs and accept scalar, NumPy, and Pandas inputs where applicable.
Built-ins also support `observables` dict input (Jarvis-HEP style), e.g.:

```python
registry.call("math:add", observables={"a": 1.0, "b": 2.0})
registry.call("stat:chi2_cov", observables={"residual": [1.0, 0.0], "cov": [[2.0, 0.0], [0.0, 1.0]]})
registry.call("helper:eggbox", observables={"x": 0.5, "y": 0.0})
```

## Query registry for external UIs (JHEP/JPlot)

```python
registry.list()
registry.list(namespace="math")
registry.info("stat:chi2_cov")
```

`registry.info(...)` returns metadata, signature, docstring summary, module/qualname, and async flag.
