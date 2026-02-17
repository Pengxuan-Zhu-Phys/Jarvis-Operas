# Jarvis-Operas

Jarvis-Operas is a standalone operator layer for registering, discovering, loading, and calling Python callables.

- Scope: operators only (likelihood, chi2, prior mapping, data transforms, etc.)
- No dependency on Jarvis-HEP internals
- Native async entrypoint for Jarvis-HEP style execution (`await registry.acall(...)`)
- Stable `"<namespace>:<name>"` naming for references from Jarvis-HEP/Jarvis-PLOT YAML

## Install

```bash
pip install .
```

## Core API

```python
from jarvis_operas import get_global_registry

registry = get_global_registry()
registry.call("core:add", a=1, b=2)
registry.call("core:identity", x={"k": 1})
```

Supported namespace convention:

- `core:<name>`: built-in/offical operators
- `user:<name>`: user-defined operators

## Async call (Jarvis-HEP Factory/Module friendly)

```python
result = await registry.acall(
    "core:chi2_cov",
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

## Logger injection

`registry` methods accept optional `logger` and operators can optionally define `logger` argument.

- If no logger is provided, Jarvis-Operas uses `loguru.logger` bound with `module="Jarvis-Operas"`
- If logger is provided, Jarvis-Operas reuses it (no duplicate handler creation)

Utility:

```python
from jarvis_operas import get_logger

logger = get_logger()
```

## Load user operators from file

```python
from jarvis_operas import OperatorRegistry, load_user_ops

registry = OperatorRegistry()
loaded = load_user_ops("./my_ops.py", registry)
print(loaded)
```

`my_ops.py` can export operators with either style:

1. Decorator (recommended)

```python
from jarvis_operas import oper

@oper("my_chi2", namespace="user")
def my_chi2(residual, cov, logger=None):
    ...
```

2. Explicit whitelist

```python
def my_op(x):
    return x

__JARVIS_OPERAS__ = {
    "my_op": my_op,
}
```

## Discover plugins by entry points

```python
from jarvis_operas import discover_entrypoints, get_global_registry

registry = get_global_registry()
discover_entrypoints(registry)
```

Supported groups:

- `jarvis_operas.core`
- `jarvis_operas.user`

## Built-in operators

- `core:identity(x)`
- `core:add(a, b)`
- `core:chi2_cov(residual, cov)`

## Query registry for external UIs (JHEP/JPlot)

```python
registry.list()
registry.list(namespace="core")
registry.info("core:chi2_cov")
```

`registry.info(...)` returns metadata, signature, docstring summary, module/qualname, and async flag.
