# Jarvis-Operas Migration Notes (1.2.x)

## Scope

This note covers migration to the declaration-driven JO layout (`core + namespaces + catalog`).

## Import Path Updates

Use these public imports:

- `from jarvis_operas import get_global_operas_registry, get_global_operas`
- `from jarvis_operas import OperasRegistry, Operas, OperaFunction`
- `from jarvis_operas import load_user_ops, persist_user_ops`

Do not import old/removed internals:

- `jarvis_operas/builtins/*` (removed)
- `jarvis_operas/registry.py` (moved to `jarvis_operas/core/registry.py`)

## Namespace Declaration Model

Built-in functions are now maintained by namespace:

- implementation: `jarvis_operas/namespaces/<ns>/defs/*.py`
- declaration: `jarvis_operas/namespaces/<ns>/decls/*.py`

Every function is an `OperaFunction` object with metadata.

## Runtime and Async Behavior

- Sync path: `registry.call(...)`
- Async path: `await registry.acall(...)`
- Numpy async calls are non-blocking to the event loop (thread offload + concurrency gate).
- Polars path returns expressions synchronously.

## User Function Persistence

Supported user workflow:

1. `jopera load /path/to/user_ops.py` (or Python `persist_user_ops(...)`)
2. JO stores source snapshots and persistent registry metadata.
3. New Python process loads persisted user functions during global registry bootstrap.

Persistence files:

- sources store: `~/.jarvis-operas/sources.json`
- override store: `~/.jarvis-operas/overrides.json`
- source snapshot dir: `~/.jarvis-operas/user_sources/`

## Downstream (Jarvis-HEP / Jarvis-PLOT) Adapter Checklist

- Replace `get_global_registry()` usage with `get_global_operas_registry()`.
- Use only:
  - `registry.resolve_name/get/list/info`
  - `registry.call/acall`
- Avoid importing JO private state or private helpers.
- Keep downstream alias/wrapper logic outside JO.

## Documentation Pipeline

Per-function docs are generated from `OperaFunction.metadata`:

```bash
python scripts/generate_function_docs.py --out docs/functions
```

Generated output:

- `docs/functions/functions.json`
- `docs/functions/<namespace>/<function>.md`
