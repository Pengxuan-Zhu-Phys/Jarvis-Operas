# Jarvis-Operas Architecture

## Design Goal

Jarvis-Operas (JO) is a function provider for downstream modules.
JO owns function declaration, registration, and dispatch.
Downstream projects should only depend on JO public APIs.

## Stable Public Surface

Use only these entry points for integration:

- `jarvis_operas.get_global_operas_registry()`
- `jarvis_operas.get_global_operas()`
- `jarvis_operas.OperasRegistry`
- `jarvis_operas.Operas`
- `jarvis_operas.oper` (for declaration-time registration)
- `jarvis_operas.load_user_ops`, `jarvis_operas.persist_user_ops`

Runtime call contract:

- Sync: `registry.call(name, *args, backend="numpy"|"polars", **kwargs)`
- Async: `await registry.acall(name, *args, backend=..., **kwargs)`
- Introspection: `registry.list()`, `registry.get()`, `registry.info()`, `registry.resolve_name()`

## Module Responsibilities

- `jarvis_operas/core/spec.py`
  - `OperaFunction` declaration object (metadata, arity, backend capabilities).
- `jarvis_operas/core/registry.py`
  - registry CRUD, backend dispatch, async gate/semaphore, diagnostics.
- `jarvis_operas/core/operas.py`
  - thin public wrapper (`call`/`acall`) over registry.
- `jarvis_operas/adapters/polars.py`
  - polars expression dispatch and numpy->polars fallback (`map_batches`).
- `jarvis_operas/namespaces/*/defs/*.py`
  - function implementations.
- `jarvis_operas/namespaces/*/decls/*.py`
  - `OperaFunction` declarations with metadata.
- `jarvis_operas/catalog/index.py`
  - static namespace declaration aggregation.
- `jarvis_operas/loading.py`
  - user source loading (`@oper` or `__JARVIS_OPERAS__` map).
- `jarvis_operas/persistence.py`
  - persisted user source and override stores.
- `jarvis_operas/curves.py` and `curve_*.py`
  - interpolation manifest/cache/registration pipeline.

## Mutation Policy

Allowed write paths:

- Developer declarations in source tree (`namespaces/*/decls` and curve manifests).
- User source registration through JO-managed APIs/CLI (`load_user_ops`, `persist_user_ops`, `jopera load`).

Not allowed at runtime:

- Arbitrary registry mutation by downstream jobs after execution starts.
- Direct modification of JO internal/private symbols.

This keeps JO behavior deterministic like a base library.

## Backend and Async Rules

- NumPy backend:
  - executes `numpy_impl`.
  - `acall` offloads sync implementations to `asyncio.to_thread` (or configured executor).
  - registry-level semaphore applies concurrency limit.
  - default concurrency is CPU-scaled (`os.cpu_count() * 3`).
- Polars backend:
  - prefers `polars_expr_impl`.
  - fallback wraps `numpy_impl` via `map_batches` and requires `return_dtype`.
  - expression construction is synchronous by design.

## Downstream Integration Guidance (HEP/PLOT)

- Depend on JO public interface only (`get_global_operas_registry`, `call/acall/list/get/info`).
- Do not import JO private/internal modules.
- Do not rely on import-time side effects.
- Treat JO as a pure function catalog service.
