# Jarvis-Operas Maintainer Guide

This document is for Jarvis-Operas maintainers and defines the workflow for maintaining and releasing built-in operators (for example: `math.*`, `stat.*`, `helper.*`).

## 1. Maintenance Goals

- Keep built-in `<namespace>.<name>` identifiers stable to avoid breaking Jarvis-HEP/Jarvis-PLOT name-based references.
- Keep the registry API stable: `register/get/call/acall/list/info`.
- Ensure tests pass before every release and keep user docs updated.

## 2. Key Files

- `jarvis_operas/core/spec.py`: `OperaFunction` declaration object
- `jarvis_operas/core/registry.py`: registry + dispatch + async behavior
- `jarvis_operas/adapters/polars.py`: polars expr dispatch and fallback wrapper
- `jarvis_operas/catalog/index.py`: static namespace declaration aggregation
- `jarvis_operas/namespaces/*/defs/*.py`: function implementations
- `jarvis_operas/namespaces/*/decls/*.py`: `OperaFunction` declarations
- `jarvis_operas/loading.py`: user source loading
- `jarvis_operas/persistence.py`: persisted user sources/overrides
- `jarvis_operas/logging.py`: logging modes and formatting (`warning/info/debug`)
- `jarvis_operas/cli.py`: `jopera` command implementation
- `scripts/generate_function_docs.py`: auto-generate per-function docs
- `tests/`: pytest coverage
- `pyproject.toml`: version, dependencies, and CLI entry point

## 3. Add a New Built-in Operator

1. Add implementation(s) in `jarvis_operas/namespaces/<namespace>/defs/*.py`.
2. Add declaration(s) in `jarvis_operas/namespaces/<namespace>/decls/*.py` using `OperaFunction`.
3. Ensure declaration metadata is complete (`summary`, `params`, `return`, `backend_support`, `examples`, etc.).
4. Export declaration constants from `jarvis_operas/namespaces/<namespace>/decls/__init__.py`.
5. If introducing a new namespace, add it to `jarvis_operas/catalog/index.py` (`NAMESPACE_DECLARATIONS`).
6. Add tests (at minimum: numpy + polars + `call` + `acall`).
7. Validate locally:

```bash
cd /path/to/Jarvis-Operas
python -m pytest
jopera list --namespace <namespace>
jopera info <namespace>.<name> --json
python scripts/generate_function_docs.py --out docs/functions
```

## 4. Logging Conventions

- Default mode: `warning`
- Optional modes: `info`, `debug`
- Default format: Jarvis-style (`module -> time - [level] >>> message`)

Python API:

```python
from jarvis_operas import set_log_mode
set_log_mode("info")
```

CLI:

```bash
jopera list --namespace math --log-mode info
jopera call math.add --kwargs '{"a":1,"b":2}' --log-mode debug
```

## 5. Write Surface Policy

- Supported user write entry:
  - `jopera load ...` / `persist_user_ops(...)`
- Developer-only write commands (`update`, `delete-func`, `delete-namespace`, `interp add/remove`) require `JARVIS_OPERAS_DEV_WRITE=1`.
- Runtime jobs should treat JO as immutable function provider.

## 6. Pre-release Checklist (Required)

1. Run tests:

```bash
cd /path/to/Jarvis-Operas
python -m pytest
```

2. Regenerate function docs:

```bash
python scripts/generate_function_docs.py --out docs/functions
```

3. Update version:

- Change `[project].version` in `pyproject.toml`
- Version must be incremented (PyPI does not allow re-upload of the same version)

4. Build and package checks:

```bash
rm -rf dist build
find . -maxdepth 1 -name '*.egg-info' -exec rm -rf {} +
python -m build
python -m twine check dist/*
```

## 7. Publish to PyPI

Prerequisite: `~/.pypirc` or `TWINE_USERNAME/TWINE_PASSWORD` is configured.

```bash
python -m twine upload dist/*
```

## 8. GitHub Main + Tag Release

```bash
cd /path/to/Jarvis-Operas

git status
git add .
git commit -m "Release: v0.1.1"
git push jopera main

git tag -a v0.1.1 -m "Release v0.1.1"
git push jopera v0.1.1
```
