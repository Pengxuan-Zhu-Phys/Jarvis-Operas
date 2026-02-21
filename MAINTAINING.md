# Jarvis-Operas Maintainer Guide

This document is for Jarvis-Operas maintainers and defines the workflow for maintaining and releasing built-in operators (for example: `math:*`, `stat:*`, `helper:*`).

## 1. Maintenance Goals

- Keep built-in `<namespace>:<name>` identifiers stable to avoid breaking Jarvis-HEP/Jarvis-PLOT name-based references.
- Keep the registry API stable: `register/get/call/acall/list/info`.
- Ensure tests pass before every release and keep user docs updated.

## 2. Key Files

- `jarvis_operas/builtins/core.py`: built-in operator implementations
- `jarvis_operas/builtins/__init__.py`: built-in registration list (`BUILTIN_OPERATORS`)
- `jarvis_operas/registry.py`: core registration and execution logic
- `jarvis_operas/logging.py`: logging modes and formatting (`warning/info/debug`)
- `jarvis_operas/cli.py`: `jopera` command implementation
- `tests/`: pytest coverage
- `pyproject.toml`: version, dependencies, and CLI entry point

## 3. Add a New Built-in Operator

1. Add the function in `jarvis_operas/builtins/core.py` (recommended: support optional `logger=None`).
2. Register it in `jarvis_operas/builtins/__init__.py` under `BUILTIN_OPERATORS`:
   - `("namespace", "name", fn, {"category": "..."})`
3. Add tests (at minimum: `call` and `info`; include `acall` when relevant).
4. Validate locally:

```bash
cd /path/to/Jarvis-Operas
python -m pytest
python -m jarvis_operas.cli list --namespace math
python -m jarvis_operas.cli info <namespace>:<name> --json
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
jopera call math:add --kwargs '{"a":1,"b":2}' --log-mode debug
```

## 5. Pre-release Checklist (Required)

```bash
cd /path/to/Jarvis-Operas
python -m pytest
```

Update version:

- Change `[project].version` in `pyproject.toml`
- Version must be incremented (PyPI does not allow re-upload of the same version)

Build and package checks:

```bash
rm -rf dist build
find . -maxdepth 1 -name '*.egg-info' -exec rm -rf {} +
python -m build
python -m twine check dist/*
```

## 6. Publish to PyPI

Prerequisite: `~/.pypirc` or `TWINE_USERNAME/TWINE_PASSWORD` is configured.

```bash
python -m twine upload dist/*
```

## 7. GitHub Main + Tag Release

```bash
cd /path/to/Jarvis-Operas

git status
git add .
git commit -m "Release: v0.1.1"
git push jopera main

git tag -a v0.1.1 -m "Release v0.1.1"
git push jopera v0.1.1
```
