from __future__ import annotations

from pathlib import Path


def test_maintaining_uses_current_module_layout() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    body = (repo_root / "MAINTAINING.md").read_text(encoding="utf-8")

    assert "jarvis_operas/core/spec.py" in body
    assert "jarvis_operas/core/registry.py" in body
    assert "jarvis_operas/namespaces/*/decls/*.py" in body
    assert "scripts/generate_function_docs.py" in body

    assert "jarvis_operas/builtins/core.py" not in body
    assert "jarvis_operas/builtins/__init__.py" not in body
    assert "jarvis_operas/registry.py" not in body


def test_architecture_and_migration_docs_exist() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    architecture = repo_root / "docs" / "architecture.md"
    migration = repo_root / "docs" / "migration.md"

    assert architecture.is_file()
    assert migration.is_file()

    architecture_body = architecture.read_text(encoding="utf-8")
    migration_body = migration.read_text(encoding="utf-8")

    assert "Stable Public Surface" in architecture_body
    assert "Mutation Policy" in architecture_body
    assert "Downstream Integration Guidance (HEP/PLOT)" in architecture_body

    assert "Import Path Updates" in migration_body
    assert "Downstream (Jarvis-HEP / Jarvis-PLOT) Adapter Checklist" in migration_body
