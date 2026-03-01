#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from jarvis_operas.catalog import get_catalog_declarations
from jarvis_operas.core.spec import OperaFunction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-function documentation from OperaFunction declarations.",
    )
    parser.add_argument(
        "--out",
        default="docs/functions",
        help="Output directory for generated docs (default: docs/functions).",
    )
    parser.add_argument(
        "--namespaces",
        default="",
        help="Optional comma-separated namespace filter, e.g. 'math,stat'.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing output directory before generation.",
    )
    return parser.parse_args()


def _callable_path(fn: Any | None) -> str:
    if fn is None:
        return ""
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return str(qualname)
    return ""


def _signature_text(fn: Any | None) -> str:
    if fn is None:
        return "()"
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "()"


def _doc_summary(declaration: OperaFunction) -> str:
    metadata = dict(declaration.metadata)
    summary = metadata.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    fn = declaration.numpy_impl or declaration.polars_expr_impl
    doc = inspect.getdoc(fn) if fn is not None else ""
    if isinstance(doc, str) and doc.strip():
        return doc.strip().splitlines()[0]

    return ""


def _to_record(declaration: OperaFunction) -> dict[str, Any]:
    metadata = dict(declaration.metadata)
    backend_support = metadata.get("backend_support", {})
    return {
        "full_name": declaration.full_name,
        "namespace": declaration.namespace,
        "name": declaration.name,
        "arity": declaration.arity,
        "return_dtype": repr(declaration.return_dtype),
        "supports_numpy": declaration.supports_numpy,
        "supports_polars_native": declaration.supports_polars_native,
        "supports_polars_fallback": declaration.supports_polars_fallback,
        "signature": _signature_text(declaration.numpy_impl or declaration.polars_expr_impl),
        "numpy_impl": _callable_path(declaration.numpy_impl),
        "polars_expr_impl": _callable_path(declaration.polars_expr_impl),
        "summary": _doc_summary(declaration),
        "backend_support": backend_support if isinstance(backend_support, dict) else {},
        "metadata": metadata,
    }


def _ensure_out_dir(out_dir: Path, *, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _function_page_content(record: dict[str, Any]) -> str:
    metadata_json = json.dumps(record["metadata"], ensure_ascii=False, indent=2, sort_keys=True)
    backend_support_json = json.dumps(
        record.get("backend_support", {}),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    summary = record.get("summary", "")
    return (
        f"# `{record['full_name']}`\n\n"
        f"## Summary\n\n"
        f"{summary or 'No summary provided.'}\n\n"
        f"## Signature\n\n"
        f"`{record['signature']}`\n\n"
        f"## Core Fields\n\n"
        f"- `namespace`: `{record['namespace']}`\n"
        f"- `name`: `{record['name']}`\n"
        f"- `arity`: `{record['arity']}`\n"
        f"- `return_dtype`: `{record['return_dtype']}`\n\n"
        f"## Backend Capability\n\n"
        f"- `supports_numpy`: `{record['supports_numpy']}`\n"
        f"- `supports_polars_native`: `{record['supports_polars_native']}`\n"
        f"- `supports_polars_fallback`: `{record['supports_polars_fallback']}`\n\n"
        f"### Backend Support Metadata\n\n"
        f"```json\n{backend_support_json}\n```\n\n"
        f"## Implementation Targets\n\n"
        f"- `numpy_impl`: `{record['numpy_impl'] or 'N/A'}`\n"
        f"- `polars_expr_impl`: `{record['polars_expr_impl'] or 'N/A'}`\n\n"
        f"## Metadata (from `OperaFunction.metadata`)\n\n"
        f"```json\n{metadata_json}\n```\n"
    )


def _namespace_index_content(namespace: str, records: list[dict[str, Any]]) -> str:
    lines = [f"# Namespace `{namespace}`", "", f"Total functions: **{len(records)}**", ""]
    for record in records:
        summary = record.get("summary", "")
        suffix = f" - {summary}" if summary else ""
        lines.append(f"- [`{record['full_name']}`]({record['name']}.md){suffix}")
    lines.append("")
    return "\n".join(lines)


def _root_index_content(records: list[dict[str, Any]]) -> str:
    by_namespace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_namespace[record["namespace"]].append(record)

    lines = [
        "# Jarvis-Operas Function Catalog",
        "",
        "This catalog is auto-generated from `OperaFunction` declarations.",
        "",
        f"Total functions: **{len(records)}**",
        "",
        "## Namespaces",
        "",
    ]
    for namespace in sorted(by_namespace.keys()):
        count = len(by_namespace[namespace])
        lines.append(f"- [`{namespace}`]({namespace}/index.md) ({count})")
    lines.append("")
    return "\n".join(lines)


def generate_function_docs(
    *,
    out_dir: Path,
    declarations: list[OperaFunction],
    clean: bool = True,
) -> None:
    _ensure_out_dir(out_dir, clean=clean)
    records = [_to_record(item) for item in sorted(declarations, key=lambda x: x.full_name)]

    by_namespace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_namespace[record["namespace"]].append(record)

    for namespace, ns_records in by_namespace.items():
        ns_dir = out_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        for record in ns_records:
            _write_text(ns_dir / f"{record['name']}.md", _function_page_content(record))
        _write_text(ns_dir / "index.md", _namespace_index_content(namespace, ns_records))

    _write_text(out_dir / "index.md", _root_index_content(records))
    (out_dir / "functions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    raw_namespaces = [item.strip() for item in str(args.namespaces).split(",") if item.strip()]
    declarations = get_catalog_declarations(
        namespaces=raw_namespaces if raw_namespaces else None,
        include_interpolations=True,
    )

    generate_function_docs(
        out_dir=out_dir,
        declarations=declarations,
        clean=not args.no_clean,
    )

    print(f"Generated {len(declarations)} function docs into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
