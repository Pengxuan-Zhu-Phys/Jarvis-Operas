from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
import textwrap
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
from typing import Any

from .api import get_global_operas_registry
from .core.registry import OperasRegistry
from .curves import (
    init_builtin_curve_cache,
    init_curve_cache,
    list_interpolation_namespace_entries,
    register_hot_curves_in_registry,
    validate_interpolation_namespace_index,
)
from .errors import OperatorNotFound
from .integration import refresh_sympy_dicts_if_global_registry
from .loading import load_user_ops
from .logging import configure_cli_logger
from .name_utils import try_split_full_name
from .namespace_policy import ensure_user_namespace_allowed, is_protected_namespace
from .persistence import (
    apply_persisted_overrides,
    get_overrides_store_path,
    get_sources_store_path,
    persist_user_ops,
)

_LOG_MODE_CHOICES = ("warning", "info", "debug")


def _resolve_version() -> str:
    try:
        return dist_version("Jarvis-Operas")
    except PackageNotFoundError:
        return "dev"


class _JarvisHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Hide argparse subparser metavar line and keep only concrete commands."""

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            parts: list[str] = []
            self._indent()
            for subaction in self._iter_indented_subactions(action):
                parts.append(self._format_action(subaction))
            self._dedent()
            return "".join(parts)
        return super()._format_action(action)


def _root_card() -> str:
    return textwrap.dedent(
        """
        Jarvis-Operas CLI

        Start here:
          jopera init
          jopera list
          jopera info helper.eggbox
          jopera call math.add --kwargs '{"a":1,"b":2}'
          jopera init --manifest ./manifest.json
          jopera load /path/to/my_ops.py

        Namespaces:
          math.*    numeric helpers
          stat.*    statistics functions
          helper.*  HEP scan helpers

        Use 'jopera --help' for full options, 'jopera help examples' for copy-paste examples.
        """
    ).strip()


def _examples_card() -> str:
    return textwrap.dedent(
        """
        Quick examples:
          jopera init
          jopera init --manifest ./manifest.json
          jopera list
          jopera list --namespace helper
          jopera info stat.chi2_cov
          jopera call math.add --kwargs '{"a":1,"b":2}'
          jopera call helper.eggbox --kwargs '{"x":0.5,"y":0.0}'
          jopera load ./my_ops.py
          jopera call my_ops.my_func --arg x=1 --arg y=2
          jopera interp list
          jopera interp validate
        """
    ).strip()


def _advanced_card() -> str:
    return textwrap.dedent(
        """
        Advanced options:
          -d, --debug         Enable debug logging (same as --log-mode debug)
          --user-ops PATH     Load operators from Python file (repeatable)
          --session-only      (load command) do not persist this source path
          --arg key=value     Append call kwargs (JSON-decoded when possible)
          --kwargs JSON       Provide call kwargs as JSON object
          --backend BACKEND   call/acall backend: numpy|polars (default: numpy)
          --log-mode MODE     warning|info|debug
          --json              Machine-readable output for supported commands
          init --force        Rebuild all curve cache pickle artifacts
          init --namespaces   Comma list of interpolation namespaces (e.g. dmdd,interp1)

        Example:
          jopera call my_ops.my_func --user-ops ./my_ops.py --arg x=1 --arg y=2 --log-mode debug
        """
    ).strip()


def _parse_kv_arg(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Invalid --arg '{raw}'. Use key=value format."
        )
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Argument key cannot be empty.")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    return key, value


def _parse_csv_namespaces(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or None


def _merge_kwargs(raw_json: str | None, key_values: list[tuple[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("--kwargs must decode to a JSON object")
        kwargs.update(parsed)

    for key, value in key_values:
        kwargs[key] = value

    return kwargs


def _to_json_safe(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - numpy is expected but keep CLI resilient
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, np.ndarray):
            return [_to_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return _to_json_safe(value.item())

    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _print_value(value: Any, as_json: bool) -> None:
    safe_value = _to_json_safe(value)
    if as_json:
        print(json.dumps(safe_value, ensure_ascii=False, indent=2, default=str))
        return

    if isinstance(safe_value, (dict, list, tuple)):
        print(json.dumps(safe_value, ensure_ascii=False, indent=2, default=str))
        return

    print(safe_value)


def _build_list_entries(registry, names: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for full_name in names:
        info = registry.info(full_name)
        entries.append(
            {
                "id": info["id"],
                "name": info["name"],
                "namespace": info["namespace"],
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            str(item.get("namespace", "")).lower(),
            str(item.get("name", "")).lower(),
        ),
    )


def _group_entries_by_namespace(
    entries: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        namespace = entry["namespace"]
        if namespace not in grouped:
            grouped[namespace] = []
        grouped[namespace].append(entry)
    ordered_namespaces = sorted(grouped.keys(), key=lambda item: item.lower())
    return [(namespace, grouped[namespace]) for namespace in ordered_namespaces]


def _print_list_human(entries: list[dict[str, Any]], namespace_filter: str | None) -> None:
    if not entries:
        if namespace_filter:
            print(f"No functions found in namespace '{namespace_filter}'.")
        else:
            print("No functions found.")
        return

    lines: list[str] = []
    groups = _group_entries_by_namespace(entries)
    for index, (namespace, items) in enumerate(groups):
        lines.append(f"{namespace} ({len(items)})")
        for item in items:
            lines.append(f"  - {item['id']}\t{item['name']}")
        if index != len(groups) - 1:
            lines.append("")

    print("\n".join(lines))


def _print_interp_list_human(
    entries: list[dict[str, Any]],
    manifest_path: str,
) -> None:
    if not entries:
        print(f"No interpolation namespaces found in: {manifest_path}")
        return

    lines = [
        f"Interpolation namespaces ({len(entries)})",
        f"Manifest:\t{manifest_path}",
        "",
    ]
    for entry in entries:
        line = f"- {entry['namespace']}\t{entry['manifest']}"
        description = entry.get("description")
        if isinstance(description, str) and description.strip():
            line = f"{line}\t# {description.strip()}"
        lines.append(line)
    print("\n".join(lines))


def _extract_cli_example(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    examples = metadata.get("examples")
    if isinstance(examples, list):
        for item in examples:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                cli = item.get("cli")
                if isinstance(cli, str) and cli.strip():
                    return cli.strip()
    if isinstance(examples, dict):
        cli = examples.get("cli")
        if isinstance(cli, str) and cli.strip():
            return cli.strip()
    return None


def _is_user_loaded_function(info: dict[str, Any]) -> bool:
    namespace = info.get("namespace")
    if not isinstance(namespace, str) or not namespace.strip():
        return False
    try:
        return not is_protected_namespace(namespace)
    except Exception:
        return False


def _print_info_human(info: dict[str, Any]) -> None:
    metadata = info.get("metadata", {})
    supports_async = bool(info.get("supports_async", info.get("is_async")))
    note = metadata.get("note") if isinstance(metadata, dict) else None
    lines = [
        f"Name:\t{info.get('name', '')}",
        f"ID:\t{info.get('id', '')}",
        f"Namespace:\t{info.get('namespace', '')}",
        f"Signature:\t{info.get('signature', '')}",
        f"Async:\t{str(supports_async).lower()}",
    ]
    category = metadata.get("category")
    if category:
        lines.append(f"Category:\t{category}")
    doc = info.get("docstring", "")
    if doc:
        lines.append(f"Summary:\t{doc}")
    if isinstance(note, str) and note.strip():
        lines.append(f"Note:\t{note.strip()}")
    if _is_user_loaded_function(info):
        lines.append(f"UserSources:\t{get_sources_store_path()}")
        lines.append(f"UserOverrides:\t{get_overrides_store_path()}")
    suggested = _extract_cli_example(metadata)
    if isinstance(suggested, str) and suggested.strip():
        lines.append("")
        lines.append("Try next:")
        lines.append(f"\t{suggested.strip()}")
    print("\n".join(lines))


def _print_not_found_error(full_name: str, names: list[str]) -> None:
    suggestions = difflib.get_close_matches(full_name, names, n=3, cutoff=0.25)
    print(f"Function '{full_name}' not found.", file=sys.stderr)
    if suggestions:
        print("Did you mean:", file=sys.stderr)
        for item in suggestions:
            print(f"  - {item}", file=sys.stderr)
    split = try_split_full_name(full_name)
    if split is not None:
        namespace, _ = split
        print("Try:", file=sys.stderr)
        print(f"  jopera list --namespace {namespace}", file=sys.stderr)
    else:
        print("Try:", file=sys.stderr)
        print("  jopera list", file=sys.stderr)


def _load_sources(user_ops: list[str]) -> list[str]:
    registry = get_global_operas_registry()
    loaded: list[str] = []

    for path in user_ops:
        loaded.extend(load_user_ops(path, registry, refresh_sympy=False))

    if user_ops:
        apply_persisted_overrides(registry)
        refresh_sympy_dicts_if_global_registry(registry)

    return sorted(set(loaded))


def _registry_with_user_sources(user_ops: list[str]) -> OperasRegistry:
    _load_sources(user_ops)
    return get_global_operas_registry()


def _ensure_optional_user_namespace(namespace: str | None, *, action: str) -> None:
    if namespace is None:
        return
    ensure_user_namespace_allowed(
        namespace,
        action=action,
    )


def _refresh_global_registry_views(registry: OperasRegistry) -> None:
    refresh_sympy_dicts_if_global_registry(registry)


def _cmd_list(args: argparse.Namespace) -> int:
    registry = _registry_with_user_sources(args.user_ops)
    names = registry.list(namespace=args.namespace)
    entries = _build_list_entries(registry, names)
    if args.json:
        _print_value(entries, as_json=True)
    else:
        _print_list_human(entries, args.namespace)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    registry = _registry_with_user_sources(args.user_ops)
    info = registry.info(args.target)
    if args.json and _is_user_loaded_function(info):
        info = dict(info)
        info["user_store"] = {
            "sources_path": str(get_sources_store_path()),
            "overrides_path": str(get_overrides_store_path()),
        }
    if args.json:
        _print_value(info, as_json=True)
    else:
        _print_info_human(info)
    return 0


def _cmd_call_common(args: argparse.Namespace, *, async_mode: bool) -> int:
    registry = _registry_with_user_sources(args.user_ops)
    kwargs = _merge_kwargs(args.kwargs, args.arg)

    if async_mode:
        result = asyncio.run(registry.acall(args.full_name, backend=args.backend, **kwargs))
    else:
        result = registry.call(args.full_name, backend=args.backend, **kwargs)
    _print_value(result, as_json=args.json)
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    return _cmd_call_common(args, async_mode=False)


def _cmd_acall(args: argparse.Namespace) -> int:
    return _cmd_call_common(args, async_mode=True)


def _cmd_load(args: argparse.Namespace) -> int:
    registry = get_global_operas_registry()
    _ensure_optional_user_namespace(
        args.namespace,
        action="CLI load namespace override",
    )
    loaded = load_user_ops(
        args.path,
        registry,
        namespace=args.namespace,
        refresh_sympy=False,
    )
    if not args.session_only:
        persist_user_ops(args.path, namespace=args.namespace)
    _refresh_global_registry_views(registry)
    _print_value(loaded, as_json=True)
    return 0


def _cmd_help(args: argparse.Namespace) -> int:
    topic = args.topic or "examples"
    if topic == "examples":
        print(_examples_card())
        return 0
    if topic == "advanced":
        print(_advanced_card())
        return 0

    print(f"Unknown help topic: {topic}", file=sys.stderr)
    return 1


def _cmd_init(args: argparse.Namespace) -> int:
    selected_namespaces = _parse_csv_namespaces(args.namespaces)
    if args.manifest:
        summary = init_curve_cache(
            args.manifest,
            source_root=args.source_root,
            cache_root=args.cache_root,
            index_path=args.index_path,
            namespaces=selected_namespaces,
            force=bool(args.force),
        )
    else:
        summary = init_builtin_curve_cache(
            source_root=args.source_root,
            cache_root=args.cache_root,
            index_path=args.index_path,
            namespaces=selected_namespaces,
            force=bool(args.force),
        )
    register_hot_curves_in_registry(
        get_global_operas_registry(),
        index_path=summary["index_path"],
        namespaces=selected_namespaces,
    )
    if args.json:
        _print_value(summary, as_json=True)
        return 0

    print(f"Manifest: {summary['manifest_path']}")
    print(f"Index:    {summary['index_path']}")
    print(f"Total:    {summary['total']}")
    print(f"Compiled: {len(summary['compiled'])}")
    print(f"Cached:   {len(summary['cached'])}")
    return 0


def _cmd_interp_list(args: argparse.Namespace) -> int:
    entries = list_interpolation_namespace_entries(manifest_path=args.manifest)
    display_manifest = args.manifest or "<built-in>"
    if args.json:
        _print_value(
            {
                "manifest_path": display_manifest,
                "namespace_count": len(entries),
                "namespaces": entries,
            },
            as_json=True,
        )
        return 0

    _print_interp_list_human(entries, manifest_path=display_manifest)
    return 0


def _cmd_interp_validate(args: argparse.Namespace) -> int:
    report = validate_interpolation_namespace_index(manifest_path=args.manifest)
    if args.json:
        _print_value(report, as_json=True)
    else:
        status = "ok" if report.get("ok") else "failed"
        print(f"Validation: {status}")
        print(f"Manifest:   {report.get('manifest_path')}")
        print(f"Namespaces: {report.get('namespace_count')}")
        print(f"Functions:  {report.get('function_count')}")
        errors = report.get("errors", [])
        if isinstance(errors, list) and errors:
            print("")
            print("Errors:")
            for item in errors:
                print(f"- {item}")
    return 0 if bool(report.get("ok")) else 1


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--user-ops",
        action="append",
        default=[],
        help="Path to a Python file with user operators. Can be provided multiple times.",
    )


def _add_log_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-mode",
        choices=_LOG_MODE_CHOICES,
        default="warning",
        help="Console log verbosity (default: warning).",
    )


def _add_call_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "full_name",
        help="Full function name, e.g. math.add or my_ops.my_op",
    )
    parser.add_argument(
        "--kwargs",
        default=None,
        help="JSON object for operator kwargs, e.g. '{\"a\": 1, \"b\": 2}'",
    )
    parser.add_argument(
        "--arg",
        action="append",
        type=_parse_kv_arg,
        default=[],
        help="Additional key=value argument. Value uses JSON decode when possible.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Force JSON output formatting.",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "polars"),
        default="numpy",
        help="Execution backend for this call (default: numpy).",
    )
    _add_source_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jopera",
        description=textwrap.dedent(
            """
            Jarvis-Operas CLI
            Register, inspect, and run functions by name.
            """
        ).strip(),
        epilog="Use 'jopera help examples' for copy-paste examples.",
        formatter_class=_JarvisHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {_resolve_version()}",
        help="Show jopera version and exit.",
    )
    parser.add_argument(
        "--help-advanced",
        action="store_true",
        help="Show advanced options and exit.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (equivalent to --log-mode debug).",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=False,
        title="commands",
    )

    parser_list = subparsers.add_parser("list", help="List available functions")
    parser_list.add_argument("--namespace", default=None, help="Filter by namespace")
    parser_list.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of grouped text.",
    )
    _add_log_mode_arg(parser_list)
    _add_source_args(parser_list)
    parser_list.set_defaults(func=_cmd_list)

    parser_info = subparsers.add_parser("info", help="Show function metadata")
    parser_info.add_argument("target", help="Full function name or function id")
    parser_info.add_argument("--json", action="store_true", help="Force JSON output")
    _add_log_mode_arg(parser_info)
    _add_source_args(parser_info)
    parser_info.set_defaults(func=_cmd_info)

    parser_call = subparsers.add_parser("call", help="Call a function synchronously")
    _add_log_mode_arg(parser_call)
    _add_call_args(parser_call)
    parser_call.set_defaults(func=_cmd_call)

    parser_acall = subparsers.add_parser("acall", help="Call a function asynchronously")
    _add_log_mode_arg(parser_acall)
    _add_call_args(parser_acall)
    parser_acall.set_defaults(func=_cmd_acall)

    parser_load = subparsers.add_parser("load", help="Load user functions from a file")
    parser_load.add_argument("path", help="Path to Python file")
    _add_log_mode_arg(parser_load)
    parser_load.add_argument(
        "--namespace",
        default=None,
        help="Override namespace. Default derives from script filename.",
    )
    parser_load.add_argument(
        "--session-only",
        action="store_true",
        help="Load only for this process. Skip persistent registration.",
    )
    parser_load.set_defaults(func=_cmd_load)

    parser_init = subparsers.add_parser(
        "init",
        help="Compile curve manifest into local runtime cache",
    )
    _add_log_mode_arg(parser_init)
    parser_init.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSON defining curves.",
    )
    parser_init.add_argument(
        "--namespaces",
        default=None,
        help="Optional comma-separated interpolation namespaces to process.",
    )
    parser_init.add_argument(
        "--source-root",
        default=None,
        help="Optional base directory for relative curve JSON paths.",
    )
    parser_init.add_argument(
        "--cache-root",
        default=None,
        help="Directory for generated pickle cache and index.json.",
    )
    parser_init.add_argument(
        "--index-path",
        default=None,
        help="Optional explicit index.json path (overrides --cache-root).",
    )
    parser_init.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even when hash matches existing cache.",
    )
    parser_init.add_argument(
        "--json",
        action="store_true",
        help="Output JSON summary.",
    )
    parser_init.set_defaults(func=_cmd_init)

    parser_interp = subparsers.add_parser(
        "interp",
        help="Manage interpolation namespace manifest index",
    )
    _add_log_mode_arg(parser_interp)
    interp_subparsers = parser_interp.add_subparsers(
        dest="interp_command",
        required=True,
        title="interp commands",
    )

    parser_interp_list = interp_subparsers.add_parser(
        "list",
        help="List interpolation namespaces from root manifest",
    )
    parser_interp_list.add_argument(
        "--manifest",
        default=None,
        help="Path to root interpolation manifest index (default: built-in).",
    )
    parser_interp_list.add_argument(
        "--json",
        action="store_true",
        help="Output JSON summary.",
    )
    parser_interp_list.set_defaults(func=_cmd_interp_list)

    parser_interp_validate = interp_subparsers.add_parser(
        "validate",
        help="Validate root interpolation manifest index and namespace manifests",
    )
    parser_interp_validate.add_argument(
        "--manifest",
        default=None,
        help="Path to root interpolation manifest index (default: built-in).",
    )
    parser_interp_validate.add_argument(
        "--json",
        action="store_true",
        help="Output JSON validation report.",
    )
    parser_interp_validate.set_defaults(func=_cmd_interp_validate)

    parser_help = subparsers.add_parser(
        "help",
        help="Show quick examples and tips",
    )
    parser_help.add_argument(
        "topic",
        nargs="?",
        default="examples",
        choices=("examples", "advanced"),
        help="Help topic",
    )
    parser_help.set_defaults(func=_cmd_help)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    force_debug = False
    parse_argv: list[str] = []
    for token in raw_argv:
        if token in ("-d", "--debug"):
            force_debug = True
            continue
        parse_argv.append(token)

    parser = build_parser()
    args = parser.parse_args(parse_argv)

    if args.help_advanced:
        print(_advanced_card())
        return 0

    if args.command is None:
        print(_root_card())
        return 0

    selected_mode = "debug" if force_debug else getattr(args, "log_mode", "warning")
    configure_cli_logger(selected_mode)

    try:
        return args.func(args)
    except OperatorNotFound as exc:
        registry = get_global_operas_registry()
        _print_not_found_error(exc.full_name, registry.list())
        return 1
    except Exception as exc:
        print(f"jopera error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
