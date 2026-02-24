from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
import textwrap
from importlib import resources
from importlib.metadata import PackageNotFoundError, version as dist_version
from typing import Any

from . import get_global_registry, load_user_ops
from .curves import init_curve_cache, register_hot_curves_in_registry
from .errors import OperatorNotFound
from .integration import refresh_sympy_dicts_if_global_registry
from .logging import configure_cli_logger
from .persistence import (
    apply_persisted_overrides,
    clear_persisted_function_overrides,
    clear_persisted_namespace_overrides,
    delete_persisted_function,
    delete_persisted_namespace,
    persist_user_ops,
)
from .registry import OperatorRegistry

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
          stat.*    statistics operators
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
          jopera call helper.eggbox --kwargs '{"inputs":{"x":0.5,"y":0.0}}'
          jopera load ./my_ops.py
          jopera call my_ops.my_func --arg x=1 --arg y=2
          jopera update ./my_ops.py
          jopera update ./my_ops.py --function my_func
          jopera delete-namespace my_ops
        """
    ).strip()


def _advanced_card() -> str:
    return textwrap.dedent(
        """
        Advanced options:
          -d, --debug         Enable debug logging (same as --log-mode debug)
          --user-ops PATH     Load operators from Python file (repeatable)
          --session-only      (load command) do not persist this source path
          --session-only      (delete/update commands) do not persist override rules
          --arg key=value     Append call kwargs (JSON-decoded when possible)
          --kwargs JSON       Provide call kwargs as JSON object
          --log-mode MODE     warning|info|debug
          --json              Machine-readable output for supported commands
          init --force        Rebuild all curve cache pickle artifacts

        Example:
          jopera call my_ops.my_func --user-ops ./my_ops.py --arg x=1 --arg y=2 --log-mode debug
        """
    ).strip()


def _split_full_name(full_name: str) -> tuple[str, str] | None:
    normalized = full_name.strip()
    if not normalized:
        return None
    if "." in normalized and ":" in normalized:
        return None
    if "." in normalized:
        return normalized.split(".", 1)
    if ":" in normalized:
        return normalized.split(":", 1)
    return None


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


def _print_value(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        return

    if isinstance(value, (dict, list, tuple)):
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        return

    print(value)


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
    return entries


def _group_entries_by_namespace(
    entries: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for entry in entries:
        namespace = entry["namespace"]
        if namespace not in grouped:
            grouped[namespace] = []
            order.append(namespace)
        grouped[namespace].append(entry)
    return [(namespace, grouped[namespace]) for namespace in order]


def _print_list_human(entries: list[dict[str, Any]], namespace_filter: str | None) -> None:
    if not entries:
        if namespace_filter:
            print(f"No operators found in namespace '{namespace_filter}'.")
        else:
            print("No operators found.")
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


def _suggest_next_info_command(full_name: str, namespace: str) -> str:
    if full_name == "helper.eggbox":
        return "jopera call helper.eggbox --kwargs '{\"inputs\":{\"x\":0.5,\"y\":0.0}}'"
    if full_name == "math.add":
        return "jopera call math.add --kwargs '{\"a\":1,\"b\":2}'"
    if namespace == "helper":
        return f"jopera call {full_name} --kwargs '{{\"inputs\":{{\"x\":0.5,\"y\":0.0}}}}'"
    return f"jopera call {full_name} --kwargs '{{}}'"


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
    lines.append("")
    lines.append("Try next:")
    lines.append(
        f"\t{_suggest_next_info_command(info.get('name', ''), info.get('namespace', ''))}"
    )
    print("\n".join(lines))


def _print_not_found_error(full_name: str, names: list[str]) -> None:
    suggestions = difflib.get_close_matches(full_name, names, n=3, cutoff=0.25)
    print(f"Operator '{full_name}' not found.", file=sys.stderr)
    if suggestions:
        print("Did you mean:", file=sys.stderr)
        for item in suggestions:
            print(f"  - {item}", file=sys.stderr)
    split = _split_full_name(full_name)
    if split is not None:
        namespace, _ = split
        print("Try:", file=sys.stderr)
        print(f"  jopera list --namespace {namespace}", file=sys.stderr)
    else:
        print("Try:", file=sys.stderr)
        print("  jopera list", file=sys.stderr)


def _load_sources(user_ops: list[str]) -> list[str]:
    registry = get_global_registry()
    loaded: list[str] = []

    for path in user_ops:
        loaded.extend(load_user_ops(path, registry))

    if user_ops:
        apply_persisted_overrides(registry)
        refresh_sympy_dicts_if_global_registry(registry)

    return sorted(set(loaded))


def _cmd_list(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops)
    registry = get_global_registry()
    names = registry.list(namespace=args.namespace)
    entries = _build_list_entries(registry, names)
    if args.json:
        _print_value(entries, as_json=True)
    else:
        _print_list_human(entries, args.namespace)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops)
    registry = get_global_registry()
    info = registry.info(args.target)
    if args.json:
        _print_value(info, as_json=True)
    else:
        _print_info_human(info)
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops)
    registry = get_global_registry()
    kwargs = _merge_kwargs(args.kwargs, args.arg)

    result = registry.call(args.full_name, **kwargs)
    _print_value(result, as_json=args.json)
    return 0


def _cmd_acall(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops)
    registry = get_global_registry()
    kwargs = _merge_kwargs(args.kwargs, args.arg)

    result = asyncio.run(registry.acall(args.full_name, **kwargs))
    _print_value(result, as_json=args.json)
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    loaded = load_user_ops(args.path, registry, namespace=args.namespace)
    if not args.session_only:
        persist_user_ops(args.path, namespace=args.namespace)
    refresh_sympy_dicts_if_global_registry(registry)
    _print_value(loaded, as_json=True)
    return 0


def _register_from_registry(
    source_registry: OperatorRegistry,
    target_registry: OperatorRegistry,
    full_name: str,
) -> None:
    info = source_registry.info(full_name)
    fn = source_registry.get(full_name)
    target_registry.register(
        name=info["short_name"],
        fn=fn,
        namespace=info["namespace"],
        metadata=info["metadata"],
    )


def _cmd_update(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    staging = OperatorRegistry()
    loaded = load_user_ops(args.path, staging, namespace=args.namespace)

    if args.function:
        targets: list[str] = []
        for full_name in loaded:
            split = _split_full_name(full_name)
            if split is None:
                continue
            namespace, short_name = split
            if short_name != args.function:
                continue
            if args.namespace is not None and namespace != args.namespace:
                continue
            targets.append(full_name)
        if not targets:
            raise ValueError(
                f"Function '{args.function}' was not found in script '{args.path}'."
            )
        if len(targets) > 1:
            raise ValueError(
                f"Function '{args.function}' appears in multiple namespaces. "
                "Use --namespace to disambiguate."
            )
    else:
        targets = list(loaded)

    if not args.function:
        target_namespaces = sorted(
            {
                split[0]
                for name in targets
                for split in [_split_full_name(name)]
                if split is not None
            }
        )
        for namespace in target_namespaces:
            registry.delete_namespace(namespace)
            if not args.session_only:
                clear_persisted_namespace_overrides(namespace)
    else:
        target_namespaces = []

    updated: list[str] = []
    for full_name in targets:
        if args.function:
            try:
                registry.delete(full_name)
            except OperatorNotFound:
                pass
            if not args.session_only:
                clear_persisted_function_overrides(full_name)
                split = _split_full_name(full_name)
                if split is not None:
                    clear_persisted_namespace_overrides(split[0])

        _register_from_registry(staging, registry, full_name)
        updated.append(full_name)

    if not args.session_only:
        persist_user_ops(args.path, namespace=args.namespace)

    refresh_sympy_dicts_if_global_registry(registry)
    _print_value(
        {
            "updated": updated,
            "mode": "function" if args.function else "namespace",
            "session_only": bool(args.session_only),
        },
        as_json=True,
    )
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
    if args.manifest:
        summary = init_curve_cache(
            args.manifest,
            source_root=args.source_root,
            cache_root=args.cache_root,
            index_path=args.index_path,
            force=bool(args.force),
        )
    else:
        resource = resources.files("jarvis_operas").joinpath(
            "manifests",
            "interpolations.manifest.json",
        )
        with resources.as_file(resource) as default_manifest_path:
            summary = init_curve_cache(
                str(default_manifest_path),
                source_root=args.source_root,
                cache_root=args.cache_root,
                index_path=args.index_path,
                force=bool(args.force),
            )
    register_hot_curves_in_registry(
        get_global_registry(),
        index_path=summary["index_path"],
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


def _cmd_delete_func(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    resolved_full_name = registry.resolve_name(args.target)
    registry.delete(args.target)
    if not args.session_only:
        delete_persisted_function(resolved_full_name)
    refresh_sympy_dicts_if_global_registry(registry)
    _print_value({"deleted": resolved_full_name, "session_only": bool(args.session_only)}, as_json=True)
    return 0


def _cmd_delete_namespace(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    deleted = registry.delete_namespace(args.namespace)
    if not args.session_only:
        delete_persisted_namespace(args.namespace)
    refresh_sympy_dicts_if_global_registry(registry)
    _print_value(
        {
            "namespace": args.namespace,
            "deleted": deleted,
            "session_only": bool(args.session_only),
        },
        as_json=True,
    )
    return 0


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
        help="Full operator name, e.g. math.add or my_ops.my_op",
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
    _add_source_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jopera",
        description=textwrap.dedent(
            """
            Jarvis-Operas CLI
            Register, inspect, and run operators by name.
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

    parser_list = subparsers.add_parser("list", help="List available operators")
    parser_list.add_argument("--namespace", default=None, help="Filter by namespace")
    parser_list.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of grouped text.",
    )
    _add_log_mode_arg(parser_list)
    _add_source_args(parser_list)
    parser_list.set_defaults(func=_cmd_list)

    parser_info = subparsers.add_parser("info", help="Show operator metadata")
    parser_info.add_argument("target", help="Full operator name or operator id")
    parser_info.add_argument("--json", action="store_true", help="Force JSON output")
    _add_log_mode_arg(parser_info)
    _add_source_args(parser_info)
    parser_info.set_defaults(func=_cmd_info)

    parser_call = subparsers.add_parser("call", help="Call an operator synchronously")
    _add_log_mode_arg(parser_call)
    _add_call_args(parser_call)
    parser_call.set_defaults(func=_cmd_call)

    parser_acall = subparsers.add_parser("acall", help="Call an operator asynchronously")
    _add_log_mode_arg(parser_acall)
    _add_call_args(parser_acall)
    parser_acall.set_defaults(func=_cmd_acall)

    parser_load = subparsers.add_parser("load", help="Load user operators from a file")
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

    parser_update = subparsers.add_parser(
        "update",
        help="Update operators from a Python file (default: whole namespace)",
    )
    parser_update.add_argument("path", help="Path to Python file")
    _add_log_mode_arg(parser_update)
    parser_update.add_argument(
        "--namespace",
        default=None,
        help="Override namespace. Default derives from script filename.",
    )
    parser_update.add_argument(
        "--function",
        default=None,
        help="Only update one function short name from this script.",
    )
    parser_update.add_argument(
        "--session-only",
        action="store_true",
        help="Update only for this process. Skip persistent override updates.",
    )
    parser_update.set_defaults(func=_cmd_update)

    parser_delete_func = subparsers.add_parser(
        "delete-func",
        help="Delete one function by full operator name or id",
    )
    _add_log_mode_arg(parser_delete_func)
    parser_delete_func.add_argument("target", help="Full operator name or operator id")
    parser_delete_func.add_argument(
        "--session-only",
        action="store_true",
        help="Delete only for this process. Skip persistent deletion rule.",
    )
    parser_delete_func.set_defaults(func=_cmd_delete_func)

    parser_delete_namespace = subparsers.add_parser(
        "delete-namespace",
        help="Delete all functions in a namespace",
    )
    _add_log_mode_arg(parser_delete_namespace)
    parser_delete_namespace.add_argument("namespace", help="Namespace to delete")
    parser_delete_namespace.add_argument(
        "--session-only",
        action="store_true",
        help="Delete only for this process. Skip persistent deletion rule.",
    )
    parser_delete_namespace.set_defaults(func=_cmd_delete_namespace)

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
        registry = get_global_registry()
        _print_not_found_error(exc.full_name, registry.list())
        return 1
    except Exception as exc:
        print(f"jopera error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
