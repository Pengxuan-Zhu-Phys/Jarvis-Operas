from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from . import discover_entrypoints, get_global_registry, load_user_ops
from .logging import set_log_mode

_LOG_MODE_CHOICES = ("warning", "info", "debug")


class _JarvisHelpFormatter(argparse.HelpFormatter):
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


def _load_sources(user_ops: list[str], discover: bool) -> list[str]:
    registry = get_global_registry()
    loaded: list[str] = []

    for path in user_ops:
        loaded.extend(load_user_ops(path, registry))

    if discover:
        loaded.extend(discover_entrypoints(registry))

    return sorted(set(loaded))


def _cmd_list(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops, args.discover)
    registry = get_global_registry()
    names = registry.list(namespace=args.namespace)
    _print_value(names, as_json=True)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops, args.discover)
    registry = get_global_registry()
    info = registry.info(args.full_name)
    _print_value(info, as_json=args.json)
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops, args.discover)
    registry = get_global_registry()
    kwargs = _merge_kwargs(args.kwargs, args.arg)

    result = registry.call(args.full_name, **kwargs)
    _print_value(result, as_json=args.json)
    return 0


def _cmd_acall(args: argparse.Namespace) -> int:
    _load_sources(args.user_ops, args.discover)
    registry = get_global_registry()
    kwargs = _merge_kwargs(args.kwargs, args.arg)

    result = asyncio.run(registry.acall(args.full_name, **kwargs))
    _print_value(result, as_json=args.json)
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    loaded = load_user_ops(args.path, registry, namespace=args.namespace)
    _print_value(loaded, as_json=True)
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    registry = get_global_registry()
    loaded = discover_entrypoints(registry)
    _print_value(loaded, as_json=True)
    return 0


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--user-ops",
        action="append",
        default=[],
        help="Path to a Python file with user operators. Can be provided multiple times.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover operators from entry points before executing command.",
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
        help="Full operator name, e.g. core:add or user:my_op",
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
        description="Jarvis-Operas terminal interface",
        formatter_class=_JarvisHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_list = subparsers.add_parser("list", help="List available operators")
    parser_list.add_argument("--namespace", default=None, help="Filter by namespace")
    _add_log_mode_arg(parser_list)
    _add_source_args(parser_list)
    parser_list.set_defaults(func=_cmd_list)

    parser_info = subparsers.add_parser("info", help="Show operator metadata")
    parser_info.add_argument("full_name", help="Full operator name")
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
        default="user",
        help="Namespace for __JARVIS_OPERAS__ exports (default: user)",
    )
    parser_load.set_defaults(func=_cmd_load)

    parser_discover = subparsers.add_parser(
        "discover",
        help="Discover operators from entry points",
    )
    _add_log_mode_arg(parser_discover)
    parser_discover.set_defaults(func=_cmd_discover)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_log_mode(args.log_mode)

    try:
        return args.func(args)
    except Exception as exc:
        print(f"jopera error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
