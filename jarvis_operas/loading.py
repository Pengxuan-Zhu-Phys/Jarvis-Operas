from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Mapping

from .core.registry import OperasRegistry
from .core.spec import OperaFunction
from .errors import OperatorLoadError
from .logging import get_logger
from .name_utils import namespace_from_source_path
from .namespace_policy import ensure_user_namespace_allowed
from .registration import make_operafunction, use_registry


def load_user_ops(
    path: str,
    registry: OperasRegistry,
    *,
    namespace: str | None = None,
    refresh_sympy: bool = True,
    logger: Any | None = None,
) -> list[str]:
    """Load user operators from a Python file via decorator or __JARVIS_OPERAS__."""

    local_logger = get_logger(logger, action="load_user_ops")
    module_path = Path(path).expanduser().resolve()

    if not module_path.exists() or not module_path.is_file():
        raise OperatorLoadError(
            str(module_path),
            f"User operator file does not exist: {module_path}",
        )

    try:
        target_namespace = ensure_user_namespace_allowed(
            namespace or namespace_from_source_path(module_path),
            action="user operator loading",
        )
        before_ops = set(registry.list())
        module_name = _build_module_name(module_path)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create import spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with use_registry(registry, default_namespace=target_namespace):
                spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        loaded = list(set(registry.list()) - before_ops)

        export_map = getattr(module, "__JARVIS_OPERAS__", None)
        if export_map is not None:
            if not isinstance(export_map, Mapping):
                raise TypeError("__JARVIS_OPERAS__ must be a mapping of name -> callable")

            for op_name, fn in export_map.items():
                loaded_name = _register_user_target(
                    registry=registry,
                    namespace=target_namespace,
                    name=str(op_name),
                    target=fn,
                    metadata={"source": "__JARVIS_OPERAS__", "path": str(module_path)},
                )
                loaded.append(loaded_name)

        if not loaded:
            raise RuntimeError(
                "No operators were loaded. Use @oper(...) or define __JARVIS_OPERAS__."
            )

        loaded_sorted = sorted(set(loaded))
        if refresh_sympy:
            from .integration import refresh_sympy_dicts_if_global_registry

            refresh_sympy_dicts_if_global_registry(registry)
        local_logger.info(
            "loaded {} operators from {}",
            len(loaded_sorted),
            str(module_path),
        )
        return loaded_sorted
    except Exception as exc:
        raise OperatorLoadError(
            str(module_path),
            f"Failed to load operators from '{module_path}': {exc}",
        ) from exc


def discover_entrypoints(
    registry: OperasRegistry,
    logger: Any | None = None,
) -> list[str]:
    """Discover and register operators from entry points."""

    local_logger = get_logger(logger, action="discover_entrypoints")
    loaded: list[str] = []

    for namespace, group in (("core", "jarvis_operas.core"), ("user", "jarvis_operas.user")):
        for ep in _iter_group_entrypoints(group):
            try:
                target = ep.load()
                dist_name = getattr(getattr(ep, "dist", None), "name", None)
                metadata = {
                    "entry_point": ep.value,
                    "group": group,
                }
                if dist_name:
                    metadata["distribution"] = dist_name

                if isinstance(target, Mapping):
                    for op_name, fn in target.items():
                        loaded_name = _register_user_target(
                            registry=registry,
                            namespace=namespace,
                            name=str(op_name),
                            target=fn,
                            metadata=metadata,
                        )
                        loaded.append(loaded_name)
                elif callable(target):
                    loaded_name = _register_user_target(
                        registry=registry,
                        namespace=namespace,
                        name=ep.name,
                        target=target,
                        metadata=metadata,
                    )
                    loaded.append(loaded_name)
                else:
                    raise TypeError(
                        f"Entry point '{ep.name}' in group '{group}' must load a callable "
                        "or mapping of callable operators"
                    )
            except Exception as exc:
                raise OperatorLoadError(
                    group,
                    f"Failed to load entry point '{ep.name}' from '{group}': {exc}",
                ) from exc

    local_logger.info("discovered {} operators from entry points", len(loaded))
    return sorted(loaded)


def _build_module_name(module_path: Path) -> str:
    digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    return f"_jarvis_operas_user_{digest}_{time.time_ns()}"


def _iter_group_entrypoints(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def _register_user_target(
    *,
    registry: OperasRegistry,
    namespace: str,
    name: str,
    target: Any,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    declaration = _to_declaration(
        namespace=namespace,
        name=name,
        target=target,
        metadata=metadata,
    )
    registry.register(declaration)
    return declaration.full_name


def _to_declaration(
    *,
    namespace: str,
    name: str,
    target: Any,
    metadata: Mapping[str, Any] | None = None,
) -> OperaFunction:
    if isinstance(target, OperaFunction):
        merged_metadata = dict(target.metadata)
        if metadata:
            merged_metadata.update(dict(metadata))
        if target.namespace == namespace and target.name == name and merged_metadata == dict(
            target.metadata
        ):
            return target
        return OperaFunction(
            namespace=namespace,
            name=name,
            arity=target.arity,
            return_dtype=target.return_dtype,
            numpy_impl=target.numpy_impl,
            polars_expr_impl=target.polars_expr_impl,
            flags=target.flags,
            metadata=merged_metadata,
        )

    if not callable(target):
        raise TypeError("target must be callable or OperaFunction")
    return make_operafunction(
        namespace=namespace,
        name=name,
        fn=target,
        metadata=metadata,
    )
