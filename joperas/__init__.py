"""Import registered Jarvis-Operas functions by namespace.

The namespace and function name are the same identifiers accepted by
``registry.call`` and ``jopera call``:

    from joperas.math import add
    from joperas.pdg import mZ
    from joperas.dmddxn import LZSI2024

    add(1, 2)
    mZ()
    LZSI2024(30.0)

Calling an imported name uses the global registry, including canonical unit
handling. Built-in operators, interpolation curves, and user operators already
loaded into that registry are all available. A function registered later in the
same process appears on the next attribute lookup.
"""

from __future__ import annotations

import inspect
import keyword
import sys
from difflib import get_close_matches
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Callable

from jarvis_operas.errors import OperatorNotFound
from jarvis_operas.name_utils import normalize_namespace

_PACKAGE = "joperas"
_FINDER_MARK = "_joperas_namespace_finder"
_HIDDEN_PARAMETERS = frozenset({"self", "logger"})


def _registry():
    from jarvis_operas.api import get_global_operas_registry

    return get_global_operas_registry()


def _registered_namespaces() -> list[str]:
    names: set[str] = set()
    for full_name in _registry().list():
        namespace, separator, short_name = full_name.partition(".")
        if separator and namespace and short_name:
            names.add(namespace)
    return sorted(names)


def _short_names(namespace: str) -> list[str]:
    prefix = f"{namespace}."
    names: list[str] = []
    for full_name in _registry().list(namespace=namespace):
        if full_name.startswith(prefix):
            short_name = full_name[len(prefix) :]
            if short_name:
                names.append(short_name)
    return sorted(names)


def _namespace_exists(namespace: str) -> bool:
    try:
        normalize_namespace(namespace)
    except ValueError:
        return False
    return bool(_short_names(namespace))


def _suggest_namespaces(namespace: str, known: list[str]) -> list[str]:
    folded = {item.casefold(): item for item in known}
    folded_name = namespace.casefold()
    if folded_name in folded and folded[folded_name] != namespace:
        return [folded[folded_name]]
    suggestions = get_close_matches(namespace, known, n=3, cutoff=0.5)
    if suggestions:
        return suggestions
    folded_hits = get_close_matches(folded_name, list(folded), n=3, cutoff=0.5)
    return [folded[hit] for hit in folded_hits]


def _is_import_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _public_signature(declaration) -> inspect.Signature:
    """Show the operator arguments, without the registry's logger hook."""

    if "constant" in declaration.flags or declaration.arity == 0:
        return inspect.Signature()
    implementation = declaration.numpy_impl or declaration.polars_expr_impl
    if implementation is None:
        return inspect.Signature()
    try:
        signature = inspect.signature(implementation)
    except (TypeError, ValueError):
        return inspect.Signature()
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.name in _HIDDEN_PARAMETERS:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and parameter.name.startswith("_"):
            continue
        parameters.append(parameter)
    try:
        return signature.replace(parameters=parameters)
    except ValueError:
        return signature


def _documentation(declaration) -> str:
    metadata = declaration.metadata
    summary = metadata.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        note = metadata.get("note")
        summary = note if isinstance(note, str) else ""
    parts = [summary.strip()] if summary.strip() else []
    implementation_doc = inspect.getdoc(declaration.numpy_impl or declaration.polars_expr_impl) or ""
    if implementation_doc and implementation_doc != summary.strip():
        parts.append(implementation_doc)
    parts.append(
        f"Jarvis-Operas function {declaration.full_name}. "
        "Calling it dispatches through the global registry, including canonical units."
    )
    return "\n\n".join(parts)


def _apply_declaration(function: Callable[..., Any], declaration) -> None:
    function.__doc__ = _documentation(declaration)
    function.__signature__ = _public_signature(declaration)  # type: ignore[attr-defined]


def _make_callable(full_name: str, namespace: str, short_name: str) -> Callable[..., Any]:
    def function(*args, **kwargs):
        return _registry().call(full_name, *args, **kwargs)

    function.__name__ = short_name
    function.__qualname__ = short_name
    function.__module__ = f"{_PACKAGE}.{namespace}"
    function.full_name = full_name  # type: ignore[attr-defined]
    function.namespace = namespace  # type: ignore[attr-defined]
    function.name = short_name  # type: ignore[attr-defined]
    return function


class _NamespaceModule(ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name == "__all__":
            return [item for item in _short_names(self.__dict__["_jo_namespace"]) if _is_import_name(item)]
        if name.startswith("_"):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        namespace = self.__dict__["_jo_namespace"]
        full_name = f"{namespace}.{name}"
        try:
            declaration = _registry().get(full_name)
        except OperatorNotFound:
            self.__dict__.get("_jo_functions", {}).pop(name, None)
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}") from None
        cache: dict[str, Callable[..., Any]] = self.__dict__.setdefault("_jo_functions", {})
        function = cache.get(name)
        if function is None:
            function = _make_callable(full_name, namespace, name)
            cache[name] = function
        _apply_declaration(function, declaration)
        return function

    def __dir__(self) -> list[str]:
        names = set(self.__dict__)
        names.discard("_jo_functions")
        names.discard("_jo_namespace")
        names.update(_short_names(self.__dict__["_jo_namespace"]))
        return sorted(names)


class _NamespaceLoader(Loader):
    def create_module(self, spec: ModuleSpec):
        return _NamespaceModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        namespace = module.__name__.split(".", 1)[1]
        module.__dict__["_jo_namespace"] = namespace
        module.__dict__["_jo_functions"] = {}
        module.__doc__ = (
            f"Registered Jarvis-Operas functions in the {namespace!r} namespace.\n\n"
            "Each attribute is a callable. A call uses the global registry, including "
            "canonical unit handling.\n\n"
            f"Example: from joperas.{namespace} import <name>"
        )


class _NamespaceFinder(MetaPathFinder):
    """Serve ``joperas.<namespace>`` from the live operator registry."""

    _joperas_namespace_finder = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        prefix = f"{_PACKAGE}."
        if not fullname.startswith(prefix):
            return None
        namespace = fullname[len(prefix) :]
        if not namespace or "." in namespace:
            return None
        if _namespace_exists(namespace):
            return ModuleSpec(fullname, _NamespaceLoader(), is_package=False)
        known = _registered_namespaces()
        suggestions = _suggest_namespaces(namespace, known)
        message = f"Jarvis-Operas has no namespace {namespace!r}."
        if suggestions:
            message += " Did you mean " + ", ".join(repr(item) for item in suggestions) + "?"
        raise ModuleNotFoundError(message, name=fullname)


def _install_finder() -> None:
    if any(getattr(finder, _FINDER_MARK, False) for finder in sys.meta_path):
        return
    sys.meta_path.append(_NamespaceFinder())


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {_PACKAGE!r} has no attribute {name!r}")
    import importlib

    try:
        return importlib.import_module(f"{_PACKAGE}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(str(exc)) from exc


def __dir__() -> list[str]:
    names = {name for name in globals() if not name.startswith("_")}
    names.update({"__all__", "__doc__", "__name__", "__package__"})
    names.update(_registered_namespaces())
    return sorted(names)


__all__: list[str] = []

_install_finder()
