from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..core.spec import OperaFunction
from ..curves import build_interpolation_declarations, list_interpolation_namespaces
from ..namespaces.helper import DECLARATIONS as HELPER_DECLARATIONS
from ..namespaces.math import DECLARATIONS as MATH_DECLARATIONS
from ..namespaces.stat import DECLARATIONS as STAT_DECLARATIONS

NAMESPACE_DECLARATIONS: dict[str, Sequence[OperaFunction]] = {
    "math": MATH_DECLARATIONS,
    "stat": STAT_DECLARATIONS,
    "helper": HELPER_DECLARATIONS,
}
# Backward-compat alias for callers importing old symbol name.
NAMESPACE_LOADERS = NAMESPACE_DECLARATIONS


def get_catalog_declarations(
    *,
    namespaces: Iterable[str] | None = None,
    include_interpolations: bool = False,
    interpolation_manifest_path: str | None = None,
    interpolation_source_root: str | None = None,
    interpolation_include_cold: bool = True,
) -> list[OperaFunction]:
    declarations: list[OperaFunction] = []
    static_namespaces: list[str] = []
    interpolation_namespaces: list[str] = []

    interpolation_available: set[str] = set()
    if include_interpolations:
        interpolation_available = set(
            list_interpolation_namespaces(manifest_path=interpolation_manifest_path)
        )

    if namespaces is None:
        static_namespaces = sorted(NAMESPACE_DECLARATIONS.keys())
        interpolation_namespaces = sorted(interpolation_available)
    else:
        selected_namespaces = sorted(
            dict.fromkeys(str(item).strip() for item in namespaces if str(item).strip())
        )
        for namespace in selected_namespaces:
            if namespace in NAMESPACE_DECLARATIONS:
                static_namespaces.append(namespace)
                continue
            if include_interpolations and namespace in interpolation_available:
                interpolation_namespaces.append(namespace)
                continue
            raise KeyError(f"Unknown namespace '{namespace}'")

    for namespace in static_namespaces:
        declarations.extend(NAMESPACE_DECLARATIONS[namespace])

    if include_interpolations and interpolation_namespaces:
        declarations.extend(
            build_interpolation_declarations(
                manifest_path=interpolation_manifest_path,
                namespaces=interpolation_namespaces,
                source_root=interpolation_source_root,
                include_cold=interpolation_include_cold,
            )
        )

    dedup: dict[str, OperaFunction] = {}
    for declaration in declarations:
        dedup[declaration.full_name] = declaration
    return sorted(dedup.values(), key=lambda item: item.full_name)
