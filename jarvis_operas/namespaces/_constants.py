"""Helpers for declaring namespace constants as arity-0 OperaFunction entries.

Constants are ordinary ``OperaFunction`` values with ``flags={"constant"}`` and
``metadata["value"]``.  The parse-time fold into ``sp.Float`` happens only in
``jarvis_operas.integration.build_sympy_dicts`` when a constants table is supplied.
"""

from __future__ import annotations

from typing import Any

from ..core.spec import OperaFunction


def constant_decl(
    namespace: str,
    name: str,
    value: float,
    **meta: Any,
) -> OperaFunction:
    """Build an arity-0 constant declaration.

    ``numpy_impl`` is derived from ``value`` so the callable return and
    ``metadata["value"]`` cannot drift.
    """

    value = float(value)
    return OperaFunction(
        namespace=namespace,
        name=name,
        arity=0,
        return_dtype=None,
        numpy_impl=lambda _v=value: _v,
        flags=frozenset({"constant"}),
        metadata={"category": "constant", "value": value, **meta},
    )


__all__ = ["constant_decl"]
