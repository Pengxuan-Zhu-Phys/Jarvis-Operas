"""Helpers for declaring namespace constants as arity-0 OperaFunction entries.

Constants are ordinary ``OperaFunction`` values with ``flags={"constant"}`` and
``metadata["value"]``.  The parse-time fold into ``sp.Float`` happens only in
``jarvis_operas.integration.build_sympy_dicts`` when a constants table is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.spec import OperaFunction
from ..units import normalize_unit


def constant_decl(
    namespace: str,
    name: str,
    value: float,
    *,
    unit: str | None = None,
    **meta: Any,
) -> OperaFunction:
    """Build an arity-0 constant declaration.

    ``numpy_impl`` is derived from ``value`` so the callable return and
    ``metadata["value"]`` cannot drift.

    A supplied ``unit`` is migrated into the root output unit contract. The
    legacy ``metadata["unit"]`` display alias is then derived from that contract.
    Constants without a supplied unit keep their original uncontracted behavior.
    """

    value = float(value)
    metadata = {"category": "constant", "value": value, **meta}
    contract = metadata.get("unit_contract")
    if contract is None and unit is not None:
        contract = {
            "version": "1.0",
            "inputs": {},
            "outputs": {
                "": {"unit": unit, "shape": "scalar", "nullable": False},
            },
        }
        metadata["unit_contract"] = contract
    if isinstance(contract, Mapping):
        outputs = contract.get("outputs")
        root = outputs.get("") if isinstance(outputs, Mapping) else None
        canonical_unit = root.get("unit") if isinstance(root, Mapping) else None
        if canonical_unit is not None:
            canonical_unit = normalize_unit(canonical_unit)
            if unit is not None and normalize_unit(unit) != canonical_unit:
                raise ValueError("constant unit conflicts with the root output unit contract")
            metadata["unit"] = canonical_unit
    return OperaFunction(
        namespace=namespace,
        name=name,
        arity=0,
        return_dtype=None,
        numpy_impl=lambda _v=value: _v,
        flags=frozenset({"constant"}),
        metadata=metadata,
    )


__all__ = ["constant_decl"]
