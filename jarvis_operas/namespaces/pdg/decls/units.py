"""Conversion / unit constants commonly used with PDG masses."""

from __future__ import annotations

from ....core.spec import OperaFunction
from ..._constants import constant_decl

_SOURCE = "PDG 2024"
_SINCE = "1.4.0"


def _unit_const(
    name: str,
    value: float,
    *,
    unit: str,
    summary: str,
    tags: list[str] | None = None,
) -> OperaFunction:
    return constant_decl(
        "pdg",
        name,
        value,
        unit=unit,
        summary=summary,
        source=_SOURCE,
        since=_SINCE,
        tags=["pdg", "unit", "constant", *(tags or [])],
        **{"return": f"{value} {unit}".strip()},
        examples=[
            f"pdg.{name}",
            f"jopera call pdg.{name}",
        ],
    )


PDG_UNIT_DECLARATIONS: tuple[OperaFunction, ...] = (
    _unit_const(
        "hbarc",
        197.3269804,
        unit="MeV·fm",
        summary="ħc conversion constant (MeV·fm). One canonical unit only; no auto-conversion.",
        tags=["conversion"],
    ),
    _unit_const(
        "c",
        299792458.0,
        unit="m/s",
        summary="Speed of light in vacuum.",
        tags=["fundamental"],
    ),
)
