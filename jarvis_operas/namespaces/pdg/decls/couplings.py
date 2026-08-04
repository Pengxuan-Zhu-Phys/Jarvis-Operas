"""PDG electroweak / strong coupling constants."""

from __future__ import annotations

from ....core.spec import OperaFunction
from ..._constants import constant_decl

_SOURCE = "PDG 2024"
_SINCE = "1.4.0"


def _coupling(
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
        tags=["pdg", "coupling", "constant", *(tags or [])],
        **{"return": f"{value} {unit}".strip()},
        examples=[
            f"pdg.{name}",
            f"jopera call pdg.{name}",
        ],
    )


PDG_COUPLING_DECLARATIONS: tuple[OperaFunction, ...] = (
    _coupling(
        "alphaEM",
        7.2973525693e-3,
        unit="1",
        summary="Fine-structure constant α (electromagnetic).",
        tags=["qed"],
    ),
    _coupling(
        "alphaSMZ",
        0.1180,
        unit="1",
        summary="Strong coupling α_s evaluated at the Z pole.",
        tags=["qcd"],
    ),
    _coupling(
        "GF",
        1.1663787e-5,
        unit="GeV^-2",
        summary="Fermi coupling constant.",
        tags=["electroweak"],
    ),
)
