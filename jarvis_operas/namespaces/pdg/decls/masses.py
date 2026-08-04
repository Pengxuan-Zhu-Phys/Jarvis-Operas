"""PDG particle masses (GeV)."""

from __future__ import annotations

from ....core.spec import OperaFunction
from ..._constants import constant_decl

_SOURCE = "PDG 2024"
_SINCE = "1.4.0"


def _mass_decl(
    name: str,
    value: float,
    *,
    summary: str,
    tags: list[str] | None = None,
) -> OperaFunction:
    return constant_decl(
        "pdg",
        name,
        value,
        unit="GeV",
        summary=summary,
        source=_SOURCE,
        since=_SINCE,
        tags=["pdg", "mass", "constant", *(tags or [])],
        **{"return": f"{value} GeV"},
        examples=[
            f"pdg.{name}",
            f"jopera call pdg.{name}",
        ],
    )


PDG_MASS_DECLARATIONS: tuple[OperaFunction, ...] = (
    _mass_decl(
        "mZ",
        91.1876,
        summary="Z boson pole mass.",
        tags=["electroweak", "boson"],
    ),
    _mass_decl(
        "mW",
        80.377,
        summary="W boson pole mass.",
        tags=["electroweak", "boson"],
    ),
    _mass_decl(
        "mt",
        172.57,
        summary="Top-quark pole mass.",
        tags=["quark"],
    ),
    _mass_decl(
        "mh",
        125.25,
        summary="Higgs boson mass.",
        tags=["boson", "higgs"],
    ),
    _mass_decl(
        "me",
        5.1099895000e-4,
        summary="Electron mass.",
        tags=["lepton"],
    ),
    _mass_decl(
        "mmu",
        0.1056583755,
        summary="Muon mass.",
        tags=["lepton"],
    ),
    _mass_decl(
        "mtau",
        1.77686,
        summary="Tau lepton mass.",
        tags=["lepton"],
    ),
)
