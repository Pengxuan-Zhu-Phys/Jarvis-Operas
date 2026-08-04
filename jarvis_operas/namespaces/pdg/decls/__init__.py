from __future__ import annotations

from ....core.spec import OperaFunction
from .couplings import PDG_COUPLING_DECLARATIONS
from .masses import PDG_MASS_DECLARATIONS
from .units import PDG_UNIT_DECLARATIONS

PDG_DECLARATIONS: tuple[OperaFunction, ...] = (
    *PDG_MASS_DECLARATIONS,
    *PDG_COUPLING_DECLARATIONS,
    *PDG_UNIT_DECLARATIONS,
)

__all__ = [
    "PDG_DECLARATIONS",
    "PDG_MASS_DECLARATIONS",
    "PDG_COUPLING_DECLARATIONS",
    "PDG_UNIT_DECLARATIONS",
]
