from __future__ import annotations

from collections.abc import Sequence

from ...core.spec import OperaFunction
from .decls import MATH_BASIC_DECLARATIONS

DECLARATIONS: tuple[OperaFunction, ...] = MATH_BASIC_DECLARATIONS


def get_declarations() -> Sequence[OperaFunction]:
    return DECLARATIONS


__all__ = ["DECLARATIONS", "get_declarations"]
