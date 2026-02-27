from __future__ import annotations

from collections.abc import Sequence

from ...core.spec import OperaFunction
from .decls import HELPER_BENCHMARK_DECLARATIONS

DECLARATIONS: tuple[OperaFunction, ...] = HELPER_BENCHMARK_DECLARATIONS


def get_declarations() -> Sequence[OperaFunction]:
    return DECLARATIONS


__all__ = ["DECLARATIONS", "get_declarations"]
