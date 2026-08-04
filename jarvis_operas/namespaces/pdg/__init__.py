"""PDG particle-data constants namespace (namespace.constant form).

Constants are arity-0 ``OperaFunction`` entries with ``flags={"constant"}``.
There is no ``defs/`` implementation layer — values live in metadata and the
helper-generated ``numpy_impl``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.spec import OperaFunction
from .decls import PDG_DECLARATIONS

DECLARATIONS: tuple[OperaFunction, ...] = PDG_DECLARATIONS


def get_declarations() -> Sequence[OperaFunction]:
    return DECLARATIONS


__all__ = ["DECLARATIONS", "get_declarations"]
