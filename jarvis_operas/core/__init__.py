from __future__ import annotations

from .operas import Operas, get_global_operas, get_global_operas_registry
from .registry import OperasRegistry
from .spec import OperaFunction

__all__ = [
    "OperaFunction",
    "OperasRegistry",
    "Operas",
    "get_global_operas_registry",
    "get_global_operas",
]
