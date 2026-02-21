from __future__ import annotations

# Compatibility module: keep legacy import path
# `jarvis_operas.builtins.core` while implementations are organized
# by namespace-specific files.
from .helper import eggbox, eggbox2d
from .math import add, identity
from .stat import chi2_cov

__all__ = ["identity", "add", "chi2_cov", "eggbox", "eggbox2d"]
